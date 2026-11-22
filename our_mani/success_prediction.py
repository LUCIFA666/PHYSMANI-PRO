"""Controller-conditioned success prediction with explicit outcome semantics.

The initial deployment model is conditioned on local handover state. A future
context-conditioned checkpoint must identify the frozen history encoder and
cannot silently substitute zeros for missing contexts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


OUTCOMES = {
    'stable_lift': 'actual contact and >=8cm lift for >=0.3s within capture budget',
    'official_task_success_immediate': 'released task.success() is true at any observed step of a full local rollout',
    'official_task_success': 'native task success sustained >=0.3s in a full local rollout',
    'stable_lift_and_official_task_success': 'both stable_lift and official_task_success recorded in a full local rollout',
}


def outcome_label(row, outcome, *, dataset_full_task=False):
    if outcome not in OUTCOMES:
        raise ValueError('unknown success outcome')
    if outcome != 'stable_lift' and row.get('full_task_executed', dataset_full_task) is not True:
        raise ValueError('censored lift-only rollout has no full-task failure label')
    keys = ('stable_lift', 'official_task_success') if outcome == 'stable_lift_and_official_task_success' else (outcome,)
    if any(type(row.get(key)) is not bool for key in keys):
        raise ValueError('success labels must be explicitly recorded booleans')
    return float(all(row[key] for key in keys))


def read_rollouts(directory, *, split, outcome):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
    rows, paths = [], []
    for path in sorted(directory.glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('initial_features') is None or row.get('failure') == 'deadline_stop':
            continue
        if row.get('split') != split or row.get('formal_test_used', False):
            raise ValueError('success predictor data have a forbidden split')
        outcome_label(row, outcome, dataset_full_task=report.get('full_task_executed') is True)
        rows.append(row)
        paths.append(path)
    if not rows:
        raise ValueError('no complete rollout outcomes')
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('duplicate rollout identity')
    identity = hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()
    return rows, identity, report.get('full_task_executed') is True


def group_partitions(rows, seed=810):
    """Disjoint TRAIN fit/selection/calibration partitions by raw episode."""
    rng = np.random.default_rng(seed)
    groups = sorted({(r['task'], r['source_episode']) for r in rows})
    result = {'fit': set(), 'select': set(), 'calibration': set()}
    for task in sorted({g[0] for g in groups}):
        selected = [g for g in groups if g[0] == task]
        if len(selected) < 5:
            raise ValueError('at least five independent TRAIN episodes per task are required')
        ordered = [selected[i] for i in rng.permutation(len(selected))]
        n = max(1, len(ordered)//5)
        result['select'].update(ordered[:n])
        result['calibration'].update(ordered[n:2*n])
        result['fit'].update(ordered[2*n:])
    return result


def feature_vector(row, context_dim=0):
    local = np.asarray(row['initial_features'], dtype=np.float64)
    if local.shape != (18,) or not np.isfinite(local).all():
        raise ValueError('expected 18 finite local handover features')
    if not context_dim:
        return local
    context = row.get('dynamics_context', {})
    values = np.asarray(context.get('values', []), dtype=np.float64)
    if values.shape != (context_dim,) or not np.isfinite(values).all():
        raise ValueError('missing or invalid causal dynamics context')
    if not context.get('encoder_sha256') or context.get('source') != 'past_observations':
        raise ValueError('context requires a frozen causal history encoder receipt')
    cutoff, decision = context.get('observed_until'), row.get('decision_timestamp')
    if cutoff is None or decision is None or not np.isfinite([cutoff, decision]).all() or cutoff > decision+1e-8:
        raise ValueError('context must not contain future observations')
    return np.r_[local, values]


class SuccessProbability:
    def __init__(self, checkpoint, *, controller_sha256=None):
        import torch
        from .local_learning import MLP
        self.metadata = checkpoint['metadata']
        if controller_sha256 is not None and self.metadata['controller_sha256'] != controller_sha256:
            raise ValueError('success probabilities belong to a different controller')
        self.context_dim = int(self.metadata.get('context_dim', 0))
        self.mean, self.std = np.asarray(checkpoint['feature_mean']), np.asarray(checkpoint['feature_std'])
        self.model = MLP(18+self.context_dim, 1, checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.temperature = float(checkpoint['temperature'])
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('invalid probability calibration temperature')

    @classmethod
    def load(cls, path, **kwargs):
        import torch
        return cls(torch.load(Path(path), map_location='cpu', weights_only=False), **kwargs)

    def __call__(self, local_features, *, context=None, context_encoder_sha256=None):
        import torch
        local = np.asarray(local_features, dtype=np.float64)
        if local.ndim not in (1, 2) or local.shape[-1] != 18:
            raise ValueError('expected local handover features with final dimension 18')
        if self.context_dim:
            values = np.asarray(context, dtype=np.float64)
            if values.shape != (*local.shape[:-1], self.context_dim):
                raise ValueError('trained dynamics context is required')
            if context_encoder_sha256 != self.metadata['context_encoder_sha256']:
                raise ValueError('context encoder differs from success model training')
            local = np.concatenate([local, values], axis=-1)
        elif context is not None:
            raise ValueError('this baseline checkpoint was not trained with dynamics context')
        if not np.isfinite(local).all():
            raise ValueError('non-finite success predictor features')
        with torch.inference_mode():
            x = torch.tensor((local-self.mean)/self.std, dtype=torch.float32)
            return torch.sigmoid(self.model(x)/self.temperature).numpy()[..., 0]
