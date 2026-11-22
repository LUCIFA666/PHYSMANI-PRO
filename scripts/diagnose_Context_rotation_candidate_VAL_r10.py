"""Read-only full VAL comparison of completed same-data rotation candidates.

Uses all existing causal validation windows, with the original masks and parent
weights. No checkpoint is changed, fitted or selected and no simulator runs.
"""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.context_se3 import MultiBodyContextDynamics, rotation_from_6d, so3_exp
from our_mani.multitask.fitting import batch_tensors
from our_mani.multitask.track_dataset import read_tracks, examples


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('diagnostic paths must remain in our-mani')
    return value


@torch.inference_mode()
def run(checkpoint, manifest, output):
    checkpoint, manifest, output = map(local, (checkpoint, manifest, output))
    if output.exists():
        raise ValueError('fresh diagnostic namespace required')
    paths = [checkpoint, checkpoint.parent / 'report.json', manifest, Path(__file__).resolve(),
             ROOT / 'our_mani/multitask/context_se3.py', ROOT / 'our_mani/multitask/fitting.py',
             ROOT / 'our_mani/multitask/track_dataset.py']
    files = {str(p): sha(p) for p in paths}
    fitted = MultiBodyContextDynamics(checkpoint)
    model = fitted.model
    if (fitted.metadata['completed_epochs'] != 30 or fitted.metadata['VAL_source_episodes'] != 275
            or fitted.metadata['training_source_manifest']['source_episode_count'] != 1400
            or fitted.metadata['track_manifest_sha256'] != sha(manifest)):
        raise ValueError('completed full30/1400/275 candidate required')
    torch.set_num_threads(4)
    _, episodes = read_tracks(ROOT, manifest)
    selected = [e for e in episodes if e.entry['split'] == 'validation']
    for episode in selected:
        path = local(episode.entry['cache'])
        if sha(path) != episode.entry['cache_sha256']:
            raise ValueError('validation cache differs')
        files[str(path)] = episode.entry['cache_sha256']
    samples, coverage = examples(selected, model, steps=22)
    if len(coverage) != 275 or any(row['source_key'][0] != 'validation' for row in samples):
        raise ValueError('all275 validation parents required; no TRAIN/TEST inference')
    parents = defaultdict(lambda: defaultdict(list))
    bodies = defaultdict(lambda: defaultdict(list))
    tasks = defaultdict(lambda: defaultdict(list))
    for start in range(0, len(samples), 256):
        rows = samples[start:start + 256]
        data = batch_tensors(samples, range(start, start + len(rows)), 'cpu')
        prediction = model(data['history'], data['length'], data['state'], steps=22)
        state = data['state']; dt = torch.arange(23, dtype=torch.float32) * .05
        p0 = state[:, :3] * model.position_scale + model.position_center
        v0 = state[:, 9:12] * model.velocity_scale
        r0 = rotation_from_6d(state[:, 3:9])
        w0 = state[:, 12:15] * model.angular_velocity_scale
        forecasts = {'context': prediction, 'CV': {
            'positions': p0[:, None] + dt[None, :, None] * v0[:, None],
            'rotations': so3_exp(dt[None, :, None] * w0[:, None]) @ r0[:, None]}}
        for name, value in forecasts.items():
            distance = torch.linalg.vector_norm(value['positions'] - data['targets']['positions'], dim=-1)
            relative = value['rotations'] @ data['targets']['rotations'].transpose(-1, -2)
            angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
            for k in (10, 22):
                for i, row in enumerate(rows):
                    if not bool(data['mask'][i, k]):
                        continue
                    error = (float(distance[i, k]), float(angle[i, k])
                             if bool(data['rotation_mask'][i, k]) else None)
                    parent = row['source_key']
                    parents[(name, k)][parent].append(error)
                    tasks[(name, k, parent[1])][parent].append(error)
                    bodies[(name, k, parent[1], row['body_id'])][parent].append(error)

    def summary(values):
        position = [np.mean([x[0] for x in rows]) for rows in values.values()]
        rotation = [np.mean([x[1] for x in rows if x[1] is not None])
                    for rows in values.values() if any(x[1] is not None for x in rows)]
        return dict(parent_sources=len(position), windows=sum(map(len, values.values())),
                    parent_macro_FDE_m=float(np.mean(position)),
                    parent_FDE_q95_m=float(np.quantile(position, .95)),
                    rotation_parent_sources=len(rotation),
                    parent_macro_angle_rad=float(np.mean(rotation)) if rotation else None,
                    parent_angle_q95_rad=float(np.quantile(rotation, .95)) if rotation else None)

    original = json.loads((checkpoint.parent / 'report.json').read_text())['validation']['metrics']
    aggregate = {}; comparison = {}
    for (name, k), values in parents.items():
        h = str(round(k * .05, 3)); metrics = summary(values)
        prior = original[name][h]
        if (metrics['parent_sources'] != prior['source_episodes'] or
                metrics['windows'] != prior['windows']):
            raise ValueError('CPU replay parent/window coverage differs from full GPU validation')
        comparison.setdefault(name, {})[h] = dict(
            FDE_absolute_difference_m=abs(metrics['parent_macro_FDE_m'] - prior['macro_FDE_m']),
            angle_absolute_difference_rad=abs(metrics['parent_macro_angle_rad'] -
                                              prior['observable_rotation_endpoint_radians']),
            parent_and_window_counts_exact=True)
        aggregate.setdefault(name, {})[h] = metrics
    per_body = [dict(method=name, horizon_seconds=k * .05, task=task, body_id=body,
                     **summary(values)) for (name, k, task, body), values in sorted(bodies.items())]
    per_task = [dict(method=name, horizon_seconds=k * .05, task=task,
                     **summary(values)) for (name, k, task), values in sorted(tasks.items())]
    if any(sha(path) != checksum for path, checksum in files.items()):
        raise ValueError('diagnostic input changed')
    report = dict(schema='r10_full275_VAL_same_data_rotation_candidate_diagnosis_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=fitted.checkpoint_sha256, completed_epochs=30,
                  validation_parent_sources=275, causal_validation_windows=len(samples),
                  aggregate=aggregate, by_task=per_task, by_body=per_body, verified_files_sha256=files,
                  TRAIN_parent_sources=1400,
                  experimental_objective_not_formal_R10_release=fitted.metadata.get(
                      'experimental_objective_not_formal_R10_release', False),
                  recorded_GPU_comparison=comparison,
                  numerical_CPU_GPU_equality_claimed=False,
                  quantile_unit='q95 of per-parent mean errors, not pooled frame errors',
                  unobservable_rotations_excluded_by_original_mask=True,
                  native_or_GPU_started=False, TRAIN_or_TEST_inference=False,
                  model_fitted_or_selected=False, closed_loop_success_claimed=False)
    output.mkdir(parents=True)
    path = output / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return dict(report=str(path.relative_to(ROOT)), sha256=sha(path), aggregate=aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.manifest, args.out), ensure_ascii=False))
