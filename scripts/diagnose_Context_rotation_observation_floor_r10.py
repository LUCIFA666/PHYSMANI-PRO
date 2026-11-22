"""Separate angular observation error from dynamics error on existing full VAL.

Native pose/twist oracle variants are diagnostic-only, never deployable inputs.
All horizons use the same full1.1s rotation-valid windows and parent weighting.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.context_se3 import MultiBodyContextDynamics, rotation_from_6d, so3_exp
from our_mani.multitask.track_dataset import read_tracks, examples
from our_mani.multitask.fitting import batch_tensors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def run():
    base = ROOT / 'artifacts/framework/multitask_r10'
    checkpoint = base / 'context_se3_full_v6/dynamics.pt'
    manifest = base / 'context_training_view_full_v6/manifest.json'
    out = base / 'Context_rotation_observation_floor_VAL_diagnosis_v2'
    if out.exists():
        raise ValueError('fresh diagnostic output required')
    torch.set_num_threads(4)
    fitted = MultiBodyContextDynamics(checkpoint)
    _, episodes = read_tracks(ROOT, manifest)
    validation = [episode for episode in episodes if episode.entry['split'] == 'validation']
    source_arrays = {episode.source_key: episode.arrays for episode in validation}
    if len(source_arrays) != len(validation):
        raise ValueError('diagnostic requires original K1 validation identities')
    samples, coverage = examples(validation, fitted.model, steps=22)
    if len(coverage) != 275 or len(samples) != 31799:
        raise ValueError('all original275/31799 validation coverage required')
    paths = [checkpoint, manifest, Path(__file__), ROOT / 'our_mani/multitask/context_se3.py',
             ROOT / 'our_mani/multitask/track_dataset.py', ROOT / 'our_mani/multitask/fitting.py']
    paths += [ROOT / episode.entry['cache'] for episode in validation]
    files = {str(path): sha(path) for path in paths}
    values = defaultdict(lambda: defaultdict(list))
    by_body = defaultdict(lambda: defaultdict(list))
    omega = defaultdict(list)
    for begin in range(0, len(samples), 256):
        rows = samples[begin:begin + 256]
        data = batch_tensors(samples, range(begin, begin + len(rows)), 'cpu')
        state = data['state']
        observed_R = rotation_from_6d(state[:, 3:9])
        observed_w = state[:, 12:15] * fitted.model.angular_velocity_scale
        # Optimization intentionally masks t0. Read its separately verified raw
        # native labels rather than unmasking the training batch or zero targets.
        target_R = torch.as_tensor(np.asarray([row['targets']['rotations'][0] for row in rows]))
        target_w = torch.as_tensor(np.asarray([row['targets']['angular_velocities'][0] for row in rows]))
        current_valid = []
        for row in rows:
            arrays = source_arrays[row['source_key']]
            body = list(map(str, arrays['body_ids'])).index(row['body_id'])
            anchor = row['anchor']
            current_valid.append(bool(arrays['target_valid'][anchor, body]
                and arrays['target_rotation_observable'][anchor, body]))
        # A common complete-rotation mask makes every reported horizon paired.
        eligible = data['rotation_mask'][:, 1:].all(dim=-1) & torch.as_tensor(current_valid)
        variants = dict(observed_pose_observed_omega=(observed_R, observed_w),
            native_pose_observed_omega=(target_R, observed_w),
            observed_pose_native_omega=(observed_R, target_w),
            native_pose_native_omega=(target_R, target_w))
        for i, row in enumerate(rows):
            if bool(eligible[i]):
                omega[row['source_key']].append(float(torch.linalg.vector_norm(observed_w[i] - target_w[i])))
        for k in (0, 1, 2, 5, 10, 22):
            for name, (R0, w0) in variants.items():
                forecast = so3_exp(w0 * (.05 * k)) @ R0
                target_at_k = target_R if k == 0 else data['targets']['rotations'][:, k]
                relative = (forecast @ target_at_k.transpose(-1, -2)).numpy()
                for i, row in enumerate(rows):
                    if not bool(eligible[i]):
                        continue
                    # Float64 geodesic avoids an acos floor from float32 traces.
                    error = float(Rotation.from_matrix(np.array(relative[i], dtype=np.float64)).magnitude())
                    parent = row['source_key']
                    values[(name, k)][parent].append(error)
                    by_body[(name, k, parent[1], row['body_id'])][parent].append(error)

    def summary(grouped):
        parent_means = [float(np.mean(rows)) for rows in grouped.values()]
        return dict(rotation_parent_sources=len(grouped), windows=sum(map(len, grouped.values())),
            parent_macro_angle_rad=float(np.mean(parent_means)),
            parent_angle_q95_rad=float(np.quantile(parent_means, .95)))

    aggregate = {}
    for (name, k), grouped in sorted(values.items()):
        aggregate.setdefault(name, {})[str(round(k * .05, 3))] = summary(grouped)
    body = [dict(method=name, horizon_seconds=.05 * k, task=task, body_id=bid, **summary(grouped))
            for (name, k, task, bid), grouped in sorted(by_body.items())]
    omega_parent = [float(np.mean(rows)) for rows in omega.values()]
    if not omega_parent or any(len(grouped) != len(omega) for grouped in values.values()):
        raise ValueError('nonempty identical paired rotation-parent coverage required')
    for path, checksum in files.items():
        if sha(path) != checksum:
            raise ValueError('diagnostic input changed: ' + path)
    report = dict(schema='r10_full275_VAL_angular_observation_floor_and_native_oracle_diagnosis_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), validation_parents=275,
        original_causal_windows=31799, paired_rotation_parent_sources=len(omega), aggregate=aggregate, by_body=body,
        observed_vs_native_backward_angular_velocity=dict(
            parent_macro_error_rad_s=float(np.mean(omega_parent)),
            parent_error_q95_rad_s=float(np.quantile(omega_parent, .95))),
        same_full1p1_rotation_valid_windows_used_at_all_horizons=True,
        angle_metric='float64 scipy SO3 geodesic from relative matrix; not bitexact to float32 acos evaluator',
        native_pose_and_backward_twist_variants_are_diagnostic_oracles_only=True,
        t0_oracle_labels_read_separately_from_verified_raw_arrays=True,
        original_optimization_t0_mask_remains_false=True,
        causal_deployable_baseline='observed_pose_observed_omega',
        target_backward_twist_is_not_a_future_difference=True,
        native_supervision_granted_to_runtime_policy=False,
        model_fitted_or_selected=False, original_checkpoints_or_masks_changed=False,
        TRAIN_inference=False, TEST_used=False, native_or_GPU_started=False,
        verified_files_sha256=files)
    out.mkdir()
    path = out / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), aggregate=aggregate,
                observed_vs_native_backward_angular_velocity=report['observed_vs_native_backward_angular_velocity'])


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False), flush=True)
