"""Real FASTER checkpoint inference, goal sensitivity, and streaming diagnostics.

Uses current VAL RGB/state and supervised future EE labels for offline action
errors only. No simulator success rate is inferred from these measurements.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/FASTER/src'), str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
os.environ.setdefault('HF_HOME', str(ROOT/'runtime/framework/huggingface'))
os.environ.setdefault('OPENPI_DATA_HOME', str(ROOT/'artifacts/framework/openpi_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'runtime/framework/cache'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(ROOT/'runtime/framework/jax-cache'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.45')
import jax
import numpy as np
from scipy.spatial.transform import Rotation
from openpi.policies.policy_config import create_trained_policy
from our_mani.faster_config import make_config
from our_mani.faster_hdf5 import PhysManiFasterDataset

STOP = ROOT/'runtime/framework/state/STOP'


def write(path, record):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def infer(policy, observation, noise, *, streaming=False, delay=0):
    observation = {k:v for k,v in observation.items() if k != 'actions'}
    if streaming:
        prefix = observation.get('action_prefix', np.empty((0, 8), dtype=np.float32))
        observation['action_prefix'], observation['delay'] = prefix, np.int32(delay)
    start = time.perf_counter()
    callbacks, first = [], []
    def ready(actions):
        if len(actions):
            if not first:
                first.append(time.perf_counter()-start)
            callbacks.append(np.asarray(actions).copy())
    if streaming:
        output = policy.infer_streaming(observation, noise=noise, on_actions_ready=ready)
    else:
        output = policy.infer(observation, noise=noise)
    elapsed = time.perf_counter()-start
    actions = np.asarray(output['actions'])
    if actions.shape != (16, 8) or not np.isfinite(actions).all():
        raise RuntimeError('FASTER action shape/finiteness check failed')
    return actions, {'wall_seconds': elapsed, 'first_ready_seconds': first[0] if first else None,
                     'partial_chunk_sizes': [len(a) for a in callbacks],
                     'finalized_actions': int(sum(map(len, callbacks))),
                     'callbacks': callbacks}


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--repo-id', default='hdf5/physmani_train_all16')
    parser.add_argument('--scope', choices=('all16', 'grasp2'), default='grasp2')
    parser.add_argument('--episodes-per-task', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    config = make_config(args.repo_id, 'gs://openpi-assets/checkpoints/pi05_base/params')
    policy = create_trained_policy(config, checkpoint, sample_kwargs={'num_steps': 10, 'alpha': .6, 'u0': .9})
    dataset = PhysManiFasterDataset(scope=args.scope, split='validation', limit_per_task=args.episodes_per_task)
    out = ROOT/args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(991).normal(size=(16, 32)).astype(np.float32)
    rows = []
    samples = []
    for episode, source in enumerate(dataset.entries):
        start = dataset.ends[episode-1] if episode else 0
        length = len(dataset.poses[episode])-1
        offset = min(5, length-1)
        samples.append((source, dataset[start+offset]))
    if not samples:
        raise RuntimeError('empty VAL inference sample set')
    # First invocation includes compilation and is reported separately.
    _, warmup = infer(policy, samples[0][1], noise)
    warmup.pop('callbacks')
    for source, observation in samples:
        if STOP.exists():
            break
        actions, timing = infer(policy, observation, noise)
        expected = observation['actions']
        position = np.linalg.norm(actions[:, :3]-expected[:, :3], axis=1)
        predicted_rotation = Rotation.from_quat(actions[:, 3:7])
        expected_rotation = Rotation.from_quat(expected[:, 3:7])
        angular = (predicted_rotation*expected_rotation.inv()).magnitude()
        timing.pop('callbacks')
        rows.append({'task': source['task'], 'raw_episode_id': source['raw_episode_id'],
                     'position_action_error_m': float(position.mean()), 'final_position_action_error_m': float(position[-1]),
                     'orientation_action_error_rad': float(angular.mean()),
                     'gripper_accuracy': float(np.mean((actions[:, 7] > .5) == (expected[:, 7] > .5))), **timing})
        write(out.with_name(out.stem+'_partial.json'), {'rows': rows, 'formal_test_used': False})
    original = samples[0][1]
    base, _ = infer(policy, original, noise)
    probes = []
    for axis in range(3):
        changed = dict(original)
        changed['handover_goal'] = original['handover_goal'].copy()
        changed['handover_goal'][axis] += .08
        shifted, timing = infer(policy, changed, noise)
        probes.append({'condition': f'goal_position_axis_{axis}_plus_8cm', 'target_shift_m': .08,
                       'mean_action_shift_m': (shifted[:, :3]-base[:, :3]).mean(0).tolist(),
                       'endpoint_action_shift_m': (shifted[-1, :3]-base[-1, :3]).tolist(),
                       'endpoint_shift_in_requested_direction_m': float(shifted[-1, axis]-base[-1, axis])})
    changed = dict(original)
    changed['handover_goal'] = original['handover_goal'].copy()
    changed['handover_goal'][16] = max(.05, float(original['handover_goal'][16])*.5)
    faster, _ = infer(policy, changed, noise)
    probes.append({'condition': 'arrival_time_halved', 'original_arrival_seconds': float(original['handover_goal'][16]),
                   'changed_arrival_seconds': float(changed['handover_goal'][16]),
                   'action_position_change_rmse_m': float(np.sqrt(np.square(faster[:, :3]-base[:, :3]).mean()))})
    changed = dict(original)
    changed['handover_goal'] = original['handover_goal'].copy()
    changed['handover_goal'][7] += .15
    velocity_changed, _ = infer(policy, changed, noise)
    probes.append({'condition': 'terminal_x_velocity_plus_0.15mps',
                   'action_position_change_rmse_m': float(np.sqrt(np.square(velocity_changed[:, :3]-base[:, :3]).mean()))})
    stream, stream_timing = infer(policy, original, noise, streaming=True)
    chunks = stream_timing.pop('callbacks')
    if sum(map(len, chunks)) != 16:
        raise RuntimeError('complete streaming run did not finalize 16 actions')
    partial_final_difference = float(np.max(np.abs(np.concatenate(chunks)-stream)))
    np.savez_compressed(out.with_name(out.stem+'_streaming_parity.npz'),
                        emitted=np.concatenate(chunks), final=stream)
    np.testing.assert_allclose(np.concatenate(chunks), stream, atol=2e-4, rtol=2e-4)
    # Warm streaming latency after its independent init/step compilation.
    _, streaming_warm = infer(policy, original, noise, streaming=True)
    streaming_warm.pop('callbacks')
    prefixed = dict(original, action_prefix=stream[:2].copy())
    prefixed_actions, prefix_timing = infer(policy, prefixed, noise, streaming=True, delay=2)
    prefix_chunks = prefix_timing.pop('callbacks')
    if sum(map(len, prefix_chunks)) != 14:
        raise RuntimeError('streaming callback must exclude the two committed prefix actions')
    np.testing.assert_allclose(prefixed_actions[:2], stream[:2], atol=2e-4, rtol=2e-4)
    checkpoint_index = int(checkpoint.name) if checkpoint.name.isdigit() else None
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'kind': 'real_faster_offline_val_inference_not_closed_loop_success', 'checkpoint': str(checkpoint),
              'checkpoint_index': checkpoint_index, 'optimizer_updates': checkpoint_index+1 if checkpoint_index is not None else None,
              'repo_id': args.repo_id, 'validation_scope': args.scope, 'formal_test_used': False,
              'samples': len(rows), 'warmup_including_compilation': warmup,
              'mean_position_action_error_m': float(np.mean([r['position_action_error_m'] for r in rows])),
              'mean_orientation_action_error_rad': float(np.mean([r['orientation_action_error_rad'] for r in rows])),
              'median_inference_wall_seconds': float(np.median([r['wall_seconds'] for r in rows])),
              'goal_condition_probes_fixed_noise': probes, 'streaming_first_run': stream_timing,
              'streaming_warm_run': streaming_warm, 'committed_prefix_probe': prefix_timing,
              'streaming_partial_final_parity_passed': True, 'committed_prefix_exclusion_passed': True,
              'streaming_ready_rule': 'HAS time exactly zero; emitted actions remain immutable',
              'partial_final_maximum_difference': partial_final_difference,
              'all_three_position_axes_follow_goal_shift': all(p['endpoint_shift_in_requested_direction_m'] > .01 for p in probes[:3]),
              'other_gpu_compute_pids_at_measurement': other_gpu_pids(), 'rows': rows,
              'closed_loop_handover_reachability_verified': False}
    write(out, report)
    dataset.close()
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, ensure_ascii=False, indent=2), flush=True)


def other_gpu_pids():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True)
        return [int(p.strip()) for p in output.splitlines() if p.strip().isdigit() and int(p.strip()) != os.getpid()]
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == '__main__':
    main()
