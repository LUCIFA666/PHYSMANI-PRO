"""Collect deterministic, sharded local outcomes from every selected TRAIN/VAL episode.

Delegates physics/control to the existing collector. Sharding only distributes
independent attempts; both predictors use the same episode/replica perturbations.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import torch
from collect_local_rollouts import TASKS, STOP, collect
from our_mani.local_capture import LocalConfig
from our_mani.pyrep_env import LocalSimulation


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sampling(task, episode, replica, split):
    identity = f'full_train_r04:{split}:{task}:{episode}:{replica}'
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    if replica == 0:
        offset, budget, delay = np.array([0., 0., .07]), 3., 0
    else:
        offset = np.r_[rng.uniform(-.20, .20, 2), rng.uniform(.025, .23)]
        budget, delay = float(rng.choice([1., 1.5, 2., 3.])), int(rng.integers(0, 5))
    return seed, offset, budget, delay, (1., 0., .5, -.5)[replica % 4], (5, 10, 20)[replica % 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', choices=('cv', 'context'), required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--latent-checkpoint')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        raise ValueError('invalid shard')
    if (args.predictor == 'context') != bool(args.latent_checkpoint):
        raise ValueError('context controller requires exactly one dynamics checkpoint')
    torch.set_num_threads(2)
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((ROOT/f'artifacts/framework/protocol/{args.split}_manifest.json').read_text())['episodes'] if r['task'] in TASKS]
    expected = 100 if args.split == 'train' else 20
    if any(sum(r['task'] == t for r in rows) != expected for t in TASKS):
        raise ValueError('full selected-task manifest coverage required')
    config = LocalConfig(max_speed=1., max_acceleration=4.)
    if args.latent_checkpoint:
        from our_mani.context_control import controller_sha256
        controller_hash = controller_sha256(ROOT/args.latent_checkpoint, config, 1.4)
    else:
        names = ('local_capture.py', 'control.py', 'benchmark_perception.py', 'goal_perception.py', 'pyrep_env.py')
        controller_hash = hashlib.sha256(b''.join((ROOT/'our_mani'/n).read_bytes() for n in names)
            +json.dumps({'controller_config': asdict(config), 'servo_lead': 1.4}, sort_keys=True).encode()).hexdigest()
    goal_offset = json.loads((ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json').read_text())['offset_train_calibrated_m']
    results = []
    for task in TASKS:
        assigned = [(i, r) for i, r in enumerate(rows) if r['task'] == task and i % args.shards == args.shard_index]
        if not assigned:
            continue
        environment = LocalSimulation(task, assigned[0][1]['path'], seed=421, servo_lead=1.4)
        try:
            for _, row in assigned:
                for replica in range(4):
                    if STOP.exists():
                        raise InterruptedError('local collection STOP marker')
                    seed, offset, budget, delay, warmup, frames = sampling(task, row['raw_episode_id'], replica, args.split)
                    key = f"{task}__{row['raw_episode_id']}__{replica}"
                    path = out/(key+'.json')
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record['controller_sha256'] != controller_hash or record['sampling_seed'] != seed or record['split'] != args.split:
                            raise ValueError('resume input identity changed')
                    else:
                        np.random.seed(seed)
                        record, arrays, _ = collect(environment, row, offset, budget, delay, seed, goal_offset,
                            full_task=True, controller_config=config, warmup_match=warmup,
                            latent_checkpoint=ROOT/args.latent_checkpoint if args.latent_checkpoint else None,
                            context_history_frames=frames)
                        if record.get('initial_features') is not None:
                            final_success, _ = environment.task.success()
                            record['official_task_success_immediate'] = bool(record['official_task_success_immediate'] or final_success)
                            record['final_native_success_observed'] = bool(final_success)
                        record.update(key=key, split=args.split, predictor=args.predictor,
                            controller='context_simnorm_capture_and_visual_goal_v1' if args.predictor == 'context' else 'cv_capture_and_visual_goal',
                            controller_sha256=controller_hash, controller_config=asdict(config), servo_lead=1.4,
                            formal_test_used=False, sampling_seed=seed, replica=replica,
                            requested_context_history_frames=frames, latent_checkpoint=args.latent_checkpoint,
                            collection_protocol='full_train_r04_deterministic_episode_replica')
                        np.savez_compressed(out/(key+'.npz'), **arrays)
                        write(path, record)
                    results.append(record)
                    valid = [r for r in results if r.get('initial_features') is not None and r.get('failure') != 'deadline_stop']
                    write(out/'report.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'completed': len(results), 'expected': sum(i % args.shards == args.shard_index for i in range(len(rows)))*4,
                        'valid_initializations': len(valid), 'valid_source_episodes': len({(r['task'], r['source_episode']) for r in valid}),
                        'controller_sha256': controller_hash, 'split': args.split, 'full_task_executed': True,
                        'formal_test_used': False, 'shard_index': args.shard_index, 'shards': args.shards,
                        'official_task_success_immediate': sum(r['official_task_success_immediate'] for r in valid)})
                    print(json.dumps({'key': key, 'valid': record.get('initial_features') is not None,
                        'success': record['official_task_success_immediate'], 'completed': len(results)}, ensure_ascii=False), flush=True)
        finally:
            environment.close()


if __name__ == '__main__':
    main()
