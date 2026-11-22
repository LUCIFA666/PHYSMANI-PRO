"""Extract causal RGB-D tracks from train/validation; labels are metrics only."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.physmani_data import RecordedEpisode
from our_mani.benchmark_perception import RedBoxTracker


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/framework/domain_study')
    parser.add_argument('--train-limit', type=int, default=8)
    parser.add_argument('--validation-limit', type=int, default=20)
    parser.add_argument('--reuse-manifest', help='Reuse verified causal track caches with the same perception implementation')
    args = parser.parse_args()
    out = ROOT/args.out
    cache = out/'tracks'
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    tasks = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
    started = time.time()
    reused = {}
    if args.reuse_manifest:
        previous = json.loads((ROOT/args.reuse_manifest).read_text())
        current_hash = hashlib.sha256((ROOT/'our_mani/benchmark_perception.py').read_bytes()).hexdigest()
        if previous.get('perception_source_sha256') != current_hash:
            raise ValueError('cached tracks use a different perception implementation')
        reused = {(r['split'], r['task'], r['raw_episode_id']):r for r in previous['entries']}
    for split, limit in [('train', args.train_limit), ('validation', args.validation_limit)]:
        source = ROOT/f'artifacts/framework/protocol/{split}_manifest.json'
        manifest = json.loads(source.read_text())
        chosen = [r for task in tasks for r in [x for x in manifest['episodes'] if x['task'] == task][:limit]]
        for entry in chosen:
            prior = reused.get((split, entry['task'], entry['raw_episode_id']))
            if prior is not None:
                path = ROOT/prior['cache']
                if prior['path'] != entry['path'] or hashlib.sha256(path.read_bytes()).hexdigest() != prior['cache_sha256']:
                    raise ValueError('cached track source or checksum differs')
                rows.append(prior)
                continue
            if (ROOT/'runtime/framework/domain_state/STOP').exists():
                write(out/'status.json', {'stage': 'stopped', 'episodes': len(rows)})
                return
            tracker = RedBoxTracker(points=256)
            with RecordedEpisode(entry['path']) as episode:
                # Stop at the recorded lift/contact transition: subsequent motion
                # depends on expert robot actions not provided to a target forecast.
                labels = episode.file['low_dim_obs/expert_trajectory_stage'][:].reshape(-1)
                stages = [s.decode() if isinstance(s, bytes) else str(s) for s in labels]
                stops = [i for i, stage in enumerate(stages) if stage == 'lift']
                end = min(stops[0] if stops else len(episode), 65)
                truth = episode.file['low_dim_obs/expert_trajectory_target_position'][:end].copy()
                positions, velocities, clouds, valid, confidence, latencies = [], [], [], [], [], []
                for index in range(end):
                    tick = time.perf_counter()
                    observation = episode.observation(index)
                    try:
                        state, meta = tracker.observe(observation)
                        positions.append(state.pose.position)
                        velocities.append(state.linear_velocity)
                        clouds.append(state.points_world)
                        valid.append(True)
                        confidence.append(state.confidence)
                    except ValueError:
                        tracker.history.clear()
                        positions.append(np.full(3, np.nan))
                        velocities.append(np.full(3, np.nan))
                        clouds.append(np.full((256, 3), np.nan))
                        valid.append(False)
                        confidence.append(0.)
                    latencies.append(time.perf_counter()-tick)
            name = f"{split}_{entry['task']}_{entry['raw_episode_id']}"
            destination = cache/f'{name}.npz'
            np.savez_compressed(destination, times=np.arange(end)*.05, positions=positions, velocities=velocities,
                                points=clouds, confidence=confidence, valid=valid, truth_metric_only=truth)
            good = np.asarray(valid)
            errors = np.linalg.norm(np.asarray(positions)[good]-truth[good], axis=-1)
            row = {**entry, 'split': split, 'cache': str(destination.relative_to(ROOT)),
                   'cache_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'frames': end,
                   'valid_frames': int(good.sum()), 'mean_position_error_m': float(errors.mean()) if len(errors) else None,
                   'p95_position_error_m': float(np.quantile(errors, .95)) if len(errors) else None,
                   'median_perception_seconds': float(np.median(latencies))}
            rows.append(row)
            write(out/'tracks_manifest.json', {'entries': rows, 'formal_test_used': False,
                  'observation_backend': 'task_specific_RGBD_red_box_known_shape',
                  'perception_backend_version': 'red_box_v5_bounded_causal_association',
                  'perception_source_sha256': hashlib.sha256((ROOT/'our_mani/benchmark_perception.py').read_bytes()).hexdigest(),
                  'ground_truth_target_positions': 'metric labels only, never perception inputs',
                  'window_scope': 'recorded pre-lift frames; future robot contact excluded using offline labels'})
            write(out/'status.json', {'stage': 'extracting_causal_tracks', 'episodes': len(rows),
                                    'latest': row, 'elapsed_seconds': time.time()-started})
            print(json.dumps(row, ensure_ascii=False), flush=True)
    write(out/'tracks_manifest.json', {'entries': rows, 'formal_test_used': False,
          'observation_backend': 'task_specific_RGBD_red_box_known_shape',
          'perception_backend_version': 'red_box_v5_bounded_causal_association',
          'perception_source_sha256': hashlib.sha256((ROOT/'our_mani/benchmark_perception.py').read_bytes()).hexdigest(),
          'ground_truth_target_positions': 'metric labels only, never perception inputs',
          'window_scope': 'recorded pre-lift frames; future robot contact excluded using offline labels'})
    write(out/'perception_report.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'episodes': len(rows),
          'valid_frames': sum(r['valid_frames'] for r in rows), 'total_frames': sum(r['frames'] for r in rows),
          'macro_mean_position_error_m': float(np.mean([r['mean_position_error_m'] for r in rows if r['mean_position_error_m'] is not None])),
          'rows': rows, 'formal_test_used': False})
    write(out/'status.json', {'stage': 'tracks_ready', 'episodes': len(rows), 'elapsed_seconds': time.time()-started})


if __name__ == '__main__':
    main()
