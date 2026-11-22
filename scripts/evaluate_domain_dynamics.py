"""Frozen NGFF transfer against causal CV/CA on recorded moving-grasp validation.

RGB-D estimates are the input. Target XYZ labels are used only for training-
split scale selection and held-out error measurement. No grasp SR is produced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from our_mani.contracts import ObjectState, Pose
from our_mani.dynamics import NGFFDynamics, NGFFFrame

OUT = ROOT/'artifacts/framework/domain_study'
STOP = ROOT/'runtime/framework/domain_state/STOP'


def write(name, obj):
    path = OUT/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def load(entry):
    path = ROOT/entry['cache']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['cache_sha256']:
        raise RuntimeError('causal track cache changed')
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def windows(entries, anchors):
    for entry in entries:
        data = load(entry)
        for anchor in anchors:
            end = anchor+12
            if end >= len(data['times']) or not data['valid'][max(0, anchor-4):end+1].all():
                continue
            # Validity is based on perception coverage, not future label accuracy.
            yield entry, data, anchor


def scores(prediction, truth, horizons=((.1, 2), (.3, 6), (.6, 12))):
    result = {}
    for horizon, end in horizons:
        errors = np.linalg.norm(prediction[1:end+1]-truth[1:end+1], axis=-1)
        result[str(horizon)] = {'mean_position_error_m': float(errors.mean()),
                               'final_position_error_m': float(errors[-1]),
                               'coordinate_rmse_m': float(np.sqrt(np.square(prediction[1:end+1]-truth[1:end+1]).mean())),
                               'final_velocity_error_m_s': float(np.linalg.norm((prediction[end]-prediction[end-1])/.05-(truth[end]-truth[end-1])/.05))}
    return result


def make_model(scale, device):
    origin = np.array([.2, 0., .75+.94*scale])
    seconds = np.sqrt(5.*scale/9.81)
    bounds = np.array([(2.-origin[0])/scale, (-2.-origin[0])/scale,
                       2./scale, -2./scale, (3.-origin[2])/scale, -.94])
    frame = NGFFFrame(scale, seconds, origin, bounds, f'train_calibrated_table_scale_{scale}')
    return NGFFDynamics(ROOT/'artifacts/framework/checkpoints/ngff_best.pt', frame, device=device, keypoints=64)


def forecast(model, data, anchor):
    state = ObjectState('target', float(data['times'][anchor]), Pose(data['positions'][anchor]),
                        data['points'][anchor], data['velocities'][anchor], np.zeros(3), float(data['confidence'][anchor]))
    prediction = model.predict([state], .6, .05)
    return prediction.positions[:, 0], prediction.metadata


def aggregate(rows, horizons=('0.1', '0.3', '0.6')):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        chosen = [r for r in rows if r['method'] == method]
        by_episode = {}
        for row in chosen:
            by_episode.setdefault((row['task'], row['raw_episode_id']), []).append(row)
        summary = {'episodes': len(by_episode), 'windows': len(chosen), 'metrics': {}}
        for horizon in horizons:
            summary['metrics'][horizon] = {}
            for metric in ('mean_position_error_m', 'final_position_error_m', 'coordinate_rmse_m', 'final_velocity_error_m_s'):
                scene_values = [np.mean([r['metrics'][horizon][metric] for r in group]) for group in by_episode.values()]
                summary['metrics'][horizon][metric] = float(np.mean(scene_values))
            finals = [r['metrics'][horizon]['final_position_error_m'] for r in chosen]
            summary['metrics'][horizon]['p95_window_final_error_m'] = float(np.quantile(finals, .95))
        latencies = [r['latency_seconds'] for r in chosen]
        summary['latency_median_seconds'] = float(np.median(latencies))
        summary['latency_p95_seconds'] = float(np.quantile(latencies, .95))
        result[method] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((OUT/'tracks_manifest.json').read_text())
    entries = manifest['entries']
    train = [r for task in ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
             for r in [e for e in entries if e['split'] == 'train' and e['task'] == task][:3]]
    calibration = []
    for scale in (.1, .25, .5, 1.):
        model = make_model(scale, args.device)
        errors, times = [], []
        for entry, data, anchor in windows(train, (5, 10)):
            if STOP.exists():
                write('status.json', {'stage': 'stopped_during_calibration'})
                return
            predicted, meta = forecast(model, data, anchor)
            truth = data['truth_metric_only'][anchor:anchor+13]
            errors.append(scores(predicted, truth)['0.6']['mean_position_error_m'])
            times.append(meta['latency_seconds'])
            write('status.json', {'stage': 'calibrating_frozen_ngff_on_train', 'scale': scale, 'windows': len(errors)})
        calibration.append({'scale': scale, 'train_window_count': len(errors),
                            'mean_position_error_m': float(np.mean(errors)), 'median_latency_seconds': float(np.median(times))})
        del model
    best = min(calibration, key=lambda r: r['mean_position_error_m'])
    write('calibration.json', {'selection_split': 'train', 'grid': calibration, 'selected': best,
                              'gravity_world_m_s2': -9.81, 'test_split_used': False})
    model = make_model(best['scale'], args.device)
    rows = []
    offsets = np.arange(13)*.05
    for entry, data, anchor in windows([r for r in entries if r['split'] == 'validation'], (5, 10, 15, 20)):
        if STOP.exists():
            write('status.json', {'stage': 'stopped_during_validation', 'completed_rows': len(rows)})
            break
        position, velocity = data['positions'][anchor], data['velocities'][anchor]
        truth = data['truth_metric_only'][anchor:anchor+13]
        start = time.perf_counter()
        cv = position+offsets[:, None]*velocity
        t = data['times'][anchor-4:anchor+1]-data['times'][anchor]
        design = np.c_[np.ones(len(t)), t, .5*t*t]
        acceleration = np.linalg.lstsq(design, data['positions'][anchor-4:anchor+1], rcond=None)[0][2]
        acceleration = np.clip(acceleration, -2., 2.)
        ca = position+offsets[:, None]*velocity+.5*offsets[:, None]**2*acceleration
        baseline_time = time.perf_counter()-start
        predicted, meta = forecast(model, data, anchor)
        supported = predicted.copy()
        # Explicit ablation: impose the known table support after NGFF prediction.
        # This is reported separately and never passed off as bare NGFF.
        supported[:, 2] = .775
        for name, prediction, elapsed in [('constant_velocity', cv, baseline_time),
                                          ('causal_acceleration', ca, baseline_time),
                                          ('ngff_frozen', predicted, meta['latency_seconds']),
                                          ('ngff_plus_known_table_projection', supported, meta['latency_seconds'])]:
            row = {'task': entry['task'], 'raw_episode_id': entry['raw_episode_id'], 'anchor': anchor,
                   'method': name, 'metrics': scores(prediction, truth), 'latency_seconds': elapsed}
            rows.append(row)
        write('status.json', {'stage': 'evaluating_rgbd_driven_motion_forecasts', 'completed_windows': len(rows)//4,
                              'latest_episode': entry['raw_episode_id'], 'device': args.device})
        write('partial_results.json', {'rows': rows, 'summary': aggregate(rows)})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'offline_causal_forecast_validation_not_grasp_success',
              'input': manifest['observation_backend'], 'perception_backend_version': manifest['perception_backend_version'],
              'known_shape_and_table_prior': True, 'truth_positions_used_as_model_input': False,
              'target_orientation': 'world-aligned task prior; rotational transfer not evaluated',
              'selected_frame_scale': best, 'checkpoint_sha256': model.checkpoint_sha256,
              'ngff_weights_updated': False, 'device': args.device, 'torch_threads': torch.get_num_threads(),
              'formal_test_used': False, 'summary': aggregate(rows), 'rows': rows}
    write('report.json', report)
    write('status.json', {'stage': 'forecast_validation_complete', 'windows': len(rows)//4,
                          'formal_grasp_success_rate': None, 'report': str(OUT/'report.json')})
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
