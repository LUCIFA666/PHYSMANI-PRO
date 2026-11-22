"""Compare a frozen-NGFF driver residual with a matched model without NGFF."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import torch
from evaluate_domain_dynamics import make_model, load, scores, aggregate
from our_mani.contracts import ObjectState, Pose
from our_mani.dynamics_adapter import DriverResidual, LoadedDriverAdapter, causal_features

OUT = ROOT/'artifacts/framework/domain_adaptation'
STOP = ROOT/'runtime/framework/domain_state/STOP'
OFFSETS = np.arange(13)*.05
HORIZON = .6


def write(name, record):
    destination = OUT/name
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(destination)


def prepare(entries, model):
    cache = OUT/'frozen_forecasts'
    cache.mkdir(exist_ok=True)
    samples = []
    for row in entries:
        data = load(row)
        future_steps = len(OFFSETS)-1
        anchors = range(5, len(data['times'])-future_steps, 3) if row['split'] == 'train' else (5, 10, 15, 20)
        for anchor in anchors:
            if STOP.exists():
                raise InterruptedError('deadline STOP')
            if anchor+future_steps >= len(data['times']) or not data['valid'][anchor-4:anchor+future_steps+1].all():
                continue
            name = f"{row['split']}_{row['task']}_{row['raw_episode_id']}_{anchor}"
            path = cache/(name+'.npz')
            if path.exists():
                with np.load(path) as z:
                    prediction, latency = z['prediction'], float(z['latency'])
                    if str(z['track_sha256']) != row['cache_sha256']:
                        raise RuntimeError('cached forecast source mismatch')
            else:
                state = ObjectState('target', float(data['times'][anchor]), Pose(data['positions'][anchor]),
                    data['points'][anchor], data['velocities'][anchor], np.zeros(3), float(data['confidence'][anchor]))
                frozen = model.predict([state], HORIZON, .05)
                prediction, meta = frozen.positions[:, 0], frozen.metadata
                latency = meta['latency_seconds']
                np.savez_compressed(path, prediction=prediction, latency=latency, track_sha256=row['cache_sha256'])
            position, velocity = data['positions'][anchor], data['velocities'][anchor]
            cv = position+OFFSETS[:, None]*velocity
            features = np.array([causal_features(data['times'][anchor-4:anchor+1], data['positions'][anchor-4:anchor+1],
                                    velocity, horizon, delta) for horizon, delta in zip(OFFSETS, prediction-cv)])
            samples.append({'task': row['task'], 'episode': row['raw_episode_id'], 'anchor': anchor, 'split': row['split'],
                            'features': features, 'truth': data['truth_metric_only'][anchor:anchor+future_steps+1],
                            'ngff': prediction, 'cv': cv, 'latency': latency})
            if len(samples) % 25 == 0:
                write('status.json', {'stage': 'preparing_frozen_ngff_adapter_training_examples', 'windows': len(samples)})
                print(f'cached frozen NGFF windows {len(samples)}', flush=True)
    return samples


def train(samples, mode):
    groups = sorted({(s['task'], s['episode']) for s in samples if s['split'] == 'train'})
    rng = np.random.default_rng(771)
    shuffled = np.array(rng.permutation(len(groups)))
    held = {groups[i] for i in shuffled[:max(1, len(groups)//5)]}
    arrays = {split: [] for split in ('fit', 'select')}
    for sample in samples:
        if sample['split'] != 'train':
            continue
        features = sample['features'][1:].copy()
        if mode == 'driver_model_without_ngff':
            features[:, -3:] = 0.
        base = sample['ngff'] if mode == 'ngff_driver_adapter' else sample['cv']
        targets = (sample['truth'][1:]-base[1:])/OFFSETS[1:, None]
        split = 'select' if (sample['task'], sample['episode']) in held else 'fit'
        arrays[split].append((features, targets))
    x_train = np.concatenate([x for x, y in arrays['fit']])
    y_train = np.concatenate([y for x, y in arrays['fit']])
    x_val = np.concatenate([x for x, y in arrays['select']])
    y_val = np.concatenate([y for x, y in arrays['select']])
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), .001)
    x_train, x_val = torch.tensor((x_train-mean)/std), torch.tensor((x_val-mean)/std)
    y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    torch.manual_seed(771)
    model = DriverResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    best_loss, best_step, best_state = float('inf'), 0, None
    history = []
    for step in range(3000):
        if STOP.exists():
            break
        indices = torch.randint(len(x_train), (min(512, len(x_train)),))
        prediction = model(x_train[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction, y_train[indices], beta=.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 100 == 0 or step == 2999:
            with torch.inference_mode():
                selection_loss = float(torch.square(model(x_val)-y_val).mean())
            history.append({'step': step+1, 'training_loss': float(loss), 'selection_velocity_mse': selection_loss})
            if selection_loss < best_loss:
                best_loss, best_step = selection_loss, step+1
                best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            write('status.json', {'stage': 'training_separate_driver_adapter', 'mode': mode, 'updates': step+1,
                                 'best_train_heldout_step': best_step})
    if best_state is None:
        raise InterruptedError('stopped before any adapter update')
    metadata = {'ngff_weights_updated': False, 'original_ngff_training_resumed': False,
                'training_scope': 'two_task_development_model', 'train_groups': len(groups)-len(held),
                'train_heldout_selection_groups': len(held), 'train_examples': len(x_train),
                'selection_split': 'TRAIN grouped by task and raw episode', 'formal_test_used': False,
                'best_step': best_step, 'completed_updates': step+1,
                'maximum_training_horizon_seconds': HORIZON,
                'ngff_checkpoint_sha256': hashlib.sha256((ROOT/'artifacts/framework/checkpoints/ngff_best.pt').read_bytes()).hexdigest()}
    path = OUT/(mode+'.pt')
    torch.save({'mode': mode, 'state_dict': best_state, 'mean': mean, 'std': std,
                'hidden': 128, 'metadata': metadata}, path)
    write(mode+'_training.json', {'metadata': metadata, 'history': history, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return LoadedDriverAdapter(path)


def reflected_forecast(features):
    position, velocity, acceleration = features[0, :3].copy(), features[0, 3:6].copy(), features[0, 6:9].copy()
    trajectory = [position.copy()]
    lower, upper = np.array([.05, -.45]), np.array([.35, .45])
    for offset in OFFSETS[1:]:
        nxt = position+velocity*.05+.5*acceleration*.05**2
        velocity += acceleration*.05
        crosses = ((position[:2] >= lower) & (nxt[:2] <= lower)) | ((position[:2] <= upper) & (nxt[:2] >= upper))
        velocity[:2][crosses] *= -1
        position = nxt
        trajectory.append(position.copy())
    return np.array(trajectory)


def main():
    global OUT, OFFSETS, HORIZON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/framework/domain_adaptation')
    parser.add_argument('--horizon', type=float, default=.6)
    args = parser.parse_args()
    OUT, HORIZON = ROOT/args.out, args.horizon
    OUT.mkdir(parents=True, exist_ok=True)
    OFFSETS = np.arange(int(round(HORIZON/.05))+1)*.05
    if not np.isclose(OFFSETS[-1], HORIZON):
        raise ValueError('horizon must be a multiple of 0.05 seconds')
    horizons = [(h, int(round(h/.05))) for h in sorted(set([.1, .3, .6, HORIZON]))]
    horizon_keys = [str(h) for h, _ in horizons]
    torch.set_num_threads(4)
    manifest = json.loads((OUT/'tracks_manifest.json').read_text())
    model = make_model(1., 'cpu')
    samples = prepare(manifest['entries'], model)
    adapters = {mode: train(samples, mode) for mode in ('driver_model_without_ngff', 'ngff_driver_adapter')}
    rows = []
    for sample in [s for s in samples if s['split'] == 'validation']:
        features = sample['features']
        start = time.perf_counter()
        cv = features[0, :3]+OFFSETS[:, None]*features[0, 3:6]
        cv_latency = time.perf_counter()-start
        start = time.perf_counter()
        reflected = reflected_forecast(features)
        reflected_latency = time.perf_counter()-start
        methods = {'constant_velocity': (cv, cv_latency), 'ngff_frozen': (sample['ngff'], sample['latency']),
                   'causal_acceleration_with_declared_boundaries': (reflected, reflected_latency)}
        for mode, adapter in adapters.items():
            x = features.copy()
            base = sample['ngff'] if mode == 'ngff_driver_adapter' else sample['cv']
            if mode == 'driver_model_without_ngff':
                x[:, -3:] = 0
            start = time.perf_counter()
            with torch.inference_mode():
                correction = adapter.model(torch.tensor((x-adapter.mean)/adapter.std)).numpy()
            prediction = base+OFFSETS[:, None]*correction
            methods[mode] = (prediction, time.perf_counter()-start+(sample['latency'] if mode == 'ngff_driver_adapter' else 0))
        for method, (prediction, latency) in methods.items():
            rows.append({'task': sample['task'], 'raw_episode_id': sample['episode'], 'anchor': sample['anchor'],
                         'method': method, 'metrics': scores(prediction, sample['truth'], horizons), 'latency_seconds': latency})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'offline_driver_domain_adaptation_not_grasp_success',
              'summary': aggregate(rows, horizon_keys), 'rows': rows, 'ngff_weights_updated': False,
              'maximum_forecast_horizon_seconds': HORIZON,
              'original_ngff_training_resumed': False, 'formal_test_used': False,
              'known_priors': 'axis-aligned red box and support; reflected CA additionally uses declared benchmark boundaries',
              'perception_backend_version': manifest['perception_backend_version'],
              'train_windows': sum(s['split'] == 'train' for s in samples),
              'validation_windows': sum(s['split'] == 'validation' for s in samples),
              'device': 'CPU under shared workstation load; latency is not an isolated RTX5090 benchmark'}
    report['latency_scope'] = 'predictor execution only; perception/file IO excluded and causal feature vectors precomputed'
    write('report.json', report)
    write('status.json', {'stage': 'adapter_validation_complete', 'formal_test_used': False, 'summary': report['summary']})
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
