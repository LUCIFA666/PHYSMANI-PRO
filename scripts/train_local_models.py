"""Train local BC and controller-specific Value without accessing formal test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from our_mani.local_learning import MLP, LocalServo, SuccessValue, probability_metrics

STOP = ROOT/'runtime/framework/local_state/STOP'


def write(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def records(directory):
    rows = []
    for path in sorted(directory.glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('initial_features') is None or row.get('failure') == 'deadline_stop':
            continue
        if row.get('split') != 'train':
            raise RuntimeError('local model training requires TRAIN rollout outcomes')
        rows.append((path, row))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='artifacts/framework/local_policy/mpc_train_v2')
    parser.add_argument('--out', default='artifacts/framework/local_policy/models_v1')
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--only-value', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory, out = ROOT/args.data, ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = records(directory)
    hashes = {r['controller_sha256'] for _, r in rows}
    if len(hashes) != 1:
        raise RuntimeError('Value outcomes must come from one frozen controller version')
    groups = sorted({(r['task'], r['source_episode']) for _, r in rows})
    rng = np.random.default_rng(810)
    heldout = set()
    for task in sorted({t for t, ep in groups}):
        selected = [g for g in groups if g[0] == task]
        indices = rng.permutation(len(selected))[:max(1, len(selected)//5)]
        heldout.update(selected[i] for i in indices)
    buckets = {name: {'servo_x': [], 'servo_y': [], 'value_x': [], 'value_y': []} for name in ('fit', 'select')}
    for path, row in rows:
        bucket = buckets['select' if (row['task'], row['source_episode']) in heldout else 'fit']
        with np.load(path.with_suffix('.npz')) as arrays:
            if len(arrays['features']):
                bucket['servo_x'].append(arrays['features'].copy())
                bucket['servo_y'].append(arrays['actions'][:, :6].copy())
        bucket['value_x'].append(row['initial_features'])
        bucket['value_y'].append(float(row['stable_lift']))
    if any(not bucket['servo_x'] or len(bucket['value_x']) < 10 for bucket in buckets.values()):
        raise RuntimeError('insufficient grouped TRAIN rollouts for fitting and selection')
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'controller_sha256': next(iter(hashes)),
        'outcome_definition': 'actual native grasp plus object >=8cm above table continuously for >=0.3s within recorded budget',
        'rollout_records': len(rows), 'train_groups': len(groups)-len(heldout), 'train_heldout_groups': len(heldout),
        'source': str(directory.relative_to(ROOT)), 'formal_test_used': False,
        'training_scope': 'two_task_development_model', 'selection_split': 'TRAIN grouped by task and raw episode',
        'dataset_sha256': hashlib.sha256(b''.join(p.read_bytes() for p, _ in rows)).hexdigest()}
    reports = {}
    modes = [('value', 18, 1, 64)] if args.only_value else [('servo', 17, 6, 128), ('value', 18, 1, 64)]
    for kind, dimension, outputs, hidden in modes:
        data = {}
        for name, bucket in buckets.items():
            if kind == 'servo':
                x, y = np.concatenate(bucket['servo_x']), np.concatenate(bucket['servo_y'])
            else:
                x, y = np.asarray(bucket['value_x'], dtype=np.float32), np.asarray(bucket['value_y'], dtype=np.float32)[:, None]
            data[name] = (x, y)
        mean = data['fit'][0].mean(0)
        std = np.maximum(data['fit'][0].std(0), .001)
        action_mean, action_std = np.zeros(outputs), np.ones(outputs)
        if kind == 'servo':
            action_mean = data['fit'][1].mean(0)
            action_std = np.maximum(data['fit'][1].std(0), .02)
        x_train, y_train = data['fit']
        x_select, y_select = data['select']
        x_train = torch.tensor((x_train-mean)/std, dtype=torch.float32)
        y_train = torch.tensor((y_train-action_mean)/action_std, dtype=torch.float32)
        x_select = torch.tensor((x_select-mean)/std, dtype=torch.float32)
        y_select_tensor = torch.tensor((y_select-action_mean)/action_std, dtype=torch.float32)
        torch.manual_seed(810)
        model = MLP(dimension, outputs, hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
        best, best_step, best_state, history = float('inf'), 0, None, []
        for step in range(args.updates):
            if STOP.exists():
                break
            indices = torch.randint(len(x_train), (min(256, len(x_train)),))
            prediction = model(x_train[indices])
            if kind == 'servo':
                loss = torch.nn.functional.smooth_l1_loss(prediction, y_train[indices], beta=.5)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            if step % 100 == 0 or step == args.updates-1:
                with torch.inference_mode():
                    select_prediction = model(x_select)
                    selection = float(torch.square(select_prediction-y_select_tensor).mean()) if kind == 'servo' else float(
                        torch.nn.functional.binary_cross_entropy_with_logits(select_prediction, y_select_tensor))
                history.append({'step': step+1, 'fit_loss': float(loss), 'selection_loss': selection})
                if selection < best:
                    best, best_step = selection, step+1
                    best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                write(out.parent/'training_status.json', {'stage': 'fitting_local_'+kind, 'updates': step+1,
                        'selected_update': best_step, 'formal_test_used': False, 'rollout_records': len(rows)})
        if best_state is None:
            raise InterruptedError('stopped before local model update')
        checkpoint = {'state_dict': best_state, 'feature_mean': mean, 'feature_std': std, 'hidden': hidden,
                      'action_mean': action_mean, 'action_std': action_std, 'temperature': 1.,
                      'metadata': {**provenance, 'kind': kind, 'selected_update': best_step, 'completed_updates': step+1}}
        path = out/(kind+'.pt')
        torch.save(checkpoint, path)
        if kind == 'servo':
            learned = LocalServo(checkpoint)
            predicted = np.array([learned(x) for x in data['select'][0]])
            metrics = {'selection_frames': len(predicted),
                       'translation_velocity_rmse_m_s': float(np.sqrt(np.square(predicted[:, :3]-y_select[:, :3]).mean())),
                       'angular_velocity_rmse_rad_s': float(np.sqrt(np.square(predicted[:, 3:]-y_select[:, 3:]).mean()))}
        else:
            value = SuccessValue(checkpoint)
            metrics = probability_metrics(value(data['select'][0]), y_select[:, 0])
            prior = float(np.mean(data['fit'][1]))
            metrics['constant_train_prevalence_baseline'] = probability_metrics(np.full(len(y_select), prior), y_select[:, 0])
        reports[kind] = {'checkpoint': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'selected_update': best_step, 'completed_updates': step+1, 'metrics': metrics}
        write(out/(kind+'_training.json'), {'provenance': provenance, 'report': reports[kind], 'history': history})
    record = {'stage': 'controller_specific_value_trained' if args.only_value else 'local_servo_and_controller_specific_value_trained',
              'provenance': provenance, 'models': reports,
              'closed_loop_learned_servo_validated': False, 'independent_validation_value_evaluated': False}
    write(out/'report.json', record)
    write(out.parent/'training_status.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
