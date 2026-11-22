"""Fit/calibrate a local success predictor using disjoint TRAIN episode groups.

Independent VAL is evaluated once after fitting and calibration. This does not
estimate success of FASTER's preceding approach or constitute a formal test.
"""
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
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.special import expit
from our_mani.local_learning import MLP, probability_metrics
from our_mani.success_prediction import OUTCOMES, feature_vector, group_partitions, outcome_label, read_rollouts


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--outcome', choices=tuple(OUTCOMES), default='stable_lift_and_official_task_success')
    parser.add_argument('--context-dim', type=int, default=0)
    parser.add_argument('--updates', type=int, default=2000)
    args = parser.parse_args()
    if args.context_dim < 0 or args.updates < 1:
        raise ValueError('invalid dimensions or training budget')
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'value.pt').exists():
        raise FileExistsError('choose a new version; existing probability weights are immutable')
    train, train_hash, train_full = read_rollouts(ROOT/args.data, split='train', outcome=args.outcome)
    val, val_hash, val_full = read_rollouts(ROOT/args.validation, split='validation', outcome=args.outcome)
    controllers = {r['controller_sha256'] for r in train+val}
    if len(controllers) != 1:
        raise ValueError('TRAIN and VAL must use the same frozen controller')
    train_groups = {(r['task'], r['source_episode']) for r in train}
    val_groups = {(r['task'], r['source_episode']) for r in val}
    if train_groups & val_groups:
        raise ValueError('independent VAL overlaps TRAIN raw episodes')
    partitions = group_partitions(train)
    x = np.asarray([feature_vector(r, args.context_dim) for r in train])
    y = np.asarray([outcome_label(r, args.outcome, dataset_full_task=train_full) for r in train])
    xv = np.asarray([feature_vector(r, args.context_dim) for r in val])
    yv = np.asarray([outcome_label(r, args.outcome, dataset_full_task=val_full) for r in val])
    context_hash = None
    if args.context_dim:
        hashes = {r['dynamics_context']['encoder_sha256'] for r in train+val}
        if len(hashes) != 1:
            raise ValueError('success predictor contexts require one frozen encoder')
        context_hash = next(iter(hashes))
    masks = {name: np.array([(r['task'], r['source_episode']) in groups for r in train])
             for name, groups in partitions.items()}
    for name, mask in masks.items():
        if len(np.unique(y[mask])) != 2:
            raise ValueError(f'{name} needs actual positive and negative outcomes; collect more TRAIN data')
    mean, std = x[masks['fit']].mean(0), np.maximum(x[masks['fit']].std(0), .001)
    torch.set_num_threads(2)
    torch.manual_seed(810)
    xt = torch.tensor((x-mean)/std, dtype=torch.float32)
    yt = torch.tensor(y[:, None], dtype=torch.float32)
    fit, select = np.flatnonzero(masks['fit']), np.flatnonzero(masks['select'])
    model = MLP(x.shape[1], 1, 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    best, best_state, selected_update, history = float('inf'), None, None, []
    for step in range(args.updates):
        if (ROOT/'runtime/framework/local_state/STOP').exists():
            raise InterruptedError('local model stop marker is set')
        indices = fit[torch.randint(len(fit), (min(128, len(fit)),)).numpy()]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt[indices]), yt[indices])
        if not torch.isfinite(loss):
            raise ValueError('non-finite success predictor loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % 100 == 0 or step == args.updates-1:
            with torch.inference_mode():
                score = float(torch.nn.functional.binary_cross_entropy_with_logits(model(xt[select]), yt[select]))
            history.append({'update': step+1, 'fit_bce': float(loss.detach()), 'selection_bce': score})
            if score < best:
                best, selected_update = score, step+1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        calibration_logits = model(xt[masks['calibration']]).numpy()[:, 0].astype(np.float64)
        val_logits = model(torch.tensor((xv-mean)/std, dtype=torch.float32)).numpy()[:, 0]
    yc = y[masks['calibration']]
    def calibration_nll(log_temperature):
        logits = calibration_logits/np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., logits)-yc*logits))
    calibrated = minimize_scalar(calibration_nll, bounds=(-3., 3.), method='bounded')
    if not calibrated.success:
        raise ValueError('probability temperature calibration failed')
    temperature = float(np.exp(calibrated.x))
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'value',
        'outcome': args.outcome, 'outcome_definition': OUTCOMES[args.outcome],
        'probability_scope': 'local handover through recorded capture/transport budgets; excludes preceding FASTER approach',
        'controller_sha256': next(iter(controllers)), 'dataset_sha256': train_hash,
        'validation_dataset_sha256': val_hash, 'source': args.data,
        'context_dim': args.context_dim, 'context_encoder_sha256': context_hash,
        'training_scope': 'two_task_development_model_not_formal_benchmark',
        'selected_update': selected_update, 'completed_updates': args.updates,
        'formal_test_used': False, 'goal_geometry_conditioned': False,
        'group_partitions': {name: sorted(groups) for name, groups in partitions.items()},
    }
    checkpoint = {'state_dict': best_state, 'feature_mean': mean, 'feature_std': std,
                  'hidden': 64, 'temperature': temperature, 'metadata': metadata}
    path = out/'value.pt'
    torch.save(checkpoint, path)
    # Freeze support using fit features and selection features only, never VAL.
    local_slack = np.r_[[.02]*3, [.05]*6, [.1]*6, 0., 0., 0.]
    slack = np.r_[local_slack, np.full(args.context_dim, .02)]
    scale = np.maximum(std, np.r_[[.02]*3, [.05]*6, [.05]*6, 1., 1., .25, np.full(args.context_dim, .05)])
    normalized = (x[masks['fit']]-mean)/scale
    tree = cKDTree(normalized)
    threshold = float(np.quantile(tree.query((x[masks['select']]-mean)/scale)[0], .99))
    lower, upper = x[masks['fit']].min(0)-slack, x[masks['fit']].max(0)+slack
    support = {
        'kind': 'TRAIN_only_support_for_controller_specific_Value', 'formal_test_used': False,
        'source': args.data, 'dataset_sha256': train_hash, 'controller_sha256': next(iter(controllers)),
        'value_checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'lower': lower.tolist(), 'upper': upper.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist(),
        'normalized_fit_features': normalized.tolist(), 'maximum_nearest_fit_distance': threshold,
        'distance_threshold_rule': 'q99 distance on disjoint TRAIN selection episodes',
        'fit_n': int(masks['fit'].sum()), 'train_group_heldout_n': int(masks['select'].sum()),
    }
    write(out/'value_support.json', support)
    probabilities, uncalibrated = expit(val_logits/temperature), expit(val_logits)
    accepted = ((xv >= lower).all(1) & (xv <= upper).all(1)
                & (tree.query((xv-mean)/scale)[0] <= threshold))
    prior = np.full(len(yv), y[masks['fit']].mean())
    metrics = probability_metrics(probabilities, yv)
    report = {
        'metadata': metadata, 'checkpoint': str(path.relative_to(ROOT)),
        'checkpoint_sha256': support['value_checkpoint_sha256'], 'temperature': temperature,
        'train_records': len(train), 'independent_validation_records': len(val),
        'partitions': {name: {'episodes': len(partitions[name]), 'records': int(mask.sum()),
                             'positive': int(y[mask].sum())} for name, mask in masks.items()},
        'validation': metrics,
        'validation_uncalibrated': probability_metrics(uncalibrated, yv),
        'constant_fit_prevalence_baseline': probability_metrics(prior, yv),
        'supported_validation': {'n': int(accepted.sum()), 'coverage': float(accepted.mean()),
            'metrics': probability_metrics(probabilities[accepted], yv[accepted]) if accepted.any() else None},
        'per_task': {task: probability_metrics(probabilities[[r['task'] == task for r in val]],
                                               yv[[r['task'] == task for r in val]])
                     for task in sorted({r['task'] for r in val})},
        'thresholds': [{'threshold': t, 'n': int((accepted & (probabilities >= t)).sum()),
                        'actual_success_rate': float(yv[accepted & (probabilities >= t)].mean())
                        if (accepted & (probabilities >= t)).any() else None} for t in (.5, .7, .8)],
        'limitations': ['small perturbed local development rollouts', 'no FASTER approach probability',
                        'no dynamics context in baseline' if not args.context_dim else 'frozen causal context only',
                        'goal geometry is not an input', 'threshold selection is not performed on VAL'],
        'training_history': history,
    }
    write(out/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('metadata', 'training_history', 'per_task')}, indent=2))


if __name__ == '__main__':
    main()
