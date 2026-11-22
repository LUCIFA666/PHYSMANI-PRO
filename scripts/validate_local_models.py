"""Independent VAL probability calibration and paired local servo outcomes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.local_learning import SuccessValue, probability_metrics
from our_mani.evaluation import wilson


def read_rows(directory):
    result = {}
    for path in directory.glob('*.json'):
        row = json.loads(path.read_text())
        if row.get('initial_features') is not None and row.get('failure') != 'deadline_stop':
            if row['split'] != 'validation':
                raise RuntimeError('independent validation must use VAL source episodes')
            result[row['key']] = row
    return result


def cluster_interval(rows, values, samples=5000):
    groups = {}
    for row, value in zip(rows, values):
        groups.setdefault((row['task'], row['source_episode']), []).append(float(value))
    # Each source episode is the independent resampling unit; its local
    # perturbations remain together, and contribute their within-episode mean.
    means = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(810)
    draws = means[rng.integers(len(means), size=(samples, len(means)))].mean(1)
    return np.quantile(draws, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpc', default='artifacts/framework/local_policy/mpc_validation_v1')
    parser.add_argument('--learned', default='artifacts/framework/local_policy/servo_validation_v1')
    parser.add_argument('--models', default='artifacts/framework/local_policy/models_v1')
    parser.add_argument('--out', default='artifacts/framework/local_policy/validation_report.json')
    args = parser.parse_args()
    mpc, learned = read_rows(ROOT/args.mpc), read_rows(ROOT/args.learned)
    value = SuccessValue.load(ROOT/args.models/'value.pt')
    if {r['controller_sha256'] for r in mpc.values()} != {value.metadata['controller_sha256']}:
        raise RuntimeError('Value controller differs from validation outcome controller')
    rows = list(mpc.values())
    features = np.asarray([r['initial_features'] for r in rows])
    labels = np.asarray([r['stable_lift'] for r in rows], dtype=int)
    probabilities = value(features)
    metrics = probability_metrics(probabilities, labels)
    fit = json.loads((ROOT/args.models/'value_training.json').read_text())
    prior_metrics = fit['report']['metrics']['constant_train_prevalence_baseline']
    prevalence = prior_metrics['calibration_bins'][0]['probability']
    metrics['constant_train_prevalence_baseline'] = probability_metrics(np.full(len(rows), prevalence), labels)
    thresholds = []
    for threshold in (.5, .7, .8):
        chosen = probabilities >= threshold
        selected = [r for r, keep in zip(rows, chosen) if keep]
        outcome = labels[chosen]
        thresholds.append({'probability_threshold': threshold, 'selected_n': int(chosen.sum()),
                           'actual_stable_lift_rate': float(outcome.mean()) if len(outcome) else None,
                           'episode_cluster_bootstrap_95': cluster_interval(selected, outcome) if len(outcome) else None})
    paired, skipped = [], []
    for key in sorted(set(mpc)&set(learned)):
        a, b = mpc[key], learned[key]
        signature = ('offset_m', 'capture_budget_seconds', 'delay_ticks', 'warmup_velocity_match_fraction',
                     'reset_target_position', 'reset_velocity', 'reset_acceleration')
        if any(a.get(field) != b.get(field) for field in signature):
            raise RuntimeError('local perturbation identities differ between policies')
        # Different post-reset physics histories can leave tiny motor residuals;
        # expose their size instead of silently asserting identical states.
        initial_difference = np.abs(np.array(a['initial_features'])-b['initial_features'])
        paired.append({'task': a['task'], 'source_episode': a['source_episode'], 'key': key,
                       'mpc_stable_lift': a['stable_lift'], 'learned_stable_lift': b['stable_lift'],
                       'mpc_official_success': a['official_task_success'], 'learned_official_success': b['official_task_success'],
                       'initial_relative_position_difference_m': float(np.linalg.norm(initial_difference[:3])),
                       'initial_relative_velocity_difference_m_s': float(np.linalg.norm(initial_difference[3:6]))})
    # Fixed engineering tolerances describe which resets are sufficiently
    # matched. Retain all rows and the descriptive full-cohort counts; do not
    # attach a paired inference claim to larger motor-state discrepancies.
    matched = [r for r in paired if r['initial_relative_position_difference_m'] <= .005
               and r['initial_relative_velocity_difference_m_s'] <= .025]
    excluded = [r for r in paired if r not in matched]
    differences = np.array([int(r['learned_stable_lift'])-int(r['mpc_stable_lift']) for r in matched])
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'heldout_perturbed_local_handover_not_official_benchmark',
              'formal_test_used': False, 'mpc_value_calibration': metrics, 'high_value_regions': thresholds,
              'per_task_value': {task: probability_metrics(probabilities[[r['task'] == task for r in rows]],
                                  labels[[r['task'] == task for r in rows]]) for task in sorted({r['task'] for r in rows})},
              'paired_servo': {'n': len(matched), 'mpc_stable_lift': sum(r['mpc_stable_lift'] for r in matched),
                  'learned_stable_lift': sum(r['learned_stable_lift'] for r in matched),
                  'stable_lift_delta': float(differences.mean()) if len(paired) else None,
                  'episode_cluster_bootstrap_95': cluster_interval(matched, differences) if len(matched) else None,
                  'mpc_official_success': sum(r['mpc_official_success'] for r in matched),
                  'learned_official_success': sum(r['learned_official_success'] for r in matched),
                  'reset_match_position_tolerance_m': .005, 'reset_match_velocity_tolerance_m_s': .025,
                  'excluded_mismatched_reset_keys': [r['key'] for r in excluded],
                  'limitation': 'development reset-matched subset; not a randomized formal benchmark comparison',
                  'maximum_initial_position_difference_m': max((r['initial_relative_position_difference_m'] for r in paired), default=None),
                  'maximum_initial_velocity_difference_m_s': max((r['initial_relative_velocity_difference_m_s'] for r in paired), default=None)},
              'full_cohort_descriptive_only': {'n':len(paired),
                  'mpc_stable_lift': sum(r['mpc_stable_lift'] for r in paired),
                  'learned_stable_lift': sum(r['learned_stable_lift'] for r in paired),
                  'mpc_official_success': sum(r['mpc_official_success'] for r in paired),
                  'learned_official_success': sum(r['learned_official_success'] for r in paired)},
              'unpaired_keys': sorted(set(mpc)^set(learned)), 'paired_rows': paired}
    out = ROOT/args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'paired_rows'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
