"""Describe observed r09 candidate selections without tuning or counterfactuals.

Only one step at each candidate attempt's selected_at is used. Persistent
selector diagnostics on subsequent tracking steps are deliberately excluded.
The output is an on-policy, attempt-weighted description, not independent
calibration validation, a causal estimate, or a model-selection input.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/aligned_r09'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
MATCH_TOLERANCE_SECONDS = 1e-7


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest_bytes(path.read_bytes())


def read_with_hash(path):
    data = path.read_bytes()
    return json.loads(data), digest_bytes(data)

def case_input_trace(record, *, predictor, ranking, config, model_sha, evaluator_sha):
    """Verify this case's own identity; distinct cases need not share identities."""
    errors = []
    receipt = record.get('case_process_receipt')
    if not isinstance(receipt, dict):
        return {}, ['case_process_receipt_missing_or_invalid']
    trace = dict(receipt=receipt, input_identity=record.get('inputs_sha256'))
    if record.get('case_isolation') != 'fresh_simulator_process_per_episode':
        errors.append('fresh_case_isolation_missing')
    if receipt.get('version') != 'fresh_process_r09_v1':
        errors.append('case_process_receipt_version')
    evaluator = 'scripts/evaluate_aligned_r09.py'
    if receipt.get('child_evaluator') != evaluator or receipt.get('child_evaluator_sha256') != evaluator_sha:
        errors.append('case_child_evaluator_source_sha256')
    parsed = {}
    for field, hash_field in (('child_inputs_path', 'child_inputs_sha256'), ('child_record_path', 'child_record_sha256')):
        try:
            raw_path = receipt[field]
            if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
                raise ValueError('expected a relative path')
            path = (ROOT / raw_path).resolve()
            if not path.is_relative_to(BASE.resolve()):
                raise ValueError('expected a path within r09 artifacts')
            value, actual_sha = read_with_hash(path)
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            if actual_sha != receipt.get(hash_field):
                errors.append(hash_field + '_mismatch')
            parsed[field] = value
            trace[field] = str(path.relative_to(ROOT))
            trace[hash_field] = actual_sha
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(field + ':' + str(error))
    inputs = parsed.get('child_inputs_path')
    if inputs is not None:
        identity = digest_bytes(json.dumps({key: value for key, value in inputs.items() if key != 'sha256'},
                                           sort_keys=True).encode())
        trace['declared_input_identity'] = inputs.get('sha256')
        trace['recomputed_input_identity'] = identity
        trace['evaluator_source_files'] = inputs.get('files', {})
        if inputs.get('sha256') != identity or record.get('inputs_sha256') != identity:
            errors.append('case_record_input_identity_mismatch')
        if inputs.get('files', {}).get(evaluator) != evaluator_sha:
            errors.append('case_inputs_child_evaluator_source_sha256')
        for field, expected in (('predictor', predictor), ('ranking', ranking), ('split', record['split']),
                                ('raw_episode_id', record['raw_episode_id']), ('raw_episode_task', record['task'])):
            if inputs.get(field) != expected:
                errors.append('case_input_' + field + '_mismatch')
        if inputs.get('config') != config:
            errors.append('case_input_config_mismatch')
        if inputs.get('value_sha256') != (model_sha if ranking == 'value' else None):
            errors.append('case_input_value_sha256_mismatch')
    child = parsed.get('child_record_path')
    if child is not None and any(key not in record or record[key] != value for key, value in child.items()):
        errors.append('parent_changed_original_child_fields')
    expected_model = model_sha if ranking == 'value' else None
    if record.get('value_checkpoint_sha256') != expected_model:
        errors.append('case_value_checkpoint_sha256_mismatch')
    trace['expected_value_checkpoint_sha256'] = expected_model
    trace['recorded_value_checkpoint_sha256'] = record.get('value_checkpoint_sha256')
    trace['accepted'] = not errors
    return trace, errors


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, minimum=None, median=None, p90=None, maximum=None)
    return dict(n=len(values), mean=float(values.mean()), minimum=float(values.min()),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def observed_metrics(events):
    """Score only completed, actually selected supported attempts."""
    selected = [e for e in events if e.get('selected_probability') is not None and e.get('outcome_complete')]
    p = np.asarray([e['selected_probability'] for e in selected], dtype=float)
    y = np.asarray([e['observed_attempt_success'] for e in selected], dtype=float)
    if not len(y):
        return dict(n=0, positive=0, brier=None, ece_5_bins=None, mean_probability=None,
                    observed_positive_rate=None, source_episodes=0, calibration_bins=[])
    bins = []
    ece = 0.
    for i in range(5):
        members = (p >= i/5) & ((p < (i+1)/5) if i < 4 else (p <= 1.))
        if members.any():
            predicted, observed = float(p[members].mean()), float(y[members].mean())
            ece += float(members.mean()) * abs(predicted-observed)
            bins.append(dict(low=i/5, high=(i+1)/5, n=int(members.sum()),
                             probability=predicted, observed_positive_rate=observed))
    return dict(n=len(y), positive=int(y.sum()), brier=float(np.mean((p-y)**2)), ece_5_bins=ece,
                mean_probability=float(p.mean()), observed_positive_rate=float(y.mean()),
                source_episodes=len({e['episode'] for e in selected}), calibration_bins=bins)


def finite_number(value, *, lower=None, upper=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        return False
    return (lower is None or value >= lower) and (upper is None or value <= upper)


def selections(record, *, ranking, path):
    """Match each attempt to exactly one selection-time step, never retained steps."""
    events, issues = [], []
    steps = record.get('steps', [])
    used_steps = set()
    attempts = record.get('candidate_attempts')
    if not isinstance(attempts, list):
        return [], [dict(file=path, attempt_id=None, error='candidate_attempts_missing_or_not_list')]
    for index, attempt in enumerate(attempts):
        identity = dict(file=path, attempt_id=attempt.get('attempt_id', index), selected_at=attempt.get('selected_at'))
        selected_at = attempt.get('selected_at')
        if not finite_number(selected_at, lower=0):
            issues.append(dict(**identity, error='invalid_selection_time'))
            continue
        matching = [i for i, step in enumerate(steps) if finite_number(step.get('time')) and
                    abs(step['time']-selected_at) <= MATCH_TOLERANCE_SECONDS]
        if len(matching) != 1:
            issues.append(dict(**identity, error='missing_selection_step' if not matching else 'ambiguous_selection_step',
                               matching_steps=len(matching)))
            continue
        step_index = matching[0]
        if step_index in used_steps:
            issues.append(dict(**identity, error='selection_step_reused_by_multiple_attempts'))
            continue
        used_steps.add(step_index)
        selector = steps[step_index].get('diagnostics', {}).get('selector')
        if not isinstance(selector, dict) or not selector:
            issues.append(dict(**identity, error='missing_selection_diagnostics'))
            continue
        feasible = selector.get('feasible_candidates')
        if type(feasible) is not int or feasible <= 0 or feasible != attempt.get('feasible_candidates'):
            issues.append(dict(**identity, error='feasible_candidate_count_mismatch'))
        event = dict(episode=record['raw_episode_id'], attempt_id=attempt.get('attempt_id', index),
                     selected_at=selected_at, step_index=step_index, step_time=steps[step_index]['time'],
                     feasible_candidates=feasible, selected_probability=None,
                     observed_attempt_success=attempt.get('official_task_success'),
                     outcome_complete=attempt.get('complete') is True,
                     outcome_reason=attempt.get('outcome_reason'),
                     outcome_observed_until=attempt.get('outcome_observed_until'))
        if ranking == 'value':
            selected = selector.get('selected_candidate')
            geometry = selector.get('geometry_candidate')
            valid_indexes = (type(selected) is int and type(geometry) is int and type(feasible) is int
                             and 0 <= selected < feasible and 0 <= geometry < feasible)
            if not valid_indexes:
                issues.append(dict(**identity, error='selected_or_geometry_candidate_missing_or_invalid'))
            supported = selector.get('supported_candidates')
            if type(supported) is not int or type(feasible) is not int or not 0 <= supported <= feasible:
                issues.append(dict(**identity, error='supported_candidate_count_invalid'))
                supported = None
            probability = selector.get('selected_success_probability')
            recorded_probability = attempt.get('predicted_probability')
            if probability is None and recorded_probability is not None or probability is not None and (
                    not finite_number(probability, lower=0, upper=1) or
                    not finite_number(recorded_probability, lower=0, upper=1) or
                    abs(probability-recorded_probability) > 1e-10):
                issues.append(dict(**identity, error='attempt_and_selection_probability_mismatch'))
            if finite_number(probability, lower=0, upper=1):
                event['selected_probability'] = probability
            bootstrap = selector.get('selected_bootstrap_std')
            if not finite_number(bootstrap, lower=0):
                issues.append(dict(**identity, error='selected_bootstrap_std_missing_or_invalid'))
                bootstrap = None
            fallback = selector.get('fallback')
            actual_override = selected != geometry if valid_indexes else None
            if actual_override and fallback is not None:
                issues.append(dict(**identity, error='override_with_geometry_fallback'))
            event.update(selected_candidate=selected, geometry_candidate=geometry,
                         proposed_candidate=selector.get('proposed_candidate'), actual_override=actual_override,
                         fallback=fallback, supported_candidates=supported,
                         supported_candidate_fraction=supported/feasible if supported is not None and feasible else None,
                         selected_bootstrap_std=bootstrap,
                         estimated_gain=selector.get('estimated_gain'), required_gain=selector.get('required_gain'),
                         learned_ranking_enabled=selector.get('learned_ranking_enabled'),
                         score_kind=selector.get('score_kind'))
        if not event['outcome_complete'] or type(event['observed_attempt_success']) is not bool:
            issues.append(dict(**identity, error='missing_or_censored_observed_attempt_outcome'))
            event['outcome_complete'] = False
        events.append(event)
    return events, issues


def usage_summary(records, events, issues, ranking):
    result = dict(episodes=len(records), native_successes=sum(r['official_task_success'] for r in records),
                  candidate_attempts=sum(len(r.get('candidate_attempts', [])) for r in records),
                  matched_selection_events=len(events), matching_issue_count=len(issues),
                  episodes_without_attempts=sum(not r.get('candidate_attempts') for r in records))
    if ranking != 'value':
        result['critic_usage'] = 'geometry-only reference; no probability or learned override claim'
        return result
    overrides = [event for event in events if event.get('actual_override') is True]
    proposed = [event for event in events if type(event.get('proposed_candidate')) is int
                and event['proposed_candidate'] != event.get('geometry_candidate')]
    candidates = [event for event in events if event.get('supported_candidate_fraction') is not None]
    total_feasible = sum(e['feasible_candidates'] for e in candidates)
    total_supported = sum(e['supported_candidates'] for e in candidates)
    probabilities = [e['selected_probability'] for e in events if e['selected_probability'] is not None]
    bootstrap = [e['selected_bootstrap_std'] for e in events if e.get('selected_bootstrap_std') is not None]
    supported_bootstrap = [e['selected_bootstrap_std'] for e in events
                           if e['selected_probability'] is not None and e.get('selected_bootstrap_std') is not None]
    result.update(actual_override_selections=len(overrides),
                  actual_override_fraction_of_matched=len(overrides)/len(events) if events else None,
                  episodes_with_actual_override=len({e['episode'] for e in overrides}),
                  proposed_non_geometry_selections=len(proposed),
                  fallback_counts=dict(sorted(Counter(e.get('fallback') or 'no_fallback_learned_override' for e in events).items())),
                  learned_ranking_enabled_selections=sum(e.get('learned_ranking_enabled') is True for e in events),
                  unknown_override_status_selections=sum(e.get('actual_override') is None for e in events),
                  unmatched_attempts=result['candidate_attempts']-len(events),
                  feasible_candidate_instances=total_feasible, supported_candidate_instances=total_supported,
                  supported_candidate_fraction_pooled=total_supported/total_feasible if total_feasible else None,
                  supported_candidate_fraction_by_selection=distribution([e['supported_candidate_fraction'] for e in candidates]),
                  selected_supported_attempts=len(probabilities), selected_unsupported_attempts=len(events)-len(probabilities),
                  calibrated_selected_probability=distribution(probabilities),
                  selected_bootstrap_std_all=distribution(bootstrap),
                  selected_bootstrap_std_supported=distribution(supported_bootstrap),
                  observed_supported_attempt_metrics=observed_metrics(events))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    manifest_path = ROOT / ('artifacts/framework/protocol/validation_manifest.json' if args.split == 'validation'
                            else 'artifacts/framework/pipeline_v1/official_test_manifest.json')
    manifest, manifest_sha = read_with_hash(manifest_path)
    config_path = ROOT / 'config/framework/aligned_protocol_r09.json'
    config, config_sha = read_with_hash(config_path)
    evaluator_sha = sha(ROOT / 'scripts/evaluate_aligned_r09.py')
    model_paths = {predictor: ROOT / config['cv_local_value' if predictor == 'cv' else 'context_local_value']
                   for predictor in ('cv', 'context')}
    model_shas = {predictor: sha(path) for predictor, path in model_paths.items()}
    expected = {(row['task'], row['raw_episode_id']) for row in manifest['episodes'] if row['task'] in TASKS}
    groups, episode_hashes, input_hashes, child_hashes, problems = {}, {}, {}, {}, []
    for predictor in ('cv', 'context'):
        for ranking in ('geometry', 'value'):
            name = f'{args.split}_{predictor}_{ranking}'
            directory = BASE / name
            records, events_by_task, issues_by_task, seen = [], {task: [] for task in TASKS}, {task: [] for task in TASKS}, Counter()
            group_errors = []
            per_case_inputs = {}
            infrastructure = sorted(str(path.relative_to(ROOT)) for path in directory.glob('*.infrastructure_error.json'))
            for path in sorted(directory.glob('pick*__episode*.json')):
                if path.name.endswith('infrastructure_error.json'):
                    continue
                label = str(path.relative_to(ROOT))
                try:
                    record, episode_hashes[label] = read_with_hash(path)
                    key = (record.get('task'), record.get('raw_episode_id'))
                    if key not in expected or record.get('split') != args.split or record.get('predictor') != predictor or record.get('ranking') != ranking:
                        group_errors.append(dict(file=label, error='episode_identity_split_or_group_mismatch'))
                        continue
                    if type(record.get('official_task_success')) is not bool:
                        group_errors.append(dict(file=label, error='native_task_success_missing_or_not_bool'))
                        continue
                    seen[key] += 1
                    records.append(record)
                    trace, input_errors = case_input_trace(record, predictor=predictor, ranking=ranking,
                        config=config, model_sha=model_shas[predictor], evaluator_sha=evaluator_sha)
                    per_case_inputs[label] = trace
                    for error in input_errors:
                        group_errors.append(dict(file=label, error=error))
                    if 'child_inputs_path' in trace:
                        input_hashes[trace['child_inputs_path']] = trace['child_inputs_sha256']
                    if 'child_record_path' in trace:
                        child_hashes[trace['child_record_path']] = trace['child_record_sha256']
                    events, issues = selections(record, ranking=ranking, path=label)
                    events_by_task[record['task']].extend(events)
                    issues_by_task[record['task']].extend(issues)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    group_errors.append(dict(file=label, error='malformed_episode:' + str(error)))
            duplicates = [dict(task=key[0], episode=key[1], copies=n) for key, n in seen.items() if n > 1]
            if duplicates:
                group_errors.append(dict(error='duplicate_episode', duplicates=duplicates))
            missing = [dict(task=task, episode=episode) for task, episode in sorted(expected-set(seen))]
            per_task = {}
            all_issues = []
            for task in TASKS:
                task_records = [record for record in records if record['task'] == task]
                task_expected = sum(key[0] == task for key in expected)
                per_task[task] = usage_summary(task_records, events_by_task[task], issues_by_task[task], ranking)
                per_task[task].update(expected_episodes=task_expected, complete_coverage=len(task_records) == task_expected,
                                      selection_events=events_by_task[task], matching_issues=issues_by_task[task])
                all_issues.extend(issues_by_task[task])
            recorded_identities = sorted({r.get('inputs_sha256', '') for r in records})
            recorded_models = sorted({r['value_checkpoint_sha256'] for r in records if r.get('value_checkpoint_sha256')})
            coverage_complete = set(seen) == expected and not duplicates
            groups[name] = dict(coverage=dict(completed_episodes=len(records), expected_episodes=len(expected),
                    complete=coverage_complete, missing=missing, infrastructure_errors=infrastructure),
                matching_complete=not all_issues, input_errors=group_errors, per_task=per_task,
                provenance=dict(recorded_versions=sorted({r.get('version', '') for r in records}),
                    case_isolation='fresh_simulator_process_per_episode', cached_geometry=False,
                    episode_inputs_sha256=recorded_identities, distinct_case_input_identities=len(recorded_identities),
                    per_case_inputs=per_case_inputs,
                    recorded_controller_sha256=sorted({r.get('controller_sha256', '') for r in records}),
                    recorded_selection_policy_sha256=sorted({r.get('selection_policy_sha256', '') for r in records}),
                    recorded_sensor_protocol_sha256=sorted({r.get('sensor_protocol_sha256', '') for r in records}),
                    recorded_value_checkpoint_sha256=recorded_models,
                    frozen_value_checkpoint=str(model_paths[predictor].relative_to(ROOT)),
                    frozen_value_checkpoint_sha256=model_shas[predictor]))
            problems.extend(dict(group=name, **error) for error in [*group_errors, *all_issues])
    source_names = ('scripts/analyze_success_usage_r09.py', 'scripts/evaluate_aligned_r09.py',
                    'config/framework/aligned_protocol_r09.json', 'our_mani/risk_v3/selector.py',
                    'our_mani/risk_v4/model.py', 'our_mani/sensor_alignment/author_rgbd.py')
    sources = {name: sha(ROOT / name) for name in source_names if (ROOT / name).exists()}
    model_files = {}
    for predictor in ('cv', 'context'):
        for filename in ('value.pt', 'value_support.json', 'report.json'):
            path = model_paths[predictor].with_name(filename)
            if path.exists():
                model_files[str(path.relative_to(ROOT))] = sha(path)
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), version='r09_descriptive_selection_usage_v1',
        split=args.split, complete=all(group['coverage']['complete'] for group in groups.values()) and not problems,
        coverage_complete=all(group['coverage']['complete'] for group in groups.values()),
        diagnostics_valid=not problems, expected_episodes_per_group=40 if args.split == 'validation' else 80,
        selection_matching_tolerance_seconds=MATCH_TOLERANCE_SECONDS, groups=groups, errors=problems,
        provenance=dict(manifest=str(manifest_path.relative_to(ROOT)), manifest_sha256=manifest_sha,
                        protocol_config=str(config_path.relative_to(ROOT)), protocol_config_sha256=config_sha,
                        analysis_and_current_code_sha256=sources, model_files_sha256=model_files,
                        input_episode_files_sha256=episode_hashes, per_case_inputs_files_sha256=input_hashes,
                        preserved_child_episode_files_sha256=child_hashes),
        interpretation=[
            'Each candidate_attempt contributes exactly once, matched by selected_at to one step.time. Retained diagnostics are excluded.',
            'Each r09 episode has its own fresh simulator process and child input identity. Group-level common input identities are neither assumed nor fabricated.',
            'All four geometry/value groups are newly evaluated fresh cases; there are no inherited geometry rollout caches.',
            'Actual override means selected_candidate differs from geometry_candidate in the same observed selection; a different proposed candidate alone is not an override.',
            'Supported selected attempts are those with an observed selected_success_probability, cross-checked against the attempt record.',
            'Brier/ECE are attempt-uniform on-policy descriptions of completed selected supported attempts, conditional on the deployed controller and support filter.',
            'Multiple attempts share one source episode and are correlated; no independent-sample confidence interval is estimated.',
            'VAL sources were used for calibration and development; these metrics are not independent calibration generalization estimates.',
            'Bootstrap std measures ensemble disagreement and is not a calibrated confidence interval.',
            'No outcomes for unselected candidates are read or inferred. Critic scoring, actual changed selections, and task success are distinct.',
            'R09 uses fresh simulator processes to remove inter-episode fixture reuse. Simulator variation and paired-condition acceptance still matter; success differences alone do not establish critic benefit.',
            'This descriptive output must not be used to modify model weights, calibration, support rules, or selection thresholds.'])
    path = BASE / f'success_selection_usage_{args.split}.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(json.dumps(dict(report=str(path.relative_to(ROOT)), complete=output['complete'],
        diagnostics_valid=output['diagnostics_valid'], errors=len(problems),
        groups={name: dict(completed=group['coverage']['completed_episodes'], expected=group['coverage']['expected_episodes'],
            per_task={task: {key: row.get(key) for key in ('candidate_attempts', 'matched_selection_events',
                'actual_override_selections', 'episodes_with_actual_override', 'matching_issue_count',
                'supported_candidate_fraction_pooled', 'observed_supported_attempt_metrics')} for task, row in group['per_task'].items()})
            for name, group in groups.items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
