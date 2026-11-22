"""Read-only saved-motion statistics for exactly ten completed VAL task reports.

No simulator, model, controller or source manifest is imported. Only already
saved velocity/acceleration values are reduced; joint positions are never used.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runtime/framework/formal_VAL_env_resume_v823/completed_ten_task_motion_summary_v838.json'
OLD = ('runtime/framework/formal_VAL_resume_v772/completed_six_task_motion_summary_v802.json',
       '7ab5df59f004a99f9735a281e5f4353c1de4a82f967046b1683d96d3c2301a0a')
LATEST = ('runtime/framework/latest_executor_continuation_v836.json',
          'dc9873b66843d6b6dfb0998822e0615786c2971a6c4105b1135ed6c2994b91d1')
EXTRA_REPORTS = (
    ('runtime/framework/formal_VAL_resume_v772/seventh_task_actual_results_v803.json', '0064b656b2f31828bf2b244bcc4931b42fe1bad78d17e5b13024612b39c6dee1'),
    ('runtime/framework/formal_VAL_resume_v772/eighth_task_actual_results_v806.json', '7192c68f03ac1f6787bb6b152d5dc38e8c2944bd66ccc8a211cebcdabe5a7411'),
    ('runtime/framework/formal_VAL_resume_v772/ninth_task_actual_results_v812.json', '58a533dd1c5b42f6d356e84ff8c6fae79961b9e5dce46deb288ae6b0beb5a784'),
    ('runtime/framework/formal_VAL_env_resume_v823/tenth_task_actual_results_v816.json', 'e5a0726f074a6544952477b625ac28ec35453d61c444ccb0a4e185f0fa64ccf4'),
)
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
POLICY = 'eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
PROFILE = 'd9a36e5ba8a73123c5deb65229d7fa151089145dca4b55c68c1e9ad1dec4e562'
PARAMETERS = dict(max_velocity=3., max_acceleration=12., max_jerk=1000., counted_dt=.05,
                  decision_ticks=10, budget_seconds=10.)
UNKNOWN = ('actual_joint_jerk_rad_s3', 'maximum_actual_joint_delta_rad',
           'actual_joint_path_length_rad', 'actual_TCP_path_length_m')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_checked(pointer, inputs, *, case=False):
    name, expected = pointer['path'], pointer['sha256']
    path = (ROOT/name).resolve(strict=True)
    require(not Path(name).is_absolute() and path.is_relative_to(ROOT), 'project-relative evidence required')
    if case:
        require('validation' in path.parts and path.name == 'case.json'
                and path.is_relative_to(ROOT/'artifacts/framework/multitask_r10'), 'VAL case evidence only')
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == expected, 'saved input SHA changed: '+name)
    require(name not in inputs or inputs[name] == expected, 'conflicting saved input SHA: '+name)
    inputs[name] = expected
    return json.loads(raw)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value):
    return value if isinstance(value, list) and len(value) == 7 and all(finite(x) for x in value) else None


def peak(values):
    saved = [float(x) for x in values if finite(x)]
    return max(saved) if saved else None


def vector_peak(vectors):
    return peak(abs(x) for v in vectors if vector(v) is not None for x in v)


def quantile(values, q):
    """Linear quantile at h=(n-1)*q; interpolate sorted neighboring values."""
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    h = (len(ordered)-1)*q
    lo, hi = math.floor(h), math.ceil(h)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(h-lo)


def statistics(values):
    values = list(values)
    valid = [float(x) for x in values if finite(x)]
    return dict(n=len(valid), total_cases=len(values), unknown_or_nonfinite_cases=len(values)-len(valid),
                min=min(valid) if valid else None, median=quantile(valid,.5), p95=quantile(valid,.95),
                max=max(valid) if valid else None)


def direct_receipts(case):
    actor = case.get('actor_metrics', {})
    if case['method'] == 'physmani':
        actions = actor.get('baseline_steps')
        if not isinstance(actions, list):
            return None
        receipts = [row.get('execution', {}).get('actual_speed_receipt') for row in actions]
    else:
        actions = actor.get('physmani_geometry_executor', {}).get('actions')
        if not isinstance(actions, list):
            return None
        receipts = [row.get('original_executor_receipt', {}).get('actual_speed_receipt') for row in actions]
    return [row for row in receipts if isinstance(row, dict)]


def reduce_saved_case(case):
    motion = case.get('all_stage_counted_joint_motion', {})
    events = motion.get('events')
    events = events if isinstance(events, list) else []
    velocities = [e.get('backward_window_velocity') for e in events]
    accelerations = [e.get('backward_window_acceleration') for e in events]
    acceleration_count = sum(vector(v) is not None for v in accelerations)
    saved_acceleration = motion.get('peak_actual_window_acceleration')
    acceleration = saved_acceleration if acceleration_count and finite(saved_acceleration) and saved_acceleration >= 0 else None
    receipts = direct_receipts(case)
    positive = [r for r in (receipts or []) if finite(r.get('command_ticks')) and r['command_ticks'] > 0]
    def direct_vector(key):
        return vector_peak(r.get(key) for r in positive)
    def direct_scalar(key):
        return peak(r.get(key) for r in positive)
    counters = [r.get('command_ticks') for r in (receipts or [])]
    direct_ticks = (sum(counters) if receipts is not None and all(finite(x) and x >= 0 for x in counters) else None)
    clock = case.get('counted_clock', {})
    alignment = case.get('alignment_receipt', {})
    metrics = dict(peak_window_joint_speed_rad_s=vector_peak(velocities),
        peak_window_joint_acceleration_rad_s2=acceleration,
        counted_ticks=clock.get('ticks'), counted_seconds=clock.get('seconds'),
        actual_physics_seconds=alignment.get('actual_physics_seconds'),
        actual_task_seconds=alignment.get('actual_task_seconds'), case_wall_seconds=case.get('wall_seconds'),
        direct_command_native_qdot_peak_rad_s=direct_vector('maximum_actual_native_reported_speed_by_joint_rad_per_s'),
        direct_command_average_speed_peak_rad_s=direct_vector('maximum_actual_average_speed_by_joint_rad_per_s'),
        direct_command_native_speed_fraction_peak=direct_scalar('maximum_actual_native_reported_speed_fraction'),
        direct_command_average_speed_fraction_peak=direct_scalar('maximum_actual_average_speed_fraction'),
        direct_command_ticks=direct_ticks, **{name:None for name in UNKNOWN})
    flag_counts = {}
    for flag in ('acceleration_passed','speed_passed','position_passed'):
        flags = [e.get(flag) for e in events]
        flag_counts[flag] = dict(passed=sum(v is True for v in flags), failed=sum(v is False for v in flags),
                                unevaluated=sum(v is None for v in flags),
                                invalid=sum(v is not True and v is not False and v is not None for v in flags))
    diagnostics = dict(window_records_present=bool(events), window_count=len(events),
        recorded_velocity_window_count=sum(vector(v) is not None for v in velocities),
        recorded_acceleration_window_count=acceleration_count, original_flag_counts=flag_counts,
        combined_window_checks_passed=motion.get('all_stage_actual_window_checks_passed'),
        direct_command_receipts=None if receipts is None else len(receipts),
        positive_tick_direct_command_receipts=len(positive))
    return metrics, diagnostics


def check_case(case, row, task):
    identity = case['identity']
    require(case.get('complete') is True and case.get('cleanup_completed') is True
            and case.get('infrastructure_error') is None, 'complete clean saved case required')
    require(case.get('split') == identity.get('split') == 'validation' and case.get('task') == task
            and identity.get('task') == task and case.get('method') == identity.get('method') == row['method'], 'VAL identity mismatch')
    raw = row['source_raw_episode_id']
    require(case.get('source_raw_episode_id') == identity.get('source_raw_episode_id') == raw,
            'original source ID mismatch')
    source_sha = identity['source_row_sha256']
    require(row.get('source_row_sha256', source_sha) == source_sha, 'source SHA mismatch')
    require(type(case.get('official_task_success')) is bool and type(case.get('native_terminal_failure')) is bool
            and case['official_task_success'] is row['official_task_success']
            and case['native_terminal_failure'] is row['native_terminal_failure'], 'native outcomes mismatch')
    require(case.get('formal_validation_metric') is True and case.get('formal_TEST_metric') is False
            and case.get('demo_only') is False and case.get('TEST_used_for_fitting') is False
            and case.get('Value_NGFF_FASTER_used') is False, 'formal three-method VAL scope required')
    frozen = case['full16_frozen_execution']
    require(frozen.get('policy_sha256') == POLICY and frozen.get('source_row_sha256') == source_sha
            and frozen.get('actual_split') == 'validation' and frozen.get('VAL_TEST_relabelled_as_TRAIN') is False,
            'frozen VAL policy/source binding mismatch')
    rml = case['common_author_RML_parameters']
    require(rml.get('actual_parameters') == PARAMETERS and rml.get('profile_sha256') == PROFILE
            and rml.get('complete') is True and rml.get('native_parameters_mutated') is False
            and rml.get('actual_motion_bound_claimed') is False, 'saved common RML attestation mismatch')
    motion = case.get('all_stage_counted_joint_motion', {})
    require(motion.get('actual_window_acceleration_criterion_rad_s2') == 6.
            and motion.get('planning_metadata_used_as_actual_acceleration_cap') is False
            and motion.get('policy_or_physics_modified') is False, 'original saved motion diagnostic semantics required')
    clock = case['counted_clock']; alignment = case['alignment_receipt']; saved_clock = alignment['clock']
    ticks = clock['ticks']
    require(type(ticks) is int and 0 < ticks <= 200 and clock.get('dt') == .05
            and finite(clock.get('seconds')) and abs(clock['seconds']-ticks*.05) <= 1e-9,
            'saved counted clock mismatch')
    require(saved_clock.get('ticks') == ticks and saved_clock.get('dt') == .05
            and saved_clock.get('budget_seconds') == 10.
            and abs(saved_clock['simulation_seconds']-clock['seconds']) <= 1e-9
            and sum(saved_clock['phases'].values()) == ticks, 'saved alignment clock mismatch')
    for key in ('actual_physics_seconds','actual_task_seconds'):
        require(finite(alignment.get(key)) and abs(alignment[key]-clock['seconds']) <= 1e-5,
                'native counted clock alignment mismatch: '+key)
    require(row.get('counted_ticks') == ticks, 'report tick count mismatch')
    return source_sha


def summarize(rows):
    metrics = {name:statistics(row['metrics'][name] for row in rows) for name in rows[0]['metrics']}
    diagnostics = dict(cases_with_window_records=0, cases_without_window_records=0, recorded_windows=0,
        acceleration_evaluated_windows=0, acceleration_unevaluated_windows=0, acceleration_invalid_flag_windows=0,
        acceleration6_diagnostic_failed_windows=0, cases_with_acceleration6_diagnostic_failure=0,
        native_success_cases_with_acceleration6_diagnostic_failure=0,
        cases_with_combined_window_check_failure=0, cases_with_unknown_combined_window_check=0,
        speed_passed_failed_windows=0, position_passed_failed_windows=0)
    for row in rows:
        d = row['recorded_window_diagnostics']; flags=d['original_flag_counts']; a=flags['acceleration_passed']
        diagnostics['cases_with_window_records' if d['window_records_present'] else 'cases_without_window_records'] += 1
        diagnostics['recorded_windows'] += d['window_count']
        diagnostics['acceleration_evaluated_windows'] += a['passed']+a['failed']
        diagnostics['acceleration_unevaluated_windows'] += a['unevaluated']
        diagnostics['acceleration_invalid_flag_windows'] += a['invalid']
        diagnostics['acceleration6_diagnostic_failed_windows'] += a['failed']
        diagnostics['cases_with_acceleration6_diagnostic_failure'] += a['failed'] > 0
        diagnostics['native_success_cases_with_acceleration6_diagnostic_failure'] += a['failed'] > 0 and row['official_task_success']
        diagnostics['cases_with_combined_window_check_failure'] += d['combined_window_checks_passed'] is False
        diagnostics['cases_with_unknown_combined_window_check'] += type(d['combined_window_checks_passed']) is not bool
        diagnostics['speed_passed_failed_windows'] += flags['speed_passed']['failed']
        diagnostics['position_passed_failed_windows'] += flags['position_passed']['failed']
    cohort = sorted({(row['task'],row['source_raw_episode_id'],row['source_row_sha256']) for row in rows})
    return dict(case_count=len(rows), source_count=len(cohort), cohort_sha256=digest(cohort),
        native_success_cases=sum(row['official_task_success'] for row in rows),
        native_false_cases=sum(not row['official_task_success'] for row in rows), all_native_outcomes_retained=True,
        metrics=metrics, recorded_window_diagnostics=diagnostics)


def compare_old(old, rows):
    previous = {row['case']['path']:row for row in old['case_evidence']}
    selected = [row for row in rows if row['case']['path'] in previous]
    require(len(selected) == 351 and len(previous) == 351, 'exact old six-task case set required')
    for row in selected:
        before = previous[row['case']['path']]
        require(before['case'] == row['case'] and before['metrics'] == row['metrics']
                and before['recorded_window_diagnostics'] == row['recorded_window_diagnostics'],
                'old per-case saved reductions not reproduced: '+row['case']['path'])
    max_error = 0.; comparisons = 0; exact = 0
    def compare(actual, expected):
        nonlocal max_error, comparisons, exact
        for name,value in expected.items():
            if isinstance(value,dict): compare(actual[name],value)
            elif finite(value):
                comparisons += 1; exact += actual[name] == value
                max_error=max(max_error,abs(actual[name]-value))
                require(math.isclose(actual[name],value,rel_tol=1e-13,abs_tol=1e-12), 'old six statistic differs: '+name)
            else: require(actual[name] == value, 'old six cohort/unknown diagnostic differs: '+name)
    for method in METHODS:
        current=summarize([row for row in selected if row['method']==method])
        reference=old['overall_by_method'][method]
        compare({key:current[key] for key in ('metrics','recorded_window_diagnostics')},
                {key:reference[key] for key in ('metrics','recorded_window_diagnostics')})
        require(current['case_count']==reference['case_count']==117 and current['native_success_cases']==reference['native_success_cases']
                and current['native_false_cases']==reference['native_false_cases'], 'old overall native cohort changed')
    for reference in old['by_task_method']:
        current=summarize([row for row in selected if row['task']==reference['task'] and row['method']==reference['method']])
        compare({key:current[key] for key in ('metrics','recorded_window_diagnostics')},
                {key:reference[key] for key in ('metrics','recorded_window_diagnostics')})
    return dict(complete=True, per_case_metrics_and_flags_exactly_reproduced=351,
        task_method_groups_reproduced=18, overall_methods_reproduced=3, cases_per_method=117,
        numeric_fields_compared=comparisons, numeric_fields_exact=exact, maximum_absolute_numeric_difference=max_error,
        float_comparison_tolerance=dict(relative=1e-13,absolute=1e-12), prior_summary=dict(path=OLD[0],sha256=OLD[1]))


def main():
    require(not OUT.exists(), 'preserve existing v838 summary; never overwrite')
    inputs={}
    old=load_checked(dict(path=OLD[0],sha256=OLD[1]),inputs)
    latest=load_checked(dict(path=LATEST[0],sha256=LATEST[1]),inputs)
    require(latest['task_reports']['full_tasks']==10 and latest['task_reports']['full_sources']==197
            and latest['task_reports']['full_cases']==591 and latest['task_reports']['tenth_task']==dict(path=EXTRA_REPORTS[-1][0],sha256=EXTRA_REPORTS[-1][1]),
            'exact reviewed ten-task continuation required')
    report_pointers=old['reports']+[dict(path=p,sha256=s) for p,s in EXTRA_REPORTS]
    rows=[];task_counts={};seen=set();job_ids=set()
    for pointer in report_pointers:
        report=load_checked(pointer,inputs);task=report['task']
        require(report.get('complete') is True and report.get('split')=='validation' and tuple(report['methods'])==METHODS
                and report.get('missing_or_invalid_fields',[])==[] and report['policy_sha256']==POLICY,
                'complete reviewed same-policy VAL task report required')
        require(task not in task_counts and report['method_cases']==3*report['source_cases']==len(report['case_evidence']),
                'unique complete task cohort required')
        task_counts[task]=report['source_cases'];cohorts={method:set() for method in METHODS}
        successes=Counter()
        for record in report['case_evidence']:
            require(record['case']['path'] not in seen and record['job_id'] not in job_ids, 'duplicate VAL case/job')
            seen.add(record['case']['path']);job_ids.add(record['job_id'])
            case=load_checked(record['case'],inputs,case=True)
            source_sha=check_case(case,record,task)
            metrics,diagnostics=reduce_saved_case(case)
            method=record['method'];cohorts[method].add((record['source_raw_episode_id'],source_sha))
            successes[method]+=case['official_task_success']
            rows.append(dict(task=task,method=method,source_raw_episode_id=record['source_raw_episode_id'],
                source_row_sha256=source_sha,source_report=pointer,case=record['case'],
                official_task_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
                metrics=metrics,recorded_window_diagnostics=diagnostics,
                saved_RML_clock_cleanup_checks_passed=True))
        require(cohorts[METHODS[0]]==cohorts[METHODS[1]]==cohorts[METHODS[2]]
                and len(cohorts[METHODS[0]])==report['source_cases'], 'three methods must retain identical raw/source-SHA tuples')
        for method in METHODS:
            require(sum(row['task']==task and row['method']==method for row in rows)==report['source_cases']
                    and successes[method]==report['statistics'][method]['native_success'], 'all native outcomes retained and report counts match')
    require(len(rows)==591 and len(task_counts)==10 and sum(task_counts.values())==197, 'exact ten-task 197x3 cohort required')
    reproduction=compare_old(old,rows)
    for pointer in old['source_semantics']+[old['hard_budget_disclosure']['source']]:
        load_checked(pointer,inputs) if pointer['path'].endswith('.json') else require(
            hashlib.sha256((ROOT/pointer['path']).read_bytes()).hexdigest()==pointer['sha256'], 'unchanged saved metric source required')
        inputs[pointer['path']]=pointer['sha256']
    inputs[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    test_path='tests/test_completed_VAL_motion_v838.py'
    inputs[test_path]=hashlib.sha256((ROOT/test_path).read_bytes()).hexdigest()
    output=dict(schema='r10_completed_ten_task_saved_motion_summary_v838',created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,read_only_analysis=True,split='validation',task_count=10,source_count=197,case_count=591,
        methods=list(METHODS),task_source_counts=task_counts,reports=report_pointers,
        same_fixed_source_cohort_for_all_methods=True,policy_sha256=POLICY,
        cohort_definition='All 591 valid cases from exactly ten complete reports; 197 identical task/raw/source-SHA tuples per method, including every native False. Preserved failed environment attempts are not valid cases or native outcomes.',
        statistical_definition=dict(old['statistical_definition'],overall='pool197 equally weighted cases per method; never average task statistics',
            quantiles='Sort finite per-case values; h=(n-1)*q; x[floor(h)]+(x[ceil(h)]-x[floor(h)])*(h-floor(h)), q=.5 or .95'),
        metric_definitions=old['metric_definitions'],diagnostic_definition=old['diagnostic_definition'],
        common_RML_planner_configuration=dict(old['common_RML_planner_configuration'],
            saved_configurations=[dict(parameters=PARAMETERS,cases=591)],saved_profile_hash_counts={PROFILE:591},
            saved_native_parameters_mutated_flag_counts={'False':591},saved_complete_flag_counts={'True':591}),
        hard_budget_disclosure=old['hard_budget_disclosure'],source_semantics=old['source_semantics'],
        by_task_method=[dict(task=task,method=method,**summarize([r for r in rows if r['task']==task and r['method']==method])) for task in task_counts for method in METHODS],
        overall_by_method={method:summarize([row for row in rows if row['method']==method]) for method in METHODS},
        old_six_task_reproduction=reproduction,case_evidence=rows,
        CPU_validation=dict(tests=dict(path=test_path,sha256=inputs[test_path]),pure_CPU_tests_passed=6,
            script_and_tests_compile_checked=True,main_imports_simulator_model_or_controller=False),
        provenance_checks=dict(all_591_case_SHAs_match_complete_reports=True,all_case_and_job_pointers_unique=True,
            all_ten_reports_complete=True,saved_RML_clock_cleanup_checked_cases=591,
            full_749_or_530_audits_reexecuted=False,physics_or_policy_metrics_recomputed=False,
            speed_peak_reduced_only_from_saved_velocity_vectors=True,acceleration_scalar_used_only_with_valid_saved_window=True,
            primary_all_stage_motion_only_not_alignment_mirror=True),
        full945_VAL_release_granted=False,TRAIN_or_TEST_data_read=False,VAL_used_for_parameter_tuning=False,
        execution_configuration_threshold_or_source_changed=False,preflight_native_or_GPU_started=False,
        TXT_or_handoff_written=False,inputs_sha256=inputs)
    with OUT.open('x') as f:
        f.write(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,output=str(OUT.relative_to(ROOT)),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
        case_count=591,methods={method:{key:output['overall_by_method'][method]['metrics'][key]
        for key in ('peak_window_joint_speed_rad_s','peak_window_joint_acceleration_rad_s2')} for method in METHODS},
        old_six_reproduction=reproduction),ensure_ascii=False))


if __name__=='__main__':
    main()
