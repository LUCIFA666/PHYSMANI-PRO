"""Resume the fixed R09 comparison with separate per-case CPU/GPU queues.

Only the scheduling layer changes. Frozen evaluators, models, thresholds,
calibration gates, failed VAL screen and paired-condition checks are retained.
The existing baseline VAL must finish before its comparison lock permits TEST.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import continue_fixed_evaluation_r09 as fixed
from our_mani.benchmark_protocol import file_sha256, verify_lock

run = fixed.run
BASE = run.BASE
SOURCE_SHA = file_sha256(Path(__file__))
CPU_WORKERS = 4
GPU_WORKERS = 4
VALIDATION_GPU_WORKERS = 4
PARALLEL_LOCK_PATH = BASE / 'parallel_evaluation_inputs.json'
TRANSITION_PATH = BASE / 'scheduling_transition.json'


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def execution_metadata():
    return dict(
        version='r09_per_case_parallel_queues_v1',
        cpu_workers=CPU_WORKERS,
        gpu_workers=GPU_WORKERS,
        baseline_validation_gpu_workers=VALIDATION_GPU_WORKERS,
        scheduling_unit='one manifest case',
        ours_submission_order='manifest row then four ours variants',
        baseline_submission_order='manifest row',
        validation_must_complete_before_TEST=True,
        wrapper=relative(Path(__file__)),
        wrapper_sha256=SOURCE_SHA,
        transition=relative(TRANSITION_PATH),
        transition_sha256=file_sha256(TRANSITION_PATH),
        previous_fixed_evaluation_inputs=relative(BASE / 'fixed_evaluation_extra_inputs.json'),
        previous_fixed_evaluation_inputs_sha256=file_sha256(BASE / 'fixed_evaluation_extra_inputs.json'),
    )


def preserve_transition():
    """Record the scheduling-only change without inspecting success labels."""
    if file_sha256(Path(__file__)) != SOURCE_SHA:
        raise ValueError('parallel continuation source changed since launch')
    if TRANSITION_PATH.exists():
        value = read(TRANSITION_PATH)
        required = dict(
            version='r09_scheduling_transition_v1',
            replacement_wrapper=relative(Path(__file__)),
            replacement_wrapper_sha256=SOURCE_SHA,
            cpu_workers=CPU_WORKERS,
            gpu_workers=GPU_WORKERS,
            baseline_validation_gpu_workers=VALIDATION_GPU_WORKERS,
            models_changed=False,
            thresholds_changed=False,
            existing_results_preserved=True,
            validation_must_complete_before_TEST=True,
        )
        if any(value.get(key) != expected for key, expected in required.items()):
            raise ValueError('existing scheduling transition differs')
        if value.get('previous_wrapper_sha256') != file_sha256(ROOT / value['previous_wrapper']):
            raise ValueError('previous continuation source changed')
        return value
    previous = ROOT / 'scripts/continue_fixed_evaluation_r09.py'
    interrupted_name = 'pick_moving_target_on_the_table_high_speed__episode586'
    parent_marker = BASE / 'validation_physmani_v6' / (interrupted_name + '.infrastructure_error.json')
    child_directory = BASE / 'validation_physmani_v6/cases' / interrupted_name
    child_marker = child_directory / (interrupted_name + '.infrastructure_error.json')
    interruption_parent = read(parent_marker)
    interruption_child = read(child_marker)
    if (interruption_parent.get('counted_as_policy_failure') is not False
            or interruption_child.get('error') != "InterruptedError('r09 stop requested')"
            or interruption_child.get('formal_test_used') is not False):
        raise ValueError('controlled interruption evidence differs from the scheduling transition')
    counts = {
        split: {
            variant: len([path for path in run.group_path(split, variant).glob('pick*__episode*.json')
                          if not path.name.endswith('infrastructure_error.json')])
            for variant in run.VARIANTS
        }
        for split in ('validation', 'test')
    }
    value = dict(
        version='r09_scheduling_transition_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        user_request='User explicitly requested VAL and formal TEST to run as concurrently as possible; relayed by root.',
        change='Replace method/task-long workers with per-case queues; TEST uses independent four-worker CPU and GPU pools.',
        previous_wrapper=relative(previous),
        previous_wrapper_sha256=file_sha256(previous),
        replacement_wrapper=relative(Path(__file__)),
        replacement_wrapper_sha256=SOURCE_SHA,
        cpu_workers=CPU_WORKERS,
        gpu_workers=GPU_WORKERS,
        baseline_validation_gpu_workers=VALIDATION_GPU_WORKERS,
        models_changed=False,
        thresholds_changed=False,
        evaluators_or_physics_changed=False,
        existing_results_preserved=True,
        existing_completed_parent_records=counts,
        controlled_scheduling_interruption=dict(
            task='pick_moving_target_on_the_table_high_speed',
            raw_episode_id='episode586',
            reason='root requested success_r09_STOP to leave the old scheduler; child observed STOP during its unfinished case',
            last_logged_simulation_seconds=5.5,
            original_baseline_VAL_completed_preserved=38,
            unfinished_cases_remaining=['episode586', 'episode588'],
            counted_as_policy_failure=False,
            parent_marker=relative(parent_marker),
            parent_marker_sha256=file_sha256(parent_marker),
            parent_marker_content=interruption_parent,
            partial_child_directory=relative(child_directory),
            child_marker=relative(child_marker),
            child_marker_sha256=file_sha256(child_marker),
            child_marker_content=interruption_child,
            log=interruption_parent.get('log'),
            existing_marker_and_partial_directory_retained=True,
            retry='frozen run_case creates a nested attempt directory and launches a new independent process; episode588 runs for the first time',
        ),
        cache_interpretation='Existing valid completed cases are retained and verified by the frozen run_case; counts do not imply execution under the new scheduler.',
        validation_must_complete_before_TEST=True,
        cross_split_overlap='All remaining baseline VAL cases finish and five-method VAL conditions pass before comparison/extra/parallel locks permit any TEST case.',
        unchanged_requirements=[
            'original failed VAL performance screen and SHA retained',
            'fixed hard_checks: TRAIN coverage and valid finite nested calibration',
            'unchanged evaluator, reset, sensor, physics and clock checks',
            'all five-method VAL and TEST paired initial-condition checks',
            'complete five-method fixed-case evaluation; no outcome-based case selection',
        ],
        performance_screen_passed=False,
        stopping_rule_changed_after_VAL=True,
        deployment_or_benefit_validated=False,
    )
    run.write(TRANSITION_PATH, value)
    return value


def parallel_lock(comparison):
    """Extend, and never rewrite, the original continuation's extra lock."""
    if file_sha256(Path(__file__)) != SOURCE_SHA:
        raise ValueError('parallel continuation source changed since launch')
    extra = fixed.extra_lock(comparison)
    verify_lock(ROOT, extra)
    preserve_transition()
    metadata = execution_metadata()
    files = dict(extra['files'])
    for source in (Path(__file__), TRANSITION_PATH, BASE / 'fixed_evaluation_extra_inputs.json'):
        name, digest = relative(source), file_sha256(source)
        if name in files and files[name] != digest:
            raise ValueError('conflicting parallel input identity: ' + name)
        files[name] = digest
    if PARALLEL_LOCK_PATH.exists():
        value = read(PARALLEL_LOCK_PATH)
        verify_lock(ROOT, value)
        if value.get('parallel_execution') != metadata:
            raise ValueError('parallel execution metadata changed')
        if any(value['files'].get(name) != digest for name, digest in files.items()):
            raise ValueError('parallel lock is not a superset of the unchanged extra lock')
        if (value.get('development_performance_screen_passed') is not False
                or value.get('stopping_rule_changed_after_VAL') is not True
                or value.get('fixed_model_comparison_not_benefit_validation') is not True):
            raise ValueError('parallel lock lacks failed-screen disclosure')
        return value
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        files=files,
        development_performance_screen_passed=False,
        stopping_rule_changed_after_VAL=True,
        fixed_model_comparison_not_benefit_validation=True,
        parallel_execution=metadata,
    )
    run.write(PARALLEL_LOCK_PATH, value)
    verify_lock(ROOT, value)
    return value


def await_cases(futures):
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        # Let already-running cases preserve their receipts, but do not start
        # the rest of either queue after an infrastructure/protocol failure.
        for future in futures:
            future.cancel()
        raise


def validation_conditions():
    """Use the same parity checks without rewriting already-frozen VAL JSON."""
    comparison_path = BASE / 'comparison_v6_inputs.json'
    if not comparison_path.exists():
        return run.parity('validation', run.VARIANTS)
    comparison = read(comparison_path)
    verify_lock(ROOT, comparison)
    tolerances = read(BASE / 'acceptance_tolerances.json')
    expected = {(row['task'], row['raw_episode_id']) for row in run.manifest('validation')}
    data = {variant: {(row['task'], row['raw_episode_id']): row
                      for row in run.records('validation', variant)} for variant in run.VARIANTS}
    errors, checks = [], []
    for variant, rows in data.items():
        if set(rows) != expected:
            errors.append(variant + ' coverage')
    for a, b in itertools.combinations(run.VARIANTS, 2):
        for key in sorted(expected & data[a].keys() & data[b].keys()):
            check = run.public_parity(data[a][key], data[b][key], tolerances)
            checks.append(dict(a=a, b=b, key=key, **check))
            if not check['accepted']:
                errors.append(dict(a=a, b=b, key=key, problems=check['problems']))
    return dict(passed=not errors, errors=errors, checks=checks,
                split='validation', tolerances_unchanged=True,
                preserved_frozen_condition_file=True)


def assert_original_screen(release):
    if file_sha256(BASE / 'success_acceptance.json') != release['original_performance_screen_sha256']:
        raise ValueError('original failed VAL screen changed')


def main():
    try:
        # STOP is deliberately owned by root; this entry point never clears it.
        if (ROOT / 'runtime/framework/success_r09_STOP').exists():
            raise InterruptedError('r09 STOP; root must finish the old wrapper transition first')
        lock, gate = fixed.hard_checks()
        release = fixed.decision(gate)
        assert_original_screen(release)
        preserve_transition()
        run.status(
            'baseline_validation',
            '逐case四GPU队列补齐基线VAL，已有结果保留；原性能筛选失败与硬门槛保持。五组VAL配对后启动四CPU＋四GPU完整TEST。',
            development_performance_screen_passed=gate['passed'],
            baseline_validation_gpu_workers=VALIDATION_GPU_WORKERS,
            planned_test_cpu_workers=CPU_WORKERS,
            planned_test_gpu_workers=GPU_WORKERS,
            scheduling_transition=relative(TRANSITION_PATH),
        )
        with ThreadPoolExecutor(max_workers=VALIDATION_GPU_WORKERS) as pool:
            futures = [pool.submit(run.run_case, 'validation', 'physmani', row, lock)
                       for row in run.manifest('validation')]
            await_cases(futures)
        # Each newly executed frozen run_case refreshes its group itself. Do
        # not rewrite any VAL report that a previous comparison lock froze.
        if not (BASE / 'comparison_v6_inputs.json').exists():
            run.update_group('validation', 'physmani')
        assert_original_screen(release)
        for variant in run.VARIANTS:
            for row in run.records('validation', variant):
                fixed.check_clock(row)
        conditions = validation_conditions()
        if not conditions['passed']:
            raise ValueError('five-method VAL initial conditions failed; no tolerance relaxation')
        comparison = run.freeze('comparison_v6_inputs.json', acceptance=True)
        stronger_lock = parallel_lock(comparison)
        verify_lock(ROOT, stronger_lock)
        metadata = stronger_lock['parallel_execution']
        run.status(
            'official_TEST',
            '五组VAL严格配对通过；冻结候选逐case并行完成400次TEST：四CPU跑四种ours，四GPU共享80次PhysMani。原性能筛选失败仍披露。',
            development_performance_screen_passed=gate['passed'],
            parallel_evaluation_inputs=relative(PARALLEL_LOCK_PATH),
            parallel_execution=metadata,
        )
        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as cpu_pool, \
                ThreadPoolExecutor(max_workers=GPU_WORKERS) as gpu_pool:
            futures = []
            rows = run.manifest('test')
            for row in rows:
                for variant in run.VARIANTS[:-1]:
                    futures.append(cpu_pool.submit(run.run_case, 'test', variant, row, stronger_lock))
            for row in rows:
                futures.append(gpu_pool.submit(run.run_case, 'test', 'physmani', row, stronger_lock))
            await_cases(futures)
        for variant in run.VARIANTS:
            run.update_group('test', variant)
        assert_original_screen(release)
        conditions = run.parity('test', run.VARIANTS)
        test_usage = run.selection_usage('test')
        if (test_usage.get('complete') is not True
                or test_usage.get('coverage_complete') is not True
                or test_usage.get('diagnostics_valid') is not True
                or test_usage.get('errors')):
            raise ValueError('full TEST selection diagnostics are incomplete or invalid')
        subprocess.run([sys.executable, 'scripts/summarize_aligned_r09_v6.py'], cwd=ROOT, check=True)
        if not conditions['passed']:
            raise ValueError('full TEST strict pairwise initial conditions failed')
        verify_lock(ROOT, stronger_lock)
        assert_original_screen(release)
        report = read(BASE / 'final_report.json')
        if (report.get('complete') is not True or report.get('protocol_acceptance_passed') is not True
                or report.get('total_test_runs') != 400 or report.get('expected_test_runs') != 400
                or report.get('acceptance_errors')):
            raise ValueError('full fixed TEST summary has incomplete or failed protocol acceptance')
        report.update(
            development_performance_screen_passed=gate['passed'],
            fixed_candidate_evaluation_release=release,
            selection_usage_report='success_selection_usage_test.json',
            evaluation_interpretation='complete frozen-candidate comparison after a disclosed failed VAL performance screen; not validated deployment or guaranteed improvement',
            parallel_evaluation_inputs=relative(PARALLEL_LOCK_PATH),
            parallel_evaluation_inputs_sha256=file_sha256(PARALLEL_LOCK_PATH),
            parallel_execution=metadata,
        )
        run.write(BASE / 'final_report.json', report)
        run.status(
            'complete',
            'TRAIN、五组完整VAL、400次正式TEST及两段视频完成；四CPU＋四GPU逐case调度，原VAL性能筛选失败和正式结果一并披露。',
            development_performance_screen_passed=gate['passed'],
            report='artifacts/framework/aligned_r09/final_report.json',
            parallel_evaluation_inputs=relative(PARALLEL_LOCK_PATH),
            parallel_execution=metadata,
        )
        run.report()
    except Exception as exc:
        run.status('needs_attention', str(exc), error=repr(exc))
        raise


if __name__ == '__main__':
    main()
