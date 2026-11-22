"""Separate original first-source runtime, motion and task outcomes for all16x3."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha

METHODS = ('cv_geometry', 'context_geometry', 'physmani')


def arg(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def motion(case):
    direct = case.get('all_stage_counted_joint_motion')
    if direct is None:
        direct = (case.get('alignment_receipt') or {}).get('all_stage_counted_joint_motion_v191')
    if direct is not None:
        passed = direct.get('all_stage_actual_window_checks_passed')
        if type(passed) is bool:
            return ('pass' if passed else 'fail'), dict(
                source_schema=direct.get('schema'), counted_windows=len(direct.get('events', [])),
                measured_peak_acceleration=direct.get('peak_actual_window_acceleration',
                    direct.get('maximum_actual_window_acceleration')))
    reference = (case.get('actor_metrics') or {}).get('continuous_reference_feedforward', {})
    failed = reference.get('observed_motion_qualification_failed')
    if type(failed) is bool and reference.get('observed_events'):
        return ('fail' if failed else 'pass'), dict(source_schema=reference.get('schema'),
            dispatch_confirmations=len(reference.get('actual_dispatch_confirmations', [])),
            observed_windows=len(reference.get('observed_events', [])))
    return 'unmeasured', dict(reason='no recognized all-stage actual motion receipt; task success is not motion proof')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        raise ValueError('fresh desktop R10 artifact namespace required')
    owner = ROOT/'runtime/framework/native_resource_owner_v19'
    status = json.loads((owner/'status.json').read_text())
    plan_path = ROOT/'artifacts/framework/multitask_r10/all16_executor_joint_train_v1/plan.json'
    plan = json.loads(plan_path.read_text())
    tasks = {r['task']: r['fixed_smoke_source'] for r in plan['tasks']}
    assert len(tasks) == 16 and tuple(plan['methods']) == METHODS
    requests = [json.loads(f.read_text()) for f in (owner/'requests').glob('*.json')]
    rows = []
    for task, raw in tasks.items():
        for method in METHODS:
            matched = [r for r in requests if arg(r['argv'], '--task') == task
                and arg(r['argv'], '--raw-episode-id') == raw
                and arg(r['argv'], '--method') == method and arg(r['argv'], '--split') == 'train'
                and arg(r['argv'], '--out') is not None]
            started = []
            pending = []
            for request in matched:
                job = request['job_id']
                launch_path = owner/'jobs'/job/'launch.json'
                if launch_path.exists():
                    launch = json.loads(launch_path.read_text())
                    assert launch['request'] == request
                    started.append((launch['created_utc'], job, request, launch))
                elif job not in status['completed'] and job not in status['active_jobs']:
                    pending.append(job)
            row = dict(task=task, raw=raw, method=method, latest_started_job=None,
                runtime_state='not_started', runtime_completed_without_infrastructure_error=False,
                task_native_success=None, motion_status='unmeasured', pending_jobs=sorted(pending),
                formal_evaluation_ready=False)
            if started:
                _, job, request, launch = max(started, key=lambda r: (r[0], r[1]))
                terminal = status['completed'].get(job)
                row.update(latest_started_job=job, launched_utc=launch['created_utc'], terminal=terminal,
                    latest_worker=request['argv'][2])
                if terminal is None:
                    proc = Path('/proc')/str(launch['launcher_pid'])/'stat'
                    live = proc.exists() and int(proc.read_text().rsplit(')', 1)[1].split()[19]) == launch['launcher_start_ticks']
                    row.update(runtime_state='running' if live else 'owner_terminal_pending',
                        launcher_live_now=live)
                elif terminal['returncode'] != 0 or not terminal['complete']:
                    row['runtime_state'] = 'infrastructure_error'
                else:
                    case_path = ROOT/arg(request['argv'], '--out')/'case.json'
                    if not case_path.exists():
                        row['runtime_state'] = 'missing_case_receipt'
                    else:
                        case = json.loads(case_path.read_text())
                        assert case['task'] == task and case['method'] == method
                        assert case['split'] == 'train' and not case.get('formal_test_used', False)
                        clean = (case.get('complete') is True and case.get('infrastructure_error') is None
                            and case.get('cleanup_completed') is True)
                        row.update(runtime_state='completed' if clean else 'incomplete_case_receipt',
                            runtime_completed_without_infrastructure_error=clean,
                            case=dict(path=str(case_path.relative_to(ROOT)), sha256=file_sha(case_path)),
                            task_native_success=case.get('official_task_success'),
                            native_terminal_failure=case.get('native_terminal_failure'),
                            policy_failure=case.get('policy_failure', case.get('failure')))
                        row['motion_status'], row['motion_receipt'] = motion(case)
                        clock = case.get('counted_clock') or (case.get('alignment_receipt') or {}).get('clock', {})
                        seconds = clock.get('seconds', clock.get('simulation_seconds', case.get('simulation_seconds')))
                        row.update(counted_clock=clock, counted_seconds=seconds,
                            full10s_observed=type(seconds) in (float, int) and math.isclose(seconds, 10., abs_tol=1e-6))
                        config_path = ROOT/arg(request['argv'], '--config')
                        config = json.loads(config_path.read_text())
                        row['configured_context_checkpoint'] = config.get('context_checkpoint')
            rows.append(row)
    assert len(rows) == 48 and len({(r['task'], r['method']) for r in rows}) == 48
    summary = {method: dict(cases=16,
        runtime_completed=sum(r['runtime_completed_without_infrastructure_error'] for r in rows if r['method'] == method),
        runtime_running=sum(r['runtime_state'] == 'running' for r in rows if r['method'] == method),
        runtime_infrastructure_error=sum(r['runtime_state'] == 'infrastructure_error' for r in rows if r['method'] == method),
        motion_pass=sum(r['motion_status'] == 'pass' for r in rows if r['method'] == method),
        motion_fail=sum(r['motion_status'] == 'fail' for r in rows if r['method'] == method),
        motion_unmeasured=sum(r['motion_status'] == 'unmeasured' for r in rows if r['method'] == method),
        native_successes=sum(r['task_native_success'] is True for r in rows if r['method'] == method)) for method in METHODS}
    report = dict(schema='r10_all16_original_first_source_runtime_motion_task_status_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), owner_snapshot_utc=status['updated_utc'],
        source_plan=dict(path=str(plan_path.relative_to(ROOT)), sha256=file_sha(plan_path)),
        latest_started_selection='latest actual owner launch per exact original first TRAIN source/method; no success selection',
        summary=summary, cases=rows,
        task_success_does_not_define_runtime_readiness=True,
        motion_unmeasured_is_not_pass=True,
        fixed_four_all_success_count_is_not_runtime_ready_task_count=True,
        policy_freeze_full16_dispatch_and_formal_release_still_required=True,
        formal_VAL_TEST=False, TEST_read=False,
        script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)))
    out.mkdir()
    atomic_json(out/'report.json', report)
    print(json.dumps(dict(summary=summary, formal_VAL_TEST=False, TEST_read=False)), flush=True)


if __name__ == '__main__':
    main()
