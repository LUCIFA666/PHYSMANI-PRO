"""Wait for the shared native slot, then run one frozen TRAIN104 pair once.

The existing formal supervisor owns the lock across VAL and TEST, so this
readiness process never takes a between-case gap or edits its frozen policy.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair, process_visibility
from scripts import run_approach_if_slow_pair_v13_r10 as pair

OUT = ROOT / 'artifacts/framework/multitask_r10/geometry_approach_if_slow_v13_wait'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    return dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=file_sha(path))


def process(pid):
    try:
        row = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(row[19]))
    except (OSError, ValueError, IndexError):
        return None


def status(stage, **extra):
    atomic_json(OUT / 'status.json', dict(stage=stage, updated_utc=utc(), **extra), overwrite=True)


def audit():
    summary = read(pair.OUTPUT / 'parent_summary.json')
    assert summary['all_children_terminal'] and summary['original_frozen_inputs_verified_after']
    methods = {}
    for method, folder in [('cv_geometry', 'cv_normal'), ('context_geometry', 'context_normal')]:
        path = pair.OUTPUT / 'development_demo_only' / folder / 'case.json'
        if not path.exists():
            methods[method] = dict(complete=False, policy_outcome=None, infrastructure_pending=True)
            continue
        case = read(path)
        metrics = case['actor_metrics']
        dispatch = metrics['approach_if_slow_dispatch_v13']
        helper = metrics['continuous_reference_feedforward']['bounded_approach_transit_v13']
        events = [x for x in helper['events'] if x['event'] == 'partial_approach_staged']
        accelerations = [np.max(np.abs(row['backward_window_acceleration']))
                         for row in dispatch['events'] if row['backward_window_acceleration'] is not None]
        methods[method] = dict(case=pointer(path), complete=case['complete'],
            infrastructure_error=case['infrastructure_error'], cleanup_completed=case['cleanup_completed'],
            policy_outcome=case['official_task_success'],
            actual_motion_qualification=dispatch['actual_observed_window_limits_passed'],
            maximum_actual_window_acceleration_rad_s2=max(accelerations, default=None),
            bounded_approach_segments=len(events), feature_exercised=bool(events),
            every_staged_segment_has_actual_and_reference_full_gate=all(x['actual_and_reference_full_gate_feasible'] for x in events),
            observable_skill_events=metrics['observable_skill_events'], counted_clock=case['counted_clock'])
    atomic_json(pair.OUTPUT / 'actual_TRAIN_acceptance.json', dict(
        schema='r10_bounded_approach_fixed_TRAIN_pair_actual_acceptance_v13', created_utc=utc(),
        methods=methods, actual_motion_fixed_TRAIN_pair_passed=all(
            x.get('complete') and x.get('cleanup_completed') and x.get('infrastructure_error') is None
            and x.get('actual_motion_qualification') for x in methods.values()),
        feature_exercised_both_methods=all(x.get('feature_exercised') for x in methods.values()),
        formal_metric=False, executor_fully_solved=False, automatic_retry=False,
        TEST_data_used_for_this_change=False, controller_parameters_selected_from_VAL_outcomes=False,
        VAL_status_and_one_baseline_case_reviewed_prior_to_change=True,
        design_basis='user requested threshold branch; existing1.1s horizon, author0.5s chunk, fixed TRAIN104 traces'))


def main():
    if OUT.exists():
        raise RuntimeError('Preserve existing queued TRAIN attempt')
    files = read(pair.SETUP / 'frozen_inputs.json')['files']
    verify_inputs(ROOT, files)
    if files.get(str(Path(__file__).resolve())) != file_sha(__file__):
        raise RuntimeError('Queued TRAIN supervisor source is not frozen')
    accepted = read(pair.SETUP / 'CPU_acceptance.json')
    assert accepted['native_parent_argv_preflight_passed']
    OUT.mkdir()
    with (OUT / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(OUT / 'launch.json', dict(created_utc=utc(), process=process(os.getpid()),
            source=pointer(Path(__file__)), frozen_inputs=pointer(pair.SETUP / 'frozen_inputs.json'),
            task='moving_basketball_in_hoop', fixed_TRAIN_source='episode104',
            methods=['cv_geometry', 'context_geometry'], native_workers=2, automatic_retry=False,
            other_frozen_formal_policy_changed=False, TEST_results_read=False))
        try:
            while True:
                if (ROOT / 'runtime/framework/multitask_r10_STOP').exists():
                    raise RuntimeError('Controlled STOP prevents queued TRAIN launch')
                acquired = False
                try:
                    with exclusive_pair():
                        acquired = True
                        inventory = process_visibility()
                        if not inventory['host_visibility_available'] or inventory['same_uid_inventory_incomplete']:
                            raise RuntimeError('Host process visibility is unavailable')
                        if inventory['live_project_native_processes']:
                            status('waiting_for_existing_native_processes', inventory=inventory, native_started=False)
                        else:
                            verify_inputs(ROOT, files)
                            status('fixed_TRAIN_pair_running', native_started=True)
                            pair.main()
                            audit()
                            status('fixed_TRAIN_pair_terminal', native_started=True,
                                   actual_report=pointer(pair.OUTPUT / 'actual_TRAIN_acceptance.json'))
                            break
                except RuntimeError as error:
                    if acquired or 'owns the run lock' not in str(error):
                        raise
                    status('waiting_for_existing_frozen_native_evaluation', native_started=False,
                           reason=str(error), active_formal_policy_changed=False)
                time.sleep(20)
        except BaseException as error:
            atomic_json(OUT / 'failure.json', dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), partial_outputs_preserved=True, automatic_retry=False))
            status('pending_failure', failure=pointer(OUT / 'failure.json'))
            raise


if __name__ == '__main__':
    main()
