"""Audit the completed guarded fixed TRAIN pair without assuming policy success."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs

BASE = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2'
SETUP = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2_setup'


def read(path):
    return json.loads(path.read_text())


def pointer(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def maximum(values):
    return max(values, default=0.)


def audit_case(child):
    out = ROOT / child['case']
    raw = BASE / 'parent_raw' / out.parent.name
    display = read(raw / 'display_readiness.json') if (raw / 'display_readiness.json').exists() else None
    row = dict(returncode=child['returncode'], timed_out=child['timed_out'],
        child_receipt=pointer(raw / 'parent_process_receipt.json'),
        raw_log=pointer(raw / 'native.log'), display_readiness=display,
        case_available=out.exists(), policy_outcome=None, actual_motion_qualification=None)
    if not out.exists():
        row['infrastructure_error'] = 'native_case_receipt_unavailable'
        return row
    case = read(out)
    row.update(case=pointer(out), complete=case.get('complete'),
        infrastructure_error=case.get('infrastructure_error'), cleanup_completed=case.get('cleanup_completed'),
        counted_clock=case.get('counted_clock'), native_task_success=case.get('official_task_success'))
    if not case.get('complete') or case.get('infrastructure_error') is not None:
        return row
    assert child['returncode'] == 0 and not child['timed_out']
    assert case['cleanup_completed'] is True and case['video'] is None
    assert case['split'] == 'train' and case['formal_test_used'] is False
    assert display['ready'] is True and display['reason'] == 'X11_connection_available'
    original = read(out.parent / 'evaluate_run_case_receipt.json')
    stripped = deepcopy(case)
    stripped.pop('task_execution_witness')
    stripped.pop('development_demo_execution')
    assert stripped == original
    hook = read(out.parent / 'trained_task_hook_receipt.json')
    assert hook['complete'] is True and hook['source_resampled'] is False
    clock = case['counted_clock']
    assert clock['dt'] == .05 and 0 < clock['ticks'] <= 200
    assert abs(clock['seconds'] - clock['ticks'] * .05) < 1e-8
    metrics = case['actor_metrics']
    assert metrics['ranking'] == 'geometry'
    assert metrics['native_state_is_policy_input'] is False and metrics['actor_advances_physics'] is False
    dispatch = metrics['continuous_direct_counted_setpoint_v5']
    events = dispatch['events']
    assert len(events) == clock['ticks']
    for index, event in enumerate(events):
        assert event['schema'] == 'r10_continuous_reference_direct_counted_setpoint_v5'
        assert event['tick_before'] == index and event['tick_after'] == index + 1
        assert event['actual_counted_dispatch_completed'] is True
        assert event['native_motor_parameters_changed'] is False
    diagnostic = metrics['native_metric_diagnostics']
    snapshots = diagnostic['events']
    assert diagnostic['diagnostic_values_supplied_to_policy'] is False
    assert diagnostic['commands_modified'] is False
    assert len(snapshots) == 4 * clock['ticks'] and all(x['clock_unchanged'] for x in snapshots)
    stages = Counter(x['stage'] for x in snapshots)
    assert all(stages[name] == clock['ticks'] for name in ('before_policy', 'after_policy', 'before_dispatch', 'after_dispatch'))
    before = {x['clock_before']['ticks']: x for x in snapshots if x['stage'] == 'before_policy'}
    after = {x['clock_before']['ticks']: x for x in snapshots if x['stage'] == 'after_policy'}
    keys = list(snapshots[0]['queries'])
    changes = {key: sum(before[t]['queries'][key] != after[t]['queries'][key] for t in before) for key in keys}
    unavailable = {key: sum(not x['queries'][key]['available'] for x in snapshots) for key in keys}
    gaps = []
    for previous, current in zip(events, events[1:]):
        certificate = current['feedforward_execution']
        if certificate is None:
            continue
        prior_certificate = previous['feedforward_execution']
        prior_v = (prior_certificate['reference_to_velocity'] if prior_certificate else
            np.zeros(len(previous['applied_joint_reference'])) if previous['command'] == 'hold' else None)
        gaps.append(dict(tick=current['tick_before'],
            max_abs_q_gap=float(np.max(np.abs(np.asarray(certificate['reference_from_joints']) - previous['applied_joint_reference']))),
            max_abs_v_gap=None if prior_v is None else float(np.max(np.abs(np.asarray(certificate['reference_from_velocity']) - prior_v)))))
    close = [x for x in metrics['gripper_timing_events'] if x.get('phase') == 'acquire' and x.get('actual_opening_request') == 0.]
    lift = [x for x in metrics['gripper_timing_events'] if x.get('phase') == 'lift']
    first_violation = next((x for x in events if x['actual_window_acceleration_within_limits'] is False), None)
    row.update(policy_outcome=bool(case['official_task_success']),
        native_grasp_requests=case['alignment_receipt']['clock']['grasp_requests'],
        native_release_requests=case['alignment_receipt']['clock']['release_requests'],
        actual_motion_qualification=dispatch['actual_observed_window_limits_passed'],
        actual_max_window_acceleration_rad_s2=maximum(float(np.max(np.abs(e['backward_window_acceleration']))) for e in events if e['backward_window_acceleration'] is not None),
        actual_max_window_speed_fraction=maximum(float(np.max(np.abs(e['backward_window_velocity']) / e['velocity_limits'])) for e in events),
        actual_motion_violations=dispatch['violations'], first_acceleration_violation=first_violation,
        observable_skill_events=metrics['observable_skill_events'], phase_counts=dict(Counter(x['phase'] for x in events)),
        actual_acquire_close_requests=close, first_lift_gripper_event=None if not lift else lift[0],
        actual_lift_open_requests=[x for x in lift if x.get('actual_opening_request', 0.) >= .5],
        acquire_lift_handover_events=metrics['continuous_reference_feedforward']['acquire_lift_handover_events'],
        final_controller_diagnostics=metrics['last_diagnostics'].get('controller'),
        max_reference_q_boundary_gap=maximum(x['max_abs_q_gap'] for x in gaps),
        max_reference_nominal_v_boundary_gap=maximum(x['max_abs_v_gap'] for x in gaps if x['max_abs_v_gap'] is not None),
        ordinary_nonstream_holds_after_warmup=[dict(tick=x['tick_before'], time=x['time_before'], phase=x['phase']) for x in events if x['command'] == 'hold' and x['tick_before'] > 10],
        diagnostic_snapshot_count=len(snapshots), diagnostic_query_unavailable_counts=unavailable,
        before_after_policy_native_state_changes=changes,
        base_case_receipt_preserved_exact=True, formal_metric=False)
    return row


def main():
    target = BASE / 'actual_smoke_report.json'
    if target.exists():
        raise RuntimeError('Preserve the completed audit; never overwrite')
    lock = read(SETUP / 'frozen_inputs.json')
    verify_inputs(ROOT, lock['files'])
    parent = read(BASE / 'parent_summary.json')
    assert parent['all_children_terminal'] and parent['original_frozen_inputs_verified_after']
    methods = {child['method']: audit_case(child) for child in parent['children']}
    assert set(methods) == {'cv_geometry', 'context_geometry'}
    report = dict(schema='r10_guarded_fixed_TRAIN104_v5_2_actual_report_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), audit_source=pointer(Path(__file__)),
        configuration=pointer(SETUP / 'configuration.json'), frozen_inputs=pointer(SETUP / 'frozen_inputs.json'),
        source_archive=pointer(SETUP / 'source_snapshot.json'), CPU_acceptance=pointer(SETUP / 'CPU_acceptance.json'),
        environment_restoration=pointer(ROOT / 'artifacts/framework/multitask_r10/native_X11_RGBD_restoration_v1/restoration_receipt.json'),
        parent_summary=pointer(BASE / 'parent_summary.json'), all225_frozen_inputs_verified_after_native=True,
        methods=methods, fixed_original_TRAIN_parent='episode104', original_physics_unchanged=True,
        original_v5_policy_identity_unchanged=True, Value_or_NGFF_or_FASTER_started=False,
        formal_VAL_or_TEST_started=False, formal_metric=False, video=None, automatic_retry=False,
        executor_fully_solved=all(row['policy_outcome'] is True and row['actual_motion_qualification'] is True for row in methods.values()),
        limitations=['Two fixed TRAIN development cases only; not a formal success-rate estimate.',
            'Reference continuity does not guarantee native motor tracking.',
            'Counted-window derivatives do not bound instantaneous/substep peaks.',
            'Force/contact diagnostics are metric-only and do not isolate the physical cause of a violation.'])
    atomic_json(target, report)
    verify_inputs(ROOT, lock['files'])
    status_path = ROOT / 'runtime/framework/handoff_runtime_latest.json'
    status = read(status_path)
    process_records = []
    for entry in status.get('native_process_ids', []):
        try:
            same = int(Path(f"/proc/{entry['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]) == entry['start_ticks']
        except OSError:
            same = False
        process_records.append(dict(entry, same_process_live=same))
    assert not any(x['same_process_live'] for x in process_records)
    status.update(status='native_pair_v5_2_terminal_audited', updated_utc=datetime.now(timezone.utc).isoformat(),
        native_process_ids=process_records, native_processes_all_terminal=True,
        actual_report=pointer(target), parent_summary=report['parent_summary'],
        native_policy_result_available=True, executor_fully_solved=report['executor_fully_solved'],
        next_required_steps=['Read the v5_2 actual report and existing traces before choosing the next minimal fix; preserve this namespace.'])
    atomic_json(status_path, status, overwrite=True)
    print(json.dumps({method: {key: row.get(key) for key in (
        'policy_outcome', 'actual_motion_qualification', 'actual_max_window_acceleration_rad_s2',
        'observable_skill_events', 'native_grasp_requests', 'native_release_requests', 'actual_lift_open_requests')}
        for method, row in methods.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
