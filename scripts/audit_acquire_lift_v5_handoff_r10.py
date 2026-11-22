"""Preserve v4 evidence and the v5 CPU/failed-display attempt without rerunning it."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from our_mani.multitask.interaction import InteractionLibrary

BASE = ROOT / 'artifacts/framework/multitask_r10'
OUT = BASE / 'geometry_acquire_lift_v5_handoff_audit'


def read(path):
    return json.loads(path.read_text())


def pointer(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def main():
    if OUT.exists():
        raise RuntimeError('Preserve the existing audit; no overwrite')
    v4 = BASE / 'geometry_native_actuator_diagnostic_v4'
    v5 = BASE / 'geometry_native_actuator_diagnostic_v5'
    setup = BASE / 'geometry_native_actuator_diagnostic_v5_setup'
    old_lock = read(BASE / 'geometry_native_actuator_diagnostic_v4_setup/frozen_inputs.json')
    new_lock = read(setup / 'frozen_inputs.json')
    verify_inputs(ROOT, old_lock['files'])
    verify_inputs(ROOT, new_lock['files'])
    assert all(new_lock['files'].get(k) == v for k, v in old_lock['files'].items())
    previous = read(ROOT / 'runtime/framework/handoff_runtime_latest.json')
    assert previous['status'] == 'terminal_audited_no_further_launch'
    assert previous['native_processes_all_terminal'] is True
    for key in ('actual_report', 'diagnostic_report', 'parent_summary'):
        p = ROOT / previous[key]['path']
        assert file_sha(p) == previous[key]['sha256']

    old_actual = read(v4 / 'actual_smoke_report.json')
    evidence = {}
    for leaf in ('cv_normal', 'context_normal'):
        case_path = v4 / 'development_demo_only' / leaf / 'case.json'
        case = read(case_path)
        metrics = case['actor_metrics']
        assert case['complete'] and case['cleanup_completed'] and case['infrastructure_error'] is None
        assert case['official_task_success'] is False
        lift = [e for e in metrics['gripper_timing_events'] if e.get('phase') == 'lift']
        dispatch = metrics['continuous_direct_counted_setpoint_v3']
        first = next((e for e in dispatch['events'] if e['actual_window_acceleration_within_limits'] is False), None)
        evidence[leaf] = dict(case=pointer(case_path),
            public_skill_events=metrics['observable_skill_events'],
            native_task_success=False,
            actual_max_window_acceleration_rad_s2=old_actual['methods'][leaf]['actual_max_window_acceleration_rad_s2'],
            first_lift_gripper_event=None if not lift else lift[0],
            first_acceleration_violation=first,
            lift_command_reasons=dict(Counter(e.get('reason') for e in metrics['public_events'] if e.get('phase') == 'lift')),
            final_controller_diagnostics=metrics['last_diagnostics'].get('controller'),
            earlier_per_tick_candidate_rejections_not_recorded=True)
    assert evidence['cv_normal']['first_lift_gripper_event']['actual_opening_request'] == 1.
    assert evidence['cv_normal']['first_lift_gripper_event']['public_contact'] is True
    assert evidence['context_normal']['first_acceleration_violation']['command'] == 'hold'
    assert evidence['context_normal']['first_acceleration_violation']['phase'] == 'lift'
    config = read(setup / 'configuration.json')
    library = InteractionLibrary.load(config['learned_artifacts']['moving_basketball_in_hoop']['interaction']['path'])
    lift_templates = library.for_phase('lift')
    assert all(t.at(u)[3] == 0. for t in lift_templates for u in (0., .5, 1.))
    offsets = config['candidate_config']['arrival_offsets_seconds']
    minimum = min(offsets)
    suffix = sum(max(minimum, max(t.duration_budget_seconds for t in library.for_phase(phase))) + .05
        for phase in library.phase_order[library.phase_order.index('lift') + 1:])
    budget = max(t.duration_budget_seconds for t in lift_templates)
    # Budget feasibility alone cannot establish IK/collision/continuity feasibility.
    evidence['context_transition_time_budget'] = dict(time=8.4, remaining_seconds=1.6,
        suffix_reserve_seconds=suffix, lift_duration_budget_seconds=budget,
        budget_only_feasible_delays=[d for d in offsets if max(d, budget) + .05 + suffix <= 1.6 + 1e-8],
        full_candidate_feasibility_claimed=False)

    parent = read(v5 / 'parent_summary.json')
    assert parent['all_children_terminal'] and parent['original_frozen_inputs_verified_after']
    failures = []
    for child in parent['children']:
        log = ROOT / child['raw_log']['path']
        assert child['returncode'] == 1 and not child['timed_out'] and not child['automatic_retry']
        assert file_sha(log) == child['raw_log']['sha256']
        assert 'could not connect to display' in log.read_text()
        assert not (ROOT / child['case']).exists()
        failures.append(dict(method=child['method'], launcher_process_id=child['launcher_process_id'],
            returncode=child['returncode'], raw_log=pointer(log), case_exists=False,
            policy_outcome=None, actual_motion_qualification=None, counted_ticks=None,
            infrastructure_error='Qt/xcb could not connect to Xvfb display',
            counted_ticks_unavailable_without_case_receipt=True))
    display = ROOT / 'runtime/framework/acquire_lift_v5_xvfb_probe.log'
    assert 'Cannot establish any listening sockets' in display.read_text()
    cpu = read(setup / 'CPU_acceptance.json')
    assert cpu['tests_each_environment'] == 26
    assert len(cpu['actual_parent_derived_preflight']) == 2
    for log in cpu['logs']:
        assert file_sha(ROOT / log['path']) == log['sha256']
    cup = BASE / 'TRAIN_high_cup_periodic10_all100_candidate_v2/manifest.json'
    cup_state = read(cup)
    OUT.mkdir()
    shutil.copyfile(ROOT / 'runtime/framework/handoff_runtime_latest.json', OUT / 'previous_v4_runtime_status.json')
    report = dict(schema='r10_acquire_lift_v5_handoff_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=pointer(Path(__file__)), previous_v4_status=pointer(OUT / 'previous_v4_runtime_status.json'),
        allowed_comparison=['cv_geometry', 'context_geometry', 'physmani'],
        prohibited_restart=['Value', 'NGFF', 'FASTER'],
        v4_evidence=evidence, v5_configuration=pointer(setup / 'configuration.json'),
        v5_frozen_inputs=pointer(setup / 'frozen_inputs.json'), v5_CPU_acceptance=pointer(setup / 'CPU_acceptance.json'),
        prior_frozen_input_count=len(old_lock['files']), frozen_input_count=len(new_lock['files']),
        all_prior_frozen_inputs_preserved=True, new_source_and_models_verified_after_attempt=True,
        v5_parent_summary=pointer(v5 / 'parent_summary.json'), v5_native_infrastructure_failures=failures,
        display_probe_log=pointer(display),
        process_visibility='exec has an isolated PID namespace; host process absence is not established by local ps',
        v4_terminal_supported_by='hash-verified parent receipts and previous PID/start_ticks audit',
        v5_owned_launchers_terminal=True, v5_actual_policy_result_available=False,
        v5_actual_acceleration_pass_claimed=False, no_formal_VAL_or_TEST_started=True,
        formal_comparison_methods=3, native_development_methods=2,
        high_cup=dict(manifest=pointer(cup), complete=cup_state['complete'],
            committed_case_receipts=len(cup_state['case_receipts']), missing_cases=len(cup_state['missing_case_ids']),
            collector_host_liveness_unverified=True, restarted=False),
        next_required_steps=[
            'Restore a native-capable execution surface with working Xvfb/Qt sockets and host PID visibility.',
            'Verify actual host processes using PID/start_ticks before any next native attempt.',
            'Preserve failed v5 namespace; prepare a fresh v5_2 setup/parent and lock the actual new wrapper sources.',
            'Run the same fixed TRAIN104 pair once; audit reference continuity, gripper requests, phase/deadline, measured limits, cleanup and official outcome separately.',
            'Do not release formal VAL/TEST, restart legacy queues, relax collision/grasp/acceleration gates or extend total10s.'
        ],
        limitations=['CPU and interface checks do not validate native motor response or task success.',
            'The v4 hold transition and contact co-occur; this is not a force-saturation or isolated causal proof.',
            'Missing v5 case/trajectory is infrastructure-pending evidence, not a policy failure or a measured zero-tick result.'])
    atomic_json(OUT / 'report.json', report)
    atomic_json(ROOT / 'runtime/framework/handoff_runtime_latest.json', dict(
        schema='our_mani_acquire_lift_v5_handoff_status_v1', status='CPU_passed_native_display_infrastructure_failed',
        updated_utc=datetime.now(timezone.utc).isoformat(), report=pointer(OUT / 'report.json'),
        previous_v4_status=pointer(OUT / 'previous_v4_runtime_status.json'),
        scope='three-method comparison; fixed normal basketball TRAIN104 CV/Context development only',
        prohibited_restart=['Value', 'NGFF', 'FASTER'], v4_terminal=True,
        v5_owned_launchers_terminal=True, v5_native_policy_result_available=False,
        host_process_visibility_available=False, automatic_retry=False, next_launch_command=None,
        configuration=pointer(setup / 'configuration.json'), frozen_inputs=pointer(setup / 'frozen_inputs.json'),
        CPU_acceptance=pointer(setup / 'CPU_acceptance.json'), parent_summary=pointer(v5 / 'parent_summary.json'),
        native_infrastructure_failures=failures, executor_fully_solved=False,
        next_required_steps=report['next_required_steps']))
    print(json.dumps(dict(report=pointer(OUT / 'report.json'), frozen_inputs=len(new_lock['files']),
        native_infrastructure_failures=len(failures), native_policy_outcomes_available=0)))


if __name__ == '__main__':
    main()
