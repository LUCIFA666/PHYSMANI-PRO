"""Read-only artifact audit of the completed two-case instrumentation run."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import file_sha,atomic_json,verify_inputs
BASE=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v4'
SETUP=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v4_setup'


def read(p):return json.loads(p.read_text())
def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
def maximum(values):return max(values,default=0.)
def contact_summary(event):
    value=event['queries']['graspable_object_last_physics_step_contacts']
    if not value['available']:return value
    contacts=[c for o in value['value'] for c in o['contacts']]
    return dict(available=True,native_contact_records=len(contacts),
        handle_pairs=sorted({tuple(c['contact_handles']) for c in contacts}),
        maximum_reported_contact_force_norm=maximum([float(np.linalg.norm(c['native_contact_payload'][3:6])) for c in contacts]),
        contact_partner_names_not_recorded=True,all_substep_contacts_may_be_aggregated=True)


def audit():
    lock=read(SETUP/'frozen_inputs.json');verify_inputs(ROOT,lock['files'])
    status=read(ROOT/'runtime/framework/handoff_runtime_latest.json')
    parent=read(BASE/'parent_summary.json')
    assert parent['all_children_terminal'] and parent['original_frozen_inputs_verified_after']
    report=dict(schema='r10_native_actuator_readonly_fixed_TRAIN104_actual_report_v4',
        created_utc=datetime.now(timezone.utc).isoformat(),configuration=pointer(SETUP/'configuration.json'),
        frozen_inputs=pointer(SETUP/'frozen_inputs.json'),source_archive=pointer(SETUP/'source_snapshot.json'),
        CPU_acceptance=pointer(SETUP/'CPU_acceptance.json'),parent_summary=pointer(BASE/'parent_summary.json'),
        all207_frozen_inputs_verified_after_native=True,methods={},processes=[],
        control_algorithm_unchanged_from_v3=True,diagnostics_supplied_to_policy=False,
        nonfitting_TRAIN_development_only=True,no_Value_or_TEST=True,no_video=True,
        automatic_retry=False,executor_fully_solved=False)
    diagnostic=dict(schema='r10_native_actuator_readonly_causal_limits_diagnostic_v4',methods={},
        source=pointer(Path(__file__)),facts_are_observational=True,
        historical_v3_first_spikes_not_instrumented_retroactively=True)
    for row in status['native_process_ids']:
        d=Path(f"/proc/{row['pid']}");same=False
        try:same=int((d/'stat').read_text().rsplit(')',1)[1].split()[19])==row['start_ticks']
        except OSError:pass
        report['processes'].append(dict(row,same_process_live=same))
        assert not same
    for leaf in ('cv_normal','context_normal'):
        out=BASE/'development_demo_only'/leaf;case=read(out/'case.json');base=read(out/'evaluate_run_case_receipt.json')
        stripped=deepcopy(case);stripped.pop('task_execution_witness');stripped.pop('development_demo_execution')
        assert stripped==base
        assert case['complete'] is True and case['cleanup_completed'] is True and case['infrastructure_error'] is None
        assert case['counted_clock']==dict(ticks=200,dt=.05,seconds=10.) and case['video'] is None
        hook=read(out/'trained_task_hook_receipt.json');assert hook['complete'] is True and not hook['source_resampled']
        m=case['actor_metrics'];dispatch=m['continuous_direct_counted_setpoint_v3'];events=dispatch['events']
        nd=m['native_metric_diagnostics'];de=nd['events']
        assert len(de)==800 and all(x['clock_unchanged'] for x in de)
        assert nd['diagnostic_values_supplied_to_policy'] is False and nd['commands_modified'] is False
        assert len(events)==200 and all(x['actual_counted_dispatch_completed'] for x in events)
        stages=Counter(x['stage'] for x in de);assert all(stages[s]==200 for s in ('before_policy','after_policy','before_dispatch','after_dispatch'))
        groups={}
        for e in de:groups.setdefault(e['clock_before']['ticks'],{})[e['stage']]=e
        keys=list(de[0]['queries']);changes={k:0 for k in keys};maxdelta={k:0. for k in keys[:5]}
        for tick,g in groups.items():
            if not {'before_policy','after_policy'}<=g.keys():continue
            for k in keys:
                a,b=g['before_policy']['queries'][k],g['after_policy']['queries'][k]
                if a!=b:changes[k]+=1
                if k in maxdelta and a['available'] and b['available']:
                    maxdelta[k]=max(maxdelta[k],float(np.max(np.abs(np.asarray(a['value'])-b['value']))))
        unavailable={k:sum(not e['queries'][k]['available'] for e in de) for k in keys}
        force=np.array([e['queries']['joint_forces_or_torques']['value'] for e in de if e['queries']['joint_forces_or_torques']['available']])
        constant={k:all(e['queries'][k]==de[0]['queries'][k] for e in de) for k in ('joint_modes','joint_control_loop_enabled','joint_motor_enabled','joint_target_velocities')}
        close=[dict(time=e['time'],phase=e['phase'],candidate_id=e.get('candidate_id'),
            position_error_m=e.get('final_grasp_position_error_m'),orientation_error_rad=e.get('final_grasp_orientation_error_rad'),reason=e.get('reason'))
            for e in m['gripper_timing_events'] if e.get('actual_opening_request')==0 and e.get('phase')=='acquire']
        row=dict(case=pointer(out/'case.json'),base_case=pointer(out/'evaluate_run_case_receipt.json'),
            hook_receipt=pointer(out/'trained_task_hook_receipt.json'),native_task_success=case['official_task_success'],
            cleanup_completed=case['cleanup_completed'],infrastructure_error=case['infrastructure_error'],
            process_id=case['process_id'],counted_clock=case['counted_clock'],
            public_skill_events=m['observable_skill_events'],final_phase_index=m['executor_observed_state_trace'][-1]['phase_index'],
            phase_counts=dict(Counter(e['phase'] for e in events)),
            actual_motion_qualification_passed=dispatch['actual_observed_window_limits_passed'],
            actual_max_window_acceleration_rad_s2=maximum([float(np.max(np.abs(e['backward_window_acceleration']))) for e in events if e['backward_window_acceleration'] is not None]),
            actual_max_window_speed_fraction=maximum([float(np.max(np.abs(e['backward_window_velocity'])/e['velocity_limits'])) for e in events]),
            actual_violations=dispatch['violations'],native_grasp_requests=case['alignment_receipt']['clock']['grasp_requests'],
            native_release_requests=case['alignment_receipt']['clock']['release_requests'],acquire_close_requests=close,
            diagnostic_snapshot_count=len(de),diagnostic_query_unavailable_counts=unavailable,
            before_after_policy_changes=changes,before_after_policy_max_abs_vector_deltas=maxdelta,
            joint_parameter_constancy=constant,initial_joint_parameters={k:de[0]['queries'][k] for k in constant},
            max_abs_reported_axis_force_or_torque=np.max(np.abs(force),axis=0).tolist(),
            base_receipt_preserved_exact=True,video=None,formal_metric=False)
        report['methods'][leaf]=row
        first=next((e for e in events if e['actual_window_acceleration_within_limits'] is False),None)
        detail=dict(policy_computation_native_state_changes=changes,all_diagnostic_clocks_unchanged=True,
            first_acceleration_violation=None,hardware_force_capacity_not_read=True,native_engine_not_identified=True,
            no_force_saturation_causal_claim=True)
        if first is not None:
            tick=first['tick_after'];previous=events[tick-2];axis=int(np.argmax(np.abs(first['backward_window_acceleration'])))
            detail['first_acceleration_violation']=dict(tick=tick,time=first['time_after'],axis_index=axis,
                acceleration_rad_s2=first['backward_window_acceleration'][axis],
                previous_dispatch=previous,violating_dispatch=first,
                applied_reference_jump_norm=float(np.linalg.norm(np.array(first['applied_joint_reference'])-previous['applied_joint_reference'])),
                ordinary_hold_after_moving_acquire_reference=(first['command']=='hold' and previous['phase']=='acquire'),
                helper_state_events=[e for e in m['continuous_reference_feedforward']['state_events'] if abs(e.get('time',-100)-first['time_before'])<1e-8],
                native_diagnostic_window=[dict(tick=t,forces=groups[t]['after_dispatch']['queries']['joint_forces_or_torques'],contacts=contact_summary(groups[t]['after_dispatch'])) for t in range(max(1,tick-2),min(200,tick+1)+1)],
                interpreted_limit='Native window acceleration violation during moving acquire to lift ordinary-hold transition. Failed continuous-brake trials and the actual-q hold target are directly recorded; contact co-occurs. These observations do not isolate contact versus command-stop effects.')
        diagnostic['methods'][leaf]=detail
    report['limitations']=[
        'Two fixed TRAIN104 executions only; neither official task succeeded; no formal success-rate or generalization claim',
        'Both observable approach/acquire completed here; this is separate from full native success and actual-motion qualification',
        'Instrumentation-only execution follows the same source-bound v3 policy, but the trajectories differ from prior v3; causal benefit of instrumentation is not claimed',
        'All 800 snapshots per case observed no clock change; no measured q/v/targets/forces/modes or contacts changed across policy computation',
        'Joint forces depend on the native engine; capacity/engine/contact partner names were not recorded; no saturation or contact-causality proof',
        'Contact API can aggregate the previous simulation pass dynamic substeps and native empty results do not prove no collision',
        'Window derivatives do not bound instantaneous/substep dynamics; per-joint model acceleration6 is not a hardware setting',
        'The original frozen feedback phase slack2 remains, subject to effective deadlines, suffix and total10seconds; no additional extension',
    ]
    report['next_action_after_handoff']='Do not restart this immutable namespace. Read the diagnostic report first. Next isolated development fix should handle acquire-to-lift reference/gripper continuity and investigate lift candidate rejection using these existing traces, preserving actual6/close/deadlines; no further native iteration launched during handoff.'
    atomic_json(BASE/'native_actuator_diagnostic_report.json',diagnostic)
    report['diagnostic_report']=pointer(BASE/'native_actuator_diagnostic_report.json')
    atomic_json(BASE/'actual_smoke_report.json',report)
    return report


if __name__=='__main__':
    report=audit()
    print(json.dumps(dict(report=pointer(BASE/'actual_smoke_report.json'),diagnostic=report['diagnostic_report'],methods={k:dict(task_success=v['native_task_success'],actual6_pass=v['actual_motion_qualification_passed'],max_window_acc=v['actual_max_window_acceleration_rad_s2'],skill_events=v['public_skill_events']) for k,v in report['methods'].items()}),ensure_ascii=False))
