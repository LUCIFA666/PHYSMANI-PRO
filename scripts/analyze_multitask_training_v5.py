"""Read-only fixed teacher+hold smoke audit; no training or label mutation."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as collection
from our_mani.multitask.specs import get_task_spec
BASE=collection.BASE
THRESHOLDS=dict(linear_motion_m_per_s=.001,angular_motion_rad_per_s=.01,
    reported_zero_m_per_s=1e-6,reported_zero_rad_per_s=1e-6)


def audit_case(path,plan,case):
    record=collection.validate_case_result(path,plan,case)
    common.require(record['policy_history_contains_native_GT'] is False,'native GT leaked into history')
    files=record['files_sha256'];entry=record['entry']
    parent=record.get('parent_process_receipt')
    common.require(parent is not None,'fresh child provenance missing')
    child=common.read(ROOT/parent['child_case_path'])
    common.require({k:v for k,v in record.items() if k!='parent_process_receipt'}==child,'canonical/raw child differ')
    common.require(record['teacher_paths_requested'] is case['teacher_used'],'teacher path scope differs')
    common.require(files.get(entry['cache'])==entry['cache_sha256'],'cache not bound to receipt')
    with np.load(ROOT/entry['cache'],allow_pickle=False) as stored:
        a={key:stored[key].copy() for key in stored.files}
    diagnostics=[ROOT/name for name in files if Path(name).name=='tracker_diagnostics.json']
    common.require(len(diagnostics)==1,'tracker diagnostics missing/duplicate')
    spec=get_task_spec(case['task']);t,ids,frames=common.validate_arrays(a,common.read(diagnostics[0]),spec)
    clock=record['alignment']['clock']
    common.require(len(t)-1==clock['ticks']<=case['maximum_collection_ticks'],'counted budget/frame coverage differs')
    common.require(abs(clock['simulation_seconds']-t[-1])<1e-6,'clock/cache differs')
    for field in ('actual_physics_seconds','actual_task_seconds'):
        common.require(abs(record['alignment'][field]-t[-1])<2e-5,'actual clock differs: '+field)
    eligibility=collection.context_window_eligibility(a)
    common.require(entry['context_window_eligibility']==eligibility,'context windows changed')
    common.require(entry['context_training_eligible']==bool(case['context_primary_view'] and eligibility['usable_context_windows']>0),
        'teacher data silently became primary context')
    common.require(np.array_equal(a['teacher_opening'],a['teacher_applied_opening']),'legacy opening alias differs')
    common.require(np.isfinite(a['gripper_open_amount']).all() and np.isfinite(a['gripper_open_amount_after']).all(),
        'continuous gripper amount nonfinite')
    recover=a['teacher_joint_limit_recovery'];settle=a['teacher_execution_stage'].astype(str)=='release_settle'
    common.require(int(recover.sum())==clock['phases'].get('TRAIN_teacher_public_joint_limit_recovery',0),
        'recovery rows/counts differ')
    common.require(int(settle.sum())==clock['phases'].get('TRAIN_teacher_release_settle',0),'settle rows/counts differ')
    common.require(np.allclose(a['applied_joint_reference'][settle],a['joint_positions'][settle],atol=1e-10,rtol=0),
        'release settling did not hold currently measured joints')
    recovery_diagnostics=[]
    for index in np.flatnonzero(recover):
        common.require(not a['teacher_command_is_new'][index] and not a['teacher_execution_valid'][index],
            'recovery re-queried teacher or entered skill supervision')
        recovery_diagnostics.append(dict(time=float(t[index]),command_id=int(a['teacher_command_id'][index]),
            raw_command_age_seconds=float(a['teacher_command_age_seconds'][index]),
            measured_joints=a['joint_positions'][index].tolist(),applied_joints=a['applied_joint_reference'][index].tolist(),
            next_measured_joints=a['joint_positions'][index+1].tolist() if index+1<len(t) else None,
            next_time=float(t[index+1]) if index+1<len(t) else None,
            teacher_query_is_new=False,skill_execution_valid=False))
    bodies={}
    for j,body in enumerate(ids):
        d=[frame['bodies'][body] for frame in frames]
        twist_a=a
        if not case['teacher_used']:
            # Computation-only hold segmentation; stored masks and raw labels remain unchanged.
            twist_a=dict(a,teacher_action_valid=np.ones(len(t),bool),teacher_stage=np.full(len(t),'hold'),teacher_phase=np.full(len(t),'hold'))
        twist=common.native_twist_audit(twist_a,j,THRESHOLDS)
        if not case['teacher_used']:twist['segmentation']='same mode, mode valid, exogenous public hold; no teacher phase requirement'
        bodies[body]=dict(valid=common.continuity(a['valid'][:,j],t,a['modes'][:,j]),
            directly_visible=common.continuity([r.get('directly_visible') is True for r in d],t),
            modes=dict(Counter(('external','attached','released')[int(v)] for v in a['modes'][:,j])),
            mode_invalid_rows=int((~a['mode_valid'][:,j]).sum()),
            target_invalid_rows=int((~a['target_valid'][:,j]).sum()),
            public_control_masked_rows=int((~a['dynamics_segment_valid'][:,j]).sum()),
            orientation_observable_rows=int(a['rotation_observable'][:,j].sum()),
            registration_rms_m=common.scalar_stats([r.get('registration_rms_m',np.nan) for r in d]),
            windows=eligibility['by_body'][body],native_twist=twist)
    press=a['public_command_kind'].astype(str)=='press';plate_checks=dict(present='plate' in ids,issued_press_rows=int(press.sum()),passed=True)
    if 'plate' in ids:
        j=ids.index('plate');mask=np.maximum.accumulate(press)
        invalid=mask&(a['valid'][:,j]|a['target_valid'][:,j]|a['dynamics_segment_valid'][:,j])
        common.require(not invalid.any(),'plate controlled press label leaked into external dynamics')
        plate_checks.update(masked_rows=int(mask.sum()),native_pose_valid_masked_rows=int(a['native_pose_valid'][mask,j].sum()))
    teacher=None
    if case['teacher_used']:
        sidecar=ROOT/entry['teacher_command_receipt']
        common.require(files.get(entry['teacher_command_receipt'])==common.sha(sidecar),'command sidecar not SHA-bound')
        data=common.read(sidecar);commands=data['commands'];events=data['phase_completion_events']
        common.require(data['raw_decisions']==len(commands)==int(a['teacher_command_is_new'].sum()),'raw decision counts differ')
        ledger={row['command_id']:row for row in commands}
        common.require(len(ledger)==len(commands),'duplicate command identity')
        command_rows=[]
        for cid,command in ledger.items():
            mask=a['teacher_command_id']==cid
            common.require(mask.any(),'ledger command not observed')
            held=a['teacher_tcp_xyzw'][mask];target=np.asarray(command['raw_waypoint_xyzw'])
            common.require(np.array_equal(held,np.broadcast_to(held[0],held.shape)) and
                np.allclose(held[:,:3],target[:3],atol=1e-10,rtol=0) and
                np.max(Rotation.from_matrix(Rotation.from_quat(held[:,3:]).as_matrix()@
                    Rotation.from_quat(target[3:]).as_matrix().T).magnitude())<1e-6,
                'held waypoint changed physically without new command')
            common.require(int(a['teacher_execution_valid'][mask].sum())==command['executed_tick_count'],'command actual execution differs')
            actual_settle=int((mask&settle).sum());completed='command_completed_at' in command
            applied=command.get('applied_opening');first=np.flatnonzero(mask)[0]
            previous=float(a['teacher_applied_opening'][first-1]) if first else float(a['robot_gripper_open'][0])
            release_edge=previous<.5 and applied is not None and applied>=.5
            if completed and release_edge:common.require(actual_settle==10,'completed real release lacks ten counted holds')
            if not release_edge:common.require(actual_settle==0,'nonrelease/initial-open command added settling')
            command_rows.append(dict(command_id=cid,phase=command['raw_phase'],executed_ticks=command['executed_tick_count'],
                completed=completed,planning_failure=command.get('planning_failure'),cutoff_reason=command.get('cutoff_reason'),
                raw_opening=command['desired_opening'],applied_opening=applied,real_release_edge=release_edge,
                release_settle_ticks=actual_settle,gripper_completed_at=command.get('gripper_completed_at'),
                command_completed_at=command.get('command_completed_at'),
                arrival_position_error_m=command.get('arrival_position_error_m'),arrival_orientation_error_rad=command.get('arrival_orientation_error_rad')))
        for event in events:
            origin=ledger[event['origin_command_id']]
            common.require(event['label_namespace']=='TRAIN_metric_only' and event['eligible_command_ids']==[origin['command_id']],
                'phase event scope/attempt identity invalid')
            common.require(event['origin_executed_tick_count']==origin['executed_tick_count']>=1 and
                event['origin_command_completed']==('command_completed_at' in origin),'phase completion execution evidence differs')
            common.require(0<=event['attempt_start_time']<event['attempt_end_time']<=t[-1]+1e-6,'phase event time invalid')
        teacher=dict(raw_decisions=len(commands),actual_held_action_rows=int(a['teacher_action_valid'].sum()),
            completed_commands=sum(row['completed'] for row in command_rows),command_diagnostics=command_rows,
            phase_completion_events=events,completed_phases=sorted({e['phase'] for e in events}),
            actual_phase_coverage=common.phase_coverage(a,spec,frames,ids),
            native_task_success_is_not_pipeline_success_rate=True)
    else:
        common.require(not a['teacher_action_valid'].any() and not a['teacher_command_is_new'].any() and
            not a['teacher_execution_valid'].any(),'hold used teacher action')
    layouts=[ROOT/name for name in files if Path(name).name=='TRAIN_rigid_layout.json']
    common.require(len(layouts)==1,'TRAIN rigid layout receipt missing')
    return dict(case_id=case['case_id'],task=case['task'],behavior=case['behavior'],
        parent_source_group=case['parent_source_group'],status='verified',case_sha256=common.sha(path),
        process_id=record['process_id'],frames=len(t),ticks=clock['ticks'],seconds=float(t[-1]),clock_verified=True,
        native_success=record['alignment']['native_success_latched'],native_terminal=record['alignment']['native_terminal_latched'],
        native_terminal_failure=record['alignment']['native_terminal_failure_latched'],short_terminal=len(t)<27,
        context_primary_view=case['context_primary_view'],context_window_eligibility=eligibility,
        parent_has_1p1_second_horizon=eligibility['full_horizon_windows']>0,
        public_contact_rows=int(a['robot_contact'].sum()),public_joint_recovery_ticks=int(recover.sum()),
        joint_recovery_diagnostics=recovery_diagnostics,
        actual_release_hold_ticks=int(settle.sum()),phase_clock=clock['phases'],
        map_queries=record['alignment']['map'].get('queries'),
        probe_unavailable_counter=record['alignment'].get('joint_limit_probe_rejections'),
        probe_counter_recorded='joint_limit_probe_rejections' in record['alignment'],
        probe_guard_not_exercised_by_teacher_pure_arm_planner=True,
        joint_intervals_not_stored_in_cache=True,
        teacher=teacher,bodies=bodies,public_plate_mask=plate_checks,
        rigid_layout=common.rigid_layout_audit(a,spec,common.read(layouts[0])),
        file_provenance=[common.pointer(path)]+[common.pointer(ROOT/name) for name in files],
        raw_native_velocity_labels_unchanged=True)


def analyze(plan_path,queue_path):
    plan=common.read(plan_path)
    common.require(plan['version']==collection.BEHAVIOR_PLAN_VERSION and plan['plan_sha256']==common.digest(
        {k:v for k,v in plan.items() if k!='plan_sha256'}),'behavior plan digest invalid')
    common.require(len(plan['cases'])==28 and len({c['parent_source_group'] for c in plan['cases']})==14,'fixed28/14 inventory differs')
    verified=[];missing=[];invalid=[];infra=[]
    for case in plan['cases']:
        path=Path(queue_path)/'cases'/case['split']/case['task']/case['case_id']/'case.json'
        if path.exists():
            try:verified.append(audit_case(path,plan,case))
            except Exception as error:invalid.append(dict(case_id=case['case_id'],task=case['task'],behavior=case['behavior'],error=str(error)))
        else:
            item=dict(case_id=case['case_id'],task=case['task'],behavior=case['behavior'])
            receipts=list((Path(queue_path)/'infrastructure_attempts'/case['case_id']).glob('*/parent_receipt.json'))
            failures=[dict(common.read(p),receipt=common.pointer(p)) for p in receipts if common.read(p).get('status')=='infrastructure_error']
            if failures:infra.append(dict(item,attempts=failures))
            else:missing.append(item)
    teachers=[r for r in verified if r['teacher'] is not None];holds=[r for r in verified if r['context_primary_view']]
    sources=defaultdict(list)
    for row in teachers:
        for phase in row['teacher']['completed_phases']:sources[row['task']+':'+phase].append(row['parent_source_group'])
    return dict(version='r10_offline_fixed_teacher_hold_smoke_v5',created_utc=datetime.now(timezone.utc).isoformat(),
        plan=common.pointer(plan_path),plan_sha256=plan['plan_sha256'],analysis_source=common.pointer(__file__),
        common_analysis_source=common.pointer(common.__file__),expected=28,verified=len(verified),infrastructure=len(infra),
        missing=len(missing),invalid=len(invalid),complete=len(verified)==28 and not invalid,
        counts_by_behavior=dict(Counter(r['behavior'] for r in verified)),
        teacher_native_successes=sum(r['native_success'] for r in teachers),teacher_cases=len(teachers),
        hold_full_horizon_parents=sum(r['parent_has_1p1_second_horizon'] for r in holds),hold_parents=len(holds),
        teacher_completed_commands=sum(r['teacher']['completed_commands'] for r in teachers),
        completed_phase_parent_support={k:len(set(v)) for k,v in sources.items()},
        public_joint_recovery_ticks=sum(r['public_joint_recovery_ticks'] for r in verified),
        actual_release_hold_ticks=sum(r['actual_release_hold_ticks'] for r in verified),
        cases=verified,missing_cases=missing,invalid_cases=invalid,infrastructure_cases=infra,
        interpretation='TRAIN acquisition diagnostics only, not pipeline or baseline SR; all fixed attempts kept, no success-selected resampling; teacher excluded from primary context; full1400/VAL/TEST not run',
        orientation='allnone diagnostic smoke: valid position windows do not prove full SE3 skill observability',
        velocity_labels='raw native get_velocity retained; pose-derived differences are offline diagnostics only',
        probe_guard='pure arm teacher/hold do not call AlignedRobotModel virtual probes; unavailable counter absent in receipts is not asserted zero')


def report_text(report):
    lines=[f"R10 v5 固定 TRAIN smoke：{report['verified']}/28 SHA验收，infra={report['infrastructure']}，invalid={report['invalid']}，pending={report['missing']}",
        f"teacher native成功 {report['teacher_native_successes']}/{report['teacher_cases']}（仅采集诊断）；hold 1.1秒预测窗口 {report['hold_full_horizon_parents']}/{report['hold_parents']}父源",
        f"真实完成commands={report['teacher_completed_commands']}；公开q恢复ticks={report['public_joint_recovery_ticks']}；实际release holdticks={report['actual_release_hold_ticks']}",
        '固定预算不按成败重采；教师轨迹不进默认context主视图。allnone不证明完整SE3技能可观测，未跑正式1400/VAL/TEST。']
    for r in sorted(report['cases'],key=lambda r:(r['task'],r['behavior'])):
        details=(f"commands={r['teacher']['completed_commands']}/{r['teacher']['raw_decisions']} phases={','.join(r['teacher']['completed_phases']) or '-'}"
            if r['teacher'] is not None else f"fullhorizon={r['context_window_eligibility']['full_horizon_windows']}")
        lines.append(f"{r['task']} | {r['behavior']} | {r['ticks']}ticks native={r['native_success']} | {details}")
    lines+=['invalid: '+json.dumps(r,ensure_ascii=False) for r in report['invalid_cases']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=BASE/'TRAIN_smoke_collection_plan_v5.json')
    parser.add_argument('--queue',type=Path,default=BASE/'TRAIN_smoke_collection_v5')
    parser.add_argument('--report',type=Path,default=BASE/'TRAIN_smoke_analysis_v5.json')
    args=parser.parse_args();report=analyze(args.plan,args.queue)
    path=collection.destination(args.report);collection.write_json(path,report,overwrite=True)
    path.with_suffix('.txt').write_text(report_text(report))
    print(json.dumps({k:report[k] for k in ('verified','infrastructure','invalid','missing','teacher_cases','teacher_native_successes','hold_parents','hold_full_horizon_parents')}))
if __name__=='__main__':main()
