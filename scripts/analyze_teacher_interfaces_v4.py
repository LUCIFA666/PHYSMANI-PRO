"""Read-only focused waypoint/gate/attachment evidence from the final v4 audit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as audit


def distribution(values):
    values=np.asarray(values,float)
    if not len(values):return dict(count=0,minimum=None,end=None,quantiles=None)
    return dict(count=len(values),minimum=float(values.min()),end=float(values[-1]),
                quantiles=dict(zip(('p0','p25','p50','p75','p95','p100'),np.quantile(values,[0,.25,.5,.75,.95,1]).tolist())))


def main():
    base=audit.BASE;source=base/'TRAIN_smoke_analysis_v4.json';accepted=audit.read(source)
    focused=[]
    tasks=('place_cups_on_rotating_frame','insert_onto_rotating_peg','put_rubbish_in_moving_bin')
    for verified in accepted['cases']:
        if verified['task'] not in tasks:continue
        for receipt in verified['provenance']:
            audit.require(audit.sha(receipt['path'])==receipt['sha256'],'audited case output changed')
        record=audit.read(verified['provenance'][0]['path'])
        with np.load(audit.absolute(record['entry']['cache']),allow_pickle=False) as loaded:
            a={key:loaded[key] for key in loaded.files}
        times=a['times'];ids=list(map(str,a['body_ids']));action=a['teacher_action_valid']
        modes=('external','attached','released');rows=[];events=[]
        before_open=bool(a['robot_gripper_open'][0]);before_contact=bool(a['robot_contact'][0])
        for i,t in enumerate(times):
            body=str(a['teacher_manipulated_body'][i]) if action[i] else ''
            j=ids.index(body) if body in ids else None;goal=a['teacher_tcp_xyzw'][i] if action[i] else None
            position=a['target_positions'][i,j] if j is not None else None
            row=dict(frame=i,time=float(t),teacher_stage=str(a['teacher_stage'][i]),teacher_phase=str(a['teacher_phase'][i]),
                teacher_action_valid=bool(action[i]),body_id=body,
                desired_opening=float(a['teacher_desired_opening'][i]) if action[i] else None,
                applied_opening=float(a['teacher_applied_opening'][i]) if action[i] else None,
                observed_gripper_open_bool=bool(a['robot_gripper_open'][i]),observed_any_attachment=bool(a['robot_contact'][i]),
                measured_tcp_xyzw=a['measured_tcp_xyzw'][i].tolist(),teacher_goal_xyzw=goal.tolist() if goal is not None else None,
                native_body_center_position=position.tolist() if position is not None else None,
                position_error_m=float(a['teacher_opening_gate_position_error_m'][i]) if action[i] else None,
                orientation_error_rad=float(a['teacher_opening_gate_orientation_error_rad'][i]) if action[i] else None,
                TCP_to_body_origin_m=float(np.linalg.norm(a['measured_tcp_xyzw'][i,:3]-position)) if position is not None else None,
                teacher_goal_to_body_origin_m=float(np.linalg.norm(goal[:3]-position)) if position is not None else None,
                teacher_goal_in_native_body_frame=(a['target_rotations'][i,j].T@(goal[:3]-position)).tolist() if position is not None else None,
                public_mode=modes[int(a['modes'][i,j])] if j is not None else None,
                mode_valid=bool(a['mode_valid'][i,j]) if j is not None else None,
                actual_joint_positions=a['joint_positions'][i].tolist())
            rows.append(row)
            after_open=bool(a['teacher_applied_opening'][i]>=.5);contact=bool(a['robot_contact'][i])
            if (action[i] and after_open!=before_open) or contact!=before_contact:
                events.append(dict(row,event='opening_and_or_attachment_transition'))
            if action[i]:before_open=after_open
            before_contact=contact
        populations={}
        for name,selected in dict(acquire=action&(a['teacher_phase']=='acquire'),
                raw_close=action&(a['teacher_desired_opening']<.5),
                delayed_close=action&a['teacher_opening_gate_delayed']).items():
            p=a['teacher_opening_gate_position_error_m'][selected];r=a['teacher_opening_gate_orientation_error_rad'][selected]
            populations[name]=dict(position_m=distribution(p),orientation_rad=distribution(r),
                position_pass_orientation_fail=int(((p<=.02)&(r>.1)).sum()),
                orientation_pass_position_fail=int(((p>.02)&(r<=.1)).sum()),both_pass=int(((p<=.02)&(r<=.1)).sum()))
        selected=action&(a['teacher_phase']=='acquire');indices=np.flatnonzero(selected)
        stable_ids=sorted(set(map(str,a['teacher_manipulated_body'][selected])))
        drift=None
        if len(indices) and len(stable_ids)==1:
            j=ids.index(stable_ids[0]);goal=a['teacher_tcp_xyzw'][indices];obj=a['target_positions'][indices,j]
            relative_angle=Rotation.from_quat(goal[:,3:]).inv()*Rotation.from_quat(goal[0,3:])
            drift=dict(body_id=stable_ids[0],teacher_goal_position_span_xyz_m=np.ptp(goal[:,:3],axis=0).tolist(),
                native_body_position_span_xyz_m=np.ptp(obj,axis=0).tolist(),teacher_goal_start_xyzw=goal[0].tolist(),
                teacher_goal_end_xyzw=goal[-1].tolist(),teacher_goal_rotation_from_start_rad=distribution(relative_angle.magnitude()),
                native_body_start_position=obj[0].tolist(),native_body_end_position=obj[-1].tolist(),
                max_teacher_quaternion_norm_error=float(np.abs(np.linalg.norm(goal[:,3:],axis=1)-1).max()))
        phases=record['alignment']['clock']['phases']
        focused.append(dict(task=record['task'],case_id=record['case_id'],provenance=verified['provenance'],
            actual_teacher_phase_counts=verified['teacher']['phases'],required_phases_missing=verified['teacher']['required_phases_missing'],
            native_success=verified['native_success_latched'],gate_error_populations=populations,
            counted_actuator_phase_totals=phases,ik_rejected_ticks=phases.get('ik_rejected',0),
            per_tick_IK_result_and_joint_target_commands_recorded=False,
            joint_state_does_not_identify_the_unlogged_servo_target=True,
            acquire_teacher_manipulated_body_ids=stable_ids,acquire_goal_and_body_drift=drift,
            opening_attachment_transition_table=events,time_series=rows))
    report=dict(version='r10_TRAIN_teacher_interface_diagnostics_v4',created_utc=datetime.now(timezone.utc).isoformat(),
        accepted_source=audit.pointer(source),analysis_source=audit.pointer(Path(__file__)),cases=focused,
        conclusions=['Ordinary place: simultaneous position/orientation arrival was never observed, with 179 aggregated IK rejects and a drifting/tipping cup-relative goal.',
            'Peg attachment follows each applied close; its first detach follows the author wp4-open release request, not an inferred loss of object identity.',
            'Bin goal/body separation grows during the arm approach while the waypoint stays fixed; a fixed 9cm coordinate-offset explanation is contradicted by the early 6mm separation.',
            'A robot-induced object push is an inference from timing and source drivers, not a measured force/contact record.'],
        thresholds_changed=False,native_or_GPU_started=False,labels_changed=False,VAL_or_TEST_used=False)
    out=base/'teacher_interface_diagnostics_v4.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['v4 TRAIN teacher/控制接口离线诊断','只读固定三例；未改阈值/重采/训练。']
    for case in focused:
        text += ['',case['task'],f"phase={case['actual_teacher_phase_counts']}；native={case['native_success']}；IK拒绝累计={case['ik_rejected_ticks']}。",
            'delayed close误差：'+json.dumps(case['gate_error_populations']['delayed_close'],ensure_ascii=False),
            '目标/物体漂移：'+json.dumps(case['acquire_goal_and_body_drift'],ensure_ascii=False)]
    text += ['','完整逐帧表、夹爪/attachment转换表在JSON。未记录逐tick IK返回/servo目标关节角，不能用实际joint状态替代这些未观测命令。']
    out.with_suffix('.txt').write_text('\n'.join(text)+'\n');print('focused_cases',len(focused))


if __name__=='__main__':main()
