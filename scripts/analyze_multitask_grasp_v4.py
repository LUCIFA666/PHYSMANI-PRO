"""Offline raw-desired versus applied close timing from real v4 TRAIN tracks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_grasp_r10 as grasp
from scripts import analyze_multitask_smoke_r10 as smoke

VERSION='r10_offline_desired_applied_close_timing_v4'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=str(smoke.BASE/'TRAIN_smoke_collection_plan_v4.json'))
    p.add_argument('--queue',default=str(smoke.BASE/'TRAIN_smoke_collection_v4'))
    p.add_argument('--out',default=str(smoke.BASE/'TRAIN_smoke_grasp_analysis_v4.json'))
    args=p.parse_args();report=grasp.analyze(args.plan,args.queue)
    configuration=smoke.read(args.plan)['protocol']['gripper_timing_config']
    for case in report['cases']:
        raw=smoke.read(case['provenance'][0]['path'])
        with np.load(smoke.absolute(raw['entry']['cache']),allow_pickle=False) as loaded:
            arrays={key:loaded[key] for key in loaded.files}
        required=('teacher_desired_opening','teacher_applied_opening','teacher_opening_gate_delayed',
            'teacher_opening_gate_position_error_m','teacher_opening_gate_orientation_error_rad')
        smoke.require(all(name in arrays for name in required),'v4 timing arrays missing')
        smoke.require(np.array_equal(arrays['teacher_opening'],arrays['teacher_applied_opening']),
                      'interaction compatibility opening is not actual applied opening')
        smoke.require(raw['teacher_opening_timing']['configuration']==configuration,
                      'case timing configuration differs from immutable plan')
        selected=arrays['teacher_action_valid'];desired=arrays['teacher_desired_opening'];applied=arrays['teacher_applied_opening']
        previous=bool(arrays['robot_gripper_open'][0]);desired_close=[];desired_open=[]
        for i in np.flatnonzero(selected):
            now=bool(desired[i]>=.5)
            if previous and not now:desired_close.append(int(i))
            if not previous and now:desired_open.append(int(i))
            previous=now
        actual=case['close_transitions'];events=[]
        for i in desired_close:
            end=next((j for j in desired_open if j>i),len(selected))
            emitted=next((row for row in actual if i<=row['frame']<end),None)
            first=case['timeline'][i]
            event=dict(raw_desired_close_frame=i,raw_desired_close_time=first['time'],
                raw_desired_stage=first['teacher_stage'],raw_desired_phase=first['teacher_phase'],
                raw_desired_TCP_to_teacher_waypoint_m=first['measured_TCP_to_teacher_goal_m'],
                raw_desired_TCP_to_body_center_m=first['measured_TCP_to_body_center_m'],
                raw_desired_orientation_error_rad=float(arrays['teacher_opening_gate_orientation_error_rad'][i]),
                actual_close_frame=emitted['frame'] if emitted is not None else None,
                actual_close_time=emitted['time'] if emitted is not None else None,
                delay_seconds=emitted['time']-first['time'] if emitted is not None else None,
                current_waypoint_gate_ever_satisfied=emitted is not None,
                desired_request_withdrawn_before_close=emitted is None and end<len(selected),
                actual_close=emitted)
            if emitted is not None:
                j=emitted['frame'];position=float(arrays['teacher_opening_gate_position_error_m'][j]);angle=float(arrays['teacher_opening_gate_orientation_error_rad'][j])
                smoke.require(emitted['teacher_phase']=='acquire'
                    and position<=configuration['close_position_tolerance_m']+1e-10
                    and angle<=configuration['close_orientation_tolerance_rad']+1e-10,
                    'applied close violated registered measured waypoint timing')
                event.update(actual_close_position_error_m=position,actual_close_orientation_error_rad=angle)
            events.append(event)
        smoke.require(not np.any(selected & (desired>=.5) & (applied<.5)),
                      'raw teacher opening/release was incorrectly held closed')
        for i,row in enumerate(case['timeline']):
            row['raw_teacher_desired_opening']=float(desired[i]) if selected[i] else None
            row['teacher_applied_opening']=float(applied[i]) if selected[i] else None
            row['opening_gate_delayed']=bool(arrays['teacher_opening_gate_delayed'][i])
        case['desired_vs_applied_close_events']=events
        case['desired_applied_mismatch_action_rows']=int((selected & (desired!=applied)).sum())
        case['timing_configuration']=configuration
        case['all_actual_close_events_satisfy_registered_gate']=True
        case['author_open_or_release_preserved']=True
    report['version']=VERSION
    report['configuration']=configuration
    report['timing_analysis_source']=smoke.pointer(Path(__file__))
    report['native_success_not_implied_by_waypoint_arrival']=True
    out=smoke.artifact_path(args.out)
    smoke.require(out.suffix=='.json','analysis must use .json')
    if out.exists():smoke.require(smoke.read(out).get('version')==VERSION,'do not overwrite different evidence')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['R10 v4 TRAIN 期望/实际闭合时序诊断',
        f"SHA验收 {report['verified_cases']}/14；无完整数据 {len(report['missing'])}；只读、未重试/改标签。",
        '阈值：位置 .02m；姿态 .10rad。raw desired 与 applied 分列；距离为native body坐标原点距离，不是proximity sensor阈值。']
    for case in report['cases']:
        lines += ['',case['task']+f"：phase={case['phase_counts']}；缺必需phase={case['required_phases_missing']}；native success={case['native_success_latched']}；grasp_requests={case['alignment_grasp_requests']}。"]
        for row in case['desired_vs_applied_close_events']:
            lines.append(f"  raw close t={row['raw_desired_close_time']:.2f}s，TCP-waypoint={row['raw_desired_TCP_to_teacher_waypoint_m']}m；applied close t={row['actual_close_time']}，delay={row['delay_seconds']}s；position={row.get('actual_close_position_error_m')}m，angle={row.get('actual_close_orientation_error_rad')}rad。")
            if row['actual_close'] is not None:
                lines.append(f"  applied时TCP-body={row['actual_close']['measured_TCP_to_body_center_m']}m；后续attachment={row['actual_close']['any_attachment_observed_before_next_open']}。")
    lines += ['','到达waypoint不代表抓持/后续阶段/native成功。移动目标无法到达、真实短terminal、失败source均保留，不按结果重采。']
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_cases':report['verified_cases'],'expected':14,'missing':len(report['missing']),
                      'close_events':sum(len(case['close_transitions']) for case in report['cases'])},ensure_ascii=False))


if __name__=='__main__':main()
