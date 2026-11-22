"""Offline first-close/attachment evidence from SHA-verified TRAIN smoke.

Body positions and teacher choices below are TRAIN metric/supervision labels.
They are never supplied to a policy and this script cannot execute an action.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as smoke
from our_mani.multitask.specs import get_task_spec

BASE=smoke.BASE
AUTHOR=ROOT.parent/'phsimani/PhysMani'
VERSION='r10_offline_first_close_attachment_analysis_v1'


def source_evidence(path,fragments,expected_sha=None):
    path=Path(path)
    actual_sha=smoke.sha(path)
    smoke.require(expected_sha is None or actual_sha == expected_sha,
                  'source changed since collection; use its archived revision: '+str(path))
    lines=path.read_text().splitlines()
    return dict(path=str(path),sha256=actual_sha,expected_collection_sha256=expected_sha,
        lines=[dict(line=i+1,text=line.strip())
        for i,line in enumerate(lines) if any(fragment in line for fragment in fragments)])


def extract_case(record,arrays,verified):
    times=arrays['times'];ids=list(map(str,arrays['body_ids']));n=len(times)
    actions=arrays['teacher_action_valid'];opening=arrays['teacher_opening']
    required=('measured_tcp_xyzw','teacher_tcp_xyzw','teacher_opening','robot_gripper_open','robot_contact')
    smoke.require(all(name in arrays for name in required),'missing logged TCP/gripper arrays')
    smoke.require(arrays['measured_tcp_xyzw'].shape == arrays['teacher_tcp_xyzw'].shape == (n,7)
        and opening.shape == arrays['robot_gripper_open'].shape == arrays['robot_contact'].shape == (n,),
        'TCP/gripper array shape differs')
    requested=bool(arrays['robot_gripper_open'][0]);close_indices=[];open_indices=[]
    for i in np.flatnonzero(actions):
        now=bool(opening[i] >= .5)
        if requested and not now:close_indices.append(int(i))
        if not requested and now:open_indices.append(int(i))
        requested=now
    timeline=[]
    for i,time in enumerate(times):
        body=str(arrays['teacher_manipulated_body'][i]) if actions[i] else ''
        j=ids.index(body) if body in ids else None
        tcp=arrays['measured_tcp_xyzw'][i,:3]
        goal=arrays['teacher_tcp_xyzw'][i,:3] if actions[i] else None
        position=arrays['target_positions'][i,j] if j is not None else None
        valid=bool(j is not None and arrays['native_pose_valid'][i,j])
        timeline.append(dict(frame=i,time=float(time),teacher_action_valid=bool(actions[i]),
            teacher_stage=str(arrays['teacher_stage'][i]),teacher_phase=str(arrays['teacher_phase'][i]),
            teacher_requested_opening=float(opening[i]) if actions[i] else None,
            measured_gripper_open_bool=bool(arrays['robot_gripper_open'][i]),
            observed_any_attachment_bool=bool(arrays['robot_contact'][i]),
            TRAIN_teacher_manipulated_body=body,native_body_pose_valid=valid,
            public_associated_motion_mode=('external','attached','released')[int(arrays['modes'][i,j])] if j is not None else None,
            public_motion_mode_valid=bool(arrays['mode_valid'][i,j]) if j is not None else None,
            measured_tcp_position_m=tcp.tolist(),teacher_tcp_goal_position_m=goal.tolist() if goal is not None else None,
            native_body_center_position_m=position.tolist() if valid else None,
            measured_TCP_to_body_center_m=float(np.linalg.norm(tcp-position)) if valid else None,
            teacher_goal_to_body_center_m=float(np.linalg.norm(goal-position)) if valid and goal is not None else None,
            measured_TCP_to_teacher_goal_m=float(np.linalg.norm(tcp-goal)) if goal is not None else None))
    closes=[]
    for i in close_indices:
        body=str(arrays['teacher_manipulated_body'][i]);smoke.require(body in ids,'close action lacks a stable TRAIN body identity')
        j=ids.index(body)
        end=next((k for k in open_indices if k > i),n)
        following=np.arange(n)>i
        following &= np.arange(n)<end
        attached=np.flatnonzero(following & arrays['robot_contact'])
        native_valid=following & arrays['native_pose_valid'][:,j]
        distances=np.linalg.norm(arrays['measured_tcp_xyzw'][:,:3]-arrays['target_positions'][:,j],axis=1)
        other_identity=[k for k in np.flatnonzero(following & actions)
            if str(arrays['teacher_manipulated_body'][k]) != body]
        actual_phases=dict(Counter(map(str,arrays['teacher_phase'][following & actions])))
        closes.append(dict(**timeline[i],close_transition=True,next_requested_open_frame=end if end < n else None,
            later_rows_before_next_open=int(following.sum()),
            first_any_attachment_frame=int(attached[0]) if len(attached) else None,
            first_any_attachment_time=float(times[attached[0]]) if len(attached) else None,
            attachment_latency_seconds=float(times[attached[0]]-times[i]) if len(attached) else None,
            any_attachment_observed_before_next_open=bool(len(attached)),
            measured_closed_bool_rows_before_next_open=int((following & ~arrays['robot_gripper_open']).sum()),
            minimum_later_TCP_to_same_body_center_m=float(distances[native_valid].min()) if native_valid.any() else None,
            later_associated_modes=dict(Counter(('external','attached','released')[int(mode)] for mode in arrays['modes'][following,j])),
            later_public_mode_ambiguous_rows=int((following & ~arrays['mode_valid'][:,j]).sum()),
            teacher_manipulated_body_changed_frames=list(map(int,other_identity)),
            later_teacher_phase_counts=actual_phases,
            first_close_neighborhood=timeline[max(0,i-3):min(n,i+9)]))
    clock=record['alignment']['clock']
    smoke.require(clock.get('grasp_requests') == len(close_indices)
        and clock.get('release_requests') == len(open_indices),'requested-opening transitions disagree with native ledger')
    phase_counts=verified['teacher']['phases']
    last_phase=next((str(arrays['teacher_phase'][i]) for i in range(n-1,-1,-1) if actions[i]),None)
    last_contact=bool(arrays['robot_contact'][-1])
    classify=('no_close_requested' if not closes else 'at_least_one_attachment_observed_after_close'
        if any(row['any_attachment_observed_before_next_open'] for row in closes) else 'close_requested_but_no_attachment_observed')
    return dict(task=record['task'],case_id=record['case_id'],frames=n,
        phase_counts=phase_counts,required_phases_missing=verified['teacher']['required_phases_missing'],
        last_teacher_phase=last_phase,last_any_attachment_bool=last_contact,
        alignment_grasp_requests=clock['grasp_requests'],alignment_release_requests=clock['release_requests'],
        native_success_latched=record['alignment']['native_success_latched'],
        native_terminal_failure_latched=record['alignment']['native_terminal_failure_latched'],
        classification=classify,close_transitions=closes,open_transition_frames=open_indices,
        total_any_attachment_observed_rows=int(arrays['robot_contact'].sum()),
        per_body_native_attachment_identity_recorded=False,
        public_contact_semantics='binary len(gripper.get_grasped_objects()) > 0; not force/tactile contact or a specific-body ID',
        metric_native_attachment_fields=record['alignment'].get('native_attachment'),
        provenance=verified['provenance'],timeline=timeline,
        teacher_source=source_evidence(AUTHOR/get_task_spec(record['task']).native_source_relative_path,
            ("is_grasp", "stage == 'wp1'", "stage == 'wp1-grasp'", "stage == 'wp2'"),
            get_task_spec(record['task']).native_source_sha256))


def analyze(plan,queue):
    accepted=smoke.analyze(plan,queue)
    plan_value=smoke.read(plan)
    cases=[]
    for verified in accepted['cases']:
        record=smoke.read(verified['provenance'][0]['path'])
        with np.load(smoke.absolute(record['entry']['cache']),allow_pickle=False) as loaded:
            arrays={key:loaded[key] for key in loaded.files}
        cases.append(extract_case(record,arrays,verified))
    failed=[row for row in cases if row['classification'] == 'close_requested_but_no_attachment_observed']
    attached=[row for row in cases if row['classification'] == 'at_least_one_attachment_observed_after_close']
    stuck=[row for row in failed if row['last_teacher_phase'] == 'acquire']
    return dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        expected_cases=accepted['expected_cases'],verified_cases=len(cases),missing=accepted['missing'],
        invalid=accepted['invalid'],complete=accepted['complete'],plan_sha256=accepted['plan_sha256'],
        classification_counts=dict(Counter(row['classification'] for row in cases)),
        no_attachment_and_last_phase_acquire_tasks=[row['task'] for row in stuck],
        attachment_observed_comparison_tasks=[row['task'] for row in attached],
        evidence_summary='Some acquire stalls coincide with a single early close request and no attachment thereafter; '
            'this is observational evidence, not a controlled retry experiment or proof every stall has one cause.',
        limitations=['TCP-to-body distances are distances to the native body frame origin, not mesh clearance or proximity-sensor detection volumes.',
            'Native attachment identity and proximity-sensor results per attempted object were not recorded; any-attachment bool is all that can be asserted.',
            'A finger-open threshold is not a grasp-success test; attachment can appear before fingers reach their closed threshold.',
            'No native replays, retries, protocol changes, label rewrites, fitting, VAL, or TEST were performed.'],
        action_order_evidence=source_evidence(ROOT/'our_mani/multitask/simulation.py',
            ('requested_opening !=','self.gripper.grasp','self.ledger.grasp_requests','self.gripper.actuate','self._tick(phase)',
             'bool(self.gripper.get_grasped_objects())'),
             plan_value['inputs'][str(ROOT/'our_mani/multitask/simulation.py')]),
        native_grasp_implementation_evidence=source_evidence(
            AUTHOR/'third_party/PyRep/pyrep/robots/end_effectors/gripper.py',
            ('def grasp(', '_proximity_sensor.is_detected','obj not in self._grasped_objects','obj.set_parent(',
             'return detected','def get_grasped_objects','return self._grasped_objects')),
        cases=cases,provenance=accepted['provenance']+[smoke.pointer(Path(__file__))],
        labels_modified=False,native_or_GPU_started=False,TEST_read=False)


def text_report(report):
    lines=['R10 TRAIN smoke 首次闭合与抓持证据',
        f"SHA验收 {report['verified_cases']}/{report['expected_cases']}；无完整数据 {len(report['missing'])}；数据错误 {len(report['invalid'])}。",
        '只读真实时间序列；未重试、改协议、训练或修改标签。',
        'contact 在当前实现中表示 gripper.get_grasped_objects() 非空，不是力传感；未记录所抓具体body或每次proximity检测值。',
        '距离以native body坐标原点计，不能直接等同于接触距离或抓持检测阈值。','']
    for case in report['cases']:
        lines.append(f"{case['task']}：{case['classification']}；grasp_requests={case['alignment_grasp_requests']}；teacher={case['phase_counts']}。")
        for event in case['close_transitions']:
            lines.append(f"  close t={event['time']:.2f}s / {event['teacher_stage']} / {event['TRAIN_teacher_manipulated_body']}：TCP-body={event['measured_TCP_to_body_center_m']:.4f}m，goal-body={event['teacher_goal_to_body_center_m']:.4f}m，TCP-goal={event['measured_TCP_to_teacher_goal_m']:.4f}m；后续attachment={event['any_attachment_observed_before_next_open']}，最近TCP-body={event['minimum_later_TCP_to_same_body_center_m']}m。")
    lines += ['', '无attachment且最终停acquire：'+', '.join(report['no_attachment_and_last_phase_acquire_tasks']),
        '观测到attachment的对照：'+', '.join(report['attachment_observed_comparison_tasks']),
        '观测支持部分任务存在过早close且仅transition尝试抓持的问题；尚未做控制实验，不能把所有卡阶段都归为单一原因。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(BASE/'TRAIN_smoke_collection_plan_v3.json'))
    parser.add_argument('--queue',default=str(BASE/'TRAIN_smoke_collection_v3'))
    parser.add_argument('--out',default=str(BASE/'TRAIN_smoke_grasp_analysis_v3.json'))
    args=parser.parse_args();report=analyze(args.plan,args.queue);out=smoke.artifact_path(args.out)
    smoke.require(out.suffix == '.json','report must use .json')
    protected={smoke.absolute(args.plan),smoke.absolute(args.queue)/'manifest.json',smoke.absolute(args.queue)/'queue_report.json'}
    protected.update(smoke.absolute(path) for path in smoke.read(args.plan)['inputs'])
    protected.update(Path(row['path']) for case in report['cases'] for row in case['provenance'])
    protected.update(Path(row['path']) for row in report['missing']+report['invalid'] if 'path' in row)
    smoke.require(out not in protected and out.with_suffix('.txt') not in protected,
                  'offline analysis cannot overwrite collection evidence')
    if out.exists():smoke.require(smoke.read(out).get('version') == VERSION,'cannot overwrite other evidence')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    out.with_suffix('.txt').write_text(text_report(report))
    print(json.dumps({key:report[key] for key in ('verified_cases','expected_cases','complete','classification_counts')},ensure_ascii=False))


if __name__ == '__main__':main()
