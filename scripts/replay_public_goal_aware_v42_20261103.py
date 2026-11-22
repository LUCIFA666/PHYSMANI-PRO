#!/usr/bin/env python3
"""Public-only causal observer replay, no Native/GT/outcome files or fitting.

Real robot/close intents and current executor h remain fixed recorded v41 data.
Observer and active-evidence replay is NOT a v42 closed-loop outcome estimate.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose,ObjectState
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.public_grasp import PublicLiftHoldMetric
from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
from our_mani.local_mpc_control_20261103.public_grasp_v42 import GoalAwareActiveVisualPublicBridge,ActiveVisualGraspV42Config
from our_mani.local_mpc_control_20261103.goal_aware_binding_v42 import public_RGBD_digest
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
from our_mani.local_mpc_20261102.records import append,write_new

CASES={
 'static':ROOT/'runtime/framework/local_mpc_20261102/public_grasp_static_active_visual_v41',
 'uniform_002':ROOT/'runtime/framework/local_mpc_20261102/public_v41_motion_canaries_v1/uniform_002',
 'speed_change_002_004':ROOT/'runtime/framework/local_mpc_20261102/public_v41_motion_canaries_v1/speed_change_002_004',
 'uniform_012':ROOT/'runtime/framework/local_mpc_20261102/public_v41_motion_canaries_v1/uniform_012'}
SOURCE_FILES=[
 'our_mani/local_mpc_control_20261103/goal_aware_binding_v42.py','our_mani/local_mpc_control_20261103/public_grasp_v42.py',
 'our_mani/local_mpc_control_20261103/goal_aware_perception.py','our_mani/local_mpc_control_20261103/public_grasp_v41.py',
 'our_mani/local_mpc_control_20261103/public_grasp_v4.py','our_mani/local_mpc_control_20261103/public_grasp.py',
 'our_mani/local_mpc_control_20261103/public_grasp_state.py','our_mani/local_mpc_control_20261103/public_attachment.py',
 'our_mani/local_mpc_control_20261103/public_shadow.py','our_mani/local_mpc_control_20261103/public_state.py',
 'our_mani/local_mpc_control_20261103/joint_goal_cem.py','our_mani/local_mpc_control_20261103/public_grasp_probe.py',
 'our_mani/local_mpc_20261102/records.py','our_mani/contracts.py','our_mani/context_dynamics.py',
 'our_mani/local_mpc_control_20261103/active_visual_attachment.py','our_mani/local_mpc_control_20261103/public_motion_bridge.py',
 'our_mani/benchmark_perception.py','our_mani/goal_perception.py','our_mani/perception.py','our_mani/multitask/context_se3.py',
 'artifacts/framework/local_policy/goal_marker_calibration.json',
 'artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
 'scripts/replay_public_goal_aware_v42_20261103.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(values):
    if not values:return dict(count=0,mean=None,p50=None,p95=None,max=None)
    a=np.asarray(values,float);return dict(count=len(a),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),max=float(a.max()))
def valid(target,now):return isinstance(target,ObjectState) and target.visible and target.confidence>=.5 and abs(target.timestamp-now)<1e-8


def replay_case(name,output,limit=None):
    case=CASES[name];public=case/'public_shadow';source=public/'public_outputs.jsonl';calibration=public/'camera_calibration.json'
    decisions=case/'public_grasp_v41_decisions.jsonl'
    original=[json.loads(line) for line in source.open()]
    control={round(float(r['time']),8):r['actuator_state_before'] for r in map(json.loads,decisions.open())}
    cameras={n:CameraCalibration(v['intrinsics'],Pose.from_xyzw(v['camera_to_world'])) for n,v in json.loads(calibration.read_text()).items()}
    sources=dict(public_outputs_sha256=sha(source),camera_calibration_sha256=sha(calibration),actual_executor_decisions_sha256=sha(decisions),raw_RGBD_files=[])
    wrappers={'original_v41':ActiveVisualPublicBridge(active_config=ActiveVisualConfig(probe_seconds=.8)),
        'goal_aware_v42':GoalAwareActiveVisualPublicBridge(active_config=ActiveVisualConfig(probe_seconds=.8))}
    metrics={k:PublicLiftHoldMetric(ActiveVisualGraspV42Config()) for k in wrappers}
    summary={k:dict(current_visual_frames=0,unknown_visual_frames=0,active_confirmed_frames=0,free_GRU_frames=0,mode_counts={},
        public_hold_max_seconds=0.,replayed_public_hold_observed=False,jump_events=[],confirmed_times=[],admissions=[],
        filter_unavailable_frames=0,filter_removed_red_pixels=[],frame_wall_seconds=[],raw_RGBD_unchanged=True) for k in wrappers}
    previous={k:None for k in wrappers};delta=[];original_delta=[];interesting=[];started=time.perf_counter()
    ledger=output/'frames.jsonl'
    for index,row in enumerate(original[:limit] if limit is not None else original):
        now=float(row['result']['timestamp']);robot=PublicRobotSensors.from_mapping(row['public_robot'])
        raw_path=public/row['rgbd']
        with np.load(raw_path,allow_pickle=False) as frame:
            observed_time=float(frame['timestamp'])
            obs={'timestamp':observed_time,'observation/images':{n:frame['images_'+n].copy() for n in cameras},
                'observation/depths':{n:frame['depths_'+n].copy() for n in cameras},'observation/calibrations':cameras}
        if abs(now-observed_time)>1e-8 or abs(now-robot.timestamp)>1e-8:raise ValueError('exact real timestamp join required')
        sources['raw_RGBD_files'].append(dict(path=str(raw_path),sha256=sha(raw_path),timestamp=now))
        raw_before=public_RGBD_digest(obs);item=dict(index=index,time=now,robot=robot,close_requested=row['close_requested'],variants={})
        targets={}
        for key,wrapper in wrappers.items():
            tick=time.perf_counter();result=wrapper.observe(obs,robot,close_requested=row['close_requested'])
            target=wrapper.binding.latest_visual_target;targets[key]=target;s=summary[key]
            isvalid=valid(target,now);s['current_visual_frames']+=int(isvalid);s['unknown_visual_frames']+=int(not isvalid)
            s['mode_counts'][result['mode']]=s['mode_counts'].get(result['mode'],0)+1
            s['active_confirmed_frames']+=int(result['mode']=='ATTACHED_PUBLIC');s['free_GRU_frames']+=int(result['free_forecast_available'])
            if result['mode']=='ATTACHED_PUBLIC':s['confirmed_times'].append(now)
            current_h=control.get(round(now,8))
            if current_h is not None and wrapper.active.spec is None and not wrapper.active.failed and row['close_requested']:
                spec=wrapper.active.begin_probe(target,robot,current_h) if target is not None else None
                if spec is not None:s['admissions'].append(dict(time=now,deadline=spec['deadline'],attempt_id=spec['attempt_id']))
            value=metrics[key].observe(result,robot,wrapper.latest_attachment,wrapper.close_visual_baseline)
            s['public_hold_max_seconds']=max(s['public_hold_max_seconds'],value['public_hold_seconds'])
            s['replayed_public_hold_observed']|=value['public_lift_hold_observed']
            jump=None
            if isvalid:
                old=previous[key]
                if old is not None and abs(now-old[0]-.05)<1e-6:
                    jump=float(np.linalg.norm(target.pose.position-old[1]))
                    if jump>.05:s['jump_events'].append(dict(time=now,jump_m=jump,position=target.pose.position.copy(),diagnostic_threshold_m=.05))
                previous[key]=(now,target.pose.position.copy())
            else:previous[key]=None
            trace=result.get('goal_aware_filter_trace')
            if trace is not None:
                s['filter_unavailable_frames']+=int(trace.get('filter_available') is not True)
                if trace.get('filter_available'):s['filter_removed_red_pixels'].append(trace['removed_red_pixels_total'])
            raw_after=public_RGBD_digest(obs)
            if raw_after!=raw_before:raise RuntimeError('raw RGBD mutated by private filter')
            s['frame_wall_seconds'].append(time.perf_counter()-tick)
            item['variants'][key]=dict(valid_current_visual=isvalid,observed_target=target,mode=result['mode'],
                reason=result['reason'],active_attachment=wrapper.latest_attachment,public_lift_hold=value,
                free_GRU_available=result['free_forecast_available'],GRU_positions=result.get('positions'),
                tracker_filter_trace=trace,tracker_error=result.get('goal_aware_tracker_error'),
                public_50ms_position_jump_m=jump,raw_RGBD_sha256=raw_before)
        if valid(targets['original_v41'],now) and valid(targets['goal_aware_v42'],now):
            delta.append(float(np.linalg.norm(targets['original_v41'].pose.position-targets['goal_aware_v42'].pose.position)))
        old_obj=row['result'].get('object_observation')
        if old_obj and abs(float(old_obj['timestamp'])-now)<1e-8 and valid(targets['original_v41'],now):
            original_delta.append(float(np.linalg.norm(np.asarray(old_obj['pose']['position'])-targets['original_v41'].pose.position)))
        old_reason=row['result'].get('attachment_estimate',{}).get('reason')
        if old_reason=='current_attached_relative_or_orientation_consistency_lost':
            interesting.append(dict(time=now,original_recorded_reason=old_reason,
                original_recorded_position=old_obj['pose']['position'] if old_obj else None,
                goal_aware_current=targets['goal_aware_v42'],goal_aware_valid=valid(targets['goal_aware_v42'],now)))
        append(ledger,item)
    for key,s in summary.items():
        s['frame_wall_seconds']=stats(s['frame_wall_seconds']);s['filter_removed_red_pixels']=stats(s['filter_removed_red_pixels'])
        s['binding']=wrappers[key].binding.receipt();s['active_config']=__import__('dataclasses').asdict(wrappers[key].active.config)
    result=dict(case=name,frames=min(len(original),limit) if limit is not None else len(original),complete=limit is None,
        actual_recorded_robot_intent_and_current_h_only=True,Native_GT_or_labels_read=False,closed_loop_v42_outcome_claimed=False,
        formal_precision_or_task_qualification=False,scope='causal_public_observer_and_evidence_replay_under_fixed_recorded_v41_trajectory',
        variants=summary,paired_observer_position_delta_m=stats(delta),
        replay_original_vs_original_log_position_delta_m=stats(original_delta),
        old_postconfirmation_inconsistency_frames=interesting,input_sources=sources,wall_seconds=time.perf_counter()-started)
    write_new(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',action='append',choices=sorted(CASES));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.limit is not None and args.limit<=0:raise ValueError('positive explicit smoke limit required')
    args.output.mkdir(parents=True,exist_ok=False);start={f:sha(ROOT/f) for f in SOURCE_FILES};results=[]
    for name in args.case or list(CASES):
        output=args.output/name;output.mkdir();result=replay_case(name,output,args.limit);results.append(dict(case=name,frames=result['frames'],path=str(output/'report.json')))
        print(name,result['frames'],{k:(v['current_visual_frames'],v['unknown_visual_frames'],v['active_confirmed_frames']) for k,v in result['variants'].items()},flush=True)
    end={f:sha(ROOT/f) for f in SOURCE_FILES}
    if start!=end:raise RuntimeError('replay source changed during fixed comparison')
    write_new(args.output/'manifest.json',dict(version='goal_aware_public_observer_replay_v42_20261103_v1',complete=args.limit is None,
        source_sha256=start,source_unchanged=True,cases=results,Native_GT_or_labels_read=False,TEST_or_VAL_read=False,
        policy_inputs_only='stored raw RGBD, public calibration, real robot/knownintent and same-time actual executor h',
        mode='observer_replay_not_new_closed_loop_controller_or_TaskSuccess',smoke_limit=args.limit))
if __name__=='__main__':main()
