"""TRAIN-only RGBD replay of frozen public perceptors; no GRU or Native step.

Native states are opened only by the separate metric function after inference.
The insert semantic checkpoint has no block class. RedBoxTracker's rotation is
an explicit world-axis prior, not an observed orientation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..benchmark_perception import RedBoxTracker
from ..contracts import Pose
from ..goal_perception import RedGoalMarker
from ..perception import CameraCalibration
from ..hierarchy_20261030.segmentation import PublicSemanticSegmenter,CLASS_NAMES

ROOT=Path(__file__).resolve().parents[2]
PUBLIC_KEYS=frozenset(('timestamp','observation/images','observation/depths','observation/calibrations'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def public_frame(case,relative):
    """Only NPZ public sensor arrays and signed camera calibration are read."""
    case=Path(case);calibration=json.loads((case/'camera_calibration.json').read_text())
    calibrations={name:CameraCalibration(np.asarray(row['intrinsics']),Pose.from_xyzw(row['camera_to_world']))
                  for name,row in calibration.items()}
    with np.load(case/relative,allow_pickle=False) as frame:
        images={name:frame['images_'+name].copy() for name in calibrations}
        depths={name:frame['depths_'+name].copy() for name in calibrations}
        observation={'timestamp':float(frame['timestamp']),'observation/images':images,
                     'observation/depths':depths,'observation/calibrations':calibrations}
    return observation


class PublicBlockPerceptor:
    def __init__(self,segment_checkpoint,goal_calibration):
        self.box=RedBoxTracker()
        self.goal=RedGoalMarker(json.loads(Path(goal_calibration).read_text())['offset_train_calibrated_m'])
        self.segmenter=PublicSemanticSegmenter(segment_checkpoint,device='cpu')

    def observe(self,observation):
        if set(observation)!=PUBLIC_KEYS:
            raise ValueError('this replay accepts only explicit current public RGBD, calibration, timestamp')
        result={'timestamp':float(observation['timestamp']),'public_input_only':True}
        start=time.perf_counter()
        try:
            state,metadata=self.box.observe(observation)
            result['target']=dict(available=True,position=state.pose.position,quaternion=state.pose.xyzw()[3:],
                linear_velocity=state.linear_velocity,confidence=state.confidence,metadata=metadata,
                rotation_observability='not_measured_world_axis_identity_prior_only')
        except Exception as error:
            result['target']=dict(available=False,error_type=type(error).__name__,error=str(error),
                rotation_observability='unavailable')
        result['target']['latency_seconds']=time.perf_counter()-start
        start=time.perf_counter()
        try:
            pose,metadata=self.goal.observe(observation)
            result['goal_marker']=dict(available=True,position=pose.position,quaternion=pose.xyzw()[3:],metadata=metadata,
                rotation_observability='fixed_task_goal_frame_prior_not_measured')
        except Exception as error:
            result['goal_marker']=dict(available=False,error_type=type(error).__name__,error=str(error))
        result['goal_marker']['latency_seconds']=time.perf_counter()-start
        start=time.perf_counter()
        try:
            segmentation=self.segmenter.predict(observation)
            result['semantic_segmenter']=dict(available=True,class_names=list(CLASS_NAMES),
                quality=segmentation['quality'],provenance=segmentation['provenance'],
                moving_block_pose_available=False,
                applicability='insert_roles_only_no_block_class_no_role_renaming')
        except Exception as error:
            result['semantic_segmenter']=dict(available=False,error_type=type(error).__name__,error=str(error),
                moving_block_pose_available=False)
        result['semantic_segmenter']['latency_seconds']=time.perf_counter()-start
        return plain(result)


def metric_truth(case,timestamp):
    """Native truth is metric-only and never returned to the perceptor."""
    case=Path(case)
    if abs(timestamp)<1e-8:
        state=json.loads((case/'initial_snapshot.json').read_text())['state']
    else:
        rows=[json.loads(line) for line in (case/'native_ticks.jsonl').read_text().splitlines() if line.strip()]
        rows=[row.get('state',row) for row in rows]
        matches=[row for row in rows if abs(row['time']-timestamp)<1e-7]
        if len(matches)!=1:raise ValueError('no unique metric-only Native state at image timestamp')
        state=matches[0]
    return {key:state[key] for key in ('object_position','object_quaternion','lift_goal_position')}


def stats(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(samples=0,mean=None,p95=None,max=None)
    if not np.isfinite(values).all():raise ValueError('nonfinite metric cannot be filtered')
    return dict(samples=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()))


def run(args):
    torch.set_num_threads(2)
    output=Path(args.output)
    if output.exists():raise ValueError('fresh replay artifact required')
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text())
    if plan.get('formal_TEST_read') is not False:raise ValueError('TRAIN-only source plan required')
    train=[row for row in plan['jobs'] if row['split']=='train']
    selected=[];scenes=set()
    # Fixed source-only selection: first of each scene, then first unused.
    for row in train:
        if row['scene'] not in scenes:selected.append(row);scenes.add(row['scene'])
    for row in train:
        if len(selected)>=4:break
        if row not in selected:selected.append(row)
    if len(selected)!=4:raise ValueError('exactly four TRAIN cases required')
    output.mkdir(parents=True)
    frames=[];ledger=[];gaps=[];initialization=[]
    for selected_case in selected:
        case=ROOT/selected_case['path'];run_inputs=json.loads((case/'run_inputs.json').read_text())
        if run_inputs.get('TEST_read') is not False or run_inputs.get('mode')!='p1':
            raise ValueError('source must be TRAIN truth-diagnostic, never formal TEST')
        source=json.loads((case/'worker_exit.json').read_text())
        if source.get('native_closed') is not True:raise ValueError('closed immutable source required')
        controls=[json.loads(line) for line in (case/'transitions.jsonl').read_text().splitlines() if line.strip()]
        records=controls[:5]
        if len(records)!=5:raise ValueError('all five fixed initial frames must exist')
        start=time.perf_counter();perceptor=PublicBlockPerceptor(args.segment_checkpoint,args.goal_calibration)
        initialization.append(time.perf_counter()-start);previous=None
        for index,record in enumerate(records):
            relative=record['rgbd'];observation=public_frame(case,relative)
            if abs(observation['timestamp']-record['before']['time'])>1e-7:
                raise ValueError('RGBD and source control clock mismatch')
            if previous is not None:gaps.append(observation['timestamp']-previous)
            previous=observation['timestamp']
            prediction=perceptor.observe(observation)
            # Prediction completes before the separate Native metric read.
            truth=metric_truth(case,observation['timestamp'])
            metrics={}
            if prediction['target']['available']:
                metrics['target_position_error_m']=float(np.linalg.norm(np.asarray(prediction['target']['position'])-truth['object_position']))
                gt=Pose.from_xyzw(np.r_[truth['object_position'],truth['object_quaternion']])
                relative_rotation=gt.rotation
                metrics['axis_prior_rotation_difference_rad']=float(np.arccos(np.clip((np.trace(relative_rotation)-1)/2,-1,1)))
                metrics['rotation_error_is_prior_consistency_not_measured_rotation_accuracy']=True
            if prediction['goal_marker']['available']:
                metrics['goal_position_error_m']=float(np.linalg.norm(np.asarray(prediction['goal_marker']['position'])-truth['lift_goal_position']))
            frames.append(dict(case=selected_case['episode_id'],frame_index=index,rgbd_path=str(case/relative),rgbd_sha256=sha(case/relative),
                               prediction=prediction,metric_only_comparison=metrics))
        ledger.append(dict(episode_id=selected_case['episode_id'],split='train',scene=selected_case['scene'],path=str(case),
            camera_calibration_sha256=sha(case/'camera_calibration.json'),run_inputs_sha256=sha(case/'run_inputs.json'),
            worker_exit_sha256=sha(case/'worker_exit.json'),fixed_first_frames=5))
    summary={}
    for module in ('target','goal_marker','semantic_segmenter'):
        predictions=[frame['prediction'][module] for frame in frames]
        summary[module]=dict(expected_frames=len(frames),available_frames=sum(p['available'] for p in predictions),
            latency_seconds=stats([p['latency_seconds'] for p in predictions]))
    summary['target']['position_error_m']=stats([f['metric_only_comparison']['target_position_error_m'] for f in frames if 'target_position_error_m' in f['metric_only_comparison']])
    summary['target']['rotation_observability']='world_axis_identity_prior_only_not_visually_estimated'
    summary['goal_marker']['position_error_m']=stats([f['metric_only_comparison']['goal_position_error_m'] for f in frames if 'goal_position_error_m' in f['metric_only_comparison']])
    report=dict(schema='P3_frozen_public_perception_TRAIN_replay_v1',complete=True,CPU_only=True,Native_started=False,
        optimizer_steps=0,formal_TEST_read=False,public_perceptor_inputs_only=sorted(PUBLIC_KEYS),Native_truth_used_only_for_metrics=True,
        source_selection='first_TRAIN_case_per_scene_then_first_unused_prior_to_metrics',case_ledger=ledger,frames=frames,summary=summary,
        segment_checkpoint=dict(path=str(args.segment_checkpoint),sha256=sha(args.segment_checkpoint),class_names=list(CLASS_NAMES),block_class_present=False),
        goal_calibration=dict(path=str(args.goal_calibration),sha256=sha(args.goal_calibration),frozen_TRAIN_offset_no_recalibration=True),
        perceptor_initialization_seconds=stats(initialization),source_frame_gap_seconds=stats(gaps),GRU_called=False,
        source_images_not_admissible_as_GRU_history='100ms_gap_exceeds_frozen_75ms_guard_no_replication_or_interpolation',
        bridge_design=dict(sensor_dt_seconds=.05,real_synchronized_camera_callback_every_physical_tick=True,
            history_inputs=21,state_outputs=19,minimum_real_causal_frames=5,maximum_admitted_gap_seconds=.075,
            prediction='frozen_context_SE3_16_real_50ms_future_steps_to_8_absolute_100ms_endpoints',
            no_fake_midpoint_observations=True,no_frozen_GRU_checkpoint_change=True,
            post_grasp='publicly_evidenced_attached_mode_never_free_body_extrapolation',
            recovery='new_bridge_reacquires_5_valid_real_free_mode_frames_after_public_release_not_old_permanent_close_block',
            unknowns='invisible_stale_unobservable_rotation_or_unresolved_attachment_explicit_unavailable'),
        source_sha256={str(Path(__file__)):sha(__file__),**{str(ROOT/relative):sha(ROOT/relative) for relative in
            ('our_mani/benchmark_perception.py','our_mani/goal_perception.py','our_mani/perception.py','our_mani/hierarchy_20261030/segmentation.py')}})
    (output/'report.json').write_text(json.dumps(plain(report),indent=2,ensure_ascii=False,allow_nan=False))
    (output/'perception_replay.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(output),summary=summary,source_frame_gap_seconds=report['source_frame_gap_seconds']),allow_nan=False))
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=str(ROOT/'runtime/framework/local_mpc_20261102/p2_native_height_data_v2/plan.json'))
    p.add_argument('--segment-checkpoint',default=str(ROOT/'artifacts/framework/multitask_r10/hierarchy_20261030/segmentation_v1/model.pt'))
    p.add_argument('--goal-calibration',default=str(ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json'))
    p.add_argument('--output',default=str(ROOT/'artifacts/framework/local_mpc_20261102/P3_public_perception_replay_v1'))
    return p


if __name__=='__main__':run(parser().parse_args())
