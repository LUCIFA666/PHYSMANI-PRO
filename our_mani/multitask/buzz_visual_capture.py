"""Candidate-only recording of already returned public buzz observations.

No environment object, native handle, simulator getter, step, tracker or model
is accepted. This helper is not connected to a live collector or protocol.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..contracts import Pose,RobotState
from ..perception import CameraCalibration

ROOT=Path(__file__).resolve().parents[2]
NAMESPACE=ROOT/'artifacts/framework/multitask_r10/buzz_visual_capture_candidates'
CAMERAS=('front','8','16','36')
COMPONENTS=('Cuboid','wand_visual','wand_visual_sub')
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
FRAME_SCHEMA='r10_candidate_existing_public_buzz_RGBD_frame_v1'
SEQUENCE_SCHEMA='r10_candidate_existing_public_buzz_RGBD_sequence_v1'


def require(value,message):
    if not value:raise ValueError('buzz visual capture candidate: '+message)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str,value.shape)).encode()+value.tobytes()).hexdigest()


def pointer(path):
    path=Path(path)
    require(path.resolve()==path and path.is_relative_to(ROOT) and path.is_file(),'noncanonical project file')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def check_pointer(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and not Path(value['path']).is_absolute(),
        'explicit relative project path/SHA required')
    path=ROOT/value['path'];require(pointer(path)==value,'project input SHA differs')
    return path


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def numeric_copy(value):
    result=np.asarray(value).copy()
    require(result.dtype.kind in 'biuf','numeric public arrays only')
    result.setflags(write=False);return result


@dataclass(frozen=True)
class PublicRGBDFrame:
    arrays:dict
    metadata:dict


def capture_existing_public_rgbd(observation,robot):
    """Copy only a whitelist of buffers already returned by env.observe().

    Missing exposure times remain explicitly marked as observation-time
    fallbacks. Empty/invalid depth pixels are preserved, not filtered away.
    Other observation keys and robot.task_success are never accessed.
    """
    require(type(robot) is RobotState and type(robot.pose) is Pose,'numeric public RobotState required')
    now=float(observation['timestamp'])
    require(np.isfinite(now) and now>=0 and abs(now-float(robot.timestamp))<=1e-9,'current observation/robot clock differs')
    arrays={};meta=dict(schema=FRAME_SCHEMA,timestamp=now,cameras={},
        task_outcome_or_native_pose_recorded=False,exposure_timestamps_invented=False)
    sensor=observation['observation/sensor_protocol']
    require(type(sensor) is dict and set(sensor)=={'version','camera_render_modes','rgb_processing'}
        and type(sensor['version']) is str and type(sensor['camera_render_modes']) is dict
        and set(sensor['camera_render_modes'])==set(CAMERAS)
        and all(type(mode) is str for mode in sensor['camera_render_modes'].values()),
        'already returned four-view sensor mode metadata required')
    # JSON serialization accepts only ordinary data, never a simulator handle.
    meta['sensor_protocol']=json.loads(json.dumps(sensor,allow_nan=False))
    channels=('observation/images','observation/depths','observation/instance_masks','observation/calibrations')
    for channel in channels:
        require(isinstance(observation[channel],dict) and set(observation[channel])==set(CAMERAS),'exact original four camera views required')
    raw=observation.get('observation/raw_rgb_255')
    require(raw is None or isinstance(raw,dict) and set(raw)==set(CAMERAS),'raw RGB camera inventory differs')
    timestamps=observation.get('observation/camera_timestamps')
    require(timestamps is None or isinstance(timestamps,dict) and set(timestamps)==set(CAMERAS),'camera timestamp inventory differs')
    original_digest=hashlib.sha256()
    for name in sorted(CAMERAS):
        image=numeric_copy(observation['observation/images'][name])
        depth=numeric_copy(observation['observation/depths'][name])
        mask=numeric_copy(observation['observation/instance_masks'][name])
        calibration=observation['observation/calibrations'][name]
        require(type(calibration) is CameraCalibration and type(calibration.camera_to_world) is Pose,
            'numeric CameraCalibration required; native handles are forbidden')
        require(image.dtype==np.uint8 and image.shape==(256,256,3) and depth.shape==(256,256)
            and mask.shape in ((256,256),(256,256,3)),'original aligned256x256 sensor shapes/dtypes required')
        camera_time=now if timestamps is None else float(timestamps[name])
        require(np.isfinite(camera_time) and np.isfinite(calibration.timestamp_offset)
            and np.isfinite(calibration.depth_scale) and calibration.depth_scale>0,'nonfinite camera calibration/time')
        transform=np.eye(4);transform[:3,:3]=calibration.camera_to_world.rotation;transform[:3,3]=calibration.camera_to_world.position
        for key,value in (('rgb',image),('depth',depth),('mask',mask),('K',calibration.intrinsics),('camera_to_world',transform),
                ('depth_scale',np.asarray(calibration.depth_scale)),('timestamp',np.asarray(camera_time)),
                ('timestamp_offset',np.asarray(calibration.timestamp_offset))):
            arrays[key+'_'+name]=numeric_copy(value)
        if raw is not None:
            values=numeric_copy(raw[name]);require(values.shape==image.shape,'raw RGB alignment differs')
            arrays['raw_rgb_255_'+name]=values
        for channel,value in zip(channels[:3],(image,depth,mask)):
            original_digest.update(channel.encode());original_digest.update(name.encode());original_digest.update(np.ascontiguousarray(value).tobytes())
        meta['cameras'][name]=dict(exposure_timestamp_source='provided_public_camera_timestamp' if timestamps is not None else 'observation_timestamp_fallback_not_exposure_measurement',
            raw_timestamp=camera_time,effective_timestamp=camera_time+float(calibration.timestamp_offset),
            effective_time_minus_observation_seconds=camera_time+float(calibration.timestamp_offset)-now,
            depth_units='stored_depth multiplied by stored_depth_scale gives metres',
            finite_positive_depth_pixels=int((np.isfinite(depth)&(depth>0)).sum()),
            invalid_or_empty_depth_pixels=int((~np.isfinite(depth)|(depth<=0)).sum()),raw_rgb_255_available=raw is not None)
    for key,value in (('observation_timestamp',np.asarray(now)),('robot_timestamp',np.asarray(robot.timestamp)),
            ('robot_pose_xyzw',robot.pose.xyzw()),('robot_joints',robot.joints),('robot_joint_velocities',robot.joint_velocities),
            ('robot_linear_velocity',robot.linear_velocity),('robot_angular_velocity',robot.angular_velocity),
            ('robot_gripper_open',np.asarray(robot.gripper_open,bool)),('robot_public_contact',np.asarray(robot.contact,bool))):
        arrays[key]=numeric_copy(value)
    meta['original_collector_sensor_digest']=original_digest.hexdigest()
    meta['channel_arrays']={key:dict(dtype=value.dtype.str,shape=list(value.shape),sha256=array_sha(value)) for key,value in arrays.items()}
    meta['all_channels_digest_including_calibration_and_proprio']=hashlib.sha256(json.dumps(meta['channel_arrays'],sort_keys=True).encode()).hexdigest()
    return PublicRGBDFrame(arrays,meta)


class CandidateVisualCaptureWriter:
    """New append-only candidate namespace; no acquisition identity override."""
    def __init__(self,directory,*,identity,component_instance_ids,canonical_CAD):
        self.directory=Path(directory)
        require(self.directory.resolve()==self.directory and self.directory.is_relative_to(NAMESPACE)
            and self.directory!=NAMESPACE and not self.directory.exists(),'fresh candidate-only project namespace required')
        require(set(identity)=={'split','task','parent_source_group','source_raw_episode_id','replica','seed','variation'}
            and identity['split']=='train' and identity['task'] in TASKS
            and identity['parent_source_group']=='train:'+identity['task']+':'+identity['source_raw_episode_id']
            and all(type(identity[k]) is int and identity[k]>=0 for k in ('replica','seed','variation')),
            'explicit TRAIN candidate identity required')
        require(set(component_instance_ids)==set(COMPONENTS),'all three declared visual components required')
        all_ids=[]
        for ids in component_instance_ids.values():
            require(isinstance(ids,(tuple,list)) and ids and all(type(i) is int and i>=0 for i in ids),'numeric current component IDs required')
            all_ids.extend(ids)
        require(len(all_ids)==len(set(all_ids)),'component mask associations overlap')
        check_pointer(canonical_CAD)
        self.identity=deepcopy(identity);self.components={k:list(v) for k,v in component_instance_ids.items()}
        self.CAD=deepcopy(canonical_CAD);self.records=[];self.closed=False;self.last_timestamp=None
        self.directory.mkdir(parents=True)
        self.source=pointer(Path(__file__))

    def append(self,observation,robot):
        require(not self.closed,'candidate sequence already closed')
        index=len(self.records);stem=self.directory/f'frame_{index:06d}'
        try:
            frame=capture_existing_public_rgbd(observation,robot)
            require(self.last_timestamp is None or frame.metadata['timestamp']>self.last_timestamp,
                'candidate observation timestamps must increase')
        except Exception as error:
            receipt=dict(schema=FRAME_SCHEMA,index=index,status='capture_structure_failure',
                error=dict(class_name=type(error).__name__,message=str(error)),raw_files_committed=False,
                task_outcome_assigned=False,simulation_step_requested=False)
            path=stem.with_suffix('.failure.json');write_json(path,receipt);self.records.append(pointer(path));raise
        path=stem.with_suffix('.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,**frame.arrays)
        receipt=dict(schema=FRAME_SCHEMA,index=index,status='captured_existing_observation',arrays=pointer(path),
            metadata=frame.metadata,simulation_step_requested=False,native_getter_requested=False)
        path=stem.with_suffix('.json');write_json(path,receipt);self.records.append(pointer(path))
        self.last_timestamp=frame.metadata['timestamp']
        return self.records[-1]

    def finish(self):
        require(not self.closed,'candidate sequence already closed')
        self.closed=True
        value=dict(schema=SEQUENCE_SCHEMA,candidate_only=True,capture_sequence_closed=True,
            native_collection_integration_approved=False,actual_acquisition_registration=None,
            identity=self.identity,canonical_CAD=self.CAD,component_instance_ids=self.components,
            helper_source=self.source,frames=self.records,attempted_observation_frames=len(self.records),
            orientation_none_promoted=False,TRAIN_label_namespace_included=False,
            original_K1_or_Context_budget_changed=False,policy_or_task_outcome_assigned=False,
            no_environment_object_or_native_handle_accepted=True)
        for item in self.records:check_pointer(item)
        check_pointer(self.CAD);check_pointer(self.source)
        path=self.directory/'manifest.json';write_json(path,value)
        result=pointer(path);validate_candidate_capture(result);return result


def validate_candidate_capture(manifest_pointer):
    """Pure JSON/NPZ integrity checks; does not grant native use permission."""
    manifest_path=check_pointer(manifest_pointer)
    require(manifest_path.parent.is_relative_to(NAMESPACE) and manifest_path.parent!=NAMESPACE,
        'candidate manifest outside dedicated namespace')
    manifest=json.loads(manifest_path.read_text())
    require(manifest['schema']==SEQUENCE_SCHEMA and manifest['candidate_only'] is True
        and manifest['native_collection_integration_approved'] is False
        and manifest['actual_acquisition_registration'] is None and manifest['orientation_none_promoted'] is False
        and manifest['capture_sequence_closed'] is True and manifest['TRAIN_label_namespace_included'] is False
        and manifest['original_K1_or_Context_budget_changed'] is False
        and manifest['policy_or_task_outcome_assigned'] is False
        and manifest['no_environment_object_or_native_handle_accepted'] is True,
        'candidate scope was widened')
    check_pointer(manifest['canonical_CAD']);check_pointer(manifest['helper_source'])
    require(manifest['attempted_observation_frames']==len(manifest['frames']),'candidate frame inventory differs')
    counts=dict(captured=0,structural_failures=0);last_timestamp=None
    for index,item in enumerate(manifest['frames']):
        row_path=check_pointer(item)
        require(row_path.parent==manifest_path.parent,'candidate frame outside sequence namespace')
        row=json.loads(row_path.read_text());require(row['schema']==FRAME_SCHEMA and row['index']==index,'frame index/schema differs')
        if row['status']=='capture_structure_failure':
            require(row['raw_files_committed'] is False and row['task_outcome_assigned'] is False
                and row['simulation_step_requested'] is False,'capture failure was relabelled')
            counts['structural_failures']+=1;continue
        require(row['status']=='captured_existing_observation' and row['simulation_step_requested'] is False
            and row['native_getter_requested'] is False,'unknown capture action/status')
        arrays_path=check_pointer(row['arrays'])
        require(arrays_path.parent==manifest_path.parent,'candidate arrays outside sequence namespace')
        with np.load(arrays_path,allow_pickle=False) as stored:
            expected=row['metadata']['channel_arrays'];require(set(stored.files)==set(expected),'raw channel inventory differs')
            for name in stored.files:
                value=stored[name];require(expected[name]==dict(dtype=value.dtype.str,shape=list(value.shape),sha256=array_sha(value)),
                    'raw channel bytes/calibration/proprio differs')
            require(row['metadata']['all_channels_digest_including_calibration_and_proprio']==
                hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),'all-channel digest differs')
            now=float(stored['observation_timestamp'])
            require(np.isfinite(now) and now==row['metadata']['timestamp']==float(stored['robot_timestamp'])
                and (last_timestamp is None or now>last_timestamp),'stored observation clock differs')
            last_timestamp=now
        counts['captured']+=1
    return dict(candidate_integrity_verified=True,native_execution_authorized=False,**counts)
