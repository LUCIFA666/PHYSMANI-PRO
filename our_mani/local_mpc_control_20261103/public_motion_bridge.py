"""Public 50ms RGBD -> unchanged v671 GRU/context -> world-space H8 bridge.

No Native object pose, held flag, success, stage, handle, mask or future frame
is accepted. RedBoxTracker supplies an explicitly unobserved identity rotation
prior. Attachment evidence is a caller assertion about public sensors, never a
claim that a closing request established contact. No contact model is added.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np

from ..benchmark_perception import RedBoxTracker
from ..contracts import ObjectState,Pose,array
from ..goal_perception import RedGoalMarker
from ..perception import CameraCalibration
from ..multitask.context_se3 import MultiBodyContextDynamics,HISTORY_DIM,STATE_DIM

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt'
CHECKPOINT_SHA256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'
GOAL_CALIBRATION=ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json'
PUBLIC_OBSERVATION_KEYS=frozenset(('timestamp','observation/images','observation/depths','observation/calibrations'))
CAMERAS=frozenset(('front','8','16','36'))
ROBOT_KEYS=frozenset(('timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts'))
PUBLIC_EVIDENCE_SOURCES=frozenset(('rgbd_relative_pose_tracking','public_robot_pose','rgbd_free_motion_consistency',
                                  'public_gripper_opening','rgbd_object_release'))


class MotionMode(str,Enum):
    FREE='FREE'
    UNRESOLVED='UNRESOLVED'
    ATTACHED_PUBLIC='ATTACHED_PUBLIC'
    STALE='STALE'


@dataclass(frozen=True)
class PublicRobotSensors:
    timestamp:float
    tcp_position:object
    tcp_quaternion:object
    q:object
    dq:object
    finger_open_amounts:object

    def __post_init__(self):
        if isinstance(self.timestamp,bool) or not np.isfinite(self.timestamp) or self.timestamp<0:
            raise ValueError('actual public robot timestamp required')
        for key,shape in (('tcp_position',(3,)),('tcp_quaternion',(4,)),('q',(7,)),('dq',(7,)),('finger_open_amounts',(2,))):
            object.__setattr__(self,key,array(getattr(self,key),shape))
        if not np.isclose(np.linalg.norm(self.tcp_quaternion),1.,atol=1e-4):
            raise ValueError('public TCP unit xyzw quaternion required')
        if np.any((self.finger_open_amounts<0)|(self.finger_open_amounts>1)):
            raise ValueError('two measured normalized finger openings required')

    @classmethod
    def from_mapping(cls,value):
        if not isinstance(value,dict) or set(value)!=ROBOT_KEYS:
            raise ValueError('only sanitized robot timestamp/TCP/q/dq/two-finger sensors are accepted')
        return cls(**value)

    @property
    def tcp_pose(self):return Pose.from_xyzw(np.r_[self.tcp_position,self.tcp_quaternion])


@dataclass(frozen=True)
class PublicMotionEvidence:
    event:str
    timestamp:float
    sources:tuple[str,...]
    tcp_relative_pose:Pose|None=None
    orientation_observable:bool=False

    def __post_init__(self):
        if self.event not in ('attachment_confirmed','release_confirmed','free_motion_confirmed','manipulation_unresolved'):
            raise ValueError('known explicit public motion evidence event required')
        if isinstance(self.timestamp,bool) or not np.isfinite(self.timestamp) or self.timestamp<0:
            raise ValueError('public evidence timestamp required')
        if (not self.sources or len(set(self.sources))!=len(self.sources)
                or not set(self.sources)<=PUBLIC_EVIDENCE_SOURCES):
            raise ValueError('Native/unknown evidence source is forbidden')
        if type(self.orientation_observable) is not bool:
            raise ValueError('explicit public orientation observability required')
        if self.event=='attachment_confirmed':
            if not isinstance(self.tcp_relative_pose,Pose) or not {'rgbd_relative_pose_tracking','public_robot_pose'}<=set(self.sources):
                raise ValueError('attachment needs public RGBD/TCP evidence and estimated TCP-relative Pose')
        elif self.tcp_relative_pose is not None:
            raise ValueError('TCP-relative attachment pose belongs only to attachment evidence')
        if self.event=='release_confirmed' and not {'rgbd_object_release','public_gripper_opening'}<=set(self.sources):
            raise ValueError('release requires public RGBD release and measured gripper-opening evidence')
        if self.event=='free_motion_confirmed' and 'rgbd_free_motion_consistency' not in self.sources:
            raise ValueError('free motion must be re-established from public RGBD consistency')


class PublicMotionBridge:
    nominal_sensor_dt=.05
    maximum_gap=.075
    minimum_frames=5

    def __init__(self,checkpoint=CHECKPOINT,*,expected_sha256=CHECKPOINT_SHA256,forecaster=None,
                 tracker=None,goal_marker=None,goal_calibration=GOAL_CALIBRATION,minimum_confidence=.5):
        if not np.isfinite(minimum_confidence) or not 0<minimum_confidence<=1:
            raise ValueError('predeclared positive confidence threshold required')
        self.minimum_confidence=float(minimum_confidence)
        self._tracker_injected=tracker is not None;self._goal_marker_injected=goal_marker is not None
        self.tracker=RedBoxTracker() if tracker is None else tracker
        self.goal_marker=RedGoalMarker(json.loads(Path(goal_calibration).read_text())['offset_train_calibrated_m']) if goal_marker is None else goal_marker
        self._mock_forecaster=forecaster is not None
        if forecaster is None:
            checkpoint=Path(checkpoint).resolve(strict=True)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected_sha256:
                raise ValueError('unchanged frozen v671 checkpoint SHA required')
            forecaster=MultiBodyContextDynamics(checkpoint)
            forecaster.model.requires_grad_(False)
            if HISTORY_DIM!=21 or STATE_DIM!=19 or forecaster.maximum_validated_horizon<.8:
                raise ValueError('frozen 21-history/19-state/50ms model contract required')
        self.forecaster=forecaster
        self.reset()

    def _clear_tracker(self):
        if hasattr(self.tracker,'reset'):self.tracker.reset()
        else:
            self.tracker.history.clear();self.tracker.last_velocity=np.zeros(3)

    def _clear_motion(self,*,clear_tracker=True):
        self.forecaster.reset();self.history_frames=0;self._last_object=None;self._last_result=None
        if clear_tracker:self._clear_tracker()

    def reset(self):
        self.forecaster.reset();self._clear_tracker();self.history_frames=0;self._last_object=None
        self.last_timestamp=None;self.mode=MotionMode.FREE;self._resume_mode=MotionMode.FREE
        self._attachment=None;self.goal_pose=None;self.goal_metadata=None;self._last_result=None

    def _base(self,timestamp):
        return dict(timestamp=float(timestamp),mode=self.mode.value,available=False,free_forecast_available=False,
            controlled_mode_available=False,reason=None,history_frames=self.history_frames,
            object_observation=self._last_object,forecast=None,endpoint_times=None,positions=None,rotations=None,
            linear_velocities=None,angular_velocities=None,controlled=None,goal_pose=self.goal_pose,
            static_goal_assumption=True,goal_cached=self.goal_pose is not None,goal_metadata=self.goal_metadata,
            rotation_observable=False,rotation_source='known_axis_identity_prior_not_visual_orientation',
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
            frozen_checkpoint_sha256=getattr(self.forecaster,'checkpoint_sha256',None),
            forecaster_is_injected_test_double=self._mock_forecaster,
            tracker_dependency_injected=self._tracker_injected,goal_marker_dependency_injected=self._goal_marker_injected)

    def _unavailable(self,now,reason):
        result=self._base(now);result['reason']=reason;self._last_result=result;return result

    def _stale(self,now,reason):
        if self.mode!=MotionMode.STALE:
            self._resume_mode=MotionMode.UNRESOLVED if self.mode in (MotionMode.UNRESOLVED,MotionMode.ATTACHED_PUBLIC) else MotionMode.FREE
        self._attachment=None;self._clear_motion();self.mode=MotionMode.STALE
        return self._unavailable(now,reason)

    def _apply_evidence(self,evidence,now,robot):
        if not isinstance(evidence,PublicMotionEvidence) or abs(evidence.timestamp-now)>1e-8:
            raise ValueError('only fresh timestamp-aligned explicit PublicMotionEvidence is accepted')
        if evidence.event=='attachment_confirmed':
            if self.mode!=MotionMode.ATTACHED_PUBLIC:self._clear_motion(clear_tracker=False)
            self.mode=MotionMode.ATTACHED_PUBLIC;self._attachment=evidence
        elif evidence.event in ('release_confirmed','free_motion_confirmed'):
            if evidence.event=='release_confirmed' and float(robot.finger_open_amounts.mean())<=.5:
                raise ValueError('public release assertion disagrees with measured open fingers')
            if self.mode!=MotionMode.FREE:self._clear_motion()
            self.mode=MotionMode.FREE;self._attachment=None
        else:
            if self.mode!=MotionMode.UNRESOLVED:self._clear_motion(clear_tracker=False)
            self.mode=MotionMode.UNRESOLVED;self._attachment=None

    def observe(self,observation,robot,*,close_requested=False,evidence=None):
        if not isinstance(observation,dict) or set(observation)!=PUBLIC_OBSERVATION_KEYS:
            raise ValueError('only public current RGBD/calibration/timestamp keys are accepted')
        for key in ('observation/images','observation/depths','observation/calibrations'):
            if not isinstance(observation[key],dict) or set(observation[key])!=CAMERAS:
                raise ValueError('exactly four public camera views/calibrations required; no Native channels')
        for camera in CAMERAS:
            image=np.asarray(observation['observation/images'][camera]);depth=np.asarray(observation['observation/depths'][camera])
            if (image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3 or depth.shape!=image.shape[:2]
                    or not isinstance(observation['observation/calibrations'][camera],CameraCalibration)):
                raise ValueError('aligned public RGB/depth arrays and CameraCalibration required')
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        now=observation['timestamp']
        if isinstance(now,bool) or not np.isfinite(now) or now<0 or abs(float(now)-robot.timestamp)>1e-8:
            raise ValueError('synchronized actual public sensor timestamps required')
        now=float(now)
        if not isinstance(close_requested,(bool,np.bool_)):
            raise ValueError('explicit boolean action request required; it is not an attachment label')
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('strictly causal unique real sensor timestamps required')
        gap=None if self.last_timestamp is None else now-self.last_timestamp
        self.last_timestamp=now
        if gap is not None and gap>self.maximum_gap:
            return self._stale(now,'public_observation_gap_over_75ms_no_interpolated_frames')
        if gap is not None and not np.isclose(gap,self.nominal_sensor_dt,rtol=0,atol=1e-6):
            return self._stale(now,'public_observation_not_on_real_50ms_sensor_period')
        if self.mode==MotionMode.STALE:self.mode=self._resume_mode
        if evidence is not None:self._apply_evidence(evidence,now,robot)
        elif close_requested and self.mode==MotionMode.FREE:
            self._clear_motion(clear_tracker=False);self.mode=MotionMode.UNRESOLVED
        if self.goal_pose is None:
            try:self.goal_pose,self.goal_metadata=self.goal_marker.observe(observation)
            except (ValueError,RuntimeError):pass
        try:
            obj,diagnostic=self.tracker.observe(observation)
        except (ValueError,RuntimeError) as error:
            return self._stale(now,'public_target_unavailable:'+str(error))
        if (not isinstance(obj,ObjectState) or obj.object_id!='target' or not obj.visible
                or abs(obj.timestamp-now)>1e-8 or obj.confidence<self.minimum_confidence):
            return self._stale(now,'low_confidence_or_invalid_current_public_target')
        if not np.allclose(obj.pose.rotation,np.eye(3),rtol=0,atol=1e-8) or not np.allclose(obj.angular_velocity,0,rtol=0,atol=1e-8):
            raise ValueError('this block bridge declares identity/unobserved rotation and zero angular-velocity prior')
        self._last_object=obj
        if self.mode==MotionMode.UNRESOLVED:
            return self._unavailable(now,'manipulation_unresolved_no_free_body_prediction')
        if self.mode==MotionMode.ATTACHED_PUBLIC:
            if self._attachment is None or now-self._attachment.timestamp>self.maximum_gap:
                return self._stale(now,'public_attachment_relative_pose_stale_requires_fresh_evidence')
            relative=self._attachment.tcp_relative_pose;tcp=robot.tcp_pose
            world=Pose(tcp.apply(relative.position),tcp.rotation@relative.rotation)
            result=self._base(now);result.update(available=True,controlled_mode_available=True,
                controlled=dict(tcp_relative_pose=relative,estimated_world_pose=world,
                    evidence_timestamp=self._attachment.timestamp,evidence_sources=self._attachment.sources,
                    orientation_observable=self._attachment.orientation_observable,
                    caller_asserted_public_evidence_not_Native_verified=True,free_body_GRU_used=False))
            self._last_result=result;return result
        self.forecaster.observe([obj],modes={'target':'external'},orientation_observability={'target':False})
        self.history_frames=min(self.history_frames+1,20)
        if self.history_frames<self.minimum_frames:
            return self._unavailable(now,'fewer_than_five_real_causal_public_frames')
        try:
            forecast=self.forecaster.predict([obj],horizon=.8,dt=.05)
            expected=now+np.arange(17)*.05
            if len(forecast.times)!=17 or not np.allclose(forecast.times,expected,atol=1e-8,rtol=0):
                raise ValueError('frozen Forecast must contain current idx0 and16 real50ms future points')
            times=now+np.arange(1,9)*.1
            samples=[forecast.at('target',timestamp) for timestamp in times]
            result=self._base(now);result.update(available=True,free_forecast_available=True,forecast=forecast,
                endpoint_times=times,positions=np.stack([s[0].position for s in samples]),
                rotations=np.stack([s[0].rotation for s in samples]),linear_velocities=np.stack([s[1] for s in samples]),
                angular_velocities=np.stack([s[2] for s in samples]),
                H8_sampling='Forecast.at(target,t0+.1*k),k=1..8; full17 indices2,4,...16',
                tracker_diagnostic=diagnostic)
            self._last_result=result;return result
        except (ValueError,RuntimeError) as error:
            return self._stale(now,'frozen_forecaster_contract_error:'+str(error))

    def latest(self,timestamp):
        if isinstance(timestamp,bool) or not np.isfinite(timestamp) or timestamp<0:
            raise ValueError('finite nonnegative actual control timestamp required')
        if self.last_timestamp is None:
            return self._unavailable(float(timestamp),'no_current_public_observation')
        if timestamp<self.last_timestamp-1e-8:raise ValueError('cannot query before the latest public observation')
        if timestamp-self.last_timestamp>self.maximum_gap:
            return self._stale(float(timestamp),'latest_public_observation_stale')
        return self._last_result if self._last_result is not None else self._unavailable(float(timestamp),'no_forecast')
