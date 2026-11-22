"""Grasp-only public state contracts, separate from frozen public_state.py.

Only the explicitly bound real RedBoxTracker/same-frame cache dependency is
approved. Forecaster and goal injection stay forbidden; the injected tracker
flag is retained, never rewritten. Pending held=False means unknown attachment,
not a free-motion assertion and never authorization to feed GRU history.
"""
from copy import deepcopy
from uuid import uuid4
from pathlib import Path
import hashlib,inspect
import numpy as np

from ..benchmark_perception import RedBoxTracker
from ..goal_perception import RedGoalMarker
from ..contracts import ObjectState,Pose
from .public_motion_bridge import PublicMotionBridge,PublicRobotSensors,PublicMotionEvidence,MultiBodyContextDynamics,CHECKPOINT_SHA256
from .public_attachment import PublicTargetFrameCache
from .public_state import PublicStateAssembler,ACTUATOR_KEYS,_phase_memory


class ApprovedTrackerCacheBinding:
    """A narrow current-frame capability for one real cache and real tracker."""
    def __init__(self,bridge,tracker,cache):
        self.bridge,self.tracker,self.cache=bridge,tracker,cache
        self.binding_id=uuid4().hex;self.latest_result=None;self.latest_visual_target=None
        self.validate_dependencies()
    def validate_dependencies(self):
        if (type(self.bridge) is not PublicMotionBridge or type(self.tracker) is not RedBoxTracker
                or type(self.cache) is not PublicTargetFrameCache or self.bridge.tracker is not self.cache
                or self.cache.public_tracker is not self.tracker):
            raise ValueError('only the real RedBoxTracker and its real same-frame cache may be approved')
        if (self.bridge._mock_forecaster is not False or self.bridge._goal_marker_injected is not False
                or self.bridge._tracker_injected is not True
                or type(self.bridge.forecaster) is not MultiBodyContextDynamics
                or self.bridge.forecaster.checkpoint_sha256!=CHECKPOINT_SHA256
                or type(self.bridge.goal_marker) is not RedGoalMarker):
            raise ValueError('frozen real forecaster and real default goal required; no blanket injection exception')
    def bind_current(self,result,target):
        self.validate_dependencies()
        if not isinstance(result,dict):raise ValueError('actual bridge result required')
        if target is not None and (not isinstance(target,ObjectState) or abs(target.timestamp-float(result['timestamp']))>1e-8):
            raise ValueError('only same-frame public tracker ObjectState may be staged for pending')
        self.latest_result=result;self.latest_visual_target=target
        self.approve(result)
    def approve(self,result):
        self.validate_dependencies()
        if result is not self.latest_result:raise ValueError('only this binding current actual bridge result is accepted')
        if (result.get('tracker_dependency_injected') is not True
                or result.get('forecaster_is_injected_test_double') is not False
                or result.get('goal_marker_dependency_injected') is not False
                or any(result.get(key) is not False for key in ('Native_inputs_used','future_observations_used','observations_fabricated'))):
            raise ValueError('actual injection facts and causal public-only provenance required')
        return True
    def receipt(self):
        paths=[inspect.getfile(cls) for cls in (RedBoxTracker,PublicTargetFrameCache,PublicMotionBridge)]
        return dict(binding_id=self.binding_id,tracker_dependency_injected=True,
            approved_dependency='real_RedBoxTracker_in_real_same_frame_PublicTargetFrameCache_only',
            forecaster_injection_approved=False,goal_injection_approved=False,allow_test_doubles_enabled=False,
            source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths})


class PublicGraspStateAssembler(PublicStateAssembler):
    """Independent strict validation with exactly one cache capability exception.

    Inherits only the unchanged real50ms robot FD observer. No permissive
    allow_test_doubles setting or rewritten result flag is used for assembly.
    """
    def __init__(self,binding):
        super().__init__()
        if type(binding) is not ApprovedTrackerCacheBinding:raise TypeError('explicit approved real tracker-cache binding required')
        self.binding=binding
    def _inputs(self,result,robot,actuator_state,phase_state):
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if self.last_robot is None or abs(robot.timestamp-self.last_robot.timestamp)>1e-9:
            raise ValueError('observe each real robot50ms frame before assembly')
        for key in ('tcp_position','tcp_quaternion','q','dq','finger_open_amounts'):
            if not np.array_equal(getattr(robot,key),getattr(self.last_robot,key)):
                raise ValueError('exact observed robot frame must be used')
        if not isinstance(actuator_state,dict) or set(actuator_state)!=ACTUATOR_KEYS:
            raise ValueError('fresh whitelisted actual executor state required')
        if actuator_state['reference_is_measured_state'] is not False or type(actuator_state['reference_valid']) is not bool:
            raise ValueError('explicit reference validity and reference-not-measurement declaration required')
        if actuator_state['gripper_done'] is not None and not isinstance(actuator_state['gripper_done'],(bool,np.bool_)):
            raise ValueError('actual gripper completion must be bool or uninitialized')
        initialized=any(actuator_state[key] is not None for key in ('reference_positions','reference_velocities','reference_accelerations','target_joints'))
        if initialized:
            clock=actuator_state['reference_clock']
            if clock is None or not np.isfinite(clock) or abs(clock-robot.timestamp)>1e-8:raise ValueError('fresh actual executor reference clock required')
            for key in ('reference_positions','reference_velocities','reference_accelerations'):
                value=np.asarray(actuator_state[key],float)
                if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('finite initialized '+key+'7 required')
            if actuator_state['reference_valid'] or actuator_state['target_joints'] is not None:
                value=np.asarray(actuator_state['target_joints'],float)
                if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('finite actual target_joints7 required')
        elif actuator_state['reference_valid'] or actuator_state['reference_clock'] is not None:
            raise ValueError('uninitialized executor cannot claim a valid/current reference')
        requested=float(actuator_state['gripper_target'])
        if requested not in (0.,1.):raise ValueError('actual known binary gripper command required')
        self.binding.approve(result)
        if abs(float(result['timestamp'])-robot.timestamp)>1e-8:raise ValueError('same-time current public result required')
        phase=_phase_memory(phase_state,robot.timestamp)
        return robot,requested,phase
    def _state(self,result,robot,h,phase,obj,pose,object_v,object_w,*,held,mode,held_source,free_motion_asserted):
        goal=result.get('goal_pose')
        if not isinstance(goal,Pose):return None
        if phase and 'lift_goal_position' in phase and not np.allclose(phase['lift_goal_position'],goal.position,atol=1e-8,rtol=0):
            raise ValueError('phase goal must remain the public marker estimate')
        velocity,omega,_=self.robot_twist
        return dict(time=robot.timestamp,tcp_position=robot.tcp_position.copy(),tcp_quaternion=robot.tcp_quaternion.copy(),
            tcp_velocity=velocity.copy(),tcp_angular_velocity=omega.copy(),q=robot.q.copy(),dq=robot.dq.copy(),
            finger_open_amounts=robot.finger_open_amounts.copy(),opening=float(robot.finger_open_amounts.mean()),
            requested_opening=float(h['gripper_target']),object_position=pose.position.copy(),object_quaternion=pose.xyzw()[3:],
            object_linear_velocity=np.asarray(object_v).copy(),object_angular_velocity=np.asarray(object_w).copy(),held=held,
            held_source=held_source,free_motion_asserted=free_motion_asserted,grasp_motion_mode=mode,
            attachment_status='public_consistency_confirmed' if held else 'unknown' if not free_motion_asserted else 'public_free_estimate',
            lift_goal_position=goal.position.copy(),actuator_state=deepcopy(h),phase_state=phase,
            Native_object_state_used=False,Native_held_used=False,Native_success_used=False,
            object_orientation_observable=bool(result['rotation_observable']),
            tracker_dependency_injected=True,approved_tracker_binding=self.binding.receipt(),
            contact_object_response_qualified=False,information_setting='public_grasp_development_prototype_not_qualified')
    def build(self,result,robot,actuator_state,*,phase_state=None):
        robot,requested,phase=self._inputs(result,robot,actuator_state,phase_state)
        if not self.robot_twist[2] or not result.get('available'):return None
        obj=result.get('object_observation')
        if not isinstance(obj,ObjectState) or not obj.visible or abs(obj.timestamp-robot.timestamp)>1e-8:return None
        if result.get('mode')=='FREE' and result.get('free_forecast_available') is True:
            return self._state(result,robot,actuator_state,phase,obj,obj.pose,obj.linear_velocity,obj.angular_velocity,
                held=False,mode='FREE_GRU',held_source='public_free_estimate_not_Native',free_motion_asserted=True)
        if result.get('mode')=='ATTACHED_PUBLIC' and result.get('controlled_mode_available') is True:
            attachment=result.get('attachment_estimate',{})
            evidence=attachment.get('evidence') if isinstance(attachment,dict) else None
            if (not isinstance(attachment,dict) or attachment.get('mode')!='ATTACHED_PUBLIC' or not isinstance(evidence,PublicMotionEvidence)
                    or evidence.event!='attachment_confirmed' or abs(evidence.timestamp-robot.timestamp)>1e-8
                    or abs(float(attachment.get('timestamp',np.inf))-robot.timestamp)>1e-8):
                raise ValueError('held=True requires this current public attachment-confirmed evidence')
            controlled=result.get('controlled',{});pose=controlled.get('estimated_world_pose')
            if not isinstance(pose,Pose):raise ValueError('current public controlled relative-pose estimate required')
            v,w,_=self.robot_twist
            return self._state(result,robot,actuator_state,phase,obj,pose,v+np.cross(w,pose.position-robot.tcp_position),w,
                held=True,mode='ATTACHED_PUBLIC',held_source='current_public_attachment_consistency_not_Native',free_motion_asserted=False)
        return None
    def build_pending(self,result,robot,actuator_state,*,phase_state=None,reopening=False):
        robot,requested,phase=self._inputs(result,robot,actuator_state,phase_state)
        if not self.robot_twist[2] or result.get('mode') not in ('UNRESOLVED','STALE'):return None
        target=self.binding.latest_visual_target
        if not isinstance(target,ObjectState) or not target.visible or target.confidence<.5 or abs(target.timestamp-robot.timestamp)>1e-8:return None
        if not reopening and requested!=0.:return None
        # This fresh visual CV proxy is explicitly NOT a FREE bridge result.
        # Nothing is fed into the frozen GRU and held remains unknown/False.
        return self._state(result,robot,actuator_state,phase,target,target.pose,target.linear_velocity,target.angular_velocity,
            held=False,mode='REOPENING_UNKNOWN_CONTACT' if reopening else 'PENDING_CLOSE_UNKNOWN_CONTACT',
            held_source='unknown_attachment_held_False_is_not_free_motion_assertion',free_motion_asserted=False)
