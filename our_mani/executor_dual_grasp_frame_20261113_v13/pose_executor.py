import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_pregrasp_posture_20261113_v12.pose_executor import PoseGoalExecutor as PostureExecutor,WorldPoseGoal
from ..executor_timed_tasks_20261112_v1.pose_executor import PoseGoalExecutor as RawTimedExecutor
from ..executor_grasp_frame_20261113_v1.grasp_frame import capture_delta,apply_delta

def grasp_frames(contact,quaternion,offset,actual):
    contact=np.asarray(contact,float)
    return capture_delta(contact,quaternion,actual),capture_delta(contact+np.asarray(offset),quaternion,actual)

class PoseGoalExecutor(PostureExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contact_delta=None;self.pending_contact_delta=None
        self.captured_segment_id=None;self.captured_action_time=None

    def _capture_receipt(self):
        if not self.compensate_grasp_frame:return super()._capture_receipt()
        # A raised approach waypoint is not the requested contact pose.
        p,q,twist,motion=RawTimedExecutor._sample_target(self)
        native=self.backend.capture_window();actual=np.asarray(self.backend.arm.get_tip().get_pose())
        error=float(np.linalg.norm(actual[:3]-p))
        angle=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(q)).magnitude())
        fresh=(self.capture_lookahead_seconds==0. and abs(self.backend.clock()-self._active_action.timestamp)<self.executor.config.clock_tolerance)
        pt,rt=self.capture_tolerances
        if pt is None:pt,rt=.03,.5
        allowed=(not native['graspables'])or(native['detected']and(motion['valid']or fresh)and error<=pt and angle<=rt)
        if allowed and native['graspables']:
            self.pending_contact_delta,self.pending_grasp_delta=grasp_frames(p,q,self.grasp_reference_world_offset,actual)
            self.capture_stamp=self._active_action.timestamp;self.capture_pose=actual.copy()
            self.captured_action_time=self._active_action.timestamp;self.captured_segment_id=self.active_segment_id
        return dict(allowed=allowed,reason='terminal_contact_pose'if allowed else'wait_for_terminal_contact_pose',
            native=native,motion_compensation=motion,grasp_position_error_m=error,grasp_rotation_error_rad=angle,
            explicit_capture_position_tolerance_m=pt,explicit_capture_rotation_tolerance_rad=rt,
            intermediate_waypoint_does_not_qualify_capture=True,
            contact_and_transport_reference_separate=True,Native_object_pose_used=False)

    def _sample_target(self):
        if not self.compensate_grasp_frame:return super()._sample_target()
        p,q,twist,motion=RawTimedExecutor._sample_target(self)
        held=bool(self.backend.gripper.get_grasped_objects())
        if held and self.grasp_delta is None and self.pending_grasp_delta is not None:
            self.grasp_delta=self.pending_grasp_delta.copy();self.contact_delta=self.pending_contact_delta.copy()
        if not held and not self.commitment.release_latched:
            self.grasp_delta=None;self.pending_grasp_delta=None;self.contact_delta=None;self.pending_contact_delta=None
        if self.grasp_delta is not None:
            same_contact_stage=(held and((self.captured_segment_id is not None
                and self.active_segment_id==self.captured_segment_id)or(
                self.captured_segment_id is None and self._active_action.timestamp==self.captured_action_time)))
            delta=self.contact_delta if same_contact_stage else self.grasp_delta
            p,q,twist=apply_delta(p,q,twist,delta)
            motion=dict(motion,rigid_grasp_frame_compensation=True,
                applied_grasp_reference='contact'if same_contact_stage else'nominal_transport',
                deliberate_contact_stroke_not_applied_twice=True,Native_object_pose_used=False)
        if self.approach is not None:
            p,diagnostic=self.approach.sample(p,q,np.asarray(self.backend.arm.get_tip().get_pose()),
                self.backend.clock(),held=held)
            motion=dict(motion,robot_approach=diagnostic)
        if self.placement_alignment is not None:
            p,twist,diagnostic=self.placement_alignment.sample(p,q,twist,
                np.asarray(self.backend.arm.get_tip().get_pose()),self.backend.clock(),
                held=held,raw_goal_z=self._active_action.position[2])
            motion=dict(motion,held_placement_alignment=diagnostic)
        return p,q,twist,motion
