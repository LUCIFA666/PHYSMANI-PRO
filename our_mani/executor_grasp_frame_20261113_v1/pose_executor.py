import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_timed_tasks_20261112_v1.pose_executor import PoseGoalExecutor as TimedExecutor,WorldPoseGoal
from .grasp_frame import capture_delta,apply_delta
from .feedback import desired_velocity
from .reference import step
from .phases import LatchedHeldContact
from .recovery import joint_recovery
from .approach import AlignedApproach,TrackingRecovery

class PoseGoalExecutor(TimedExecutor):
    def __init__(self,*args,compensate_grasp_frame=False,taskspace_objective=True,event_driven_gripper=True,
                 grasp_reference_world_offset=(0.,0.,0.),align_grasp_before_descent=False,latch_held_contact=False,
                 aligned_hover_approach=False,servo_tracking_recovery=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.compensate_grasp_frame=compensate_grasp_frame
        self.taskspace_objective=taskspace_objective
        if event_driven_gripper:self.schedule.delay_ticks=1
        self.grasp_delta=None;self.pending_grasp_delta=None
        self.grasp_reference_world_offset=np.asarray(grasp_reference_world_offset,float)
        self.align_grasp_before_descent=align_grasp_before_descent
        if latch_held_contact:self.held_descent=LatchedHeldContact()
        self.capture_stamp=None;self.capture_pose=None
        self._recovery_target=None;self._recovery_receipt=None
        self.approach=AlignedApproach()if aligned_hover_approach else None
        self.tracking_recovery=TrackingRecovery()if servo_tracking_recovery else None

    def _sample_target(self):
        position,quaternion,twist,motion=super()._sample_target()
        if self.compensate_grasp_frame:
            held=bool(self.backend.gripper.get_grasped_objects())
            if held and self.grasp_delta is None and self.pending_grasp_delta is not None:
                self.grasp_delta=self.pending_grasp_delta.copy()
            if not held and not self.commitment.release_latched:
                self.grasp_delta=None;self.pending_grasp_delta=None
            if self.grasp_delta is not None:
                if self.capture_stamp is not None and abs(self._active_action.timestamp-self.capture_stamp)<1e-9:
                    return self.capture_pose[:3].copy(),self.capture_pose[3:].copy(),np.zeros(6),dict(
                        valid=True,reason='new_grasp_hold_until_next_goal_observation',rigid_grasp_frame_compensation=True)
                position,quaternion,twist=apply_delta(position,quaternion,twist,self.grasp_delta)
                motion=dict(motion,rigid_grasp_frame_compensation=True,
                    correction_source='nominal_grasp_command_and_measured_robot_TCP_at_native_capture',
                    Native_object_pose_used_by_correction=False)
        if self.approach is not None:
            actual=np.asarray(self.backend.arm.get_tip().get_pose(),float)
            position,approach=self.approach.sample(position,quaternion,actual,self.backend.clock(),
                held=bool(self.backend.gripper.get_grasped_objects()))
            motion=dict(motion,robot_approach=approach)
        return position,quaternion,twist,motion

    def _capture_receipt(self):
        if not self.compensate_grasp_frame:return super()._capture_receipt()
        position,quaternion,twist,motion=self._sample_target()
        native=self.backend.capture_window()
        elapsed=self.backend.clock()-self._active_action.timestamp
        fresh_current=(self.capture_lookahead_seconds==0. and abs(elapsed)<self.executor.config.clock_tolerance)
        actual=np.asarray(self.backend.arm.get_tip().get_pose())
        nominal_position=position+self.grasp_reference_world_offset
        position_error=float(np.linalg.norm(actual[:3]-nominal_position))
        angle_error=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
        quality=position_error<=.03 and angle_error<=.5
        allowed=(not native['graspables'])or(native['detected']and(motion['valid']or fresh_current)and quality)
        if allowed and native['graspables']:
            self.pending_grasp_delta=capture_delta(nominal_position,quaternion,actual)
            self.capture_stamp=self._active_action.timestamp;self.capture_pose=actual.copy()
        return dict(allowed=allowed,reason='native_capture_with_recorded_grasp_frame'if allowed else'wait_for_timed_native_capture',
            native=native,motion_compensation=motion,fresh_current_sample_used=fresh_current,
            candidate_nominal_to_actual_TCP=None if self.pending_grasp_delta is None else self.pending_grasp_delta.tolist(),
            grasp_position_error_m=position_error,grasp_rotation_error_rad=angle_error,
            grasp_quality_position_bound_m=.03,grasp_quality_rotation_bound_rad=.5,
            physical_capture_and_held_feedback_required_before_correction=True,
            Native_object_pose_used=False,clock=self.backend.clock())

    def _next(self,state,goal,actual,A,b,*,retained):
        if retained:
            velocity=np.zeros(len(actual))if self._recovery_target is None else 6.*(self._recovery_target-state.position)
            return step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                        speed_norm=self.speed_norm,joint_margin=self.joint_margin)
        recovery=None
        if self.tracking_recovery is not None:
            target,recovery=self.tracking_recovery.update(state.position,actual,self.backend.clock())
            if target is not None:
                next_state,governor=step(state,6.*(target-state.position),self.executor.limits,
                    actual_q=actual,barrier_A=A,barrier_b=b,speed_norm=self.speed_norm,joint_margin=self.joint_margin)
                governor['servo_tracking_recovery']=recovery
                return next_state,governor
        position,quaternion,twist,motion=self._sample_target()
        actual_height=float(self.clearance.fk(actual)[2,3])
        near_contact=self.feedback_mode=='compliant_insert'and self._contact_segment and actual_height-position[2]<.10
        velocity,feedback=desired_velocity(self.clearance.fk,actual,state,position,quaternion,twist,self.executor.limits,
            mode=self.feedback_mode,near_contact=near_contact,joint_avoidance=self.joint_avoidance)
        if (self.align_grasp_before_descent and self._active_action.opening==0.
                and not self.backend.gripper.get_grasped_objects()and feedback['actual_rotation_error_rad']>.3):
            feedback['desired_Cartesian_twist'][2]=max(0.,feedback['desired_Cartesian_twist'][2])
            feedback['grasp_descent_waits_for_orientation']=True
        task=dict(task_jacobian=feedback['geometric_jacobian'],task_velocity=feedback['desired_Cartesian_twist'],
                  null_velocity=feedback['preferred_null_velocity'])if self.taskspace_objective else{}
        next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                               speed_norm=self.speed_norm,joint_margin=self.joint_margin,**task)
        governor.update(Cartesian_feedback=feedback,timed_goal_motion=motion,reference_speed_norm_cap=self.speed_norm,
                        reference_joint_bound_reserve_rad=self.joint_margin)
        if recovery is not None:governor['servo_tracking_recovery']=recovery
        return next_state,governor

    def execute(self,action,**kwargs):
        result=super().execute(action,**kwargs)
        result['grasp_frame_correction']=dict(enabled=self.compensate_grasp_frame,
            committed_nominal_to_actual_TCP=None if self.grasp_delta is None else self.grasp_delta.tolist(),
            Native_object_pose_used=False,taskspace_objective=self.taskspace_objective,
            grasp_reference_world_offset=self.grasp_reference_world_offset.tolist())
        return result

    def advance_retained_goal(self,**kwargs):
        start=self.backend.clock()
        actual=np.asarray(self.backend.get_joint_positions(),float)
        target,diagnostic=joint_recovery(self._state().position,actual,self.executor.limits.lower,self.executor.limits.upper)
        self._recovery_target=target if diagnostic['active']else None
        self._recovery_receipt=diagnostic
        try:
            result=super().advance_retained_goal(**kwargs)
            result['joint_limit_recovery']=diagnostic
            return result
        except RuntimeError as error:
            receipt=getattr(self,'last_dispatch_receipt',None)
            if (not str(error).startswith('Robot body FK disagrees with Native geometry:')
                    or receipt is None or receipt.get('start_clock')!=start):raise
            return dict(receipt,status='hardware_model_diverged',hardware_fault=str(error),
                mechanical_model_qualification_passed=False,dispatcher_remains_faulted=True)
        finally:self._recovery_target=None
