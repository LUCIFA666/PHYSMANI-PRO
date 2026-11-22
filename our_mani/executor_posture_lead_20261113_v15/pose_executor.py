"""Honor the declared control lead in the robot posture planning branch too."""
import numpy as np
from ..executor_drop_alignment_20261113_v14.pose_executor import PoseGoalExecutor as DropExecutor,WorldPoseGoal
from ..executor_grasp_frame_20261113_v1.reference import step
from ..executor_pregrasp_posture_20261113_v12.feedback import desired_velocity
from ..executor_control_lead_20261113_v9.lead import lead_pose

class PoseGoalExecutor(DropExecutor):
    def _next(self,state,goal,actual,A,b,*,retained):
        if retained:return super()._next(state,goal,actual,A,b,retained=True)
        unload=None
        if self.live_unload is not None:
            velocity,unload=self.live_unload.update(state.position,actual,
                self.executor.limits.lower,self.executor.limits.upper,self.backend.clock())
            if velocity is not None:
                next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                    speed_norm=self.speed_norm,joint_margin=self.joint_margin)
                governor['live_reference_unload']=unload
                return next_state,governor
        p_now,q_now,twist_now,motion=self._sample_target()
        p,q,twist=lead_pose(p_now,q_now,twist_now,self.control_lead_seconds)
        actual_height=float(self.clearance.fk(actual)[2,3])
        near_contact=self.feedback_mode=='compliant_insert'and self._contact_segment and actual_height-p_now[2]<.10
        preferred=None;posture_receipt=None
        if (self.posture_prior is not None and(self.posture_activation=='always'or self.backend.gripper.get_grasped_objects())
                and self.executor.gripper_done is not False):
            horizon=(1./self.executor.config.reference_frequency+.025 if self.posture_mode=='joint_goal'else.25)+self.control_lead_seconds
            ahead_p,ahead_q,_=lead_pose(p_now,q_now,twist_now,horizon)
            preferred,posture_receipt=self.posture_prior.update(self.clearance,self.executor.limits,
                state.position,ahead_p,ahead_q,self.backend.clock())
            if preferred is not None and self.posture_mode=='joint_goal':
                velocity=self.executor.config.reference_frequency*(np.asarray(preferred)-state.position)
                next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,
                    barrier_A=A,barrier_b=b,speed_norm=self.speed_norm,joint_margin=self.joint_margin)
                governor.update(robot_posture_prior=posture_receipt,posture_mode=self.posture_mode,
                    current_goal_position=p_now.tolist(),current_goal_quaternion=q_now.tolist(),
                    intercept_horizon_seconds=horizon,control_target_lead_seconds=self.control_lead_seconds,current_goal_motion=motion,
                    joint_target_is_not_measured_state=True)
                if unload is not None:governor['live_reference_unload']=unload
                return next_state,governor
        velocity,feedback=desired_velocity(self.clearance.fk,actual,state,p,q,twist,self.executor.limits,
            mode=self.feedback_mode,near_contact=near_contact,joint_avoidance=self.joint_avoidance,
            alignment_reference=(p_now,q_now),preferred_posture=preferred,posture_gain=self.posture_gain)
        if (self.align_grasp_before_descent and self._active_action.opening==0.
                and not self.backend.gripper.get_grasped_objects()and feedback['current_alignment_rotation_error_rad']>.3):
            feedback['desired_Cartesian_twist'][2]=max(0.,feedback['desired_Cartesian_twist'][2])
            feedback['grasp_descent_waits_for_orientation']=True
        task=dict(task_jacobian=feedback['geometric_jacobian'],task_velocity=feedback['desired_Cartesian_twist'],
            null_velocity=feedback['preferred_null_velocity'])if self.taskspace_objective else{}
        next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
            speed_norm=self.speed_norm,joint_margin=self.joint_margin,**task)
        governor.update(Cartesian_feedback=feedback,timed_goal_motion=motion,
            control_target_lead_seconds=self.control_lead_seconds,
            completion_grasp_and_release_still_evaluated_at_current_time=True)
        if posture_receipt is not None:governor['robot_posture_prior']=posture_receipt
        if unload is not None:governor['live_reference_unload']=unload
        return next_state,governor
