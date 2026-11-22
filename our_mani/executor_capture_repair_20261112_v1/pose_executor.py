from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_task_goals_20261112_v1.pose_executor import PoseGoalExecutor as PhysicalExecutor,WorldPoseGoal
from ..executor_contract_repair_20261112_v4.reference import step,ReferenceInfeasible
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .capture import qualify
from .precision import refine
from .goal_motion import current_pose
from .gripper_commitment import GripperCommitment

class PoseGoalExecutor(PhysicalExecutor):
    def __init__(self,*args,refine_ik=False,capture_lookahead_seconds=None,commit_release=False,
                 release_position_tolerance=.002,release_angle_tolerance=.015,**kwargs):
        super().__init__(*args,**kwargs)
        self.refine_ik=refine_ik
        self.capture_lookahead_seconds=capture_lookahead_seconds
        self.commit_release=commit_release
        self.commitment=GripperCommitment()
        self.release_position_tolerance=release_position_tolerance
        self.release_angle_tolerance=release_angle_tolerance
        self.last_precision_diagnostic=None
        self._goal_history=deque(maxlen=5)
        self._active_action=None

    def _decode(self,action):
        goal=super()._decode(action)
        self.last_precision_diagnostic=None
        if goal is None or not self.refine_ik:return goal
        refined,receipt=refine(self.clearance.fk,goal,action.position,action.quaternion_xyzw,
                               self.executor.limits.lower,self.executor.limits.upper)
        qualified,qualification=self._qualify(refined,action)
        receipt.update(qualified=qualified,qualification=qualification)
        self.last_precision_diagnostic=receipt
        return refined if qualified else goal

    def execute(self,action,**kwargs):
        self._active_action=action
        result=super().execute(action,**kwargs)
        result['IK_precision_refinement']=self.last_precision_diagnostic
        if result['executed_ticks']:
            self._goal_history.append((action.timestamp,np.asarray(action.position),np.asarray(action.quaternion_xyzw)))
        return result

    def _capture_receipt(self):
        action=self._active_action
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        error=float(np.linalg.norm(measured[:3]-action.position))
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(action.quaternion_xyzw)).magnitude())
        history=list(self._goal_history)+[(action.timestamp,np.asarray(action.position),np.asarray(action.quaternion_xyzw))]
        speed=0.;angular_speed=0.
        if len(history)>1 and history[-1][0]-history[0][0]>1e-6:
            duration=history[-1][0]-history[0][0]
            speed=float(np.linalg.norm(history[-1][1]-history[0][1])/duration)
            angular_speed=float((Rotation.from_quat(history[0][2]).inv()*Rotation.from_quat(history[-1][2])).magnitude()/duration)
        stationary=speed<.03 and angular_speed<.1
        native=self.backend.capture_window()
        motion=None
        if self.capture_lookahead_seconds is not None:
            capture_position,capture_quaternion,motion=current_pose(history,self.capture_lookahead_seconds)
            error=float(np.linalg.norm(measured[:3]-capture_position))
            angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(capture_quaternion)).magnitude())
            if motion['valid']:
                stationary=motion['goal_speed_m_s']<.03 and motion['goal_angular_speed_rad_s']<.1
        allowed,reason=qualify(graspables=native['graspables'],detected=native['detected'],stationary=stationary,
                              position_error=error,rotation_error=angle,
                              require_moving_precision=self.capture_lookahead_seconds is not None)
        if native['graspables']and motion is not None and not motion['valid']:
            allowed=False;reason='wait_for_consistent_timed_goal_history'
        return dict(native= native,allowed=allowed,reason=reason,goal_stationary_estimate=stationary,
            motion_compensation=motion,
            goal_speed_m_s=speed,goal_angular_speed_rad_s=angular_speed,
            measured_TCP_position_error_m=error,measured_TCP_rotation_error_rad=angle)

    def _release_receipt(self):
        action=self._active_action
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        history=list(self._goal_history)+[(action.timestamp,np.asarray(action.position),np.asarray(action.quaternion_xyzw))]
        if self.capture_lookahead_seconds is None:
            position=np.asarray(action.position);quaternion=action.quaternion_xyzw;motion=dict(valid=True)
        else:
            position,quaternion,motion=current_pose(history,self.capture_lookahead_seconds)
        error=float(np.linalg.norm(measured[:3]-position))
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
        allowed=motion['valid']and error<=self.release_position_tolerance and angle<=self.release_angle_tolerance
        return dict(allowed=allowed,reason='precise_release_pose'if allowed else'wait_for_precise_release_pose',
            measured_TCP_position_error_m=error,measured_TCP_rotation_error_rad=angle,
            motion_compensation=motion,task_object_pose_read=False,Native_held_feedback_used=True)

    def _dispatch(self,goal,*,opening=None,stop_requested=lambda:False,retained=False):
        receipts=[];issued=False;start=float(self.backend.clock());safety=None;safety_diagnostic=None
        if retained:
            self.schedule.cancel()
            self.commitment.brake(actual_target=self.executor.gripper_target,done=self.executor.gripper_done)
        elif opening is not None:
            selected=opening
            if self.commit_release:
                selected=self.commitment.resolve(opening,actual_target=self.executor.gripper_target,
                    done=self.executor.gripper_done,held=bool(self.backend.gripper.get_grasped_objects()))
            if selected is not None:self.schedule.request(selected,self.executor.gripper_target,self.backend._ticks)
        for _ in range(self.control_ticks):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            self.clearance.refresh(self.backend)
            actual=np.asarray(self.backend.get_joint_positions(),float)
            if (np.all(actual>=self.executor.limits.lower)and np.all(actual<=self.executor.limits.upper)
                    and self.clearance.safe(actual)[0]):self._last_safe_actual=actual.copy()
            state=self._state();A,b,barriers=self._barriers(state,actual)
            try:
                next_ref,governor=step(state,goal,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                                     frequency=self.executor.config.reference_frequency)
                if not self.clearance.safe(next_ref.position)[0]:
                    raise ReferenceInfeasible('Next governed reference violates exact static hardware clearance')
            except ReferenceInfeasible as error:
                safety=str(error);safety_diagnostic=getattr(error,'diagnostic',None);self.schedule.cancel();break
            op=None if retained else self.schedule.due(self.backend._ticks)
            if self.commit_release and op is not None and op!=self.executor.gripper_target and self.executor.gripper_done is False:
                op=None
            capture=None
            if op==0.:
                capture=self._capture_receipt()
                if not capture['allowed']:op=None
            elif op==1. and self.backend.gripper.get_grasped_objects():
                capture=self._release_receipt()
                if not capture['allowed']:op=None
            command=JointTrajectory(start_time=self.backend.clock(),positions=[next_ref.position],times_from_start=[.05],opening=op)
            receipt=self.executor.execute_command(command,phase='feedback_brake_and_observe'if retained else'feedback_governed_pose',
                planner_contract=dict(reference_source='coupled_measured_feedback_governor',
                    target_and_measured_state_unmodified=True,governor=governor,barriers=barriers,
                    barrier_acceleration_matrix=A.tolist(),barrier_lower_bound=b.tolist(),
                    gripper_schedule_uses_physical_ticks=True,capture_gate=capture,
                    release_event_latched=self.commitment.release_latched if self.commit_release else None))
            receipts.append(receipt.to_dict())
            if receipt.executed_ticks and op is not None:self.schedule.acknowledge(op);issued=True
            if not receipt.executed_ticks:break
        count=sum(r['executed_ticks']for r in receipts)
        result=dict(status=('retained_goal_chunk_complete'if retained else'chunk_complete')if count==self.control_ticks else
                    'control_barrier_infeasible'if safety else'truncated',safety_stop=safety,safety_diagnostic=safety_diagnostic,
            requested_ticks=self.control_ticks,executed_ticks=count,start_clock=start,end_clock=float(self.backend.clock()),execution=receipts,
            gripper_intent_issued=issued,scheduled_gripper_delay_ticks=10,gripper_pending=self.schedule.pending,
            joint_goal=np.asarray(goal).tolist(),gripper_done=self.executor.gripper_done,
            actual_persistent_opening=self.executor.gripper_target,whole_RML_path_waited=False,measured_arrival_claimed=False,
            grasp_or_task_success_claimed=False,reference_mode='coupled_actual_feedback_discrete_joint_references',
            reference_reseeded_on_target_change=False,reference_difference_limits_checked=True,
            exact_unmodified_author_executor=False,all_physics_and_task_steps_counted=True)
        self.last_dispatch_receipt=result
        return result
