"""Opt-in timed-world-trajectory interface; no original physics/source changes."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_capture_repair_20261112_v1.pose_executor import PoseGoalExecutor as CaptureExecutor,WorldPoseGoal
from ..executor_task_goals_20261112_v1.pose_executor import PoseGoalExecutor as PhysicalExecutor
from ..executor_contract_repair_20261112_v4.reference import step as brake_step,ReferenceInfeasible
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .reference import step
from .feedback import desired_velocity
from .trajectory import sample
from .phases import HeldDescent

class PoseGoalExecutor(CaptureExecutor):
    def __init__(self,*args,capture_lookahead_seconds=None,feedback_mode='actual',**kwargs):
        if capture_lookahead_seconds is None or capture_lookahead_seconds<.2:
            raise ValueError('Explicit timed-goal lookahead required; no silent time inference')
        super().__init__(*args,capture_lookahead_seconds=capture_lookahead_seconds,**kwargs)
        if feedback_mode not in ('actual','bounded_bias','compliant_insert'):raise ValueError('Explicit feedback mode required')
        self.feedback_mode=feedback_mode
        self.held_descent=HeldDescent()
        self._contact_segment=False
        self._motion_receipt=None

    def _sample_target(self):
        return sample(list(self._goal_history),self.capture_lookahead_seconds,
                      self.backend.clock()-self._active_action.timestamp)

    def _next(self,state,goal,actual,A,b,*,retained):
        if retained:
            return brake_step(state,goal,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                              frequency=self.executor.config.reference_frequency)
        p,q,twist,motion=self._sample_target()
        actual_height=float(self.clearance.fk(actual)[2,3])
        near_contact=(self.feedback_mode=='compliant_insert'and self._contact_segment and actual_height-p[2]<.10)
        velocity,feedback=desired_velocity(self.clearance.fk,actual,state,p,q,twist,self.executor.limits,
                                          mode=self.feedback_mode,near_contact=near_contact)
        next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b)
        governor.update(Cartesian_feedback=feedback,timed_goal_motion=motion,
                        endpoint_IK_branch_not_used_as_reference_direction=True)
        return next_state,governor

    def _qualify(self,goal,action):
        if goal is None:return False,dict(reason='IK_no_solution')
        goal=np.asarray(goal,float)
        try:self.executor._validate_positions(goal)
        except ValueError as error:return False,dict(reason='IK_solution_outside_native_contract',error=str(error))
        transform=self.clearance.fk(goal)
        position=float(np.linalg.norm(transform[:3,3]-action.position))
        angle=float((Rotation.from_matrix(transform[:3,:3]).inv()*Rotation.from_quat(action.quaternion_xyzw)).magnitude())
        if position>.002 or angle>.02:return False,dict(reason='IK_pose_residual',position_m=position,rotation_rad=angle)
        depths=self.clearance.depths(goal)[0]
        if np.any(depths>1e-7):
            body=int(np.argmax(depths));return False,dict(reason='hardware_table_intersection',
                body=self.clearance.bodies[body]['name'],reference_index=0,planning_margin_violation_m=float(depths[body]))
        state=self._state();actual=np.asarray(self.backend.get_joint_positions(),float)
        A,b,_=self._barriers(state,actual)
        try:first,_=self._next(state,goal,actual,A,b,retained=False)
        except ReferenceInfeasible:return False,dict(reason='first_Cartesian_reference_infeasible')
        if not self.clearance.safe(first.position)[0]:return False,dict(reason='first_Cartesian_reference_table_intersection')
        return True,dict(reason='current_timed_target_endpoint_and_first_actual_feedback_reference_clear',
                         position_residual_m=position,rotation_residual_rad=angle)

    def execute(self,action,**kwargs):
        if type(action)is not WorldPoseGoal:raise TypeError('Explicit world goal required')
        if abs(action.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future observation timestamp')
        self._active_action=action
        self._contact_segment=self.held_descent.observe(held=bool(self.backend.gripper.get_grasped_objects()),goal_z=action.position[2])
        row=(action.timestamp,np.asarray(action.position),np.asarray(action.quaternion_xyzw))
        if self._goal_history and abs(self._goal_history[-1][0]-action.timestamp)<1e-9:self._goal_history[-1]=row
        else:self._goal_history.append(row)
        p,q,twist,motion=self._sample_target()
        # The caller explicitly declares future target samples, as distinct
        # from a destination-only WorldPoseGoal. Current-trajectory admission
        # and actual Cartesian tracking are derived from those samples.
        current=WorldPoseGoal(action.timestamp,p,q,action.opening)
        start=self.backend.clock()
        try:result=PhysicalExecutor.execute(self,current,**kwargs)
        except RuntimeError as error:
            # Preserve every physically executed sample when post-step model
            # qualification fails. The dispatcher remains faulted; no replay.
            receipt=getattr(self,'last_dispatch_receipt',None)
            if (not str(error).startswith('Robot body FK disagrees with Native geometry:')
                    or receipt is None or receipt.get('start_clock')!=start):raise
            result=dict(receipt,status='hardware_model_diverged',hardware_fault=str(error),
                        mechanical_model_qualification_passed=False,dispatcher_remains_faulted=True)
        result['timed_world_trajectory']=dict(raw_future_position=list(action.position),
            raw_future_quaternion=list(action.quaternion_xyzw),declared_lookahead_seconds=self.capture_lookahead_seconds,
            derived_current_position=p.tolist(),derived_current_quaternion=q.tolist(),motion=motion,
            observation_timestamp=action.timestamp,IK_is_current_endpoint_admission_only=True,
            commanded_held_descent_segment=self._contact_segment,
            no_task_dynamic_state_read_by_controller=True,goal_history_scope='received_timed_goals_including_rejected_observations')
        return result

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
                next_ref,governor=self._next(state,goal,actual,A,b,retained=retained)
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
                planner_contract=dict(reference_source='actual_Cartesian_feedback_timed_world_trajectory',
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
