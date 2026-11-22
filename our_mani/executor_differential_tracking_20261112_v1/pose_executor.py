import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_cartesian_tracking_20261112_v1.pose_executor import PoseGoalExecutor as CartesianExecutor,WorldPoseGoal
from ..executor_contract_repair_20261112_v4.tool_pose import housing_clearance
from ..executor_contract_repair_20261112_v6.reach import outer_reach
from ..executor_contract_repair_20261112_v4.reference import ReferenceInfeasible
from ..executor_capture_repair_20261112_v1.capture import qualify
from .feedback import desired_velocity
from .reference import step

class PoseGoalExecutor(CartesianExecutor):
    def __init__(self,*args,native_reference_speed=False,joint_margin=0.,joint_avoidance='centre',**kwargs):
        super().__init__(*args,**kwargs)
        self.speed_norm=float(np.linalg.norm(self.executor.limits.effective_velocity))if native_reference_speed else 3.
        self.joint_margin=joint_margin
        self.joint_avoidance=joint_avoidance

    def _next(self,state,goal,actual,A,b,*,retained):
        if retained:
            return step(state,np.zeros(len(actual)),self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                        speed_norm=self.speed_norm,joint_margin=self.joint_margin)
        p,q,twist,motion=self._sample_target()
        actual_height=float(self.clearance.fk(actual)[2,3])
        near_contact=self.feedback_mode=='compliant_insert'and self._contact_segment and actual_height-p[2]<.10
        velocity,feedback=desired_velocity(self.clearance.fk,actual,state,p,q,twist,self.executor.limits,
                                          mode=self.feedback_mode,near_contact=near_contact,joint_avoidance=self.joint_avoidance)
        next_state,governor=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                               speed_norm=self.speed_norm,joint_margin=self.joint_margin)
        governor.update(Cartesian_feedback=feedback,timed_goal_motion=motion,
                        reference_speed_norm_cap=self.speed_norm,reference_joint_bound_reserve_rad=self.joint_margin,
                        joint_avoidance='nullspace_reciprocal_distance_to_native_limits')
        return next_state,governor

    def _capture_receipt(self):
        # Capture is judged at the current physical tick, including the
        # second 50 ms tick after a model/goal query, not its old timestamp.
        position,quaternion,twist,motion=self._sample_target()
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        error=float(np.linalg.norm(measured[:3]-position))
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
        stationary=np.linalg.norm(twist[:3])<.03 and np.linalg.norm(twist[3:])<.1
        native=self.backend.capture_window()
        allowed,reason=qualify(graspables=native['graspables'],detected=native['detected'],stationary=stationary,
                              position_error=error,rotation_error=angle,require_moving_precision=True)
        if native['graspables']and not motion['valid']:allowed=False;reason='wait_for_consistent_timed_goal_history'
        return dict(native=native,allowed=allowed,reason=reason,motion_compensation=motion,
            measured_TCP_position_error_m=error,measured_TCP_rotation_error_rad=angle,
            current_physical_tick_time=self.backend.clock(),goal_observation_time=self._active_action.timestamp)

    def _release_receipt(self):
        position,quaternion,twist,motion=self._sample_target()
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        error=float(np.linalg.norm(measured[:3]-position))
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
        allowed=motion['valid']and error<=self.release_position_tolerance and angle<=self.release_angle_tolerance
        return dict(allowed=allowed,reason='precise_release_pose'if allowed else'wait_for_precise_release_pose',
            measured_TCP_position_error_m=error,measured_TCP_rotation_error_rad=angle,motion_compensation=motion,
            current_physical_tick_time=self.backend.clock(),goal_observation_time=self._active_action.timestamp)

    def _admit(self,position,quaternion):
        bound=outer_reach(self.clearance.snapshot,position)
        if not bound['admissible']:return False,dict(reason='outside_outer_robot_reach',**bound)
        housing=next(b for b in self.clearance.bodies if b['name']=='Panda_gripper')
        index=self.clearance.bodies.index(housing)
        tool=housing_clearance(position,quaternion,tcp_reference=self.clearance.fk.reference_tcp,
            housing_reference=housing['reference_matrix'],vertices=housing['vertices'],
            table_matrix=self.clearance.table_matrix,table_bounds=self.clearance.bounds,
            margin=self.clearance.body_margins[index])
        if not tool['admissible']:return False,dict(reason='hardware_table_intersection',**tool)
        state=self._state();actual=np.asarray(self.backend.get_joint_positions(),float)
        A,b,_=self._barriers(state,actual)
        try:first,governor=self._next(state,state.position,actual,A,b,retained=False)
        except ReferenceInfeasible as error:return False,dict(reason='first_differential_reference_infeasible',error=str(error))
        if not self.clearance.safe(first.position)[0]:
            return False,dict(reason='first_differential_reference_table_intersection')
        return True,dict(reason='exact_next_differential_reference_qualified',
            endpoint_IK_certificate=False,Native_IK_called=False,physical_steps=0,
            global_pose_reachability_claimed=False,governor=governor)

    def execute(self,action,*,stop_requested=lambda:False):
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        if type(action)is not WorldPoseGoal:raise TypeError('Explicit world target sample required')
        if abs(action.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future observation timestamp')
        if self.backend.terminal()or self.backend.remaining_ticks()==0:
            return dict(status='terminal_or_budget',executed_ticks=0,execution=[],gripper_intent_issued=False)
        actual=np.asarray(self.backend.get_joint_positions(),float)
        if np.any(actual<self.executor.limits.lower)or np.any(actual>self.executor.limits.upper):
            return dict(status='measured_start_outside_bounds',executed_ticks=0,execution=[],gripper_intent_issued=False)
        self._active_action=action
        self._contact_segment=self.held_descent.observe(held=bool(self.backend.gripper.get_grasped_objects()),goal_z=action.position[2])
        row=(action.timestamp,np.asarray(action.position),np.asarray(action.quaternion_xyzw))
        if self._goal_history and abs(self._goal_history[-1][0]-action.timestamp)<1e-9:self._goal_history[-1]=row
        else:self._goal_history.append(row)
        self.clearance.refresh(self.backend)
        position,quaternion,twist,motion=self._sample_target()
        admitted,qualification=self._admit(position,quaternion)
        self.last_clearance_diagnostic=dict(accepted=admitted,checks=[qualification],Native_IK_called=False)
        if not admitted:
            return dict(status='hardware_clearance_rejected',executed_ticks=0,execution=[],gripper_intent_issued=False,
                hardware_clearance=self.last_clearance_diagnostic)
        # There is no joint destination: the next constrained reference is
        # generated directly from the timed Cartesian target. The state value
        # only fills the inherited dispatch receipt/braking API.
        result=self._dispatch(self._state().position,opening=action.opening,stop_requested=stop_requested)
        try:self._validate_hardware(result)
        except RuntimeError as error:
            if not str(error).startswith('Robot body FK disagrees with Native geometry:'):raise
            result.update(status='hardware_model_diverged',hardware_fault=str(error),
                          mechanical_model_qualification_passed=False,dispatcher_remains_faulted=True)
        result.update(hardware_clearance=self.last_clearance_diagnostic,joint_goal=None,
            Cartesian_endpoint_IK_required=False,global_pose_reachability_claimed=False,
            timed_world_trajectory=dict(raw_future_position=list(action.position),
                raw_future_quaternion=list(action.quaternion_xyzw),declared_lookahead_seconds=self.capture_lookahead_seconds,
                derived_current_position=position.tolist(),derived_current_quaternion=quaternion.tolist(),motion=motion,
                commanded_held_descent_segment=self._contact_segment,no_task_dynamic_state_read_by_controller=True))
        return result
