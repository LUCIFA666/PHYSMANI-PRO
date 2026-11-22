"""Public interception, actual membership lift/delivery, original Native judge."""
from copy import deepcopy
import numpy as np
from ..local_mpc_control_20261103.public_grasp import qualify_public_grasp_candidates
from ..local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from ..local_mpc_p4_20261103.runtime import ProposalUnavailable
from .candidate_runtime import NativeContactCandidatesV1,NativeContactSymmetricCandidatesV1,NativeContactMatchedCandidatesV1,NativeContactMatchedSymmetricCandidatesV1
from .planner import NativePendingForecastPlannerV1,NativeContactHoldPlannerV1


class NativeContactCoordinatorV1:
    def __init__(self,core,helper,backend,*,pi_symmetry_representative=True):
        self.core,self.helper,self.backend=core,helper,backend
        if backend.candidate_mode=='matched_candidate_with_corrected_forecast':
            cls=NativeContactMatchedSymmetricCandidatesV1 if pi_symmetry_representative else NativeContactMatchedCandidatesV1
        else:cls=NativeContactSymmetricCandidatesV1 if pi_symmetry_representative else NativeContactCandidatesV1
        self.geometry=cls(backend.pilot_inference,backend.candidate_cache)
        self.phase=self.warm=self.candidates=None;self.mode='UNAVAILABLE';self.mode_events=[]
        self.carry_stage=None;self.lift_TCP=None;self.lift_object=None;self.hold_quaternion=None
        self._last_held=False

    def _mode(self,mode,now):
        if mode!=self.mode:self.mode_events.append(dict(time=now,previous=self.mode,mode=mode))
        self.mode=mode

    def _state(self,robot,h,pose,v,w,*,held,goal,mode):
        velocity,angular,valid=self.backend.robot_history.robot_twist
        if not valid:return None
        return dict(time=robot.timestamp,q=robot.q.copy(),dq=robot.dq.copy(),tcp_position=robot.tcp_position.copy(),tcp_quaternion=robot.tcp_quaternion.copy(),
            tcp_velocity=velocity.copy(),tcp_angular_velocity=angular.copy(),finger_open_amounts=robot.finger_open_amounts.copy(),opening=float(robot.finger_open_amounts.mean()),
            object_position=pose.position.copy(),object_quaternion=pose.xyzw()[3:],object_linear_velocity=np.asarray(v).copy(),object_angular_velocity=np.asarray(w).copy(),
            held=held,requested_opening=h['gripper_target'],actuator_state=deepcopy(h),phase_state=deepcopy(self.phase),lift_goal_position=np.asarray(goal).copy(),
            grasp_motion_mode=mode,free_motion_asserted=not held,held_source='actual_Native_gripper_target_membership',
            Native_contact_feedback_used=True,Native_held_used=True,Native_object_state_used=False,Native_success_used=False,
            object_orientation_observable=False)

    def decide(self,robot,h):
        backend=self.backend;now=robot.timestamp;held=backend.latest_contact['held'];result=backend.latest
        if abs(backend.latest_contact['timestamp']-now)>1e-9:raise ValueError('Same-frame actual membership required')
        if backend.goal_pose is None:
            self._mode('UNAVAILABLE',now);return dict(mode=self.mode,action=None,reason='public_goal_not_observed')
        if held:
            carry=backend.latest_carry;pose=carry.get('estimated_pose')
            if pose is None:
                self._mode('NATIVE_HELD_OFFSET_UNKNOWN',now)
                return dict(mode=self.mode,action=None,reason='no_public_visual_or_in_horizon_forecast_carry_anchor',keep_closed=True)
            if not self._last_held:
                self.carry_stage='lift';self.lift_TCP=robot.tcp_position+np.array([0.,0.,.10])
                self.lift_object=pose.position+np.array([0.,0.,.10]);self.hold_quaternion=robot.tcp_quaternion.copy()
                self.phase=self.warm=None
            if self.carry_stage=='lift'and np.linalg.norm(robot.tcp_position-self.lift_TCP)<.015:self.carry_stage='deliver'
            goal=self.lift_object if self.carry_stage=='lift'else backend.goal_pose.position
            self.phase=dict(phase='hold',dwell=0.,lost_hold=0.,updated_at=now,lift_goal_position=goal.copy(),hold_tcp_quaternion=self.hold_quaternion.copy())
            state=self._state(robot,h,pose,carry['linear_velocity'],carry['angular_velocity'],held=True,goal=goal,mode='NATIVE_CARRY')
            self._last_held=True;self._mode('NATIVE_'+self.carry_stage.upper(),now)
            if state is None or self.candidates is None:return dict(mode=self.mode,action=None,state=state,reason='robot_twist_or_original_candidate_pool_pending',keep_closed=True)
            planner=NativeContactHoldPlannerV1(self.core.config,execution_model=self.core.execution_model,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,backend=backend)
            plan=planner.plan(state,deepcopy(self.candidates),warm_start=None)
            action=deepcopy(plan.get('action'))
            if action is not None:
                if action['opening']!=0.:raise ValueError('Native held carry cannot generate an open action')
                action['candidate_id']=plan['selected_candidate_id']
            return dict(mode=self.mode,state=state,plan=plan,action=action,carry=carry,
                qualification=planner.fresh_nominal_qualification,Native_contact_feedback_used=True,visual_attachment_gate_used=False)
        was_held=self._last_held;self._last_held=False
        if h['gripper_target']==0. and(was_held or h['gripper_done']is True):
            self.phase=self.warm=None;self.carry_stage=None;self._mode('NATIVE_CONFIRMED_MISS_OR_DROP_OPEN',now)
            return dict(mode=self.mode,action=dict(joint_goal=robot.q.copy(),opening=1.),state=None,
                direct_membership_miss_open=True,reason='actual_contact_lost'if was_held else'completed_close_actual_membership_false',
                Native_contact_feedback_used=True,visual_proof_failure_used=False)
        obj=result.get('object_observation')
        if not result.get('free_forecast_available')or obj is None:
            self.warm=None;self.phase=None;self._mode('UNAVAILABLE',now)
            return dict(mode=self.mode,action=None,reason='current_public_measurement_or_five_frame_forecast_missing',keep_closed=h['gripper_target']==0.)
        pose,v,w=result['forecast'].at('target',now)
        # Keep real measured position/R; only derivatives use the exact OLS anchor.
        state=self._state(robot,h,obj.pose,v,w,held=False,goal=backend.goal_pose.position,mode='FREE_GRU')
        if state is None:self._mode('UNAVAILABLE',now);return dict(mode=self.mode,action=None,reason='robot_twist_not_yet_observed')
        pending=h['gripper_target']==0.
        if pending:
            state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=now,close_entered_at=now,reopening=False)
        generation=self.geometry.generation
        try:self.candidates=self.geometry.update(state)
        except ProposalUnavailable as error:
            self._mode('UNAVAILABLE',now);return dict(mode=self.mode,action=None,reason=str(error),keep_closed=pending)
        if self.geometry.generation!=generation:self.warm=None
        bound=BoundForecastExecutionModel.from_public_bridge(self.core.execution_model,result)
        candidates,qualification=qualify_public_grasp_candidates(backend,state,deepcopy(self.candidates),self.helper,mode='FREE_GRU',bound=bound)
        planner=NativePendingForecastPlannerV1(self.core.config,execution_model=bound,goal_helper=self.helper,
            constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,backend=backend)
        plan=planner.plan(state,candidates,warm_start=None if pending else self.warm)
        self.phase=deepcopy(plan['phase_state']);self.warm=plan.get('warm_start')
        action=deepcopy(plan.get('action'))
        if action is not None:
            action['candidate_id']=plan['selected_candidate_id']
            if pending and action['opening']!=0.:raise ValueError('Actual unfinished close must not be reopened by predicted visual phase')
        self._mode('NATIVE_PENDING_CLOSE_FOLLOW'if pending else'FREE_GRU',now)
        return dict(mode=self.mode,state=state,plan=plan,action=action,qualification=qualification,
            candidate_pool=deepcopy(self.geometry.last_snapshot),Native_contact_feedback_used=True,visual_attachment_gate_used=False)
