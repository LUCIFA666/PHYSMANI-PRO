from dataclasses import asdict
import numpy as np
from ..executor_contract_repair_20261112_v5.pose_executor import PoseGoalExecutor as PhysicalExecutor,WorldPoseGoal
from .press_cycle import PressState,plan
from .reach import outer_reach

class PoseGoalExecutor(PhysicalExecutor):
    """Opt-in button operation: exact destination plus explicit retract waypoint.

    This is a task-specific execution prior, not an unchanged network action or
    a generic 14-task replacement. Opening=1 cancels the press cycle. Success is
    still sampled by the unchanged native backend after every physical tick.
    """
    def __init__(self,backend,executor,**kwargs):
        if backend.task_name not in ('push_moving_button','push_moving_button_high_speed'):
            raise ValueError('Explicit button operation profile only')
        super().__init__(backend,executor,**kwargs)
        self.press_state=PressState()

    def _decode(self,action):
        bound=outer_reach(self.clearance.snapshot,action.position)
        if not bound['admissible']:
            self.last_clearance_diagnostic=dict(accepted=False,alternative_IK=False,checks=[dict(
                source='outer_robot_reach_precheck',reason='outside_outer_robot_reach',**bound)])
            self.backend.last_ik_receipt=dict(success=False,reason='outside_outer_robot_reach',Native_IK_called=False,physics_steps=0)
            return None
        return super()._decode(action)

    def execute(self,action,**kwargs):
        if type(action)is not WorldPoseGoal:raise TypeError('World pose goal required')
        if abs(action.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future pose goal')
        actual_z=float(self.backend.arm.get_tip().get_position()[2])
        planned,offset=plan(self.press_state,now=self.backend.clock(),opening=action.opening,
                            goal_z=action.position[2],tcp_z=actual_z)
        if offset:
            # A retract waypoint must not make an inadmissible final request
            # appear admissible. Qualify the original destination read-only.
            self.clearance.refresh(self.backend)
            if self._decode(action)is None:
                result=dict(status='IK_failed',executed_ticks=0,execution=[],gripper_intent_issued=False,
                            hardware_clearance=self.last_clearance_diagnostic)
                self._validate_hardware(result)
                return result
        position=np.asarray(action.position).copy();position[2]+=offset
        intermediate=WorldPoseGoal(action.timestamp,position,action.quaternion_xyzw,action.opening)
        result=super().execute(intermediate,**kwargs)
        if result['executed_ticks']:self.press_state=planned
        result['button_press_cycle']=dict(state=asdict(self.press_state),proposed_state=asdict(planned),
            measured_tcp_z=actual_z,retract_waypoint_offset_m=offset,raw_requested_position=list(action.position),
            dispatched_world_position=position.tolist(),task_specific_execution_prior=True,
            target_pose_velocity_contact_or_success_used_to_plan=False,NN_output_modified_into_explicit_waypoint=bool(offset))
        return result

    def advance_retained_goal(self,**kwargs):
        # A newly rejected model destination cancels an unissued contact phase.
        self.press_state=PressState()
        return super().advance_retained_goal(**kwargs)
