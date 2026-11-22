import numpy as np
from ..executor_placement_alignment_20261113_v5.pose_executor import PoseGoalExecutor as AlignmentExecutor,WorldPoseGoal
from ..executor_reference_unload_20261113_v4.pose_executor import PoseGoalExecutor as TerminalExecutor
from .continuity import GoalContinuity,endpoint_errors

class PoseGoalExecutor(AlignmentExecutor):
    def __init__(self,*args,target_continuity=False,static_completion_segments=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.continuity=GoalContinuity()if target_continuity else None
        self.static_completion_segments=frozenset(static_completion_segments)
        if any(not isinstance(s,str)or not s for s in self.static_completion_segments):
            raise ValueError('Explicit static segment identifiers required')
        self.active_segment_id=None;self.active_packet=None

    def _continue_active(self,stop_requested):
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        actual=np.asarray(self.backend.get_joint_positions(),float)
        if np.any(actual<self.executor.limits.lower)or np.any(actual>self.executor.limits.upper):
            return dict(status='measured_start_outside_bounds',executed_ticks=0,execution=[],gripper_intent_issued=False)
        self.clearance.refresh(self.backend)
        p,q,twist,motion=self._sample_target()
        admitted,qualification=self._admit(p,q)
        if not admitted:
            return dict(status='hardware_clearance_rejected',executed_ticks=0,execution=[],gripper_intent_issued=False,
                hardware_clearance=qualification)
        result=self._dispatch(self._state().position,opening=self._active_action.opening,stop_requested=stop_requested)
        try:self._validate_hardware(result)
        except RuntimeError as error:
            if not str(error).startswith('Robot body FK disagrees with Native geometry:'):raise
            result.update(status='hardware_model_diverged',hardware_fault=str(error),
                mechanical_model_qualification_passed=False,dispatcher_remains_faulted=True)
        result.update(active_target_observation_time=self._active_action.timestamp,
            original_reference_state_retained=True,goal_history_not_filled_with_predictions=True)
        return result

    def execute(self,action,*,segment_id=None,stop_requested=lambda:False):
        if (self.continuity is not None or self.static_completion_segments)and(not isinstance(segment_id,str)or not segment_id):
            raise ValueError('Phase-aware controller requires explicit segment identifiers')
        if abs(action.pose.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future observation timestamp')
        completion=None
        if (self.active_packet is not None and segment_id!=self.active_segment_id
                and self.active_segment_id in self.static_completion_segments
                and self.backend.gripper.get_grasped_objects()):
            if self.active_packet.lookahead_seconds!=0.:
                raise ValueError('A moving forecast cannot be declared a static completion segment')
            p,q,twist,motion=TerminalExecutor._sample_target(self)
            position,angle=endpoint_errors(self.backend.arm.get_tip().get_pose(),p,q)
            accepted=position<=.003 and angle<=.03
            completion=dict(active_segment_id=self.active_segment_id,proposed_segment_id=segment_id,
                transition_accepted=accepted,position_error_m=position,rotation_error_rad=angle,
                command_completion_is_not_task_success=True)
            if not accepted:
                result=self._continue_active(stop_requested)
                result['static_segment_completion']=completion
                return result
        continuity=None
        if self.continuity is not None:
            accepted,continuity=self.continuity.observe(action,segment_id)
            if not accepted:
                result=self._continue_active(stop_requested)
                result['goal_continuity']=continuity
                return result
        self.active_segment_id=segment_id;self.active_packet=action
        result=super().execute(action,stop_requested=stop_requested)
        if completion is not None:result['static_segment_completion']=completion
        if continuity is not None:result['goal_continuity']=continuity
        return result
