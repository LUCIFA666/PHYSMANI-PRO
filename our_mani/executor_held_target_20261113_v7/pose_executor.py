from ..executor_goal_continuity_20261113_v6.pose_executor import PoseGoalExecutor as SegmentExecutor,WorldPoseGoal
from ..executor_goal_continuity_20261113_v6.continuity import GoalContinuity

class HeldGoalContinuity(GoalContinuity):
    def __init__(self,held):
        super().__init__();self.held=held

    def observe(self,action,segment):
        if not self.held():
            self.history.clear();self.segment=None;self.lookahead=None
            return True,dict(accepted=True,segment_id=segment,reason='unheld_target_reselection_allowed',
                continuity_scope='held_operation_only',synthetic_observations_added=False,task_body_state_used=False)
        accepted,diagnostic=super().observe(action,segment)
        return accepted,dict(diagnostic,continuity_scope='held_operation_only')

class PoseGoalExecutor(SegmentExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.continuity is not None:
            self.continuity=HeldGoalContinuity(lambda:bool(self.backend.gripper.get_grasped_objects()))
