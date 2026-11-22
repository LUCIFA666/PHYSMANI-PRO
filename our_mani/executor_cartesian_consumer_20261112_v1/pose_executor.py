import numpy as np
from ..executor_cartesian_tracking_20261112_v1.pose_executor import PoseGoalExecutor as CartesianExecutor,WorldPoseGoal

class PoseGoalExecutor(CartesianExecutor):
    def _sample_target(self):
        action=self._active_action
        held=bool(self.backend.gripper.get_grasped_objects())
        if action.opening==1. and not held and not self.commitment.release_latched:
            # Reviewed basketball wp0 is a current approach waypoint; its
            # grasp/place branches use a 0.5 s prediction. This is an explicit
            # operation prior outside the unchanged neural network.
            return np.asarray(action.position),np.asarray(action.quaternion_xyzw),np.zeros(6),dict(
                valid=False,reason='current_approach_destination_without_future_retiming',
                declared_lookahead_seconds=0.,task_specific_operation_prior=True,
                Native_held_feedback_outside_NN=True,Native_body_pose_read=False)
        return super()._sample_target()

    def execute(self,action,**kwargs):
        result=super().execute(action,**kwargs)
        motion=result.get('timed_world_trajectory',{}).get('motion',{})
        result.setdefault('timed_world_trajectory',{}).update(
            effective_lookahead_seconds=motion.get('declared_lookahead_seconds',self.capture_lookahead_seconds),
            phase_rule='open_unheld_approach_destination_else_half_second_target_sample',
            task_specific_operation_prior=True,neural_time_prediction_claimed=False)
        return result
