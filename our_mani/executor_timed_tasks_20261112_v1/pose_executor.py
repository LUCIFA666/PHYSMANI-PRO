from ..executor_differential_tracking_20261112_v1.pose_executor import PoseGoalExecutor as DifferentialExecutor,WorldPoseGoal
from .commands import TimedWorldTarget
from .trajectory import sample

class PoseGoalExecutor(DifferentialExecutor):
    def _sample_target(self):
        return sample(list(self._goal_history),self.capture_lookahead_seconds,
                      self.backend.clock()-self._active_action.timestamp)

    def execute(self,action,**kwargs):
        if type(action)is not TimedWorldTarget:raise TypeError('Explicit per-command target timing required')
        if abs(action.pose.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future observation timestamp')
        if abs(action.lookahead_seconds-self.capture_lookahead_seconds)>1e-9:self._goal_history.clear()
        self.capture_lookahead_seconds=action.lookahead_seconds
        result=super().execute(action.pose,**kwargs)
        result.setdefault('timed_world_trajectory',{}).update(timing_source=action.timing_source,
            timestamp_semantics='pose.timestamp_is_observation_time_sample_time_is_timestamp_plus_lookahead',
            per_command_timing_explicit=True)
        return result
