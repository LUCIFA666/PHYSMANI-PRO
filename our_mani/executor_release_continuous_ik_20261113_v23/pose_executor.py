from ..executor_release_settle_20261113_v18.pose_executor import PoseGoalExecutor as ReleaseExecutor,WorldPoseGoal
from ..executor_continuous_posture_20261113_v22.posture import ContinuousRobotPosturePrior

class PoseGoalExecutor(ReleaseExecutor):
    def __init__(self,*args,continuous_posture_ik=False,**kwargs):
        super().__init__(*args,**kwargs)
        if continuous_posture_ik:
            if self.posture_prior is None:raise ValueError('Explicit robot posture planning required')
            self.posture_prior=ContinuousRobotPosturePrior(self.posture_prior.bank,period=self.posture_prior.period)
