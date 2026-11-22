from ..executor_reachable_approach_20261113_v21.pose_executor import PoseGoalExecutor as GuidedExecutor,WorldPoseGoal
from .posture import ContinuousRobotPosturePrior

class PoseGoalExecutor(GuidedExecutor):
    def __init__(self,*args,continuous_posture_ik=False,**kwargs):
        super().__init__(*args,**kwargs)
        if continuous_posture_ik:
            if self.posture_prior is None:raise ValueError('Explicit robot posture planning required')
            self.posture_prior=ContinuousRobotPosturePrior(self.posture_prior.bank,period=self.posture_prior.period)
