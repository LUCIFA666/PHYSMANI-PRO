import numpy as np
from ..executor_posture_lead_20261113_v15.pose_executor import PoseGoalExecutor as LeadExecutor,WorldPoseGoal
from .retreat import ReleaseRetreat

class PoseGoalExecutor(LeadExecutor):
    def __init__(self,*args,release_retreat=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.release_retreat=ReleaseRetreat()if release_retreat else None

    def _sample_target(self):
        p,q,twist,motion=super()._sample_target()
        if self.release_retreat is not None:
            p,q,twist,receipt=self.release_retreat.sample(p,q,twist,
                np.asarray(self.backend.arm.get_tip().get_pose()),self.backend.clock(),
                held=bool(self.backend.gripper.get_grasped_objects()),
                opening=self.executor.gripper_target,done=self.executor.gripper_done)
            motion=dict(motion,release_retreat=receipt)
        return p,q,twist,motion
