import numpy as np
from ..executor_dual_grasp_frame_20261113_v13.pose_executor import PoseGoalExecutor as ContactExecutor,WorldPoseGoal
from .approach import GuidedApproach

class PoseGoalExecutor(ContactExecutor):
    def __init__(self,*args,guided_approach=False,guided_descent_speed=.08,**kwargs):
        super().__init__(*args,**kwargs)
        self.guided_approach=guided_approach
        if guided_approach:
            if self.approach is None:raise ValueError('Explicit aligned approach profile required')
            self.approach=GuidedApproach(speed=guided_descent_speed)

    def _sample_target(self):
        p,q,twist,motion=super()._sample_target()
        if self.guided_approach and not self.backend.gripper.get_grasped_objects():
            twist=np.asarray(twist,float).copy();rate=abs(self.approach.vertical_velocity)
            # The known Cartesian segment ends at its target. Predicting a
            # full-rate descent beyond that endpoint would overshoot contact.
            horizon=1./self.executor.config.reference_frequency+.025
            twist[2]=np.clip((self.approach.active_target_z-p[2])/horizon,-rate,rate)
            motion=dict(motion,guided_approach_velocity_used=True,guided_prediction_bounded_by_segment_endpoint=True)
        return p,q,twist,motion
