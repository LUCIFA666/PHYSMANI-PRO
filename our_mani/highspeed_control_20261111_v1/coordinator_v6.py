"""Complete-task QP MPC, with unchanged public perception and Native judging."""
from .coordinator_v5 import PredictiveCompleteTrackingCoordinator
from .qp_tracking import QPTrackingPlanner,PredictiveTrackingConfig


class QPCompleteTrackingCoordinator(PredictiveCompleteTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        super().__init__(*args,tracking_config=tracking_config,**kwargs)
        old=self.reference_planner
        self.reference_planner=QPTrackingPlanner(old.kinematics,old.joint_response,old.limits,config=PredictiveTrackingConfig(**(tracking_config or {})))
