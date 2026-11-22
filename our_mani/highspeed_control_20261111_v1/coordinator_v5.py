"""Complete-task predictive tracking with the unchanged public/contact sources."""
from .coordinator_v4 import StoppableCompleteTrackingCoordinator
from .predictive_tracking import PredictiveTrackingPlanner,PredictiveTrackingConfig


class PredictiveCompleteTrackingCoordinator(StoppableCompleteTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        extra={'horizon_ticks','position_scale','orientation_scale','relative_speed_weight','orientation_weight','terminal_velocity_zero','optimizer_iterations'}
        super().__init__(*args,tracking_config={k:v for k,v in (tracking_config or {}).items() if k not in extra},**kwargs)
        old=self.reference_planner
        self.reference_planner=PredictiveTrackingPlanner(old.kinematics,old.joint_response,old.limits,config=PredictiveTrackingConfig(**(tracking_config or {})))
