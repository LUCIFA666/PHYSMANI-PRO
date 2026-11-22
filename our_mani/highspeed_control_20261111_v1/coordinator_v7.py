"""Stage global approach, local MPC, and actual-contact carry explicitly."""
from .coordinator_v6 import QPCompleteTrackingCoordinator
from .hybrid_tracking import HybridTrackingPlanner,HybridTrackingConfig


class HybridCompleteTrackingCoordinator(QPCompleteTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        extra={'local_mpc_radius','execution_pregrasp_offset'}
        super().__init__(*args,tracking_config={k:v for k,v in (tracking_config or {}).items() if k not in extra},**kwargs)
        old=self.reference_planner
        self.reference_planner=HybridTrackingPlanner(old.kinematics,old.joint_response,old.limits,config=HybridTrackingConfig(**(tracking_config or {})))
