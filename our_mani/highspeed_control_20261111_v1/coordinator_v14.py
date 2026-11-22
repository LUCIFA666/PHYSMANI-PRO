"""Public reflection tracking with sequential nonlinear future kinematics."""
from .coordinator_v12 import EarlyReflectionTrackingCoordinator
from .nonlinear_qp import NonlinearHybridTrackingPlanner,HybridTrackingConfig

class NonlinearReflectionTrackingCoordinator(EarlyReflectionTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        super().__init__(*args,tracking_config=tracking_config,**kwargs)
        old=self.reference_planner
        self.reference_planner=NonlinearHybridTrackingPlanner(old.kinematics,old.joint_response,old.limits,config=HybridTrackingConfig(**(tracking_config or {})),workspace_lower=self.helper.config.lower,workspace_upper=self.helper.config.upper)
