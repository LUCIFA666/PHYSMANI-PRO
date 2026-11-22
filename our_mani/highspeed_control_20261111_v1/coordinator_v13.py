"""Numerically reserved MPC and past-public reflection control."""
from .coordinator_v12 import EarlyReflectionTrackingCoordinator
from .hybrid_tracking_v2 import ReservedHybridTrackingPlanner,HybridTrackingConfig

class ReservedReflectionTrackingCoordinator(EarlyReflectionTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        super().__init__(*args,tracking_config=tracking_config,**kwargs)
        old=self.reference_planner
        self.reference_planner=ReservedHybridTrackingPlanner(old.kinematics,old.joint_response,old.limits,config=HybridTrackingConfig(**(tracking_config or {})))
