"""Tracking controller with a position-input-calibrated physical response."""
from pathlib import Path
from .coordinator import TrackingCoordinator
from .tracking import TrackingReferencePlanner,TrackingConfig
from .direct_response import DirectPositionResponse


class PositionResponseTrackingCoordinator(TrackingCoordinator):
    def __init__(self,*args,tracking_config=None,response_parameters,response_sha256,**kwargs):
        super().__init__(*args,tracking_config=tracking_config,**kwargs)
        root=Path(__file__).resolve().parents[2]
        response=DirectPositionResponse(root/response_parameters,response_sha256)
        self.reference_planner=TrackingReferencePlanner(self.helper.kinematics,response,self.helper.limits,
            TrackingConfig(**(tracking_config or {})))
