"""Preserve causal motion evidence while clearing public-camera occlusions."""
from .reacquisition import ReacquiringPublicTrackingCoordinator
from .public_motion_v2 import GapAwarePublicReflectionObserver


class GapAwareReacquiringTrackingCoordinator(ReacquiringPublicTrackingCoordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.backend.motion=GapAwarePublicReflectionObserver()
