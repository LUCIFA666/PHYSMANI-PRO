"""Control forecast uses two actually observed opposing extrema as evidence."""
from .coordinator_v8 import PublicMotionCompleteTrackingCoordinator
from .public_motion_v3 import EarlyPublicReflectionObserver

class EarlyReflectionTrackingCoordinator(PublicMotionCompleteTrackingCoordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.backend.motion=EarlyPublicReflectionObserver()
