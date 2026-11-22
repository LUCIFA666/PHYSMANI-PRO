"""Public causal motion residual for control; frozen NN inputs stay intact."""
from .coordinator_v7 import HybridCompleteTrackingCoordinator
from .public_motion import PublicReflectionObserver


class PublicMotionPlanningView:
    """Read-only planning view, separate from the authenticated NN row cache."""
    def __init__(self,backend):
        self.backend=backend;self.motion=PublicReflectionObserver()
        self._time=None;self._latest=None;self._was_held=False

    def __getattr__(self,name):return getattr(self.backend,name)

    @property
    def latest(self):
        original=self.backend.latest
        at=original['timestamp']
        if self._time==at:return self._latest
        held=self.backend.latest_contact['held']
        if held or self._was_held:self.motion.reset()
        self._was_held=held
        result=dict(original)
        obj=original.get('object_observation')
        if not held and obj is not None:
            self.motion.observe(at,obj.pose.position)
            if original.get('forecast') is not None:
                forecast,evidence=self.motion.fuse(original['forecast'])
                result.update(forecast=forecast,effective_forecast_source=forecast.metadata['source'],
                    causal_reflection_evidence=evidence,GRU_checkpoint_sha256=original.get('frozen_checkpoint_sha256'),
                    frozen_checkpoint_sha256=None,neural_candidate_cache_modified=False,
                    Native_inputs_used=False,future_observations_used=False)
        self._time,self._latest=at,result
        return result


class PublicMotionCompleteTrackingCoordinator(HybridCompleteTrackingCoordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.backend=PublicMotionPlanningView(self.backend)

    def decide(self,robot,h):
        result=super().decide(robot,h)
        result['control_motion_source']=self.backend.latest.get('effective_forecast_source','frozen_GRU')
        result['causal_reflection_evidence']=self.backend.latest.get('causal_reflection_evidence',[])
        result['neural_candidate_cache_modified']=False
        return result
