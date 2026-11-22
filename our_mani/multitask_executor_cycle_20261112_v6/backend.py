from ..multitask_executor_cycle_20261112_v4.backend import TaskExecutionBackend as PreviousBackend

class TaskExecutionBackend(PreviousBackend):
    def __init__(self,*args,**kwargs):
        self._public_cache=None
        self.on_public_frame=None
        super().__init__(*args,**kwargs)
        self.provenance.update(public_history_cadence_seconds=.05,history_is_not_action_endpoint_subsample=True)

    def public_sample(self):
        now=self.clock()
        if self._public_cache is None or abs(self._public_cache['timestamp']-now)>1e-8:
            self._public_cache=super().public_sample()
            if self.on_public_frame is not None:
                self.on_public_frame(self._public_cache,self._ticks)
        return self._public_cache

    def tick(self,phase='continuous_public_history'):
        result=super().tick(phase)
        if self.on_public_frame is not None:
            self.public_sample()
        return result
