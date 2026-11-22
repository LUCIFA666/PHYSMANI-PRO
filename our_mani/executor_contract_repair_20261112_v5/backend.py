from ..executor_contract_repair_20261112_v4.backend import TaskExecutionBackend as PreviousBackend

class TaskExecutionBackend(PreviousBackend):
    def __init__(self,*args,**kwargs):
        self.on_diagnostic_tick=None
        super().__init__(*args,**kwargs)

    def tick(self,phase='executor_timing'):
        result=super().tick(phase)
        if self.on_diagnostic_tick is not None:self.on_diagnostic_tick()
        return result
