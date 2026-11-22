"""All16-task feedback backend with the common newly authorized time limit."""
from ..physmani_feedback_interface_20261104_v1.backend import PhysManiFeedbackBackendV1
from . import TaskBudget


class ExtendedTimeFeedbackBackend(PhysManiFeedbackBackendV1):
    def __init__(self,*args,budget_ticks=None,task_time_limit_seconds=None,**kwargs):
        config={}
        if budget_ticks is not None:config['budget_ticks']=budget_ticks
        if task_time_limit_seconds is not None:config['task_time_limit_seconds']=task_time_limit_seconds
        budget=TaskBudget.from_config(config)
        if budget.ticks<2:raise ValueError('At least two physical ticks required by the generic backend')
        # The immutable old constructor caps200. It creates/resets the scene
        # and samples the initial metric, without controlled actions. Install
        # the declared runtime budget immediately, before exposing this object.
        super().__init__(*args,budget_ticks=min(budget.ticks,200),**kwargs)
        try:
            if self._ticks!=0:raise RuntimeError('Cannot extend a task after a controlled tick')
            self.budget_ticks=budget.ticks
            self.provenance.update(requested_budget_ticks=budget.ticks,task_budget=budget.receipt(),
                initial_constructor_admission_ticks=min(budget.ticks,200),budget_installed_before_first_tick=True)
        except BaseException:self.close();raise
