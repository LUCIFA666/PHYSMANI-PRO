"""One physical-time limit for every task using the repaired runtime."""
from dataclasses import dataclass
import math

DEFAULT_TASK_TIME_LIMIT_SECONDS=16.
PHYSICS_DT=.05


@dataclass(frozen=True)
class TaskBudget:
    seconds: float=DEFAULT_TASK_TIME_LIMIT_SECONDS

    def __post_init__(self):
        if isinstance(self.seconds,bool) or not isinstance(self.seconds,(float,int)) or not math.isfinite(self.seconds) or self.seconds<=0:
            raise ValueError('positive finite task time limit required')
        if abs(self.seconds/PHYSICS_DT-round(self.seconds/PHYSICS_DT))>1e-8:
            raise ValueError('task time limit must contain whole50ms physical ticks')

    @property
    def ticks(self):return round(self.seconds/PHYSICS_DT)

    @classmethod
    def from_config(cls,config):
        seconds=config.get('task_time_limit_seconds')
        ticks=config.get('budget_ticks')
        if ticks is not None and (type(ticks) is not int or ticks<1):raise ValueError('positive integer physical ticks required')
        if seconds is None:return cls(DEFAULT_TASK_TIME_LIMIT_SECONDS if ticks is None else ticks*PHYSICS_DT)
        result=cls(seconds)
        if ticks is not None and ticks!=result.ticks:raise ValueError('time limit and physical tick budget disagree')
        return result

    def receipt(self):
        return dict(task_time_limit_seconds=self.seconds,budget_ticks=self.ticks,physics_dt=PHYSICS_DT,
            applies_to='all_tasks_without_task_specific_timeout_override',previous_comparison_seconds=8.,
            success_sampling_period_seconds=PHYSICS_DT,stop_at_first_original_success=True,
            time_limit_is_wall_clock=False,Native_physics_or_success_changed=False)
