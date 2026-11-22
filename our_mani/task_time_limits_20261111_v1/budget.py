"""User-selected60second common default; historical16second policy is intact."""
from dataclasses import dataclass
from ..highspeed_control_20261111_v1.task_budget import TaskBudget as HistoricalTaskBudget

DEFAULT_TASK_TIME_LIMIT_SECONDS=60.


@dataclass(frozen=True)
class TaskBudget(HistoricalTaskBudget):
    seconds: float=DEFAULT_TASK_TIME_LIMIT_SECONDS

    @classmethod
    def from_config(cls,config):
        if config.get('task_time_limit_seconds') is None and config.get('budget_ticks') is None:return cls()
        declared=HistoricalTaskBudget.from_config(config)
        return cls(declared.seconds)

    def receipt(self):
        return dict(super().receipt(),common_default_seconds=DEFAULT_TASK_TIME_LIMIT_SECONDS,
            previous_common_default_seconds=16.,legacy16_previous_seconds=10.)
