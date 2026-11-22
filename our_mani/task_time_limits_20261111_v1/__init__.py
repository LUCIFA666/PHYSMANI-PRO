"""Shared physical-time policy for Cube, generic feedback and legacy16 tasks."""
from .budget import TaskBudget,DEFAULT_TASK_TIME_LIMIT_SECONDS

__all__=['TaskBudget','DEFAULT_TASK_TIME_LIMIT_SECONDS']
