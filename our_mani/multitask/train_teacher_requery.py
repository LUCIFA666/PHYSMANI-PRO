"""Opt-in TRAIN augmentation: requery after a counted failed-planning wait.

The original committed teacher is unchanged. This separate follower records
every failed command, holds for a fixed number of real collector ticks, then
asks the original teacher for a fresh goal in the same evolving episode. It
never resamples a scene, advances physics, retries an infrastructure error,
or turns a failed command into completed supervision.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .train_teacher import CommittedWaypointFollower


@dataclass(frozen=True)
class PlanningRequeryConfig:
    wait_ticks: int = 10
    maximum_requeries: int = 20
    dt: float = .05

    def __post_init__(self):
        if (type(self.wait_ticks) is not int or self.wait_ticks != 10
                or type(self.maximum_requeries) is not int or self.maximum_requeries != 20
                or self.dt != .05):
            raise ValueError('fixed10 actual ticks/.5s and maximum20 same-episode TRAIN requeries required')


class CountedPlanningRequeryFollower(CommittedWaypointFollower):
    def __init__(self, *args, requery_configuration, **kwargs):
        if not isinstance(requery_configuration, PlanningRequeryConfig):
            raise ValueError('explicit fixed TRAIN planning-requery configuration required')
        super().__init__(*args, **kwargs)
        self.requery_configuration = requery_configuration
        self.requery_count = 0
        self._failure_wait_ticks = 0
        self._last_feedback_timestamp = None
        self._pending_previous_failure = None

    def command(self, robot, lower, upper):
        if not np.isfinite(float(robot.timestamp)) or float(robot.timestamp) < 0:
            raise ValueError('fresh TRAIN query requires a finite actual robot timestamp')
        before = self.command_id
        result = super().command(robot, lower, upper)
        if self.command_id != before:
            command = self.commands[-1]
            command['TRAIN_planning_requery_configuration'] = asdict(self.requery_configuration)
            if self._pending_previous_failure is not None:
                command['fresh_query_after_failed_command_id'] = self._pending_previous_failure
                command['same_episode_after_counted_wait'] = True
                self.requery_count += 1
                command['actual_planning_requery_index'] = self.requery_count
                self._pending_previous_failure = None
            if self.failed is not None:
                self._failure_wait_ticks = 0
                command['planning_failure_is_completed_skill'] = False
                command['planning_requery_wait_started_at'] = float(robot.timestamp)
        return result

    def after_tick(self, feedback):
        now = float(feedback['timestamp'])
        if (not np.isfinite(now) or now < 0
                or self._last_feedback_timestamp is not None and
                abs(now - self._last_feedback_timestamp - self.requery_configuration.dt) > 1e-6):
            raise ValueError('planning requery requires distinct consecutive actual collector tick feedback')
        self._last_feedback_timestamp = now
        was_failed = self.failed is not None
        completed = super().after_tick(feedback)
        if not was_failed:
            return completed
        command = self.commands[-1]
        started = command['planning_requery_wait_started_at']
        if abs(now - started - (self._failure_wait_ticks + 1) * self.requery_configuration.dt) > 1e-6:
            raise ValueError('failed planning wait is not bound to the real post-query frame clock')
        self._failure_wait_ticks += 1
        command['planning_requery_counted_wait_ticks'] = self._failure_wait_ticks
        command['planning_requery_last_actual_feedback_at'] = now
        if (self._failure_wait_ticks < self.requery_configuration.wait_ticks
                or self.requery_count >= self.requery_configuration.maximum_requeries):
            return False
        command.update(command_incomplete_at_cutoff=True,
                       cutoff_reason='failed_planning_fresh_query_after_counted_wait',
                       cutoff_at=now, planning_requery_wait_completed_at=now,
                       planning_failure_preserved=True)
        self._pending_previous_failure = self.command_id
        self.close_path()
        self.failed = None
        self.completed = False
        self.state = 'query'
        return False
