"""Separate TRAIN candidate: interrupt active arm plans after real feedback.

The existing committed follower and failure-only augmentation are unchanged.
This pure state machine never calls a simulator, a getter, or a physics step.
Actual gripper actuation and release settling complete before a fresh query.
"""
from dataclasses import asdict, dataclass
import math

from .train_teacher_requery import CountedPlanningRequeryFollower, PlanningRequeryConfig


@dataclass(frozen=True)
class PeriodicReplanningConfig:
    feedback_ticks: int = 10
    maximum_periodic_requeries: int = 20
    dt: float = .05

    def __post_init__(self):
        if (type(self.feedback_ticks) is not int or self.feedback_ticks != 10
                or type(self.maximum_periodic_requeries) is not int
                or self.maximum_periodic_requeries != 20 or self.dt != .05):
            raise ValueError('fixed10 actual feedback ticks/.5s and max20 periodic requeries required')


class PeriodicFeedbackReplanningFollower(CountedPlanningRequeryFollower):
    """A typed extra teacher policy, not the author's global-step cadence.

The counter restarts at each real query. A naturally completed command can
query sooner; active arm plans may be interrupted at ten actual feedbacks.
Failure-only waits retain the frozen parent's own count and ledger fields.
"""

    def __init__(self, *args, periodic_configuration, failure_configuration, **kwargs):
        if (not isinstance(periodic_configuration, PeriodicReplanningConfig)
                or not isinstance(failure_configuration, PlanningRequeryConfig)):
            raise ValueError('explicit periodic and failure-only TRAIN configurations required')
        super().__init__(*args, requery_configuration=failure_configuration, **kwargs)
        self.periodic_configuration = periodic_configuration
        self.periodic_requery_count = 0
        self.post_query_feedback_ticks = 0
        self._pending_periodic_command_id = None

    def command(self, robot, lower, upper):
        before = self.command_id
        pending = self._pending_periodic_command_id
        result = super().command(robot, lower, upper)
        if self.command_id != before:
            command = self.commands[-1]
            command['TRAIN_periodic_replanning_configuration'] = asdict(self.periodic_configuration)
            self.post_query_feedback_ticks = 0
            if pending is not None:
                if 'fresh_query_after_failed_command_id' in command:
                    raise ValueError('periodic successful-plan query cannot impersonate a failed-planning query')
                previous = self.commands[-2]
                if previous['command_id'] != pending or 'command_completed_at' in previous:
                    raise ValueError('periodic query must follow its exact censored prior command')
                self.periodic_requery_count += 1
                command.update(fresh_query_after_periodic_interrupted_command_id=pending,
                               same_episode_after_periodic_actual_feedback=True,
                               actual_periodic_requery_index=self.periodic_requery_count)
                previous['periodic_interruption'].update(actual_fresh_query_performed=True,
                    fresh_command_id=self.command_id, fresh_generated_at=float(robot.timestamp))
                self._pending_periodic_command_id = None
        return result

    def after_tick(self, feedback):
        had_command = self.raw is not None
        was_failed = self.failed is not None
        now = float(feedback['timestamp'])
        if had_command and (not math.isfinite(now) or abs(now - float(self.generated_at)
                - (self.post_query_feedback_ticks + 1) * self.periodic_configuration.dt) > 1e-6):
            raise ValueError('periodic actual feedback must match the exact post-query tick clock')
        completed = super().after_tick(feedback)
        if not had_command:
            return completed
        self.post_query_feedback_ticks += 1
        command = self.commands[-1]
        command['post_query_actual_feedback_ticks'] = self.post_query_feedback_ticks
        command['post_query_last_actual_feedback_at'] = now
        if (was_failed or self.failed is not None or completed or self.completed
                or self._pending_periodic_command_id is not None
                or self.post_query_feedback_ticks < self.periodic_configuration.feedback_ticks
                or self.periodic_requery_count >= self.periodic_configuration.maximum_periodic_requeries):
            return completed
        if feedback.get('joint_limit_recovery_action', False):
            command['periodic_replan_deferred_reason'] = 'actual_public_joint_limit_recovery'
            command['periodic_replan_last_deferred_feedback_at'] = now
            return completed
        if self.state in ('actuate', 'release_settle'):
            command['periodic_replan_deferred_reason'] = 'actual_gripper_actuation_or_release_settling'
            command['periodic_replan_last_deferred_feedback_at'] = now
            return completed
        if self.state not in ('follow_path', 'wait_arrival'):
            raise ValueError('unexpected active successful-plan state at periodic feedback boundary')
        if abs(now - float(command['generated_at']) - self.post_query_feedback_ticks * .05) > 1e-6:
            raise ValueError('periodic replan feedback count is not bound to the actual issue clock')
        self.close_path()
        command.update(command_incomplete_at_cutoff=True,
            cutoff_reason='successful_arm_plan_interrupted_after_periodic_actual_feedback', cutoff_at=now,
            periodic_interruption=dict(actual_feedback_ticks=self.post_query_feedback_ticks,
                last_actual_feedback_at=now, actual_path_closed=True,
                successful_plan_is_completed_skill=False, actual_fresh_query_performed=False))
        self._pending_periodic_command_id = self.command_id
        self.completed = False
        self.state = 'query'
        return False
