"""Measured-state bounded execution; original skill search and gates are reused.

Only a rejected next execution segment may be replaced. Candidate feasibility,
skill timers, grasp timing and public completion stay in InteractionController.
The new command integrates a constant acceleration from measured q/v and is
checked by the original complete joint/collision gate. No sensor or physics
query is introduced. A bounded step is not completion of a nominal trajectory.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
import numpy as np

from ..reachability import TimedReachability
from .evaluation import digest, file_sha
from .interaction import InteractionController

SCHEMA = 'r10_observed_measured_feedback_controller_v1'
ROOT = Path(__file__).resolve().parents[2]
RECOVERABLE = frozenset(('insufficient_arrival_time', 'trajectory_joint_speed_limit',
    'trajectory_joint_acceleration_limit', 'terminal_joint_speed_limit'))


@dataclass(frozen=True)
class FeedbackConfig:
    version: str = SCHEMA
    acceleration_fraction: float = .5
    phase_tracking_slack_seconds: float = 0.

    def __post_init__(self):
        if (self.version != SCHEMA or isinstance(self.acceleration_fraction, bool)
                or not np.isfinite(self.acceleration_fraction)
                or not 0 < self.acceleration_fraction <= 1
                or isinstance(self.phase_tracking_slack_seconds, bool)
                or not np.isfinite(self.phase_tracking_slack_seconds)
                or not 0 <= self.phase_tracking_slack_seconds <= 10):
            raise ValueError('feedback requires declared schema and 0 < acceleration_fraction <= 1')


def feedback_binding(config):
    config = config if isinstance(config, FeedbackConfig) else FeedbackConfig(**config)
    return dict(normalized_config=asdict(config), configuration_sha256=digest(asdict(config)),
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)))


def bounded_joint_segment(robot, target, model, dt, acceleration_fraction):
    """Return new command values; never mutate measured state or a nominal plan."""
    target = np.asarray(target, float)
    if (dt <= 0 or target.shape != robot.joints.shape or not np.isfinite(target).all()
            or not 0 < acceleration_fraction <= 1):
        raise ValueError('invalid measured-feedback segment input')
    maximum_change = np.asarray(model.acceleration_limits) * acceleration_fraction * dt
    desired = np.clip((target - robot.joints) / dt, -model.velocity_limits, model.velocity_limits)
    terminal = robot.joint_velocities + np.clip(desired - robot.joint_velocities,
                                               -maximum_change, maximum_change)
    joints = robot.joints + .5 * (robot.joint_velocities + terminal) * dt
    return joints, terminal


class _CaptureIK:
    """Observe the existing IK return without an additional IK/FK/query."""
    def __init__(self, model): self.original, self.target = model, None
    def __getattr__(self, name): return getattr(self.original, name)
    def inverse(self, pose, seed):
        result = self.original.inverse(pose, seed)
        self.target = None if result is None else np.asarray(result, float).copy()
        return result


class _FeedbackReachability(TimedReachability):
    def __init__(self, original, owner):
        self.original, self.owner = original, owner
        # Preserve all original gate parameters, including any configured ones.
        self.__dict__.update({key:value for key,value in original.__dict__.items()})

    def check(self, robot, pose, terminal_velocity, terminal_angular_velocity, duration):
        capture = _CaptureIK(self.model)
        model = self.model
        prior = len(self.owner.feedback_events)
        try:
            self.model = capture
            result = TimedReachability.check(self, robot, pose, terminal_velocity,
                                             terminal_angular_velocity, duration)
        finally:
            self.model = model
        # check() can reject terminal speed before invoking check_joints(). The
        # already obtained IK target is reused, and no probe is repeated.
        if (not result.feasible and result.reason == 'terminal_joint_speed_limit'
                and len(self.owner.feedback_events) == prior and capture.target is not None):
            return self._recover(robot, capture.target, None, duration, result)
        return result

    def check_joints(self, robot, joints, velocities, duration):
        result = TimedReachability.check_joints(self.original, robot, joints, velocities, duration)
        if result.feasible: return result
        return self._recover(robot, joints, velocities, duration, result)

    def _recover(self, robot, joints, velocities, duration, original_result):
        owner = self.owner
        if (owner.goal is None or abs(duration-owner.generator.config.dt) > 1e-10
                or original_result.reason not in RECOVERABLE):
            return original_result
        bounded, terminal = bounded_joint_segment(robot, joints, self.original.model,
            duration, owner.feedback_config.acceleration_fraction)
        # Mandatory original continuous joint extrema and sampled dynamic
        # collision checks apply to the actual command, with no weaker limits.
        result = TimedReachability.check_joints(self.original, robot, bounded, terminal, duration)
        event = dict(schema=SCHEMA, observed_time=float(robot.timestamp), duration=float(duration),
            phase=owner.library.phase_order[owner.phase_index],
            candidate_id=owner.goal.candidate_id, original_rejection=original_result.reason,
            measured_joints=robot.joints.tolist(), measured_joint_velocities=robot.joint_velocities.tolist(),
            nominal_joints=np.asarray(joints).tolist(),
            nominal_joint_velocities=None if velocities is None else np.asarray(velocities).tolist(),
            bounded_joints=bounded.tolist(), bounded_joint_velocities=terminal.tolist(),
            bounded_gate_feasible=bool(result.feasible), bounded_gate_reason=result.reason,
            bounded_gate_evidence=deepcopy(result.evidence),
            command_issued=False, nominal_trajectory_completed=False,
            phase_deadline_extended=owner._phase_has_slack() if hasattr(owner,'_phase_has_slack') else False,
            total_budget_extended=False,
            additional_sensor_or_physics_queries=False)
        owner.feedback_events.append(event)
        if result.feasible: owner._feedback_command = (result, event)
        return result


class ObservedFeedbackController(InteractionController):
    @classmethod
    def from_controller(cls, original, config):
        if type(original) is not InteractionController:
            raise ValueError('install only on the original unwrapped InteractionController')
        if original.goal is not None or original.phase_index != 0 or original.attempts.records:
            raise ValueError('feedback must be installed before the first policy step')
        result = cls.__new__(cls)
        result.__dict__.update(original.__dict__)
        result.feedback_config = config if isinstance(config, FeedbackConfig) else FeedbackConfig(**config)
        result.feedback_events, result.feedback_step_calls = [], 0
        result.phase_timing_events, result._phase_slack_tokens = [], set()
        result._feedback_command = None
        result.feedback_binding = feedback_binding(result.feedback_config)
        result.original_reachability = original.generator.reachability
        result.feedback_reachability = _FeedbackReachability(result.original_reachability, result)
        return result

    def _phase_has_slack(self):
        return any(event['candidate_id']==self.goal.candidate_id
            and event['selected_at']==self.goal.selected_at and event['granted_slack_seconds']>0
            for event in self.phase_timing_events)

    def _grant_phase_slack(self, state, remaining_seconds):
        if self.goal is None: return
        goal=self.goal; now=float(state.robot.timestamp)
        token=(goal.candidate_id,goal.selected_at,len(self.attempts.records))
        if token in self._phase_slack_tokens: return
        self._phase_slack_tokens.add(token)
        original_remaining=max(0.,goal.selected_at+goal.phase_budget-now)
        reserve=max(0.,float(goal.suffix_budget))
        available=max(0.,float(remaining_seconds)-original_remaining-reserve)
        requested=self.feedback_config.phase_tracking_slack_seconds
        granted=min(requested,available)
        self.phase_timing_events.append(dict(candidate_id=goal.candidate_id,selected_at=goal.selected_at,
            observed_time=now,original_phase_budget_seconds=goal.phase_budget,
            requested_slack_seconds=requested,granted_slack_seconds=granted,
            total_task_remaining_seconds=float(remaining_seconds),suffix_reserve_seconds=reserve,
            original_phase_deadline=goal.selected_at+goal.phase_budget,
            actual_phase_deadline=goal.selected_at+goal.phase_budget+granted,
            total_budget_extended=False,nominal_trajectory_completed=False,
            source='declared_feedback_config_capped_by_current_task_remaining_and_original_suffix'))
        if granted>0:self.goal=replace(goal,phase_budget=goal.phase_budget+granted)

    def step(self, state, forecast, remaining_seconds):
        if not np.isfinite(remaining_seconds) or not 0 <= remaining_seconds <= 10.+1e-8:
            raise ValueError('feedback requires the unchanged counted task budget of at most 10 seconds')
        self.feedback_step_calls += 1
        self._feedback_command = None
        self._grant_phase_slack(state,remaining_seconds)
        original = self.generator.reachability
        if original is not self.original_reachability:
            raise ValueError('original reachability changed after feedback installation')
        try:
            self.generator.reachability = self.feedback_reachability
            command = super().step(state, forecast, remaining_seconds)
        finally:
            self.generator.reachability = original
        self._grant_phase_slack(state,remaining_seconds)
        if self._feedback_command is not None and command.get('command') == 'joint_trajectory':
            check, event = self._feedback_command
            nominal = np.asarray(event['nominal_joints'])
            if not (np.allclose(command['joints'], nominal, rtol=0, atol=1e-12)
                    or np.allclose(command['joints'], check.joints, rtol=0, atol=1e-12)):
                raise ValueError('bounded gate and returned command target differ')
            # Approach super.step holds its own nominal q local; replace the
            # returned public value explicitly, rather than altering that q.
            command = dict(command, joints=check.joints.copy(), joint_velocities=check.joint_velocities.copy(),
                measured_feedback_command=True, feedback_schema=SCHEMA,
                nominal_trajectory_completion_claimed=False,
                original_nominal_arrival_time=float(self.goal.arrival_time),
                phase_deadline_extended=self._phase_has_slack(), total_budget_extended=False)
            event['command_issued'] = True
            event['phase_deadline_extended'] = self._phase_has_slack()
            self.last_diagnostics['measured_feedback_execution'] = deepcopy(event)
        return command

    def feedback_metrics(self):
        return dict(schema=SCHEMA,
            actual_controller_class='our_mani.multitask.feedback_interaction.ObservedFeedbackController',
            **deepcopy(self.feedback_binding),
            base_source=dict(path='our_mani/multitask/interaction.py',
                sha256=file_sha(ROOT/'our_mani/multitask/interaction.py')),
            continuation_sha256=self.continuation_sha256, installed_before_first_step=True,
            step_calls=self.feedback_step_calls, fallback_events=deepcopy(self.feedback_events),
            phase_timing_events=deepcopy(self.phase_timing_events),
            candidate_selection_unchanged=True, public_skill_completion_unchanged=True,
            grasp_timing_unchanged=True,
            phase_budget_unchanged=not any(event['granted_slack_seconds']>0 for event in self.phase_timing_events),
            nominal_completion_claimed=False, total_budget_extended=False,
            additional_sensor_or_physics_queries=False)


def install_feedback_controller(actor, config):
    actor.controller = ObservedFeedbackController.from_controller(actor.controller, config)
    original_metrics = actor.metrics_report
    def metrics(instance):
        result = dict(original_metrics())
        result['observed_feedback_controller'] = instance.controller.feedback_metrics()
        return result
    actor.metrics_report = MethodType(metrics, actor)
    return actor
