"""Persistent verified Hermite reference, distinct from measured motor motion.

The selected full-duration trajectory is unchanged. Its next reference segment
is checked against the same analytic joint limits and fresh sampled collision
scene. Observed q-window velocity/acceleration is monitored separately. Direct
position-setpoint dispatch must be supplied and source-bound by the backend.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
import numpy as np

from .evaluation import digest, file_sha
from ..contracts import Reachability
from ..reachability import TimedReachability

SCHEMA = 'r10_persistent_verified_reference_feedforward_v1'
COMMAND_SCHEMA = SCHEMA
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PersistentReferenceConfig:
    version: str = SCHEMA
    dt: float = .05
    observed_velocity_tolerance: float = .00001
    observed_acceleration_tolerance: float = .001

    def __post_init__(self):
        if (self.version != SCHEMA or self.dt != .05
                or any(isinstance(x, bool) for x in (self.dt, self.observed_velocity_tolerance, self.observed_acceleration_tolerance))
                or not np.isfinite([self.dt, self.observed_velocity_tolerance, self.observed_acceleration_tolerance]).all()
                or min(self.observed_velocity_tolerance, self.observed_acceleration_tolerance) < 0):
            raise ValueError('declare the original50ms and finite observed-window tolerances')


def persistent_reference_binding(config=None):
    config = config if isinstance(config, PersistentReferenceConfig) else PersistentReferenceConfig(**(config or {}))
    sources = ('our_mani/multitask/feedback_interaction.py', 'our_mani/multitask/interaction.py',
        'our_mani/reachability.py', 'our_mani/intercept_planner.py')
    return dict(normalized_config=asdict(config), configuration_sha256=digest(asdict(config)),
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)),
        original_sources_sha256={p:file_sha(ROOT/p) for p in sources},
        empirical_servo_model_used=False, actual_tracking_guaranteed=False)


class PersistentReferenceManager:
    def __init__(self, controller, config=None):
        self.controller = controller
        self.config = config if isinstance(config, PersistentReferenceConfig) else PersistentReferenceConfig(**(config or {}))
        self.binding = persistent_reference_binding(self.config)
        self.previous_robot = self.previous_window_velocity = None
        self.observed_events, self.reference_events = [], []
        self.pending = None
        self.forecast_observed_until = None
        self.active_time = None
        self.qualification_failed = False

    def observe(self, robot):
        c, model = self.config, self.controller.original_reachability.model
        previous = self.previous_robot
        if previous is not None and robot.timestamp <= previous.timestamp + 1e-8:
            raise ValueError('persistent reference observes one increasing controller step per tick')
        violations = []
        if np.any(robot.joints < model.lower - 1e-7) or np.any(robot.joints > model.upper + 1e-7):
            violations.append('observed_joint_position_limit')
        if np.any(abs(robot.joint_velocities) > model.velocity_limits + c.observed_velocity_tolerance):
            violations.append('observed_reported_joint_velocity_limit')
        window = acceleration = None
        if previous is not None:
            elapsed = robot.timestamp - previous.timestamp
            window = (robot.joints - previous.joints) / elapsed
            if np.any(abs(window) > model.velocity_limits + c.observed_velocity_tolerance):
                violations.append('observed_backward_window_velocity_limit')
            if abs(elapsed - c.dt) < 1e-8 and self.previous_window_velocity is not None:
                acceleration = (window - self.previous_window_velocity) / c.dt
                if np.any(abs(acceleration) > model.acceleration_limits + c.observed_acceleration_tolerance):
                    violations.append('observed_backward_window_acceleration_limit')
            # A gap does not identify a50ms velocity window for next-step
            # acceleration; never mix incompatible window durations.
            self.previous_window_velocity = window.copy() if abs(elapsed-c.dt) < 1e-8 else None
        row = dict(time=float(robot.timestamp), measured_joints=robot.joints.tolist(),
            native_reported_velocity_proxy=robot.joint_velocities.tolist(),
            backward_window_velocity=None if window is None else window.tolist(),
            backward_window_acceleration=None if acceleration is None else acceleration.tolist(),
            first_window_acceleration_not_defined=acceleration is None,
            violations=violations, actual_tracking_guaranteed=False,
            observed_velocity_is_not_instantaneous_endpoint_velocity=True)
        self.previous_robot = robot
        self.qualification_failed |= bool(violations)
        self.observed_events.append(row)
        return row

    def check_reference(self, robot, duration):
        controller, c = self.controller, self.config
        goal, plan = controller.goal, controller.trajectory
        if (goal is None or plan is None or abs(duration-c.dt) > 1e-8
                or goal.phase != 'approach' or goal.template.reference_time_basis != 'command_issue'
                or robot.timestamp < plan.start-1e-8):
            return None
        stationary_endpoint = robot.timestamp >= plan.end-1e-8
        if ((stationary_endpoint and np.any(abs(plan.v1) > 1e-10))
                or (not stationary_endpoint and robot.timestamp+c.dt > plan.end+1e-8)):
            return None
        check = goal.reachability
        if (not check.feasible or abs(plan.end-goal.arrival_time) > 1e-8
                or not np.allclose(plan.q1, check.joints, atol=1e-12, rtol=0)
                or not np.allclose(plan.v1, check.joint_velocities, atol=1e-12, rtol=0)):
            raise ValueError('persistent reference must be the original fully verified candidate trajectory')
        if (self.forecast_observed_until is None or
                not -1e-8 <= robot.timestamp-self.forecast_observed_until
                    <= controller.generator.config.maximum_observation_age):
            return Reachability(False, 'persistent_reference_requires_current_causal_forecast')
        if stationary_endpoint:
            q0, v0 = plan.q1.copy(), np.zeros_like(plan.v1)
            q1, v1 = q0.copy(), v0.copy()
        else:
            q0, v0 = plan.at(robot.timestamp)
            q1, v1 = plan.at(min(robot.timestamp+c.dt, plan.end))
        reference_state = replace(robot, joints=q0, joint_velocities=v0)
        result = TimedReachability.check_joints(controller.original_reachability,
            reference_state, q1, v1, c.dt)
        event = dict(schema=SCHEMA, time=float(robot.timestamp), candidate_id=goal.candidate_id,
            reference_mode='verified_zero_velocity_endpoint' if stationary_endpoint else 'verified_original_Hermite_segment',
            plan_start=float(plan.start), plan_end=float(plan.end), phase_epoch=float(goal.selected_at),
            phase_deadline=float(goal.selected_at+goal.phase_budget),
            reference_from_joints=q0.tolist(), reference_from_velocity=v0.tolist(),
            reference_to_joints=q1.tolist(), reference_to_velocity=v1.tolist(),
            measured_joints=robot.joints.tolist(), measured_velocity_proxy=robot.joint_velocities.tolist(),
            reference_tracking_error_rad=float(np.linalg.norm(robot.joints-q0)),
            original_full_candidate_gate_feasible=bool(check.feasible),
            reference_gate_passed=bool(result.feasible), reference_gate_reason=result.reason,
            reference_gate_evidence=deepcopy(result.evidence),
            forecast_observed_until=float(self.forecast_observed_until),
            checked_state_is_reference_not_measured=True, reference_gate_is_not_actual_motor_proof=True,
            actual_tracking_guaranteed=False, sampled_collision_semantics_unchanged=True,
            endpoint_reference_not_physical_phase_completion=True, phase_clock_reset=False,
            total_budget_extended=False, extra_sensor_or_physics_queries=False, issued=False)
        self.reference_events.append(event)
        if result.feasible:
            self.pending = event
        return result

    def metrics(self):
        result = deepcopy(self.binding)
        result.update(schema=SCHEMA, observed_events=deepcopy(self.observed_events),
            reference_events=deepcopy(self.reference_events), observed_motion_qualification_failed=self.qualification_failed,
            actual_tracking_guaranteed=False, phase_completion_claimed=False, total_budget_extended=False,
            empirical_servo_model_used=False, native_or_GPU_launched_by_module=False)
        return result


class _PersistentReferenceReachability:
    def __init__(self, manager, original_feedback):
        self.manager, self.original_feedback = manager, original_feedback

    def __getattr__(self, key): return getattr(self.original_feedback, key)

    def check_joints(self, robot, q, v, duration):
        result = self.manager.check_reference(robot, duration)
        if result is not None:
            return result
        return self.original_feedback.check_joints(robot, q, v, duration)

    def check(self, robot, pose, velocity, angular, duration):
        # Original execution switches to MPC one tick before nominal arrival.
        # Preserve the last verified reference segment through its actual end;
        # after end, ordinary measured-feedback MPC resumes without this tag.
        result = self.manager.check_reference(robot, duration)
        if result is not None:
            return result
        return self.original_feedback.check(robot, pose, velocity, angular, duration)


def install_persistent_reference(actor, config=None):
    """Install after feedback/optional causal retarget, before any act()."""
    from .feedback_interaction import ObservedFeedbackController
    controller = actor.controller
    if (type(controller) is not ObservedFeedbackController or controller.feedback_step_calls
            or controller.goal is not None or hasattr(actor, 'persistent_reference_manager')
            or hasattr(actor, 'servo_response_compensator') or controller.ranking != 'geometry'
            or controller.library.task_spec.task_name != 'moving_basketball_in_hoop'):
        raise ValueError('persistent reference installs once on normal basketball geometry after feedback, before first act')
    manager = PersistentReferenceManager(controller, config)
    previous_step, previous_metrics = controller.step, actor.metrics_report
    controller.feedback_reachability = _PersistentReferenceReachability(manager, controller.feedback_reachability)

    def step(instance, state, forecast, remaining_seconds):
        if not np.isfinite(remaining_seconds) or not 0 <= remaining_seconds <= 10.+1e-8:
            raise ValueError('persistent reference keeps the original10-second task budget')
        observation = manager.observe(state.robot)
        manager.pending = None
        manager.active_time = float(state.robot.timestamp)
        cutoff = forecast.metadata.get('observed_until')
        manager.forecast_observed_until = float(cutoff) if type(cutoff) in (float, int) and np.isfinite(cutoff) else None
        if observation['violations']:
            instance.last_diagnostics['persistent_observed_motion_violation'] = deepcopy(observation)
            return dict(command='hold', phase=instance.library.phase_order[instance.phase_index]
                if instance.phase_index < len(instance.library.phase_order) else 'verify',
                reason='persistent_observed_motion_limit_violation', actual_motion_qualification_failed=True,
                hold_is_not_verified_acceleration_limited_braking=True)
        command = previous_step(state, forecast, remaining_seconds)
        event = manager.pending
        if event is not None and command.get('command') == 'joint_trajectory':
            # Original approach keeps q nominal as a local variable; the last
            # segment may have entered MPC. Return the checked reference q
            # explicitly and never substitute an unverified IK/MPC setpoint.
            command = dict(command, joints=np.asarray(event['reference_to_joints']),
                joint_velocities=np.asarray(event['reference_to_velocity']),
                persistent_reference_verified=True, feedforward_execution=deepcopy(event),
                measured_feedback_command=False, nominal_constraints_verified=True,
                actual_motor_tracking_guaranteed=False)
            event['issued'] = True
            command['feedforward_execution']['issued'] = True
            instance.last_diagnostics['persistent_reference_feedforward'] = deepcopy(event)
        return command

    def metrics(instance):
        report = dict(previous_metrics())
        report['persistent_reference_feedforward'] = manager.metrics()
        return report
    controller.step, actor.metrics_report = MethodType(step, controller), MethodType(metrics, actor)
    actor.persistent_reference_manager = manager
    return actor
