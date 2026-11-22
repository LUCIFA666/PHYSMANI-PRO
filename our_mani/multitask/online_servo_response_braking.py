"""Velocity-anchored empirical servo control and explicit continuous braking.

The frozen empirical estimator remains the causal observation/fit mechanism.
Every predicted response sample uses the original joint/sampled-collision gate.
Unknown response uses the preceding bounded-reference control semantics and is
explicitly not a guarantee of actual tracking. A rejected tracking step does
not abandon a skill or reset its clock.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np

from .evaluation import digest, file_sha
from .online_servo_response import (OnlineServoResponseCompensator, ServoResponseConfig,
    servo_response_binding, actuator_reference_bounds)
from ..reachability import TimedReachability

SCHEMA = 'r10_online_empirical_servo_response_braking_v1'
ROOT = Path(__file__).resolve().parents[2]


def braking_response_binding(config):
    config = dict(config or {})
    if set(config) - {'version', 'servo_response_configuration', 'braking_acceleration_fractions'}:
        raise ValueError('unknown servo braking configuration')
    if config.get('version', SCHEMA) != SCHEMA:
        raise ValueError('wrong servo braking configuration version')
    fractions = config.get('braking_acceleration_fractions', [0., .125, .25, .375, .5])
    if (not isinstance(fractions, (list, tuple)) or not fractions
            or any(isinstance(x, bool) for x in fractions)
            or not np.isfinite(fractions).all() or any(not 0 <= x <= .5 for x in fractions)
            or list(sorted(set(fractions))) != list(fractions) or fractions[0] != 0.):
        raise ValueError('declare increasing braking fractions including constant velocity')
    base = servo_response_binding(config.get('servo_response_configuration', {}))
    normalized = dict(version=SCHEMA, servo_response_configuration=base['normalized_config'],
        braking_acceleration_fractions=[float(x) for x in fractions])
    return dict(normalized_config=normalized, configuration_sha256=digest(normalized),
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)),
        base_source=base['source'])


class OnlineServoResponseBrakingCompensator(OnlineServoResponseCompensator):
    def __init__(self, model, config=None):
        binding = braking_response_binding(config)
        super().__init__(model, ServoResponseConfig(**binding['normalized_config']['servo_response_configuration']))
        self.binding = binding
        self.braking_fractions = binding['normalized_config']['braking_acceleration_fractions']
        self.guard_events = []

    def observe(self, robot):
        guard = super().observe(robot)
        if (guard['violations'] and (not self.guard_events
                or self.guard_events[-1]['observed_at'] != guard['observed_at'])):
            self.guard_events.append(dict(**deepcopy(guard),
                cap_violations=[x for x in guard['violations'] if x != 'observed_empirical_response_interval_violation'],
                model_violations=[x for x in guard['violations'] if x == 'observed_empirical_response_interval_violation'],
                empirical_samples_cleared=True))
        return guard

    def _event(self, robot, guard):
        violations = guard['violations']
        return dict(schema=SCHEMA, observed_time=float(robot.timestamp),
            measured_joints=robot.joints.tolist(), measured_joint_velocities=robot.joint_velocities.tolist(),
            observed_guard=deepcopy(guard),
            observed_cap_violations=[x for x in violations if x != 'observed_empirical_response_interval_violation'],
            observed_model_violations=[x for x in violations if x == 'observed_empirical_response_interval_violation'],
            response_model_reset=bool(violations), actual_tracking_guaranteed=False,
            sampled_interval_approximation=True, full_uncertainty_region_proved=False,
            servo_reference_original_hermite_gate_verified=False,
            additional_sensor_or_physics_queries=False, phase_completion_claimed=False,
            goal_abandoned_by_outer_controller=False, phase_clock_reset=False, total_budget_extended=False)

    @staticmethod
    def _check(reachability, robot, q, v, dt):
        result = TimedReachability.check_joints(reachability, robot, q, v, dt)
        return dict(feasible=bool(result.feasible), reason=result.reason, evidence=deepcopy(result.evidence))

    def _trial(self, robot, displacement, estimate, reachability, **description):
        c = self.config
        inverse = displacement / np.asarray(estimate['median'])
        inverse = np.clip(inverse, -c.maximum_reference_amplification * abs(displacement),
            c.maximum_reference_amplification * abs(displacement))
        reference = actuator_reference_bounds(robot, robot.joints + inverse, self.model, c.dt)
        delta = reference - robot.joints
        checks = []
        for name, gain in self._gain_samples(estimate):
            q = robot.joints + gain * delta
            v = gain * delta / c.dt
            check = self._check(reachability, robot, q, v, c.dt)
            checks.append(dict(sample=name, gain=gain.tolist(), predicted_actual_joints=q.tolist(),
                predicted_reported_velocity_proxy=v.tolist(), **check))
            if not check['feasible']:
                break
        return dict(**description, servo_reference=reference.tolist(), samples=checks,
            passed=all(x['feasible'] for x in checks))

    def _accept(self, event, trial, action, reason):
        event.update(action=action, reason=reason, servo_reference=trial['servo_reference'],
            servo_reference_independent_bounds_verified=True, predicted_actual_samples=trial.get('samples', []),
            selected_scale=trial.get('scale'), selected_braking_fraction=trial.get('braking_fraction'),
            braking_reference_gate=trial.get('reference_gate'),
            predicted_actual_response_available=bool(trial.get('samples')))
        self.events.append(event)
        return event

    def _brake(self, robot, reachability, event, trigger):
        """Try nonincreasing velocity continuations; never invent a safe stop."""
        c = self.config
        estimate = self.estimate()
        event.update(braking_trigger=trigger, braking_response_estimate=estimate, braking_trials=[])
        for fraction in reversed(self.braking_fractions):
            change = np.minimum(abs(robot.joint_velocities), self.model.acceleration_limits * fraction * c.dt)
            # Known response predicts the next reported velocity proxy, which
            # is also the interval-average q displacement model. The original
            # Hermite gate independently checks this boundary interpretation.
            if estimate['ready']:
                velocity = robot.joint_velocities - np.sign(robot.joint_velocities) * change
                trial = self._trial(robot, velocity * c.dt, estimate, reachability,
                    braking_fraction=float(fraction), candidate_basis='velocity_anchored_braking_proxy')
            else:
                terminal = robot.joint_velocities - np.sign(robot.joint_velocities) * change
                desired_q = robot.joints + .5 * (robot.joint_velocities + terminal) * c.dt
                reference = actuator_reference_bounds(robot, desired_q, self.model, c.dt)
                check = self._check(reachability, robot, reference, terminal, c.dt)
                trial = dict(braking_fraction=float(fraction),
                    candidate_basis='unknown_response_original_bounded_reference',
                    servo_reference=reference.tolist(), reference_gate=check, samples=[], passed=check['feasible'],
                    checked_reference_terminal_velocity=terminal.tolist(),
                    predicted_actual_response_unavailable=True)
            event['braking_trials'].append(trial)
            if trial['passed']:
                event['braking_command_is_continuation_not_phase_completion'] = True
                return self._accept(event, trial,
                    'emit_empirical_braking_reference' if estimate['ready'] else 'emit_original_bounded_braking_reference',
                    'continuous_braking_passes_original_sampled_gate')
        event.update(action='unverified_hold_no_feasible_braking', reason='no_original_gate_feasible_braking_reference',
            acceleration_limited_stop_verified=False, servo_reference=None,
            predicted_actual_response_available=False)
        self.events.append(event)
        return event

    def brake(self, robot, reachability, *, trigger):
        guard = self.observe(robot)
        return self._brake(robot, reachability, self._event(robot, guard), trigger)

    def prepare(self, robot, desired_joints, desired_velocities, reachability):
        guard = self.observe(robot)
        event = self._event(robot, guard)
        q, v = np.asarray(desired_joints, float), np.asarray(desired_velocities, float)
        event.update(desired_measured_state_joints=q.tolist(), desired_measured_state_velocities=v.tolist())
        if guard['hold_replan']:
            return self._brake(robot, reachability, event, 'observed_caps_or_empirical_model_violation')
        desired = self._check(reachability, robot, q, v, self.config.dt)
        event['desired_actual_path_original_gate'] = desired
        if not desired['feasible']:
            return self._brake(robot, reachability, event, 'desired_actual_path_rejected')
        estimate = self.estimate()
        event['response_estimate'] = estimate
        if not estimate['ready']:
            trial = dict(servo_reference=actuator_reference_bounds(robot, q, self.model, self.config.dt).tolist())
            return self._accept(event, trial, 'fallback_original_bounded', estimate['reason'])
        anchor = robot.joint_velocities * self.config.dt
        event.update(candidate_displacement_anchor=anchor.tolist(),
            candidate_scaling_basis='measured_velocity_dt_plus_scale_times_desired_delta_minus_measured_velocity_dt', trials=[])
        for scale in self.config.candidate_scales:
            displacement = anchor + scale * (q - robot.joints - anchor)
            trial = self._trial(robot, displacement, estimate, reachability, scale=float(scale),
                candidate_basis='velocity_anchored_tracking')
            event['trials'].append(trial)
            if trial['passed']:
                return self._accept(event, trial, 'emit_compensated_reference',
                    'velocity_anchored_empirical_actual_samples_pass_original_gate')
        return self._brake(robot, reachability, event, 'tracking_interval_samples_rejected')

    def metrics(self):
        result = super().metrics()
        result.update(schema=SCHEMA, goal_abandoned_by_outer_controller=False,
            phase_clock_reset=False, source_bound_estimator_reused=True, observed_guard_events=deepcopy(self.guard_events))
        return result


def install_servo_response_braking(actor, config):
    """Install after measured feedback and before first act, with final receipts."""
    from .feedback_interaction import ObservedFeedbackController
    if (type(actor.controller) is not ObservedFeedbackController or actor.controller.feedback_step_calls
            or hasattr(actor, 'servo_response_compensator')):
        raise ValueError('install servo braking once after feedback and before first act')
    compensator = OnlineServoResponseBrakingCompensator(actor.robot_model, config)
    original_act, original_metrics = actor.act, actor.metrics_report

    class DeferredSegments:
        def __init__(self, segments): self.original = segments
        def __getattr__(self, name): return getattr(self.original, name)
        def issue(self, kind):
            return np.asarray([body not in self.original.non_exogenous for body in self.original.body_ids], bool)

    def act(instance, observation, robot, remaining_seconds):
        guard = compensator.observe(robot)
        controller = instance.controller
        previous_goal = controller.goal
        previous_close = controller._close_edge_issued_at
        old_opening = getattr(instance, 'requested_opening', None)
        if old_opening is None:
            old_opening = float(robot.gripper_open)
        segments = getattr(instance, 'command_segments', None)
        if segments is not None: instance.command_segments = DeferredSegments(segments)
        try:
            command = original_act(observation, robot, remaining_seconds)
        finally:
            if segments is not None: instance.command_segments = segments
        decision = None
        scene_ready = (not hasattr(instance, 'collision_port') or
            instance.last_diagnostics.get('collision_scene_readiness', {}).get('ready') is True)
        if guard['hold_replan'] and scene_ready and not command.get('joint_limit_recovery'):
            decision = compensator.brake(robot, controller.original_reachability,
                trigger='observed_caps_or_empirical_model_violation')
        elif command.get('measured_feedback_command') and command.get('command') == 'joint_trajectory':
            decision = compensator.prepare(robot, command['joints'], command['joint_velocities'],
                controller.original_reachability)
        elif (scene_ready and (previous_goal is not None or controller.goal is not None)
                and command.get('command') == 'hold' and np.any(abs(robot.joint_velocities) > 1e-7)):
            decision = compensator.brake(robot, controller.original_reachability, trigger='inner_hold_with_nonzero_velocity')
        if decision is not None:
            braking = 'braking' in decision['action']
            if braking:
                # A safety continuation cannot issue a newly proposed close or
                # press interaction. Existing actual closing edges remain only
                # when the original controller retains that same active goal.
                instance.requested_opening = old_opening
                controller._close_edge_issued_at = previous_close if controller.goal is previous_goal else None
            if decision['action'] == 'unverified_hold_no_feasible_braking':
                command = dict(command='hold', phase=command.get('phase'), reason=decision['reason'],
                    public_command_kind='hold', acceleration_limited_stop_verified=False)
            else:
                inner = command
                opening = old_opening if braking and old_opening is not None else inner.get('opening', 1.)
                command = dict(inner, command='joint_trajectory', joints=np.asarray(decision['servo_reference']),
                    opening=float(opening), servo_response_schema=SCHEMA, servo_response_action=decision['action'],
                    servo_reference_independent_bounds_verified=True,
                    servo_reference_original_hermite_gate_verified=False,
                    predicted_actual_path_constraints_verified=decision['predicted_actual_response_available'],
                    original_bounded_braking_reference_verified=bool(decision.get('braking_reference_gate', {}).get('feasible'))
                        if decision.get('braking_reference_gate') else False,
                    nominal_constraints_verified=False, joints_are_servo_setpoint_not_predicted_measured_state=True,
                    servo_reference_terminal_velocity_declared=False, actual_tracking_guaranteed=False,
                    sampled_interval_approximation=True, phase_clock_reset=False)
                if 'joints' in inner:
                    command['inner_desired_measured_state_joints'] = np.asarray(inner['joints']).copy()
                if 'joint_velocities' in inner:
                    command['inner_desired_measured_state_velocities'] = np.asarray(inner['joint_velocities']).copy()
                command.pop('joint_velocities', None)
                if braking:
                    command.update(public_command_kind='joint_trajectory', reason=decision['reason'])
            feedback = [e for e in controller.feedback_events if e.get('observed_time') == float(robot.timestamp)]
            gripper = [e for e in controller.gripper_timing_events if e.get('time') == float(robot.timestamp)]
            decision['inner_feedback_proposals'], decision['inner_gripper_proposals'] = deepcopy(feedback), deepcopy(gripper)
            for event in feedback:
                original_issued = bool(event.get('command_issued'))
                same = (command['command'] == 'joint_trajectory'
                    and np.allclose(command['joints'], event['bounded_joints'], rtol=0, atol=1e-12))
                event.update(inner_command_proposed=original_issued, command_issued=bool(original_issued and same),
                    actual_servo_reference_issued=command['command'] == 'joint_trajectory',
                    outer_command_action=decision['action'], actual_servo_reference=None if command['command'] != 'joint_trajectory'
                        else np.asarray(command['joints']).tolist())
            for event in gripper:
                event.update(inner_opening_proposal=event.get('commanded_opening'),
                    commanded_opening=None if command['command'] != 'joint_trajectory' else float(command['opening']),
                    actual_opening_request_issued=command['command'] == 'joint_trajectory',
                    actual_opening_request=None if command['command'] != 'joint_trajectory' else float(command['opening']),
                    outer_returned_command=command['command'],
                    hold_retains_previous_native_opening_request=command['command'] == 'hold')
            if feedback:
                controller.last_diagnostics['measured_feedback_execution'] = deepcopy(feedback[-1])
                if isinstance(instance.last_diagnostics.get('controller'), dict):
                    instance.last_diagnostics['controller']['measured_feedback_execution'] = deepcopy(feedback[-1])
            instance.last_diagnostics['online_servo_response_braking'] = deepcopy(decision)
        if segments is not None: segments.issue(command.get('public_command_kind', command['command']))
        events = getattr(instance, 'events', None)
        if events and events[-1].get('time') == float(robot.timestamp):
            if decision is not None: decision['inner_public_event'] = deepcopy(events[-1])
            events[-1].update(command=command['command'], phase=command.get('phase'),
                public_command_kind=command.get('public_command_kind', command['command']), reason=command.get('reason'))
        instance.last_command = command
        reference = command['joints'] if command.get('command') == 'joint_trajectory' else robot.joints
        compensator.record_issued(robot, reference, kind=command['command'])
        return command

    def metrics(instance):
        result = dict(original_metrics())
        result['online_servo_response_braking'] = compensator.metrics()
        return result
    actor.act, actor.metrics_report = MethodType(act, actor), MethodType(metrics, actor)
    actor.servo_response_compensator = compensator
    return actor
