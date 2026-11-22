"""Owned direct reference submission with unchanged native motors and tick.

This changes the controller's reference filter. Position bounds, incremental
gripper semantics and the shared single counted physics tick are retained.
Reference derivatives and measured backward-window derivatives are distinct.
"""
from copy import deepcopy
import numpy as np

from .actuation import dispatch_counted_command
from .protocol import MultitaskProtocolError, MultitaskInfrastructureError

SCHEMA = 'r10_continuous_reference_direct_counted_setpoint_v22'


def require(value, message):
    if not value:
        raise MultitaskProtocolError(message)


def apply_direct_setpoint(env, joints, *, opening, phase):
    """Position-bound-only reference; exactly one unchanged counted tick."""
    env._before_action()
    current = np.asarray(env.arm.get_joint_positions(), float)
    requested = np.asarray(joints, float)
    require(requested.shape == current.shape and np.isfinite(requested).all(),
        'invalid persistent direct joint reference')
    require(isinstance(opening, (int, float, np.number)) and np.isfinite(opening)
        and 0 <= opening <= 1, 'invalid persistent direct gripper reference')
    bounded = np.clip(requested, env.robot_model.lower, env.robot_model.upper)
    try:
        env.arm.set_joint_target_positions(bounded.tolist())
        requested_opening = float(opening >= .5)
        if requested_opening != env._requested_opening:
            if requested_opening:
                env.gripper.release()
                env.ledger.release_requests += 1
            else:
                for obj in env.task.get_graspable_objects():
                    env.gripper.grasp(obj)
                env.ledger.grasp_requests += 1
            env._requested_opening = requested_opening
        env.gripper.actuate(requested_opening, velocity=env.protocol['gripper_velocity'])
    except Exception as exc:
        raise MultitaskInfrastructureError('persistent direct arm/gripper application failed') from exc
    env._tick(phase)
    require(np.allclose(env.arm.get_joint_target_positions(), bounded, rtol=0, atol=1e-6),
        'persistent direct applied target state invariant failed')
    return bounded


class CountedDirectDispatcher:
    """Direct only for the source-bound feedforward helper's explicit commands."""
    def __init__(self):
        self.events = []
        self.manager = None
        self._environment = None
        self._last_average = None
        self._last_tick = None

    def bind_manager(self, manager):
        require(self.manager is None and not self.events, 'bind the actual-issued manager before execution')
        self.manager = manager

    def dispatch(self, env, command):
        from .task_reference_v37 import SCHEMA as COMMAND_SCHEMA
        require(self.manager is not None, 'continuous execution manager is not bound')
        require(self._environment is None or self._environment is env,
            'persistent dispatcher cannot mix native environments')
        self._environment = env
        before_tick = int(env.ledger.ticks)
        require(self._last_tick is None or self._last_tick == before_tick,
            'persistent dispatcher physical tick sequence changed')
        q0 = np.asarray(env.arm.get_joint_positions(), float)
        native0 = np.asarray(env.arm.get_joint_velocities(), float)
        evidence = command.get('feedforward_execution')
        direct = evidence is not None
        if direct:
            require(command.get('command') == 'joint_trajectory' and isinstance(evidence, dict)
                and evidence.get('schema') == COMMAND_SCHEMA
                and evidence.get('reference_gate_passed') is True
                and evidence.get('dispatch_authorized') is True,
                'direct execution requires the persistent helper reference certificate')
            certified = np.asarray(evidence.get('reference_to_joints'), float)
            requested = np.asarray(command.get('joints'), float)
            require(certified.shape == q0.shape and np.isfinite(certified).all()
                and np.array_equal(certified, requested)
                and isinstance(evidence.get('candidate_id'), str) and bool(evidence['candidate_id'])
                and type(evidence.get('time')) in (float, int)
                and abs(float(evidence['time'])-before_tick*float(env.ledger.dt)) < 1e-8,
                'persistent certificate differs from the current actual issued joint reference')
            bounded = apply_direct_setpoint(env, command.get('joints'),
                opening=command.get('opening'), phase='ours_' + str(command.get('phase', 'control')))
            execution = dict(ticks=1, phase=command.get('phase'),
                public_command_kind=command.get('public_command_kind'),
                counted_dt=env.ledger.dt, command=command['command'])
        else:
            execution = dispatch_counted_command(env, command)
            bounded = np.asarray(env.arm.get_joint_target_positions(), float)
        require(env.ledger.ticks == before_tick + 1,
            'persistent dispatcher must consume exactly one common physical tick')
        q1 = np.asarray(env.arm.get_joint_positions(), float)
        native1 = np.asarray(env.arm.get_joint_velocities(), float)
        require(q0.shape == q1.shape == native0.shape == native1.shape
            and all(np.isfinite(x).all() for x in (q0, q1, native0, native1)),
            'nonfinite persistent dispatcher measured joint state')
        dt = float(env.ledger.dt)
        average = (q1-q0)/dt
        acceleration = None if self._last_average is None else (average-self._last_average)/dt
        velocity_limits = np.asarray(env.robot_model.velocity_limits, float)
        acceleration_limits = np.asarray(env.robot_model.acceleration_limits, float)
        velocity_ok = bool(np.all(np.abs(average) <= velocity_limits + 1e-6))
        acceleration_ok = (None if acceleration is None else
            bool(np.all(np.abs(acceleration) <= acceleration_limits + 1e-6)))
        event = dict(schema=SCHEMA, tick_before=before_tick, tick_after=int(env.ledger.ticks),
            time_before=before_tick*dt, time_after=int(env.ledger.ticks)*dt,
            actual_issued_at=before_tick*dt, reference_target_at=int(env.ledger.ticks)*dt,
            actual_counted_dispatch_completed=True,
            command=command.get('command'), phase=command.get('phase'),
            reference_filter='position_bounds_only' if direct else 'original_counted_dispatch',
            persistent_reference=direct, feedforward_execution=deepcopy(evidence),
            measured_q_before=q0.tolist(), measured_q_after=q1.tolist(),
            native_reported_velocity_before=native0.tolist(),
            native_reported_velocity_after=native1.tolist(),
            applied_joint_reference=bounded.tolist(),
            backward_window_velocity=average.tolist(),
            backward_window_acceleration=None if acceleration is None else acceleration.tolist(),
            velocity_limits=velocity_limits.tolist(), acceleration_limits=acceleration_limits.tolist(),
            actual_window_velocity_within_limits=velocity_ok,
            actual_window_acceleration_within_limits=acceleration_ok,
            derivative_semantics='backward counted-window differences; no instantaneous or substep peak bound',
            native_motor_parameters_changed=False)
        self.events.append(event)
        self.manager.confirm_actual_issued(time=float(env.time), joints=bounded.copy(),
            nominal_velocity=None if not direct else np.asarray(evidence['reference_to_velocity'], float),
            command=deepcopy(command), execution=deepcopy(event))
        self._last_average = average.copy()
        self._last_tick = int(env.ledger.ticks)
        return dict(execution, persistent_reference_execution=deepcopy(event))

    def receipt(self):
        violations = [dict(tick=row['tick_after'], velocity=row['actual_window_velocity_within_limits'],
            acceleration=row['actual_window_acceleration_within_limits']) for row in self.events
            if not row['actual_window_velocity_within_limits']
            or row['actual_window_acceleration_within_limits'] is False]
        return dict(schema=SCHEMA, events=deepcopy(self.events),
            actual_observed_window_limits_passed=not violations, violations=violations,
            actual_tracking_guaranteed=False, reference_filter_is_controller_algorithm=True,
            hardware_acceleration_limit_claimed=False, formal_metric=False)
