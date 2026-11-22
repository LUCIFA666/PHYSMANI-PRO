"""Continuous tracking of declared moving-frame task offsets.

The learned offset stays fixed. The control policy explicitly changes a
sample-and-hold command target into a moving-frame target, using only the
current public observation and its causal forecast. Native success is untouched.
"""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import numpy as np
from ..contracts import Reachability
from ..reachability import TimedReachability
from .interaction import CURRENT, COMMAND_GOAL, COMMAND_ISSUE
from .continuous_transit import ConstraintContract

SCHEMA = 'r10_causal_moving_reference_velocity_and_rolling_v1'


def install_moving_reference(actor, *, phases, replan_seconds=.2):
    c, m = actor.controller, actor.continuous_reference_manager
    if (not isinstance(phases, tuple) or not phases or len(set(phases)) != len(phases)
            or any(p not in c.library.phase_order for p in phases)
            or m.confirmations or c.feedback_step_calls
            or not .05 <= replan_seconds <= .5):
        raise ValueError('declare moving phases and install before execution')
    contract = ConstraintContract.from_manager(m)
    original_templates = c.library.templates
    converted = []
    for template in original_templates:
        if template.phase not in phases:
            converted.append(template)
            continue
        if (template.reference_mode != 'external'
                or template.reference_kind not in ('translation_only', 'observed_se3')):
            raise ValueError('moving control requires an observed external reference')
        if template.reference_time_basis in (COMMAND_GOAL, COMMAND_ISSUE):
            template = replace(template, reference_time_basis=CURRENT,
                reference_time_rule_sha256=None, completion_geometry_xyzw=None)
        converted.append(template)
    c.library.templates = tuple(converted)
    if c.generator.library is not c.library:
        raise ValueError('controller and generator must share the actual task library')
    events, last_attempt = [], {}
    previous_joint_check = m.dual_gate.check_joints

    def check_joints(instance, robot, q, v, duration):
        phase = c.library.phase_order[c.phase_index] if c.phase_index < len(c.library.phase_order) else None
        if phase not in phases or duration <= m.config.dt + 1e-8:
            return previous_joint_check(robot, q, v, duration)
        contract.verify(m)
        # Match the moving target's endpoint velocity. Neither the actual-state
        # proof nor the reference-state proof may silently replace it with zero.
        actual = TimedReachability.check_joints(m.native_gate, robot, q, v, duration)
        if not actual.feasible:
            return actual
        if m.braking_active(robot.timestamp):
            return Reachability(False, 'continuous_braking_in_progress')
        boundary = m.continuity_state(robot)
        if boundary is None:
            return Reachability(False, 'confirmed_reference_boundary_unavailable')
        q0, v0, basis = boundary
        reference = replace(robot, joints=q0, joint_velocities=v0)
        gate = getattr(m, 'reference_reserve_gate', m.native_gate)
        checked = TimedReachability.check_joints(gate, reference, q, v, duration)
        evidence = deepcopy(actual.evidence)
        evidence['actual_state_reachability_retained'] = True
        evidence['continuous_reference_proof'] = dict(start_time=float(robot.timestamp),
            duration=float(duration), start_joints=q0.tolist(), start_velocity=v0.tolist(),
            end_joints=np.asarray(q).tolist(), end_velocity=np.asarray(v).tolist(),
            boundary_basis=basis, feasible=bool(checked.feasible), reason=checked.reason,
            evidence=deepcopy(checked.evidence), moving_target_velocity_retained=True)
        return Reachability(bool(checked.feasible), actual.reason if checked.feasible else
            'reference_continuity_'+checked.reason, actual.minimum_duration,
            actual.joints, actual.joint_velocities, evidence)

    m.dual_gate.check_joints = MethodType(check_joints, m.dual_gate)
    previous_stream = m.check_stream

    def stream(instance, robot, duration):
        contract.verify(m)
        goal, plan, now = c.goal, m.plan, float(robot.timestamp)
        eligible = (goal is not None and goal.phase in phases
            and goal.template.reference_time_basis == CURRENT
            and not m.braking_active(now) and not m.qualification_failed
            and abs(duration-m.config.dt) < 1e-8)
        if eligible:
            due = (plan is None or now >= plan.end-1e-8
                   or now >= last_attempt.get(goal.candidate_id, goal.selected_at)+replan_seconds-1e-8)
            if due:
                last_attempt[goal.candidate_id] = now
                renewed = m.roll_current_reference(robot)
                events.append(dict(time=now, phase=goal.phase, renewed=bool(renewed),
                    phase_epoch=float(goal.selected_at),
                    phase_deadline=float(goal.selected_at+goal.phase_budget),
                    original_target_velocity_retained=True, native_success_claimed=False,
                    phase_clock_reset=False, additional_physics_ticks=0))
                if not renewed and (plan is None or now >= plan.end-1e-8):
                    return Reachability(False, 'moving_reference_requires_fresh_feasible_roll')
        return previous_stream(robot, duration)

    m.check_stream = MethodType(stream, m)
    previous_metrics = actor.metrics_report
    def metrics(instance):
        return dict(previous_metrics(), moving_reference=dict(schema=SCHEMA,
            phases=list(phases), replan_seconds=replan_seconds,
            learned_offsets_unchanged=True, command_time_semantics_explicitly_changed=True,
            native_success_unchanged=True, events=deepcopy(events),
            constraint_contract=deepcopy(contract.__dict__), motor_tracking_guaranteed=False))
    actor.metrics_report = MethodType(metrics, actor)
    return dict(events=events, constraint_contract=contract)
