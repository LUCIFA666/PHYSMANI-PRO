"""Follow above a button before committing its learned downward command.

The standoff height is the original TRAIN approach offset. The contact disk
comes from the shared TRAIN plate mesh, reduced by the existing position-fit
tolerance. No task joint, private target pose, or native success value is read.
"""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import numpy as np
from scipy.spatial import ConvexHull
from .interaction import CURRENT

SCHEMA = 'r10_public_button_standoff_and_velocity_alignment_v1'


def contact_radius(asset, position_tolerance):
    vertices = np.asarray(asset['bodies']['plate']['collision_vertices_body'], float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('shared TRAIN plate geometry required')
    hull = ConvexHull(vertices[:, :2])
    equations = hull.equations
    radius = float(np.min(-equations[:, 2]/np.linalg.norm(equations[:, :2], axis=1)))
    if not 0 < position_tolerance < radius:
        raise ValueError('plate cannot support the declared registration tolerance')
    return radius-float(position_tolerance)


def aligned(relative_xy, relative_velocity_xy, radius, control_interval):
    """Stay in the current contact disk for one replanning interval."""
    return bool(np.linalg.norm(relative_xy)+control_interval*np.linalg.norm(relative_velocity_xy)
                <= radius)


def install(actor, asset, *, control_interval=.2):
    c, m = actor.controller, actor.continuous_reference_manager
    if m.confirmations or c.feedback_step_calls or c.library.phase_order != ('approach', 'press'):
        raise ValueError('install on a fresh actual button controller')
    approach, press = c.library.for_phase('approach')[0], c.library.for_phase('press')[0]
    if press.reference_time_basis != CURRENT:
        raise ValueError('standoff requires explicit moving-frame press control')
    radius = contact_radius(asset, press.position_tolerance_m)
    hover_positions = press.positions.copy()
    hover_positions[:, 2] = approach.positions[-1, 2]
    hover = replace(press, positions=hover_positions)
    if not hover.positions[0, 2] > press.positions[-1, 2]:
        raise ValueError('TRAIN approach must be above the press command')
    events, mode = [], {'press': False}
    previous_step = c.step

    def step(instance, state, forecast, remaining_seconds):
        now = float(state.robot.timestamp)
        if instance.phase_index != 1:
            return previous_step(state, forecast, remaining_seconds)
        ids = state.role_bindings.get(press.reference_role, ())
        allow, reason, relative, velocity = False, 'current_unique_reference_unavailable', None, None
        try:
            if len(ids) != 1:
                raise ValueError('unique public button required')
            body_id = ids[0]
            reference = instance.generator.observed_reference(state, body_id, press, instance.origin)
            _, target_velocity, _ = forecast.at(body_id, now)
            relative = state.robot.pose.position[:2]-reference.pose.position[:2]
            velocity = state.robot.linear_velocity[:2]-np.asarray(target_velocity)[:2]
            allow = aligned(relative, velocity, radius, control_interval)
            reason = 'within_plate_disk_and_relative_velocity_envelope' if allow else 'track_above_before_descent'
        except (KeyError, ValueError, IndexError):
            pass
        desired = press if allow else hover
        current = instance.library.for_phase('press')[0]
        changed = current is not desired
        if changed:
            old_goal = instance.goal
            instance.library.templates = tuple(desired if t.phase == 'press' else t for t in instance.library.templates)
            if old_goal is not None and old_goal.phase == 'press':
                # Changing height is a new physical trajectory request. Retain
                # the same phase epoch/deadline and require complete proofs.
                instance.goal = replace(old_goal, template=desired)
                m.current_state, m.current_forecast, m.current_robot = state, forecast, state.robot
                cutoff = forecast.metadata.get('observed_until')
                m.forecast_observed_until = float(cutoff) if type(cutoff) in (float, int) else None
                accepted = m.roll_current_reference(state.robot)
                if not accepted:
                    instance.goal = old_goal
                    instance.library.templates = tuple(current if t.phase == 'press' else t for t in instance.library.templates)
                    if not allow:
                        m.invalidate('lost_button_alignment_requires_fresh_standoff_or_brake', now)
                    reason += '_height_retarget_rejected_full_original_gates'
                else:
                    mode['press'] = allow
            else:
                mode['press'] = allow
        events.append(dict(time=now, descent_permitted=bool(mode['press']),
            alignment_envelope_passed=allow, contact_radius_m=radius,
            relative_xy=None if relative is None else relative.tolist(),
            relative_velocity_xy=None if velocity is None else velocity.tolist(), reason=reason,
            actual_height_offset_m=float(instance.library.for_phase('press')[0].positions[-1, 2]),
            original_phase_epoch=None if instance.goal is None else float(instance.goal.selected_at),
            original_phase_deadline=None if instance.goal is None else float(instance.goal.selected_at+instance.goal.phase_budget),
            native_success_claimed=False, additional_physics_ticks=0))
        return previous_step(state, forecast, remaining_seconds)

    c.step = MethodType(step, c)
    previous_metrics = actor.metrics_report
    def metrics(instance):
        return dict(previous_metrics(), button_standoff=dict(schema=SCHEMA, events=deepcopy(events),
            contact_radius_m=radius, control_interval_seconds=control_interval,
            TRAIN_approach_offset_m=approach.positions[-1].tolist(),
            TRAIN_press_offset_m=press.positions[-1].tolist(),
            native_task_success_unchanged=True, source_models_retrained=False))
    actor.metrics_report = MethodType(metrics, actor)
    return events
