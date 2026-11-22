"""Differentiable Torch copy of the frozen P0 joint-reference equations.

This predicts dispatched references, never actual motion. Every 50 ms closed
chain step must feed the *predicted actual* q/dq into this function, dispatch
its q_ref/dq_ref/ddq_ref to the learned response model, then use that model's
next actual state. The initial q/dq and actuator h are real episode anchors.

Reference math remains in actual_q's float32/float64 dtype. Discrete protocol
predicates run on detached float64 values, so the strict >1e-10 retarget rule,
native bounds and feasibility tolerances are not rounded away in float32.
These predicates have no gradient; selected continuous branches retain their
autograd graph, including measured/predicted retarget seeds and carried a.

No Native, physics, FK/IK, I/O, teacher forcing, or Cartesian fallback occurs.
"""
from copy import deepcopy

import torch

from .reference import (
    DT, FEASIBILITY_TOLERANCE, START_BOUNDS_TOLERANCE,
    TARGET_CHANGE_TOLERANCE, FailureCode, ReferenceLimits,
)


_ALIASES = {
    "goal_q": ("goal_q", "last_goal", "target_joints"),
    "q_ref": ("q_ref", "reference_positions"),
    "dq_ref": ("dq_ref", "reference_velocities"),
    "ddq_ref": ("ddq_ref", "reference_accelerations"),
    "valid": ("valid", "reference_valid"),
    "initialized": ("initialized", "reference_initialized"),
}
_ALL_ALIASES = {key for values in _ALIASES.values() for key in values}


def _copy(value):
    # deepcopy(non-leaf tensor) is unsupported and would also be the wrong
    # way to preserve a multi-tick graph. clone is differentiable.
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    return deepcopy(value)


def _first(mapping, names):
    return next((mapping[key] for key in names if key in mapping), None)


def _matrix(value, batch, joints, name, like):
    if value is None:
        return like.new_zeros((batch, joints))
    result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if result.shape == (joints,):
        result = result.unsqueeze(0).expand(batch, joints)
    if result.shape != (batch, joints):
        raise ValueError(name + " must be [J] or [B,J]")
    return result.clone()


def _mask(value, batch, name, like, *, default=True):
    if value is None:
        return torch.full((batch,), default, dtype=torch.bool, device=like.device)
    result = torch.as_tensor(value, dtype=torch.bool, device=like.device).detach()
    if result.ndim == 0:
        result = result.expand(batch)
    if result.shape != (batch,):
        raise ValueError(name + " must be scalar or [B]")
    return result.clone()


def _rows(mask, selected, previous):
    return torch.where(mask.unsqueeze(-1), selected, previous)


def _limits(limits, joints, like):
    double, native = {}, {}
    for name in ("lower", "upper", "velocity", "acceleration", "jerk", "reference_velocity_cap"):
        value = getattr(limits, name)
        # Preserve the original native parameter values for protocol decisions.
        vector64 = torch.as_tensor(value, dtype=torch.float64, device=like.device)
        if vector64.ndim == 0 and name not in ("lower", "upper"):
            vector64 = vector64.expand(joints)
        if vector64.shape != (joints,) or not bool(torch.isfinite(vector64.detach()).all()):
            raise ValueError("finite " + name + "[J] required")
        if name not in ("lower", "upper") and bool((vector64.detach() <= 0).any()):
            raise ValueError("positive " + name + " required")
        double[name], native[name] = vector64, vector64.to(dtype=like.dtype)
    if bool((double["lower"].detach() >= double["upper"].detach()).any()):
        raise ValueError("ordered native joint bounds required")
    for values in (double, native):
        values["effective_velocity"] = torch.minimum(values["velocity"], values["reference_velocity_cap"])
    return native, double


def _normalize_h(h, batch, joints, like):
    h = {} if h is None else h
    raw = {key: _matrix(_first(h, _ALIASES[key]), batch, joints, key, like)
           for key in ("goal_q", "q_ref", "dq_ref", "ddq_ref")}
    valid = _mask(_first(h, _ALIASES["valid"]), batch, "valid", like, default=False)
    initialized = _mask(_first(h, _ALIASES["initialized"]), batch, "initialized", like,
                        default=_first(h, _ALIASES["ddq_ref"]) is not None)
    if bool((valid & ~initialized).any()):
        raise ValueError("valid reference requires initialized q/dq/ddq")
    for key in ("q_ref", "dq_ref", "ddq_ref"):
        if not bool(torch.isfinite(raw[key].detach()[initialized]).all()):
            raise ValueError("initialized reference must be finite: " + key)
    if not bool(torch.isfinite(raw["goal_q"].detach()[valid]).all()):
        raise ValueError("valid reference goal must be finite")
    result = {key: _copy(value) for key, value in h.items() if key not in _ALL_ALIASES}
    result["goal_q"] = _rows(valid, raw["goal_q"], like.new_zeros((batch, joints)))
    for key in ("q_ref", "dq_ref", "ddq_ref"):
        result[key] = _rows(initialized, raw[key], like.new_zeros((batch, joints)))
    result.update(valid=valid, initialized=initialized)
    return result, raw


def _interval(q, v, a, limits, dt):
    vmax = limits["effective_velocity"]
    amax, jmax = limits["acceleration"], limits["jerk"]
    # Exact scalar expression / operation order from ShortHorizonExecutor.
    low = torch.maximum(torch.maximum(-amax, a-jmax*dt),
                        torch.maximum((-vmax-v)/dt, 2*(limits["lower"]-q-v*dt)/(dt*dt)))
    high = torch.minimum(torch.minimum(amax, a+jmax*dt),
                         torch.minimum((vmax-v)/dt, 2*(limits["upper"]-q-v*dt)/(dt*dt)))
    return low, high


def reference_step(actual_q, actual_dq, goal_q, h, limits, *, dt=DT,
                   frequency=6., active_mask=None, apply_mask=None):
    """Differentiable dispatched reference for one native 50 ms tick.

    Canonical h: goal_q/q_ref/dq_ref/ddq_ref[B,J], valid/initialized[B], with
    non-reference extras (including gripper_target) copied unchanged. NumPy
    wrapper aliases and the real executor.state() names are accepted.

    Padding active_mask=False never changes any h tensor, including invalid
    placeholders. apply_mask=False prepares a valid goal but dispatches no
    reference, matching a zero-budget P0 command. A failed active row returns
    feasible=False and must be reported as an unpredictable horizon by callers;
    the return value is not a fallback action or a successful hold.
    """
    actual_input = torch.as_tensor(actual_q)
    if actual_input.dtype not in (torch.float32, torch.float64):
        raise TypeError("actual_q must use float32 or float64")
    if dt != DT or not isinstance(frequency, (int, float)) or not 0 < frequency < float("inf"):
        raise ValueError("native dt=.05 and positive finite reference frequency required")
    goal_input = torch.as_tensor(goal_q, dtype=actual_input.dtype, device=actual_input.device)
    batch = goal_input.shape[0] if goal_input.ndim == 2 else actual_input.shape[0] if actual_input.ndim == 2 else 1
    joints = len(limits.lower)
    actual = _matrix(actual_input, batch, joints, "actual_q", actual_input)
    velocity = _matrix(actual_dq, batch, joints, "actual_dq", actual_input)
    goal = _matrix(goal_input, batch, joints, "goal_q", actual_input)
    native, double = _limits(limits, joints, actual_input)
    next_h, raw_h = _normalize_h(h, batch, joints, actual_input)
    active = _mask(active_mask, batch, "active_mask", actual_input)
    apply = _mask(apply_mask, batch, "apply_mask", actual_input) & active
    goal64, actual64, velocity64 = (value.detach().to(torch.float64) for value in (goal, actual, velocity))
    finite_goal = torch.isfinite(goal64).all(dim=-1)
    inside_goal = ((goal64 >= double["lower"].detach()) & (goal64 <= double["upper"].detach())).all(dim=-1)
    codes = torch.zeros(batch, dtype=torch.int8, device=actual_input.device)
    codes = torch.where(active & ~finite_goal, int(FailureCode.NONFINITE_GOAL), codes)
    codes = torch.where(active & finite_goal & ~inside_goal, int(FailureCode.GOAL_OUTSIDE_NATIVE_BOUNDS), codes)
    prepared = active & finite_goal & inside_goal
    finite_actual = torch.isfinite(actual64).all(dim=-1) & torch.isfinite(velocity64).all(dim=-1)
    bad_actual = prepared & ~finite_actual
    codes = torch.where(bad_actual, int(FailureCode.NONFINITE_ACTUAL_STATE), codes)
    next_h["valid"] = next_h["valid"] & ~bad_actual
    prepared = prepared & finite_actual
    changed = prepared & (~next_h["valid"] | (torch.abs(
        goal64-next_h["goal_q"].detach().to(torch.float64)) > TARGET_CHANGE_TOLERANCE).any(dim=-1))
    inside_start = ((actual64 >= double["lower"].detach()-START_BOUNDS_TOLERANCE) &
                    (actual64 <= double["upper"].detach()+START_BOUNDS_TOLERANCE)).all(dim=-1)
    bad_start = prepared & changed & ~inside_start
    codes = torch.where(bad_start, int(FailureCode.MEASURED_START_OUTSIDE_NATIVE_BOUNDS), codes)
    next_h["valid"] = next_h["valid"] & ~bad_start
    prepared = prepared & ~bad_start
    reseeded = prepared & changed
    # Sanitize rejected nonfinite rows before doing continuous arithmetic so
    # their unused branches cannot poison gradients of other batch rows.
    actual_safe = _rows(finite_actual, actual, torch.zeros_like(actual))
    velocity_safe = _rows(finite_actual, velocity, torch.zeros_like(velocity))
    goal_safe = _rows(finite_goal, goal, torch.zeros_like(goal))
    next_h["q_ref"] = _rows(reseeded, actual_safe, next_h["q_ref"])
    next_h["dq_ref"] = _rows(reseeded, velocity_safe, next_h["dq_ref"])
    next_h["initialized"] = next_h["initialized"] | reseeded
    next_h["goal_q"] = _rows(prepared, goal_safe, next_h["goal_q"])
    next_h["valid"] = next_h["valid"] | prepared
    q, v, a = next_h["q_ref"], next_h["dq_ref"], next_h["ddq_ref"]
    low, high = _interval(q, v, a, native, dt)
    low64, high64 = _interval(*(value.detach().to(torch.float64) for value in (q, v, a)),
                              {key: value.detach() for key, value in double.items()}, dt)
    failed_joints = low64 > high64+FEASIBILITY_TOLERANCE
    evaluate = prepared & apply
    bad_reference = evaluate & failed_joints.any(dim=-1)
    codes = torch.where(bad_reference, int(FailureCode.NO_FEASIBLE_BOUNDED_REFERENCE), codes)
    next_h["valid"] = next_h["valid"] & ~bad_reference
    applied = evaluate & ~bad_reference
    desired = frequency*frequency*(next_h["goal_q"]-q)-2*frequency*v
    acceleration = torch.minimum(high, torch.maximum(low, desired))
    position = q+v*dt+0.5*acceleration*dt*dt
    speed = v+acceleration*dt
    next_h["q_ref"] = _rows(applied, position, q)
    next_h["dq_ref"] = _rows(applied, speed, v)
    next_h["ddq_ref"] = _rows(applied, acceleration, a)
    # Unlike malformed active state, inactive padded memory is opaque; do not
    # turn its invalid placeholders into zeros or modify a stored goal.
    for key, value in raw_h.items():
        next_h[key] = _rows(active, next_h[key], value)
    return dict(next_h=next_h, q_ref=next_h["q_ref"].clone(), dq_ref=next_h["dq_ref"].clone(),
                ddq_ref=next_h["ddq_ref"].clone(), applied=applied, feasible=codes == 0,
                failure_code=codes, target_changed=changed, reference_reseeded=reseeded,
                failed_joints=failed_joints & bad_reference.unsqueeze(-1), dt=dt,
                predicate_dtype="float64_detached", actual_state_prediction_performed=False,
                native_steps=0)


__all__ = ["ReferenceLimits", "FailureCode", "reference_step"]
