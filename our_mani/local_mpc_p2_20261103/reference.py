"""Pure NumPy copy of the frozen P0 joint-reference state machine.

This module generates references, not actual robot motion.  In a learned
50 ms rollout call ``reference_step`` using the current predicted actual q/dq,
then predict that physical tick from the returned applied reference.  A fixed
100 ms goal normally calls it twice; only a changed goal reseeds from actual
q/dq.  Same-goal references continue independently of actual tracking error.

All angular units are radian based. Acceleration / jerk are controller
reference constraints, with jerk defined as acceleration difference / 50 ms;
they are not assertions about native motor physics. No Native/Torch import,
I/O, FK, IK, physics step, gripper model, or actual-state prediction occurs.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


DT = .05
TARGET_CHANGE_TOLERANCE = 1e-10
START_BOUNDS_TOLERANCE = 1e-8
FEASIBILITY_TOLERANCE = 1e-9


class FailureCode(IntEnum):
    OK = 0
    NONFINITE_GOAL = 1
    GOAL_OUTSIDE_NATIVE_BOUNDS = 2
    NONFINITE_ACTUAL_STATE = 3
    MEASURED_START_OUTSIDE_NATIVE_BOUNDS = 4
    NO_FEASIBLE_BOUNDED_REFERENCE = 5


def _positive_vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(size, result.item())
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("positive finite " + name + "[J] required")
    return result.copy()


@dataclass(frozen=True)
class ReferenceLimits:
    lower: object
    upper: object
    velocity: object
    acceleration: object = 12.
    jerk: object = 1000.
    reference_velocity_cap: object = 3.

    def __post_init__(self):
        lower, upper = np.asarray(self.lower, dtype=np.float64), np.asarray(self.upper, dtype=np.float64)
        if lower.ndim != 1 or not len(lower) or upper.shape != lower.shape:
            raise ValueError("native lower/upper[J] required")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("ordered finite native joint position bounds required")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        for name in ("velocity", "acceleration", "jerk", "reference_velocity_cap"):
            object.__setattr__(self, name, _positive_vector(getattr(self, name), len(lower), name))

    @property
    def effective_velocity(self):
        return np.minimum(self.velocity, self.reference_velocity_cap)

    @classmethod
    def from_executor_limits(cls, limits):
        """Copy an existing JointLimits object without importing its module."""
        return cls(**{key: getattr(limits, key) for key in (
            "lower", "upper", "velocity", "acceleration", "jerk", "reference_velocity_cap")})


_ALIASES = {
    "goal_q": ("goal_q", "last_goal", "target_joints"),
    "q_ref": ("q_ref", "reference_positions"),
    "dq_ref": ("dq_ref", "reference_velocities"),
    "ddq_ref": ("ddq_ref", "reference_accelerations"),
    "valid": ("valid", "reference_valid"),
    "initialized": ("initialized", "reference_initialized"),
}
_ALL_ALIASES = {key for values in _ALIASES.values() for key in values}


def _first(mapping, names):
    return next((mapping[key] for key in names if key in mapping), None)


def _matrix(value, batch, joints, name, *, default=0.):
    if value is None:
        return np.full((batch, joints), default, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (joints,):
        array = np.broadcast_to(array, (batch, joints))
    if array.shape != (batch, joints):
        raise ValueError(name + " must be [J] or [B,J]")
    return array.copy()


def _mask(value, batch, name, *, default=True):
    if value is None:
        return np.full(batch, default, dtype=bool)
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        array = np.full(batch, array.item(), dtype=bool)
    if array.shape != (batch,):
        raise ValueError(name + " must be scalar or [B]")
    return array.copy()


def normalize_h(h, batch, joints):
    """Copy canonical h, accepting actual executor.state() or wrapper aliases.

    ``valid`` means a prior goal exists. ``initialized`` separately remembers
    that q_ref/dq_ref/ddq_ref exist after an execution failure invalidates the
    goal.  Carrying that prior ddq_ref is required by the real executor.
    Extra non-reference fields (e.g. gripper_target) are copied unchanged.
    """
    h = {} if h is None else h
    result = {key: deepcopy(value) for key, value in h.items() if key not in _ALL_ALIASES}
    for key in ("goal_q", "q_ref", "dq_ref", "ddq_ref"):
        result[key] = _matrix(_first(h, _ALIASES[key]), batch, joints, key)
    result["valid"] = _mask(_first(h, _ALIASES["valid"]), batch, "valid", default=False)
    # Explicit initialized is preferred. For a real executor snapshot, None
    # acceleration means no reference was initialized, while an array exists
    # even when an earlier failure set reference_valid=False.
    has_acceleration = _first(h, _ALIASES["ddq_ref"]) is not None
    result["initialized"] = _mask(_first(h, _ALIASES["initialized"]), batch,
                                   "initialized", default=has_acceleration)
    if np.any(result["valid"] & ~result["initialized"]):
        raise ValueError("valid reference requires initialized q/dq/ddq")
    for key in ("q_ref", "dq_ref", "ddq_ref"):
        if not np.isfinite(result[key][result["initialized"]]).all():
            raise ValueError("initialized reference must be finite: " + key)
        result[key][~result["initialized"]] = 0.
    if not np.isfinite(result["goal_q"][result["valid"]]).all():
        raise ValueError("valid reference goal must be finite")
    result["goal_q"][~result["valid"]] = 0.
    return result


def state_from_executor(snapshot, batch_size=1):
    """Convert one actual executor.state() snapshot to replicated canonical h."""
    size = next((len(snapshot[key]) for key in ("reference_positions", "target_joints")
                 if snapshot.get(key) is not None), None)
    if size is None:
        raise ValueError("uninitialized executor snapshot needs normalize_h(snapshot,B,J) with explicit J")
    return normalize_h(snapshot, batch_size, size)


def _inputs(actual_q, actual_dq, goal_q, limits):
    if not isinstance(limits, ReferenceLimits):
        limits = ReferenceLimits.from_executor_limits(limits)
    goal = np.asarray(goal_q, dtype=np.float64)
    actual = np.asarray(actual_q, dtype=np.float64)
    batch = goal.shape[0] if goal.ndim == 2 else actual.shape[0] if actual.ndim == 2 else 1
    joints = len(limits.lower)
    return (_matrix(actual_q, batch, joints, "actual_q"),
            _matrix(actual_dq, batch, joints, "actual_dq"),
            _matrix(goal_q, batch, joints, "goal_q"), limits, batch, joints)


def reference_step(actual_q, actual_dq, goal_q, h, limits, *, dt=DT,
                   frequency=6., active_mask=None, apply_mask=None):
    """Generate one 50 ms dispatched joint reference per batch row.

    Invalid rows fail independently and return feasible=False, never a clipped
    substitute trajectory. ``apply_mask=False`` still prepares/commits the goal
    exactly as a zero-budget P0 call does; it dispatches no reference. Inactive
    rows are not processed at all. Actual q/dq are only reference seeds when
    the goal changes (strict max absolute change > 1e-10) or h.valid is False.
    """
    if dt != DT or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("native dt=.05 and positive finite reference frequency required")
    actual, velocity, goal, limits, batch, joints = _inputs(actual_q, actual_dq, goal_q, limits)
    next_h = normalize_h(h, batch, joints)
    active = _mask(active_mask, batch, "active_mask")
    apply = _mask(apply_mask, batch, "apply_mask") & active
    codes = np.zeros(batch, dtype=np.int8)
    changed = np.zeros(batch, dtype=bool)
    reseeded = np.zeros(batch, dtype=bool)
    finite_goal = np.isfinite(goal).all(axis=-1)
    inside_goal = ((goal >= limits.lower) & (goal <= limits.upper)).all(axis=-1)
    codes[active & ~finite_goal] = FailureCode.NONFINITE_GOAL
    codes[active & finite_goal & ~inside_goal] = FailureCode.GOAL_OUTSIDE_NATIVE_BOUNDS
    # Direct joint-target validation precedes _execute in the actual executor:
    # bad goal rows preserve old reference state, including h.valid.
    prepared = active & finite_goal & inside_goal
    finite_actual = np.isfinite(actual).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
    bad_actual = prepared & ~finite_actual
    codes[bad_actual] = FailureCode.NONFINITE_ACTUAL_STATE
    next_h["valid"][bad_actual] = False
    prepared &= finite_actual
    changed[prepared] = (~next_h["valid"][prepared] |
                         np.any(np.abs(goal[prepared] - next_h["goal_q"][prepared]) > TARGET_CHANGE_TOLERANCE, axis=-1))
    inside_start = ((actual >= limits.lower - START_BOUNDS_TOLERANCE) &
                    (actual <= limits.upper + START_BOUNDS_TOLERANCE)).all(axis=-1)
    bad_start = prepared & changed & ~inside_start
    codes[bad_start] = FailureCode.MEASURED_START_OUTSIDE_NATIVE_BOUNDS
    next_h["valid"][bad_start] = False
    prepared &= ~bad_start
    reseeded = prepared & changed
    next_h["q_ref"][reseeded] = actual[reseeded]
    next_h["dq_ref"][reseeded] = velocity[reseeded]
    # normalize_h already supplies ddq=0 only when never initialized. Otherwise
    # preserve its full prior acceleration, including after h.valid=False.
    next_h["initialized"][reseeded] = True
    next_h["goal_q"][prepared] = goal[prepared]
    next_h["valid"][prepared] = True
    evaluate = prepared & apply
    q, v, a = next_h["q_ref"], next_h["dq_ref"], next_h["ddq_ref"]
    vmax, amax, jmax = limits.effective_velocity, limits.acceleration, limits.jerk
    low = np.maximum.reduce(np.broadcast_arrays(-amax, a - jmax * dt, (-vmax - v) / dt,
                                                2 * (limits.lower - q - v * dt) / (dt * dt)))
    high = np.minimum.reduce(np.broadcast_arrays(amax, a + jmax * dt, (vmax - v) / dt,
                                                 2 * (limits.upper - q - v * dt) / (dt * dt)))
    failed_joints = low > high + FEASIBILITY_TOLERANCE
    bad_reference = evaluate & failed_joints.any(axis=-1)
    codes[bad_reference] = FailureCode.NO_FEASIBLE_BOUNDED_REFERENCE
    next_h["valid"][bad_reference] = False
    applied = evaluate & ~bad_reference
    desired = frequency * frequency * (next_h["goal_q"] - q) - 2 * frequency * v
    acceleration = np.minimum(high, np.maximum(low, desired))
    position = q + v * dt + 0.5 * acceleration * dt * dt
    speed = v + acceleration * dt
    # A row is atomic: if any joint is infeasible the scalar implementation
    # raises before dispatching *any* joint and before updating any reference.
    next_h["q_ref"][applied] = position[applied]
    next_h["dq_ref"][applied] = speed[applied]
    next_h["ddq_ref"][applied] = acceleration[applied]
    return dict(next_h=next_h, q_ref=next_h["q_ref"].copy(), dq_ref=next_h["dq_ref"].copy(),
                ddq_ref=next_h["ddq_ref"].copy(), applied=applied, feasible=codes == 0,
                failure_code=codes, target_changed=changed, reference_reseeded=reseeded,
                failed_joints=failed_joints & bad_reference[:, None], dt=dt)


def reference_rollout(actual_q, actual_dq, goal_q, h, limits, *, ticks=2,
                      tick_budget=None, frequency=6.):
    """Reference-only 0/1/2-tick macrostep; it predicts no actual q/dq.

    Current actual q/dq can be repeated here because the fixed goal remains the
    same within this macrostep. For a learned physical rollout, instead call
    reference_step between each predicted 50 ms physical transition. The
    output [B,ticks,J] arrays are NaN where no reference was dispatched; always
    respect applied_mask/feasible before giving references to a motion model.
    """
    if type(ticks) is not int or not 0 <= ticks <= 2:
        raise ValueError("P0 supports only 0, 1, or 2 physical reference ticks")
    actual, velocity, goal, limits, batch, joints = _inputs(actual_q, actual_dq, goal_q, limits)
    budgets = np.full(batch, ticks, dtype=int) if tick_budget is None else np.asarray(tick_budget)
    if budgets.dtype.kind == 'b':
        raise ValueError("boolean is not an integer physical tick budget")
    if budgets.ndim == 0:
        budgets = np.full(batch, budgets.item())
    if budgets.shape != (batch,) or not np.isfinite(budgets).all() or np.any(budgets != np.floor(budgets)) or np.any(budgets < 0):
        raise ValueError("nonnegative integer tick_budget[B] required")
    budgets = np.minimum(budgets.astype(int), ticks)
    references = {key: np.full((batch, ticks, joints), np.nan) for key in ("q_ref", "dq_ref", "ddq_ref")}
    applied = np.zeros((batch, ticks), dtype=bool)
    codes = np.zeros(batch, dtype=np.int8)
    failure_step = np.full(batch, -1, dtype=int)
    active = np.ones(batch, dtype=bool)
    changed = reseeded = np.zeros(batch, dtype=bool)
    current_h = normalize_h(h, batch, joints)
    for step in range(max(1, ticks)):
        result = reference_step(actual, velocity, goal, current_h, limits,
            frequency=frequency, active_mask=active,
            apply_mask=(step < budgets) if ticks else np.zeros(batch, dtype=bool))
        current_h = result["next_h"]
        if step == 0:
            changed, reseeded = result["target_changed"], result["reference_reseeded"]
        failed = active & ~result["feasible"]
        codes[failed], failure_step[failed] = result["failure_code"][failed], step
        active &= ~failed
        if step < ticks:
            applied[:, step] = result["applied"]
            for key in references:
                references[key][result["applied"], step] = result[key][result["applied"]]
    return dict(next_h=current_h, reference_positions=references["q_ref"],
                reference_velocities=references["dq_ref"], reference_accelerations=references["ddq_ref"],
                applied_mask=applied, executed_ticks=applied.sum(axis=-1), feasible=codes == 0,
                failure_code=codes, failure_step=failure_step, target_changed=changed,
                reference_reseeded=reseeded, requested_ticks=ticks,
                generated_duration=applied.sum(axis=-1) * DT,
                actual_state_prediction_performed=False, native_steps=0)
