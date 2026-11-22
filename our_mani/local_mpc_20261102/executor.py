"""Counted 100 ms execution, independent of the frozen PhysMani executors.

Only ``backend.tick`` may advance physics.  The backend owns native grasp /
release semantics, native task outcomes, and the simulation budget.  This file
does not import PyRep or change robot motor / task physics parameters.
"""
from dataclasses import asdict, dataclass, field
from math import isfinite, sqrt
from time import perf_counter
from typing import Any, Mapping, Optional, Protocol, Sequence


def _vector(values, name, size=None):
    result = tuple(float(value) for value in values)
    if (size is not None and len(result) != size) or not result or not all(map(isfinite, result)):
        raise ValueError("finite " + name + " with the expected shape required")
    return result


def _per_joint(values, size, name):
    try:
        result = _vector(values, name, size)
    except TypeError:
        result = (float(values),) * size
    if not all(isfinite(value) and value > 0 for value in result):
        raise ValueError("positive finite " + name + " required")
    return result


class ExecutionBackend(Protocol):
    def get_joint_positions(self) -> Sequence[float]: ...
    def get_joint_velocities(self) -> Sequence[float]: ...
    def solve_ik(self, position, quaternion) -> Sequence[float]: ...
    def set_joint_target_positions(self, positions) -> None: ...
    def actuate_gripper(self, opening: float) -> bool: ...
    def tick(self, phase: str) -> None: ...
    def clock(self) -> float: ...
    def remaining_ticks(self) -> int: ...
    def terminal(self) -> bool: ...


@dataclass(frozen=True)
class JointLimits:
    """Native position / speed limits plus explicit controller reference limits.

    ``acceleration``, ``jerk``, and ``reference_velocity_cap`` constrain our
    generated references; they are not claims about native motor physics.
    The author RML's scalar 3 / 12 / 1000 parameters are separately reported.
    Pass the native arrays and their source via ``provenance``.
    """
    lower: Sequence[float]
    upper: Sequence[float]
    velocity: Sequence[float]
    acceleration: Any = 12.0
    jerk: Any = 1000.0
    reference_velocity_cap: Any = 3.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        lower = _vector(self.lower, "native lower joint limits")
        upper = _vector(self.upper, "native upper joint limits", len(lower))
        velocity = _per_joint(self.velocity, len(lower), "native joint velocity limits")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("ordered native joint position bounds required")
        for name, values in (("lower", lower), ("upper", upper), ("velocity", velocity)):
            object.__setattr__(self, name, values)
        for name in ("acceleration", "jerk", "reference_velocity_cap"):
            object.__setattr__(self, name, _per_joint(getattr(self, name), len(lower), name))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def effective_velocity(self):
        return tuple(min(a, b) for a, b in zip(self.velocity, self.reference_velocity_cap))

    def receipt(self):
        return dict(native_joint_position_lower=list(self.lower),
                    native_joint_position_upper=list(self.upper),
                    native_joint_velocity_limits=list(self.velocity),
                    controller_reference_velocity_cap=list(self.reference_velocity_cap),
                    effective_reference_velocity_limits=list(self.effective_velocity),
                    controller_reference_acceleration_limits=list(self.acceleration),
                    controller_reference_jerk_limits=list(self.jerk),
                    reference_jerk_interpretation="acceleration change / native control tick; not a native physical jerk guarantee",
                    controller_limits_are_native_physical_parameters=False,
                    author_RML_path_scalar_parameters=dict(max_velocity=3.0,
                        max_acceleration=12.0, max_jerk=1000.0,
                        interpretation="source path parameterization; not per-joint physical limits"),
                    provenance=dict(self.provenance))


@dataclass(frozen=True)
class ExecutionConfig:
    physics_dt: float = 0.05
    control_ticks: int = 2
    reference_frequency: float = 6.0
    clock_tolerance: float = 2e-5

    def __post_init__(self):
        if self.physics_dt != 0.05 or type(self.control_ticks) is not int or self.control_ticks != 2:
            raise ValueError("this protocol requires two native 50 ms ticks per control cycle")
        if not isfinite(self.reference_frequency) or self.reference_frequency <= 0:
            raise ValueError("positive finite reference frequency required")
        if not isfinite(self.clock_tolerance) or not 0 < self.clock_tolerance < self.physics_dt / 2:
            raise ValueError("bounded positive clock tolerance required")


@dataclass(frozen=True)
class PoseTarget:
    position: Sequence[float]
    quaternion: Sequence[float]  # xyzw, world frame; backend implements its IK frame.
    opening: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "position", _vector(self.position, "target position", 3))
        quat = _vector(self.quaternion, "xyzw target quaternion", 4)
        norm = sqrt(sum(value * value for value in quat))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("unit xyzw target quaternion required")
        quat = tuple(value / norm for value in quat)
        if next((value for value in reversed(quat) if abs(value) > 1e-12), 1.0) < 0:
            quat = tuple(-value for value in quat)
        object.__setattr__(self, "quaternion", quat)
        if self.opening is not None:
            _opening(self.opening)


def _opening(value):
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite gripper opening in [0, 1] required")
    return float(value >= 0.5)


@dataclass
class ExecutionReceipt:
    command_id: int
    phase: str
    requested_ticks: int = 2
    executed_ticks: int = 0
    tick_calls_attempted: int = 0
    tick_calls_completed: int = 0
    start_clock: float = 0.0
    end_clock: float = 0.0
    requested_duration: float = 0.1
    executed_duration: float = 0.0
    physical_ticks_observed: Optional[int] = 0
    unexpected_physics_ticks: Optional[int] = 0
    wall_seconds: float = 0.0
    ik_wall_seconds: float = 0.0
    planning_wall_seconds: float = 0.0
    target_changed: bool = False
    reference_reseeded: bool = False
    initial_measured_positions: tuple = ()
    initial_measured_velocities: tuple = ()
    target_joints: tuple = ()
    gripper_target: float = 1.0
    gripper_done: Optional[bool] = None
    status: str = "running"
    truncated_reason: Optional[str] = None
    error: Optional[str] = None
    limits: dict = field(default_factory=dict)
    ticks: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ClockInvariantError(RuntimeError):
    pass


class ReferenceLimitError(RuntimeError):
    pass


class ExecutionFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class ShortHorizonExecutor:
    """Persist references / gripper state, apply at most two synchronous ticks.

    A changed target seeds reference position and velocity from measured state.
    The last reference acceleration is retained, so small retargets do not
    restart at zero velocity.  An unchanged target continues the prior reference.
    Reference and measured state are logged separately on every physical tick.
    """
    def __init__(self, backend: ExecutionBackend, limits: JointLimits,
                 config: Optional[ExecutionConfig] = None, *, initial_gripper_opening=1.0):
        self.backend, self.limits = backend, limits
        self.config = config or ExecutionConfig()
        self.gripper_target = _opening(initial_gripper_opening)
        self.gripper_done = None
        self._goal = self._q = self._v = self._a = None
        self._pose_key = self._pose_joints = None
        self._command_id = 0
        self._reference_clock = None
        self.last_receipt = None

    def state(self):
        """Snapshot actuator planning state; these references are not measurements."""
        return dict(reference_valid=self._goal is not None,
                    reference_positions=self._q, reference_velocities=self._v,
                    reference_accelerations=self._a, target_joints=self._goal,
                    reference_clock=self._reference_clock,
                    gripper_target=self.gripper_target, gripper_done=self.gripper_done,
                    reference_is_measured_state=False)

    def _clock(self):
        clock = float(self.backend.clock())
        if not isfinite(clock):
            raise ClockInvariantError("backend clock must be finite")
        return clock

    def _no_step(self, operation, function):
        before = self._clock()
        try:
            return function()
        finally:
            if abs(self._clock() - before) > self.config.clock_tolerance:
                raise ClockInvariantError(operation + " advanced physics outside tick")

    def _state(self):
        size = len(self.limits.lower)
        return self._no_step("joint state read", lambda: (
            _vector(self.backend.get_joint_positions(), "measured joints", size),
            _vector(self.backend.get_joint_velocities(), "measured joint velocities", size)))

    def _validate_goal(self, values):
        goal = _vector(values, "joint target", len(self.limits.lower))
        if any(not lo <= value <= hi for value, lo, hi in zip(goal, self.limits.lower, self.limits.upper)):
            raise ValueError("joint target outside native position limits")
        return goal

    def _next_reference(self):
        dt, omega = self.config.physics_dt, self.config.reference_frequency
        positions, velocities, accelerations = [], [], []
        for q, v, a, goal, lo, hi, vmax, amax, jmax in zip(
                self._q, self._v, self._a, self._goal, self.limits.lower,
                self.limits.upper, self.limits.effective_velocity,
                self.limits.acceleration, self.limits.jerk):
            # Piecewise constant acceleration; jerk bounds its change per tick.
            # Intersect all constraints before choosing a reference, never clip
            # position after integration (which would break derivative bounds).
            low = max(-amax, a - jmax * dt, (-vmax - v) / dt,
                      2 * (lo - q - v * dt) / (dt * dt))
            high = min(amax, a + jmax * dt, (vmax - v) / dt,
                       2 * (hi - q - v * dt) / (dt * dt))
            if low > high + 1e-9:
                raise ReferenceLimitError("measured/reference state has no feasible bounded next tick")
            desired = omega * omega * (goal - q) - 2 * omega * v
            acceleration = min(high, max(low, desired))
            positions.append(q + v * dt + 0.5 * acceleration * dt * dt)
            velocities.append(v + acceleration * dt)
            accelerations.append(acceleration)
        return tuple(positions), tuple(velocities), tuple(accelerations)

    def execute(self, target: PoseTarget, *, phase="local_mpc"):
        if not isinstance(target, PoseTarget):
            raise TypeError("PoseTarget required")
        key = (target.position, target.quaternion)

        def resolve(receipt):
            if key == self._pose_key:
                return self._pose_joints
            started = perf_counter()
            try:
                joints = self._no_step("target IK", lambda: self.backend.solve_ik(
                    target.position, target.quaternion))
                joints = self._validate_goal(joints)
            finally:
                receipt.ik_wall_seconds += perf_counter() - started
            self._pose_key, self._pose_joints = key, joints
            return joints

        return self._execute(resolve, target.opening, phase)

    def execute_joint_target(self, joints, *, opening=None, phase="local_mpc"):
        goal = self._validate_goal(joints)
        # A direct command invalidates the pose cache, not the persistent state.
        self._pose_key = self._pose_joints = None
        return self._execute(lambda receipt: goal, opening, phase)

    def _execute(self, resolve, opening, phase):
        requested_opening = self.gripper_target if opening is None else _opening(opening)
        self._command_id += 1
        receipt = ExecutionReceipt(self._command_id, str(phase), limits=self.limits.receipt())
        self.last_receipt = receipt
        started = perf_counter()
        receipt.start_clock = self._clock()
        try:
            planning_started = perf_counter()
            try:
                goal = self._validate_goal(resolve(receipt))
                measured_q, measured_v = self._state()
                receipt.initial_measured_positions = measured_q
                receipt.initial_measured_velocities = measured_v
                changed = self._goal is None or any(abs(a - b) > 1e-10 for a, b in zip(goal, self._goal))
                receipt.target_changed = changed
                if changed:
                    if any(not lo - 1e-8 <= q <= hi + 1e-8 for q, lo, hi in zip(
                            measured_q, self.limits.lower, self.limits.upper)):
                        raise ReferenceLimitError("measured start outside native joint bounds")
                    self._q, self._v = measured_q, measured_v
                    self._a = self._a or (0.0,) * len(goal)
                    receipt.reference_reseeded = True
                self._goal = goal
                self.gripper_target = requested_opening
                receipt.target_joints = goal
                receipt.gripper_target = self.gripper_target
            finally:
                receipt.planning_wall_seconds += perf_counter() - planning_started
            for index in range(self.config.control_ticks):
                if self._no_step("terminal query", self.backend.terminal):
                    receipt.truncated_reason = "terminal"
                    break
                remaining = self._no_step("budget query", self.backend.remaining_ticks)
                if isinstance(remaining, bool) or int(remaining) != remaining or remaining < 0:
                    raise ValueError("nonnegative integer remaining tick budget required")
                if remaining == 0:
                    receipt.truncated_reason = "budget"
                    break
                measured_q, measured_v = self._state()
                q, v, a = self._next_reference()
                row = dict(index=index, clock_before=self._clock(),
                           measured_positions_before=measured_q, measured_velocities_before=measured_v,
                           reference_positions=q, reference_velocities=v, reference_accelerations=a,
                           gripper_target=self.gripper_target, tick_completed=False)
                receipt.ticks.append(row)
                self._no_step("arm target application", lambda: self.backend.set_joint_target_positions(q))
                self.gripper_done = bool(self._no_step("incremental gripper application",
                    lambda: self.backend.actuate_gripper(self.gripper_target)))
                row["gripper_done"] = self.gripper_done
                receipt.tick_calls_attempted += 1
                self.backend.tick(str(phase))
                receipt.tick_calls_completed += 1
                row["clock_after"] = self._clock()
                if abs(row["clock_after"] - row["clock_before"] - self.config.physics_dt) > self.config.clock_tolerance:
                    raise ClockInvariantError("tick must advance exactly one native 50 ms step")
                receipt.executed_ticks += 1
                row["tick_completed"] = True
                self._q, self._v, self._a = q, v, a
                self._reference_clock = row["clock_after"]
                actual_q, actual_v = self._state()
                row.update(measured_positions_after=actual_q, measured_velocities_after=actual_v,
                           tracking_error_positions=tuple(actual - ref for actual, ref in zip(actual_q, q)),
                           maximum_tracking_error_rad=max(abs(actual - ref) for actual, ref in zip(actual_q, q)))
            receipt.status = "completed" if receipt.executed_ticks == self.config.control_ticks else "truncated"
            return receipt
        except Exception as error:
            receipt.status, receipt.error = "failed", type(error).__name__ + ": " + str(error)
            # A backend failure may have happened after moving.  Force the next
            # command to read actual q/qdot rather than resume a stale reference.
            self._goal = None
            raise ExecutionFailure(receipt.error, receipt) from error
        finally:
            receipt.end_clock = self._clock()
            receipt.executed_duration = receipt.end_clock - receipt.start_clock
            observed = round(receipt.executed_duration / self.config.physics_dt)
            if observed >= 0 and abs(receipt.executed_duration - observed * self.config.physics_dt) <= self.config.clock_tolerance:
                receipt.physical_ticks_observed = observed
                receipt.unexpected_physics_ticks = observed - receipt.executed_ticks
            else:
                receipt.physical_ticks_observed = receipt.unexpected_physics_ticks = None
            receipt.gripper_done = self.gripper_done
            receipt.wall_seconds = perf_counter() - started
