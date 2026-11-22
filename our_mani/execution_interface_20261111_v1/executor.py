"""Single-clock execution with explicit goal/trajectory semantics.

The goal mode reuses the frozen MPC's pure reference kernel, so the planner
model and real executor have the same reference-generation implementation.
Trajectory mode validates the complete supplied prefix before the first write
and sends its position references unchanged. Both use the same counted tick,
native position servo and native gripper APIs. No actual q is teleported.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
from time import perf_counter

import numpy as np

from ..local_mpc_20261102.executor import (
    ClockInvariantError, ExecutionFailure, JointLimits, PoseTarget, ReferenceLimitError,
)
from ..local_mpc_p2_20261103.reference import reference_step
from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from .commands import JointGoal, JointTrajectory, vector, opening, names


@dataclass(frozen=True)
class ExecutionConfig:
    physics_dt: float = .05
    control_ticks: int = 2
    reference_frequency: float = 6.
    clock_tolerance: float = 2e-5

    def __post_init__(self):
        if self.physics_dt != .05 or type(self.control_ticks) is not int or self.control_ticks not in (1, 2):
            raise ValueError("native 50ms physics and one/two-tick execution required")
        if not isfinite(self.reference_frequency) or self.reference_frequency <= 0:
            raise ValueError("positive finite reference frequency required")
        if not isfinite(self.clock_tolerance) or not 0 < self.clock_tolerance < .025:
            raise ValueError("bounded positive clock tolerance required")


@dataclass
class ExecutionReceipt:
    command_id: int
    phase: str
    command_mode: str
    requested_ticks: int
    requested_duration: float
    start_clock: float
    end_clock: float
    executed_ticks: int = 0
    tick_calls_attempted: int = 0
    tick_calls_completed: int = 0
    executed_duration: float = 0.
    physical_ticks_observed: object = 0
    unexpected_physics_ticks: object = 0
    target_changed: bool = False
    reference_reseeded: bool = False
    target_joints: tuple = ()
    initial_measured_positions: tuple = ()
    initial_measured_velocities: tuple = ()
    gripper_target: float = 1.
    gripper_done: object = None
    status: str = "running"
    truncated_reason: object = None
    error: object = None
    wall_seconds: float = 0.
    planning_wall_seconds: float = 0.
    ik_wall_seconds: float = 0.
    velocity_reference_actuated: bool = False
    trajectory_reshaped: bool = False
    planner_contract: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    ticks: list = field(default_factory=list)
    feedback: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class PhysManiExecutor:
    """Structural backend API: no concrete observer/robot class is required.

    Backend commands/read methods must not advance simulation. Only tick()
    advances one actual 50ms physics/task step. Optional same-frame contact and
    task-judge channels are reported separately; a terminal flag is not success.
    A failed dispatch faults this executor; construct a fresh one after the
    backend has been explicitly recovered, never replay a partially sent chunk.
    """
    def __init__(self, backend, limits: JointLimits, config=None, *,
                 initial_gripper_opening=1., joint_names=None):
        required = ("get_joint_positions", "get_joint_velocities", "set_joint_target_positions",
                    "actuate_gripper", "tick", "clock", "remaining_ticks", "terminal")
        missing = [n for n in required if not callable(getattr(backend, n, None))]
        if missing:
            raise TypeError("backend methods missing: " + ", ".join(missing))
        if not isinstance(limits, JointLimits):
            raise TypeError("native JointLimits are required")
        self.backend, self.limits = backend, limits
        self.config = config or ExecutionConfig()
        self.joint_names = names(joint_names)
        if self.joint_names is not None and len(self.joint_names) != len(limits.lower):
            raise ValueError("joint names must match native degrees of freedom")
        self.gripper_target = opening(initial_gripper_opening)
        if self.gripper_target is None:
            raise ValueError("initial gripper intent must be explicit")
        self.gripper_done = None
        self._gripper_driver = CompletedGripperBackend(backend)
        self._h = {}
        self._mode = "idle"
        self._reference_clock = None
        self._last_direct = None
        self._command_id = 0
        self._fault = None
        self.last_receipt = None

    def state(self):
        h = self._h
        def row(key):
            return None if key not in h else tuple(np.asarray(h[key])[0])
        return dict(reference_valid=bool(np.asarray(h.get("valid", [False]))[0]),
                    reference_positions=row("q_ref"), reference_velocities=row("dq_ref"),
                    reference_accelerations=row("ddq_ref"), target_joints=row("goal_q"),
                    reference_clock=self._reference_clock, gripper_target=self.gripper_target,
                    gripper_done=self.gripper_done, reference_is_measured_state=False,
                    command_mode=self._mode, faulted=self._fault is not None)

    def capabilities(self):
        return dict(schema="PhysMani_execution_interface_20261111_v1",
                    physics_dt=.05, degrees_of_freedom=len(self.limits.lower),
                    joint_names=self.joint_names, position_units="absolute_radians",
                    motor_interface="native_joint_position_servo",
                    command_modes=["joint_goal", "joint_trajectory"],
                    maximum_ticks_per_command=2, velocity_command_supported=False,
                    gripper_commands={"CLOSE": 0., "OPEN": 1., "RETAIN": None},
                    trajectory_interpolation="caller_supplies_50ms_samples_no_reshaping",
                    task_success_source="original_native_task_only")

    def read_feedback(self):
        """Read measured state and optional Native outcomes without stepping."""
        return self._feedback()

    def _clock(self):
        value = float(self.backend.clock())
        if not isfinite(value):
            raise ClockInvariantError("backend clock must be finite")
        return value

    def _no_step(self, name, call):
        before = self._clock()
        try:
            return call()
        finally:
            if abs(self._clock() - before) > self.config.clock_tolerance:
                raise ClockInvariantError(name + " advanced physics outside tick")

    def _measure(self):
        size = len(self.limits.lower)
        return self._no_step("robot state read", lambda: (
            vector(self.backend.get_joint_positions(), "measured q", size),
            vector(self.backend.get_joint_velocities(), "measured dq", size)))

    def _validate_positions(self, value):
        q = vector(value, "absolute joint positions", len(self.limits.lower))
        if any(v < lo or v > hi for v, lo, hi in zip(q, self.limits.lower, self.limits.upper)):
            raise ValueError("joint positions outside native bounds; commands are never clipped")
        return q

    def _feedback(self):
        q, dq = self._measure()
        result = dict(timestamp=self._clock(), q=q, dq=dq,
                      gripper_requested=self.gripper_target, gripper_done=self.gripper_done,
                      held=None, native_task_success=None)
        contact = getattr(self.backend, "read_contact_feedback", None)
        if callable(contact):
            value = self._no_step("contact feedback", contact)
            if type(value.get("held")) is not bool or abs(float(value["timestamp"]) - result["timestamp"]) > self.config.clock_tolerance:
                raise ValueError("same-frame explicit Native membership required")
            result["held"] = value["held"]
        judge = getattr(self.backend, "native_task_judge_receipt", None)
        if callable(judge):
            value = self._no_step("native task judge", judge)
            success = value.get("native_task_success_latched")
            if type(success) is not bool:
                raise ValueError("native task success must be an explicit boolean")
            result.update(native_task_success=success, native_task_judge=deepcopy(value))
        return result

    def _trajectory_references(self, command, measured_q, measured_dq, now):
        # These are discrete reference differences, not motor velocity commands.
        dt = self.config.physics_dt
        if self._last_direct is not None and abs(self._last_direct[0] - now) <= self.config.clock_tolerance:
            _, q, v, a = self._last_direct
            q, v, a = np.asarray(q), np.asarray(v), np.asarray(a)
        else:
            q, v, a = np.asarray(measured_q), np.asarray(measured_dq), np.zeros(len(measured_q))
        refs = []
        for values in command.positions:
            p = np.asarray(self._validate_positions(values))
            speed = (p - q) / dt
            acceleration = (speed - v) / dt
            jerk = (acceleration - a) / dt
            for name, val, limit in (("velocity", speed, self.limits.effective_velocity),
                                     ("acceleration", acceleration, self.limits.acceleration),
                                     ("jerk", jerk, self.limits.jerk)):
                if np.any(np.abs(val) > np.asarray(limit) + 1e-8):
                    raise ReferenceLimitError("trajectory discrete " + name + " exceeds reference limits")
            refs.append((tuple(p), tuple(speed), tuple(acceleration)))
            q, v, a = p, speed, acceleration
        return refs

    def execute_command(self, command, *, phase="execution_interface", planner_contract=None):
        if self._fault is not None:
            raise RuntimeError("executor faulted; partial commands cannot be retried: " + self._fault)
        if not isinstance(command, (JointGoal, JointTrajectory)):
            raise TypeError("explicit JointGoal or JointTrajectory required")
        if not isinstance(phase, str) or not phase:
            raise ValueError("an explicit execution phase is required")
        if command.joint_names is not None and command.joint_names != self.joint_names:
            raise ValueError("joint order differs or backend joint names are undeclared")
        now = self._clock()
        expected = command.expected_time if isinstance(command, JointGoal) else command.start_time
        if expected is not None and abs(expected - now) > self.config.clock_tolerance:
            raise ValueError("stale/future command timestamp; references are never silently retimed")
        if self._reference_clock is not None and abs(now - self._reference_clock) > self.config.clock_tolerance:
            raise ClockInvariantError("simulation advanced outside this executor; stale reference state")
        mode = "joint_goal" if isinstance(command, JointGoal) else "joint_trajectory"
        count = command.ticks if isinstance(command, JointGoal) else len(command.positions)
        if isinstance(command, JointGoal):
            goal = self._validate_positions(command.positions)
        else:
            for point in command.positions:
                self._validate_positions(point)
            goal = tuple(command.positions[-1])
        self._command_id += 1
        receipt = ExecutionReceipt(self._command_id, phase, mode, count, count * .05, now, now,
            target_joints=goal, gripper_target=self.gripper_target,
            planner_contract=deepcopy(planner_contract or {}), limits=self.limits.receipt())
        self.last_receipt = receipt
        began = perf_counter()
        dispatch_started = False
        try:
            measured_q, measured_dq = self._measure()
            receipt.initial_measured_positions, receipt.initial_measured_velocities = measured_q, measured_dq
            # Validate all trajectory points before even the first arm/gripper write.
            refs = self._trajectory_references(command, measured_q, measured_dq, now) if mode == "joint_trajectory" else None
            request = self.gripper_target if command.opening is None else command.opening
            for index in range(count):
                if self._no_step("terminal query", self.backend.terminal):
                    receipt.truncated_reason = "terminal"
                    break
                remaining = self._no_step("budget query", self.backend.remaining_ticks)
                if isinstance(remaining, (bool, np.bool_)) or not isfinite(remaining) or int(remaining) != remaining or remaining < 0:
                    raise ValueError("remaining_ticks must be a nonnegative integer")
                if remaining == 0:
                    receipt.truncated_reason = "budget"
                    break
                q_actual, dq_actual = self._measure()
                planning_started = perf_counter()
                if mode == "joint_goal":
                    h = deepcopy(self._h)
                    if self._mode != "joint_goal":
                        h = {}  # An explicit mode switch reseeds from measured q/dq.
                    ref = reference_step(q_actual, dq_actual, goal, h, self.limits,
                                         frequency=self.config.reference_frequency)
                    if not ref["feasible"][0] or not ref["applied"][0]:
                        raise ReferenceLimitError("no feasible bounded goal reference; code=" + str(ref["failure_code"][0]))
                    p, v, a = (tuple(ref[k][0]) for k in ("q_ref", "dq_ref", "ddq_ref"))
                    next_h = ref["next_h"]
                    receipt.target_changed |= bool(ref["target_changed"][0])
                    receipt.reference_reseeded |= bool(ref["reference_reseeded"][0])
                else:
                    p, v, a = refs[index]
                    next_h = {}  # Trajectory differences are not legacy goal-kernel state.
                receipt.planning_wall_seconds += perf_counter() - planning_started
                row = dict(index=index, clock_before=self._clock(), measured_positions_before=q_actual,
                           measured_velocities_before=dq_actual, reference_positions=p,
                           reference_velocities=v, reference_accelerations=a,
                           derivative_semantics="goal_kernel" if mode == "joint_goal" else "sample_differences",
                           gripper_target=request, tick_completed=False)
                receipt.ticks.append(row)
                dispatch_started = True
                self._no_step("joint reference write", lambda: self.backend.set_joint_target_positions(p))
                done = self._no_step("gripper command", lambda: self._gripper_driver.actuate_gripper(request))
                if type(done) is not bool:
                    raise ValueError("gripper completion must be an explicit boolean")
                self.gripper_target, self.gripper_done = request, done
                receipt.gripper_target = request
                row["gripper_done"] = done
                receipt.tick_calls_attempted += 1
                self.backend.tick(phase)
                receipt.tick_calls_completed += 1
                row["clock_after"] = self._clock()
                if abs(row["clock_after"] - row["clock_before"] - .05) > self.config.clock_tolerance:
                    raise ClockInvariantError("tick must advance exactly one native 50ms step")
                receipt.executed_ticks += 1
                row["tick_completed"] = True
                self._h, self._mode = next_h, mode
                self._reference_clock = row["clock_after"]
                self._last_direct = (self._reference_clock, p, v, a)
                feedback = self._feedback()
                row.update(measured_positions_after=feedback["q"], measured_velocities_after=feedback["dq"],
                           tracking_error_positions=tuple(x-y for x, y in zip(feedback["q"], p)),
                           maximum_tracking_error_rad=max(abs(x-y) for x, y in zip(feedback["q"], p)),
                           held=feedback["held"], native_task_success=feedback["native_task_success"])
                receipt.feedback = feedback
                if feedback["native_task_success"] is True:
                    receipt.truncated_reason = "native_task_success"
                    break
            receipt.status = "completed" if receipt.executed_ticks == count else "truncated"
            return receipt
        except Exception as error:
            receipt.status, receipt.error = "failed", type(error).__name__ + ": " + str(error)
            if dispatch_started or abs(self._clock() - now) > self.config.clock_tolerance:
                self._fault = receipt.error
                self._h = {}
            raise ExecutionFailure(receipt.error, receipt) from error
        finally:
            receipt.end_clock = self._clock()
            receipt.executed_duration = receipt.end_clock - receipt.start_clock
            observed = round(receipt.executed_duration / .05)
            if observed >= 0 and abs(receipt.executed_duration - observed * .05) <= self.config.clock_tolerance:
                receipt.physical_ticks_observed = observed
                receipt.unexpected_physics_ticks = observed - receipt.executed_ticks
            else:
                receipt.physical_ticks_observed = receipt.unexpected_physics_ticks = None
            receipt.gripper_done = self.gripper_done
            receipt.wall_seconds = perf_counter() - began

    def execute_joint_target(self, joints, *, opening=None, phase="local_mpc", ticks=None, expected_time=None):
        """Compatibility entry: joints are a destination, not a trajectory sample."""
        return self.execute_command(JointGoal(joints, opening, self.config.control_ticks if ticks is None else ticks,
                                              expected_time), phase=phase)

    def execute(self, target, *, phase="local_mpc"):
        """Compatibility pose destination: explicit world-frame unit xyzw PoseTarget."""
        if not isinstance(target, PoseTarget):
            raise TypeError("world-frame PoseTarget with xyzw quaternion required")
        solve = getattr(self.backend, "solve_ik", None)
        if not callable(solve):
            raise TypeError("pose destinations require a backend solve_ik method")
        started = perf_counter()
        goal = self._no_step("pose IK", lambda: solve(target.position, target.quaternion))
        elapsed = perf_counter() - started
        if goal is None:
            raise ValueError("pose IK failed; no action or physical step was issued")
        result = self.execute_joint_target(goal, opening=target.opening, phase=phase)
        result.ik_wall_seconds = elapsed
        return result

    def execute_planner_action(self, action, *, velocity_reference_role=None, phase="planner", expected_time=None):
        """Adapt the current CEM action without silently claiming twist actuation.

        tcp_velocity_reference/angular_velocity_reference are ONLY permitted
        with an explicit objective_only declaration. The CEM currently uses
        these as tracking objectives. An external solver producing per-tick
        positions must use JointTrajectory instead of this destination API.
        """
        if not isinstance(action, dict) or "joint_goal" not in action or "opening" not in action:
            raise ValueError("planner action requires joint_goal and binary opening")
        unsupported = set(action) & {"joint_velocity", "joint_velocities", "joint_target_velocities",
                                     "joint_goal_velocity", "dq_command", "dq", "torque", "effort", "velocity_command"}
        if unsupported:
            raise ValueError("native position interface does not actuate: " + ", ".join(sorted(unsupported)))
        references = {}
        for key in ("tcp_velocity_reference", "angular_velocity_reference"):
            if key in action:
                references[key] = vector(action[key], key, 3)
        if references and velocity_reference_role != "objective_only":
            raise ValueError("declare velocity_reference_role='objective_only'; these fields are not motor commands")
        if "dt" in action:
            dt = float(action["dt"])
            if not isfinite(dt) or abs(dt - self.config.control_ticks * .05) > 1e-9:
                raise ValueError("planner action duration differs from execution contract")
        return self.execute_command(JointGoal(action["joint_goal"], action["opening"], self.config.control_ticks,
                                              expected_time), phase=phase,
            planner_contract=dict(action_semantics="smoothed_joint_destination", velocity_reference_role=velocity_reference_role,
                                  objective_references=references, velocity_reference_actuated=False))
