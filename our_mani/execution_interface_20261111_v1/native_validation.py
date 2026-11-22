"""Real Native command-boundary and current Cube pipeline checks, TRAIN only."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from ..local_mpc_20261102.executor import JointLimits, ExecutionFailure
from ..local_mpc_20261102.records import append, write_new
from .commands import JointGoal, JointTrajectory
from .executor import PhysManiExecutor


class AuditedNativeBackend:
    """Observe the actual servo target after each write; do not alter dynamics."""
    def __init__(self, backend, output):
        self.backend, self.output = backend, Path(output)
        self.commands, self.ticks = [], []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def set_joint_target_positions(self, positions):
        before = self.backend.clock()
        self.backend.set_joint_target_positions(positions)
        actual_target = np.asarray(self.backend.arm.get_joint_target_positions())
        error = float(np.max(np.abs(actual_target - positions)))
        if error > 1e-6 or self.backend.clock() != before:
            raise RuntimeError("actual Native position-servo target differs or write advanced physics")
        row = dict(timestamp=before, requested=list(positions), native_servo_target=actual_target,
                   maximum_difference_rad=error, physical_steps=0)
        self.commands.append(row)
        append(self.output / "native_commands.jsonl", row)

    def tick(self, phase):
        state = self.backend.tick(phase)
        row = dict(timestamp=self.backend.clock(), q=self.backend.get_joint_positions(),
                   dq=self.backend.get_joint_velocities(), phase=phase,
                   judge=self.backend.native_task_judge_receipt())
        self.ticks.append(row)
        append(self.output / "native_feedback.jsonl", row)
        return state


def make_executor(backend, config):
    native = backend.joint_limits()
    limits = JointLimits(native["lower"], native["upper"], native["velocity"],
        acceleration=config.get("joint_reference_acceleration", 12.),
        jerk=config.get("joint_reference_jerk", 1000.), provenance=native.get("provenance", {}))
    joint_names = tuple(joint.get_name() for joint in backend.arm.joints)
    return PhysManiExecutor(backend, limits, initial_gripper_opening=1., joint_names=joint_names)


def boundary_checks(backend, executor, output):
    """Exercise bounded motion near TRAIN home, binary gripper, and stale input."""
    output = Path(output)
    receipts = []
    q0 = np.asarray(backend.get_joint_positions())
    points = np.broadcast_to(q0, (2, 7)).copy()
    points[:, 0] += [.002, .004]
    command = JointTrajectory(backend.clock(), points, [.05, .1], opening=1., joint_names=executor.joint_names)
    result = executor.execute_command(command, phase="interface_direct_trajectory")
    receipts.append(result)
    np.testing.assert_array_equal([r["requested"] for r in backend.commands], points)
    # No bad second point may leave the first point or a gripper side effect.
    at, writes = backend.clock(), len(backend.commands)
    bad = np.broadcast_to(np.asarray(backend.get_joint_positions()), (2, 7)).copy()
    bad[:, 0] += [.001, .8]
    try:
        executor.execute_command(JointTrajectory(at, bad, [.05, .1], opening=0.))
    except ExecutionFailure:
        pass
    else:
        raise AssertionError("invalid trajectory unexpectedly executed")
    if backend.clock() != at or len(backend.commands) != writes:
        raise AssertionError("trajectory preflight rejection had side effects")
    try:
        executor.execute_command(command)
    except ValueError:
        pass
    else:
        raise AssertionError("stale trajectory was silently retimed")
    target = np.asarray(backend.get_joint_positions())
    target[0] += .025
    for opening in (0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.):
        receipts.append(executor.execute_command(JointGoal(target, opening=opening, expected_time=backend.clock(),
                                                          joint_names=executor.joint_names), phase="interface_goal_and_gripper"))
    receipts.append(executor.execute_joint_target(target, ticks=1, expected_time=backend.clock(), phase="interface_single_tick"))
    for result in receipts:
        if result.unexpected_physics_ticks != 0:
            raise AssertionError("unexpected physics during boundary checks")
        append(output / "receipts.jsonl", result)
    return dict(complete=True, actual_ticks=sum(r.executed_ticks for r in receipts),
        servo_writes=len(backend.commands), trajectory_points_dispatched_unchanged=True,
        bad_prefix_rejected_before_any_write=True, stale_trajectory_rejected=True,
        single_tick_executed=receipts[-1].executed_ticks == 1,
        maximum_servo_command_difference_rad=max(r["maximum_difference_rad"] for r in backend.commands),
        maximum_actual_tracking_error_rad=max(row["maximum_tracking_error_rad"] for r in receipts for row in r.ticks),
        final_feedback=executor.read_feedback(), physical_accuracy_claimed=False,
        task_success_claimed=False, Native_task_or_physics_modified=False)


def run_case(spec, output, stop_requested):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = None
    closed, error, result = False, None, None
    try:
        from ..cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
        # Retain partial construction for shutdown if initialization raises.
        backend = NativeContactCubeBackendV1.__new__(NativeContactCubeBackendV1)
        NativeContactCubeBackendV1.__init__(backend, spec["task"], spec["calibration_episode"],
            seed=spec["seed"], variation=0, budget_ticks=160,
            gripper_velocity=spec["config"]["gripper_velocity"], execute_native=True)
        write_new(output / "initial_snapshot.json", backend.initial_snapshot)
        if spec["kind"] == "pipeline":
            from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
            from ..cube_native_contact_feedback_live_20261104_v1.observer import ObservedNativeContactCubeBackendV1
            root = Path(__file__).resolve().parents[2]
            c = spec["config"]
            backend = CompletedGripperBackend(backend)
            backend = ObservedNativeContactCubeBackendV1(backend, output / "public_observer",
                candidate_mode="matched_candidate_with_corrected_forecast",
                checkpoint=root / c["candidate_transformer_checkpoint"],
                checkpoint_sha256=c["candidate_transformer_checkpoint_sha256"],
                forecast_checkpoint=root / c["corrected_forecast_checkpoint"],
                forecast_checkpoint_sha256=c["corrected_forecast_checkpoint_sha256"],
                refiner_checkpoint=root / c["position_refiner_checkpoint"],
                refiner_checkpoint_sha256=c["position_refiner_checkpoint_sha256"], save_frames=False)
            backend.sample()
        backend = AuditedNativeBackend(backend, output)
        executor = make_executor(backend, spec["config"])
        write_new(output / "interface_capabilities.json", executor.capabilities())
        if spec["kind"] == "boundary":
            result = boundary_checks(backend, executor, output)
        else:
            from .producer import run
            result = run(backend, executor, output, spec["config"], stop_requested=stop_requested)
            result.update(actual_ticks=len(backend.ticks), servo_writes=len(backend.commands),
                maximum_servo_command_difference_rad=max((r["maximum_difference_rad"] for r in backend.commands), default=0.),
                Native_task_or_physics_modified=False)
        if len(backend.commands) != len(backend.ticks):
            raise AssertionError("servo write / actual tick counts differ")
        result.update(task=spec["task"], seed=spec["seed"], kind=spec["kind"],
                      dataset_role="generated_TRAIN_development", formal_VAL_TEST=False)
        write_new(output / "result.json", result)
        print(json.dumps(dict(case=spec["id"], complete=result["complete"], actual_ticks=result["actual_ticks"],
                              episode_success=result.get("episode_success")), ensure_ascii=False), flush=True)
        return result
    except BaseException as exception:
        error = type(exception).__name__ + ": " + str(exception)
        raise
    finally:
        if backend is not None:
            backend.close()
            closed = bool(getattr(backend, "_closed", False))
        write_new(output / "worker_exit.json", dict(native_closed=closed, error=error, result_recorded=result is not None))
