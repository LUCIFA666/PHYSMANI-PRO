"""Owned truth-only moving-block backend for the new local-MPC diagnostic.

This is deliberately not a public-input benchmark adapter. Native object and
attachment identities are permitted only in this diagnostic. No expert action,
waypoint, stage, target-state archive, validation or TEST package is loaded.
The optional TRAIN HDF5 calibration is consumed by the inherited simulator only
for initial home joints and camera calibration. Native task/scene files are
never edited. ``tick`` alone advances one 50 ms physics/task step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping
import math
import time

import numpy as np

from .static_collision import check_static_arm_collision


DT = .05
TASK_NAME = "pick_moving_target_on_the_table"


class NativeBackendError(RuntimeError):
    pass


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"finite {name}[{size}] required")
    return result.copy()


def _quaternion(value, name="quaternion"):
    result = _vector(value, 4, name)
    if not np.isclose(np.linalg.norm(result), 1., atol=1e-5, rtol=1e-5):
        raise ValueError(f"unit xyzw {name} required")
    return result


def _angular_difference(previous, current, dt):
    """World-frame rotation increment from two observed xyzw quaternions."""
    p, q = _quaternion(previous), _quaternion(current)
    xyz = q[3] * -p[:3] + p[3] * q[:3] + np.cross(q[:3], -p[:3])
    w = float(q[3] * p[3] + np.dot(q[:3], p[:3]))
    if w < 0:
        xyz, w = -xyz, -w
    norm = float(np.linalg.norm(xyz))
    return np.zeros(3) if norm < 1e-12 else xyz * (2 * math.atan2(norm, w) / (norm * dt))


@dataclass(frozen=True)
class MovingBlockScenario:
    name: str
    initial_position: tuple[float, float, float]
    initial_velocity: tuple[float, float, float]
    switch_tick: int | None = None
    changed_velocity: tuple[float, float, float] | None = None
    initial_quaternion: tuple[float, float, float, float] = (0., 0., 0., 1.)
    goal_position: tuple[float, float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("explicit diagnostic scenario name required")
        _vector(self.initial_position, 3, "initial_position")
        _vector(self.initial_velocity, 3, "initial_velocity")
        _quaternion(self.initial_quaternion)
        if self.goal_position is not None:
            _vector(self.goal_position, 3, "goal_position")
        if (self.switch_tick is None) != (self.changed_velocity is None):
            raise ValueError("switch_tick and changed_velocity must be declared together")
        if self.switch_tick is not None:
            if type(self.switch_tick) is not int or self.switch_tick < 1:
                raise ValueError("positive integer switch_tick required")
            _vector(self.changed_velocity, 3, "changed_velocity")


class TruthNativeBackend:
    """One owned simulator with side-effect-checked state and IK interfaces.

    The constructor performs reset staging and finishes at control time zero,
    with the robot at TRAIN-calibrated home, not above the moving target.
    ``_raw``/``_physics_clock``/``_ik_errors`` are CPU-fixture injection points;
    ordinary construction lazily imports and launches the original simulator.
    """

    def __init__(self, calibration_episode, scenario: MovingBlockScenario | Mapping, *,
                 budget_ticks=200, seed=20261102, headless=True, gripper_velocity=.2,
                 preserve_native_height=False,
                 _raw=None, _physics_clock=None, _ik_errors=None):
        if type(budget_ticks) is not int or budget_ticks < 1:
            raise ValueError("positive integer physical budget required")
        if not math.isfinite(gripper_velocity) or gripper_velocity <= 0:
            raise ValueError("positive native gripper velocity required")
        self.budget_ticks, self.gripper_velocity = budget_ticks, float(gripper_velocity)
        self.preserve_native_height = bool(preserve_native_height)
        self.calibration_episode = str(calibration_episode)
        self.scenario = self._scenario(scenario)
        self._closed, self._fault, self._staged = False, None, False
        self.tick_records, self.motion_events = [], []
        self.last_ik_receipt = None
        self.last_collision_probe = None
        self._last_native_result = dict(success=False, terminate=False, terminal_failure=False)
        self._last_pose_sample = None
        if _raw is None:
            if not headless:
                raise ValueError("this diagnostic backend uses explicit headless native execution")
            calibration = Path(calibration_episode).resolve(strict=True)
            if "train_package_compressed" not in calibration.parts or calibration.name != "data.h5":
                raise ValueError("only an explicit TRAIN calibration HDF5 may supply home/camera fields")
            from our_mani.aligned.simulation import AlignedLocalSimulation, physics_time
            from pyrep.errors import ConfigurationError, IKError
            protocol = dict(dt=DT, simulation_seconds=budget_ticks * DT,
                            gripper_velocity=self.gripper_velocity, current_instance_masks=False)
            raw = AlignedLocalSimulation.__new__(AlignedLocalSimulation)
            try:
                AlignedLocalSimulation.__init__(raw, TASK_NAME, calibration_episode, protocol=protocol,
                                                seed=seed, servo_lead=1.)
            except BaseException:
                if hasattr(raw, "environment"):
                    raw.environment.shutdown()
                raise
            self.raw = raw
            self._physics_clock = physics_time
            self._ik_errors = (ConfigurationError, IKError)
        else:
            if _physics_clock is None or _ik_errors is None:
                raise ValueError("CPU fixture must supply its physics clock and recoverable IK error types")
            self.raw, self._physics_clock, self._ik_errors = _raw, _physics_clock, tuple(_ik_errors)
        self.arm, self.gripper, self.task = self.raw.arm, self.raw.gripper, self.raw.task
        try:
            self._limits = self._read_limits()
        except BaseException:
            self.close()
            raise
        self.provenance = dict(
            schema="local_mpc_truth_native_backend_20261102_v1", truth_diagnostic_only=True,
            official_public_input_evaluation=False, privileged_policy_state=True,
            expert_actions_or_stages_read=False, TEST_read=False,
            calibration_scope="TRAIN_home_joints_and_camera_calibration_only",
            calibration_episode=self.calibration_episode, task=TASK_NAME,
            physical_dt=DT, control_period_ticks=2, physical_budget_ticks=budget_ticks,
            native_grasp_semantics="original_proximity_gated_native_attachment_not_force_verified_friction_grasp",
            close_semantics="one_native_attempt_per_open_to_close_edge_continuous_actuation_each_requested_tick",
            first_native_success_is_terminal=False,
            scenario_motion_phase="native_task_evaluates_preincrement_time; segment_t0_is_task_time_minus_one_dt",
            native_source_files_edited=False,
        )
        try:
            self.reset_scenario(self.scenario)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _scenario(value):
        return value if isinstance(value, MovingBlockScenario) else MovingBlockScenario(**dict(value))

    def _read_limits(self):
        cyclic, intervals = self.arm.get_joint_intervals()
        intervals = np.asarray(intervals, dtype=np.float64)
        if intervals.shape != (7, 2) or np.asarray(cyclic).shape != (7,) or any(cyclic):
            raise NativeBackendError("this Panda backend requires seven finite noncyclic native joint intervals")
        lower, upper = intervals[:, 0], intervals[:, 0] + intervals[:, 1]
        velocity = _vector(self.arm.get_joint_upper_velocity_limits(), 7, "native_joint_velocity_limits")
        if not np.isfinite(intervals).all() or not (np.all(upper > lower) and np.all(velocity > 0)):
            raise NativeBackendError("invalid native joint bounds or velocity limits")
        return dict(lower=lower.copy(), upper=upper.copy(), velocity=velocity,
                    source="PyRep_get_joint_intervals_minimum_plus_range_and_get_joint_upper_velocity_limits",
                    controller_reference_acceleration=12., controller_reference_jerk=1000.,
                    reference_parameters_are_not_native_physical_limits=True)

    def joint_limits(self):
        return deepcopy(self._limits)

    def _clock_tuple(self):
        return (float(self._physics_clock()), float(self.task.t), int(self.task.step_id),
                int(self.raw.ledger.ticks) if hasattr(self.raw, "ledger") else None)

    def _assert_unchanged(self, before, operation):
        if self._clock_tuple() != before:
            self._fault = f"{operation} advanced physics/task/control time"
            raise NativeBackendError(self._fault)

    def _check_live_clock(self):
        if self._closed:
            raise NativeBackendError("closed native backend")
        physics, task_time, _, _ = self._clock_tuple()
        tolerance = max(2e-5, 2 * float(np.spacing(np.float32(max(1., physics)))))
        if (abs(physics - self._physics_origin - self.clock()) > tolerance
                or abs(task_time - self._task_origin - self.clock()) > 2e-5):
            self._fault = "physics/task/control clocks diverged"
            raise NativeBackendError(self._fault)

    def _can_command(self):
        self._check_live_clock()
        if self.terminal():
            raise NativeBackendError("physical budget exhausted or backend faulted")

    def reset_scenario(self, scenario=None):
        if self._closed:
            raise NativeBackendError("cannot reset a closed backend")
        if scenario is not None:
            self.scenario = self._scenario(scenario)
        if self.scenario.switch_tick is not None and self.scenario.switch_tick >= self.budget_ticks:
            raise ValueError("declared speed switch must lie inside the physical budget")
        self._staged, self._fault = False, None
        self.tick_records.clear(); self.motion_events.clear()
        self._switch_handled, self._last_pose_sample = False, None
        self._staging_before = self._clock_tuple()
        self.raw.task_environment.reset()
        self.native_default_target_pose = list(self.task.target_block.get_pose())
        if self.preserve_native_height:
            position = list(self.scenario.initial_position)
            position[2] = self.native_default_target_pose[2]
            self.scenario = replace(self.scenario, initial_position=tuple(position))
        self.task.disable_expert_plan()
        self.finish_home_reset()
        return deepcopy(self.initial_snapshot)

    def finish_home_reset(self):
        if self._staged:
            if self.raw.ledger.ticks:
                raise NativeBackendError("home reset forbidden after the first control tick")
            return deepcopy(self.staging_receipt)
        # disable_dynamics deliberately advances hidden reset-staging steps.
        # It is never used by solve_ik or during a controlled rollout.
        self.gripper.release()
        self.gripper.set_joint_positions([.04, .04], disable_dynamics=True)
        self.gripper.set_joint_target_positions([.04, .04])
        home = _vector(self.raw.home_joints, 7, "TRAIN home joints")
        self.arm.set_joint_positions(home.tolist(), disable_dynamics=True)
        self.arm.set_joint_target_positions(home.tolist())
        for _ in range(3):
            self.raw.pyrep.step()
        self.task.t, self.task.step_id = 0., 0
        pose = np.r_[self.scenario.initial_position, self.scenario.initial_quaternion]
        self.task.target_block.set_pose(pose.tolist(), reset_dynamics=True)
        self.task.target_state_list = [dict(x=np.asarray(self.scenario.initial_position, dtype=float),
                                          v=np.asarray(self.scenario.initial_velocity, dtype=float),
                                          a=np.zeros(3), t0=-DT)]
        if self.scenario.goal_position is not None:
            self.task.success_detector.set_position(list(self.scenario.goal_position))
        self.raw._start_alignment()
        self.raw.ledger.budget = self.budget_ticks * DT
        self.raw._requested_opening = 1.
        self.requested_opening = 1.
        self._physics_origin, self._task_origin = float(self._physics_clock()), float(self.task.t)
        self._staged = True
        self._sample_native_result()
        self.staging_receipt = dict(
            truth_diagnostic_only=True, excluded_from_control_budget=True,
            clock_before=self._staging_before, clock_after=self._clock_tuple(),
            raw_physics_seconds=float(self._physics_clock()) - self._staging_before[0],
            reset_home_joints=home.tolist(), controlled_ticks=0,
            disable_dynamics_used_only_during_staging=True, extra_raw_settle_steps=3,
            scenario=asdict(self.scenario), motion_segment_t0=-DT,
        )
        state = self.read_state()
        self.initial_snapshot = dict(state=state, scenario=asdict(self.scenario),
                                     native_default_target_pose_before_override=self.native_default_target_pose,
                                     native_height_preserved=self.preserve_native_height,
                                     staging=deepcopy(self.staging_receipt),
                                     native_bounds=self._geometry_snapshot(), joint_limits=self.joint_limits(),
                                     truth_diagnostic_only=True, official_public_input_evaluation=False)
        return deepcopy(self.staging_receipt)

    def _geometry_snapshot(self):
        target_bbox = _vector(self.task.target_block.get_bounding_box(), 6, "target local bounding box")
        gripper_bbox = _vector(self.gripper.get_model_bounding_box(), 6, "gripper model bounding box")
        return dict(task_area=_vector(self.task.area, 6, "native task area").tolist(),
                    task_area_order="xmin_ymin_zmin_xmax_ymax_zmax",
                    target_local_bbox=target_bbox.tolist(), gripper_model_bbox=gripper_bbox.tolist(),
                    bbox_order="xmin_xmax_ymin_ymax_zmin_zmax",
                    target_dimensions=(target_bbox[1::2] - target_bbox[::2]).tolist(),
                    gripper_model_dimensions=(gripper_bbox[1::2] - gripper_bbox[::2]).tolist(),
                    note="native geometry measurements_not_grasp_or_success_tolerances")

    def _sample_native_result(self):
        before = self._clock_tuple()
        value = self.task.success()
        if not isinstance(value, tuple) or len(value) != 2:
            raise NativeBackendError("native task.success must return success, terminate")
        success, terminate = map(bool, value)
        self._last_native_result = dict(success=success, terminate=terminate,
                                        terminal_failure=terminate and not success)
        self._assert_unchanged(before, "native outcome read")

    def read_state(self):
        self._check_live_clock()
        before = self._clock_tuple()
        tip = self.arm.get_tip()
        tcp_pose = _vector(tip.get_pose(), 7, "TCP pose xyzw")
        object_pose = _vector(self.task.target_block.get_pose(), 7, "object pose xyzw")
        _quaternion(tcp_pose[3:]); _quaternion(object_pose[3:])
        tcp_v, tcp_w = tip.get_velocity()
        object_v, object_w = self.task.target_block.get_velocity()
        native_object_v = _vector(object_v, 3, "native object velocity")
        native_object_w = _vector(object_w, 3, "native object angular velocity")
        fd_velocity, fd_angular, fd_valid = np.zeros(3), np.zeros(3), False
        if self._last_pose_sample is not None:
            last_time, last_position, last_quaternion = self._last_pose_sample
            if self.clock() > last_time:
                delta_time = self.clock() - last_time
                fd_velocity = (object_pose[:3] - last_position) / delta_time
                fd_angular = _angular_difference(last_quaternion, object_pose[3:], delta_time)
                fd_valid = True
        opening = np.asarray(self.gripper.get_open_amount(), dtype=float)
        if opening.ndim != 1 or len(opening) < 1 or not np.isfinite(opening).all():
            raise NativeBackendError("finite native finger open amounts required")
        held = self.task.target_block in self.gripper.get_grasped_objects()
        result = dict(
            time=self.clock(), counted_tick=int(self.raw.ledger.ticks),
            physics_time=before[0], task_time=before[1],
            tcp_position=tcp_pose[:3], tcp_quaternion=tcp_pose[3:],
            tcp_velocity=_vector(tcp_v, 3, "TCP velocity"),
            tcp_angular_velocity=_vector(tcp_w, 3, "TCP angular velocity"),
            object_position=object_pose[:3], object_quaternion=object_pose[3:],
            object_linear_velocity=fd_velocity.copy(), object_angular_velocity=fd_angular.copy(),
            object_native_linear_velocity=native_object_v, object_native_angular_velocity=native_object_w,
            object_causal_fd_velocity=fd_velocity.copy(), object_causal_fd_angular_velocity=fd_angular.copy(),
            object_causal_twist_valid=fd_valid,
            q=_vector(self.arm.get_joint_positions(), 7, "joint positions"),
            dq=_vector(self.arm.get_joint_velocities(), 7, "joint velocities"),
            servo_target_positions=_vector(self.arm.get_joint_target_positions(), 7, "servo target readback"),
            opening=float(opening.mean()), finger_open_amounts=opening.copy(),
            finger_positions=np.asarray(self.gripper.get_joint_positions(), dtype=float),
            finger_velocities=np.asarray(self.gripper.get_joint_velocities(), dtype=float),
            gripper_actuator_history=dict(previous_positions=list(getattr(self.gripper, "_prev_positions", [])),
                                          previous_velocity_commands=list(getattr(self.gripper, "_prev_vels", []))),
            requested_opening=self.requested_opening, held=bool(held),
            native_success=self._last_native_result["success"],
            native_result=deepcopy(self._last_native_result),
            lift_goal_position=_vector(self.task.success_detector.get_position(), 3, "native lift goal"),
            velocity_source=("causal_world_SE3_difference_over_actual_previous_counted_tick" if fd_valid
                             else "initial_zero_pending_first_causal_tick_pair_not_driver_or_future_schedule"),
            native_velocity_source="native_Object_get_velocity_logged_separately_may_be_unreliable_for_kinematic_set_pose",
            truth_diagnostic_only=True, official_public_input_evaluation=False,
        )
        self._assert_unchanged(before, "truth-state read")
        return result

    observe_truth = read_state

    def observe_rgbd(self):
        self._check_live_clock()
        before = self._clock_tuple()
        observation, _ = self.raw.observe()
        self._assert_unchanged(before, "RGBD read")
        return observation

    def get_joint_positions(self):
        before = self._clock_tuple()
        result = _vector(self.arm.get_joint_positions(), 7, "joint positions")
        self._assert_unchanged(before, "joint position read")
        return result

    def get_joint_velocities(self):
        before = self._clock_tuple()
        result = _vector(self.arm.get_joint_velocities(), 7, "joint velocities")
        self._assert_unchanged(before, "joint velocity read")
        return result

    def _restoration_errors(self, actual_q, servo_q, ik_pose):
        expected = np.asarray(ik_pose, dtype=float)
        observed = np.asarray(self.arm._ik_target.get_pose(), dtype=float)
        qa, qb = expected[3:] / np.linalg.norm(expected[3:]), observed[3:] / np.linalg.norm(observed[3:])
        errors = dict(joint_position_rad=float(np.max(np.abs(np.asarray(self.arm.get_joint_positions()) - actual_q))),
                      servo_target_rad=float(np.max(np.abs(np.asarray(self.arm.get_joint_target_positions()) - servo_q))),
                      IK_position_m=float(np.max(np.abs(observed[:3] - expected[:3]))),
                      IK_rotation_rad=float(2 * np.arccos(np.clip(abs(np.dot(qa, qb)), 0., 1.))))
        tolerances = dict(joint_position_rad=1e-6, servo_target_rad=1e-6,
                          IK_position_m=1e-6, IK_rotation_rad=5e-6)
        return errors, tolerances, all(errors[k] <= tolerances[k] for k in errors)

    def solve_ik(self, position, quaternion):
        self._can_command()
        position, quaternion = _vector(position, 3, "IK position"), _quaternion(quaternion, "IK quaternion")
        before, started = self._clock_tuple(), time.monotonic()
        actual_q = self.get_joint_positions()
        servo_q = _vector(self.arm.get_joint_target_positions(), 7, "original joint servo targets")
        ik_pose = list(self.arm._ik_target.get_pose())
        solution, reason = None, "jacobian"
        try:
            try:
                solution = self.arm.solve_ik_via_jacobian(position.tolist(), quaternion=quaternion.tolist())
            except self._ik_errors:
                reason = "sampling_fallback"
                try:
                    solutions = self.arm.solve_ik_via_sampling(position.tolist(), quaternion=quaternion.tolist(),
                        ignore_collisions=False, trials=8, max_configs=1, max_time_ms=10)
                    if len(solutions):
                        solution = solutions[0]
                except self._ik_errors:
                    reason = "ik_failed"
            if solution is not None:
                solution = _vector(solution, 7, "IK joint solution")
                if np.any(solution < self._limits["lower"] - 1e-8) or np.any(solution > self._limits["upper"] + 1e-8):
                    solution, reason = None, "ik_outside_native_joint_limits"
        finally:
            # Default False is essential: True hides two physical steps.
            self.arm.set_joint_positions(actual_q.tolist(), disable_dynamics=False)
            self.arm.set_joint_target_positions(servo_q.tolist())
            self.arm._ik_target.set_pose(ik_pose)
            self._assert_unchanged(before, "IK/probe restoration")
            errors, tolerances, restored = self._restoration_errors(actual_q, servo_q, ik_pose)
            self.last_ik_receipt = dict(success=solution is not None, reason=reason,
                                       wall_seconds=time.monotonic() - started,
                                       physics_steps=0, actual_q_restored=restored, servo_targets_restored=restored,
                                       restoration_errors=errors, restoration_tolerances=tolerances)
            if not restored:
                self._fault = "IK restoration error exceeds native float precision: " + repr(errors)
                raise NativeBackendError(self._fault)
        return None if solution is None else solution.copy()

    def set_joint_target_positions(self, q):
        self._can_command()
        q = _vector(q, 7, "joint target")
        if np.any(q < self._limits["lower"] - 1e-8) or np.any(q > self._limits["upper"] + 1e-8):
            raise ValueError("joint target lies outside actual native joint limits")
        before = self._clock_tuple()
        self.arm.set_joint_target_positions(q.tolist())
        self._assert_unchanged(before, "joint target command")

    def probe_joint_collision(self, q):
        """Read-only sampled arm-vs-static-map collision at a candidate q.

        The native frozen-map service excludes the live dynamic target; held
        object geometry is also excluded. False is not a swept-path certificate.
        This probe preserves actual q/qdot, servo targets and IK target without
        advancing physics. Native cached velocities have no no-step setter: if
        a probe changes them, fail closed rather than inventing their recovery.
        """
        self._can_command()
        q = _vector(q, 7, "collision probe joint configuration")
        if np.any(q < self._limits["lower"] - 1e-8) or np.any(q > self._limits["upper"] + 1e-8):
            self.last_collision_probe = dict(valid=False, collision=True, reason="outside_native_joint_limits",
                                             physics_steps=0, native_query_performed=False)
            return True
        before, started = self._clock_tuple(), time.monotonic()
        actual_q, actual_dq = self.get_joint_positions(), self.get_joint_velocities()
        servo_q = _vector(self.arm.get_joint_target_positions(), 7, "original servo targets")
        ik_pose = np.asarray(self.arm._ik_target.get_pose(), dtype=float).copy()
        target_pose = np.asarray(self.task.target_block.get_pose(), dtype=float).copy()
        native_v, native_w = [np.asarray(v, dtype=float).copy() for v in self.task.target_block.get_velocity()]
        collision = None
        query_receipt = {}
        try:
            self.arm.set_joint_positions(q.tolist(), disable_dynamics=False)
            collision = check_static_arm_collision(self.raw.map_service, clock=self._clock_tuple,
                                                   receipt=query_receipt)
            return collision
        finally:
            self.arm.set_joint_positions(actual_q.tolist(), disable_dynamics=False)
            self.arm.set_joint_target_positions(servo_q.tolist())
            self.arm._ik_target.set_pose(ik_pose.tolist())
            self._assert_unchanged(before, "static collision probe/restoration")
            after_v, after_w = self.task.target_block.get_velocity()
            errors, tolerances, restored = self._restoration_errors(actual_q, servo_q, ik_pose)
            after_pose = np.asarray(self.task.target_block.get_pose(), dtype=float)
            qa, qb = target_pose[3:] / np.linalg.norm(target_pose[3:]), after_pose[3:] / np.linalg.norm(after_pose[3:])
            errors.update(joint_velocity_rad_s=float(np.max(np.abs(np.asarray(self.arm.get_joint_velocities()) - actual_dq))),
                          object_position_m=float(np.max(np.abs(after_pose[:3] - target_pose[:3]))),
                          object_rotation_rad=float(2 * np.arccos(np.clip(abs(np.dot(qa, qb)), 0., 1.))),
                          object_native_velocity_m_s=float(np.max(np.abs(np.asarray(after_v) - native_v))),
                          object_native_angular_velocity_rad_s=float(np.max(np.abs(np.asarray(after_w) - native_w))))
            tolerances.update(joint_velocity_rad_s=1e-9, object_position_m=1e-6,
                              object_rotation_rad=5e-6, object_native_velocity_m_s=1e-9,
                              object_native_angular_velocity_rad_s=1e-9)
            restored = restored and all(errors[key] <= tolerances[key] for key in errors)
            self.last_collision_probe = dict(
                valid=collision is not None and restored, collision=collision,
                reason="sampled_static_configuration" if collision is not None else "native_collision_query_failed",
                wall_seconds=time.monotonic() - started, physics_steps=0, native_query_performed=True,
                actual_q_qdot_servo_ik_and_target_native_state_preserved=restored,
                scope="arm_vs_frozen_static_map_dynamic_target_and_held_geometry_excluded",
                sampled_configuration_only_not_swept_path_certificate=True)
            self.last_collision_probe.update(restoration_errors=errors, restoration_tolerances=tolerances)
            self.last_collision_probe["static_query"] = query_receipt
            if not restored:
                self._fault = "collision probe changed actual q/qdot, servo/IK target, or object native pose/velocity: " + repr(errors)
                raise NativeBackendError(self._fault)

    def actuate_gripper(self, opening):
        self._can_command()
        if not math.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("gripper target must be in [0,1], with 0=closed and 1=open")
        requested = float(opening >= .5)
        before = self._clock_tuple()
        if requested != self.requested_opening:
            if requested:
                self.gripper.release(); self.raw.ledger.release_requests += 1
            else:
                for obj in self.task.get_graspable_objects():
                    self.gripper.grasp(obj)
                self.raw.ledger.grasp_requests += 1
            self.requested_opening = requested
            self.raw._requested_opening = requested
        done = bool(self.gripper.actuate(requested, velocity=self.gripper_velocity))
        self._assert_unchanged(before, "nonblocking gripper command")
        return done

    def _motion_event(self):
        scenario = self.scenario
        if self._switch_handled or scenario.switch_tick is None or self.raw.ledger.ticks != scenario.switch_tick:
            return
        self._switch_handled = True
        held = self.task.target_block in self.gripper.get_grasped_objects()
        event = dict(counted_tick=int(self.raw.ledger.ticks), time=self.clock(),
                     first_affected_tick=int(self.raw.ledger.ticks) + 1, applied=not held,
                     changed_velocity=list(scenario.changed_velocity),
                     reason="native_target_already_held" if held else "declared_single_velocity_change")
        if not held:
            segment = dict(x=np.asarray(self.task.target_block.get_position(), dtype=float),
                           v=np.asarray(scenario.changed_velocity, dtype=float), a=np.zeros(3),
                           t0=float(self.task.t) - DT)
            self.task.target_state_list.append(segment)
            event["continuous_position"] = segment["x"].tolist()
            event["native_segment_t0"] = segment["t0"]
        self.motion_events.append(event)

    def tick(self, phase="local_mpc"):
        self._can_command()
        if not isinstance(phase, str) or not phase:
            raise ValueError("explicit counted-tick phase required")
        before_state = self.read_state()
        physics_before, task_before, _, _ = self._clock_tuple()
        self.raw.ledger.before()
        self._motion_event()
        try:
            self.raw.scene.step()
            physics_delta = float(self._physics_clock()) - physics_before
            task_delta = float(self.task.t) - task_before
            tolerance = max(2e-5, 2 * float(np.spacing(np.float32(max(1., physics_before)))))
            self.raw.ledger.after(physics_delta, task_delta, phase, physics_tolerance=tolerance)
            self.raw.time = self.raw.ledger.time
            self._sample_native_result()
            self.raw._native_success_latched |= self._last_native_result["success"]
            self._last_pose_sample = (before_state["time"], before_state["object_position"].copy(),
                                      before_state["object_quaternion"].copy())
            state = self.read_state()
            state["phase"] = phase
            self.tick_records.append(deepcopy(state))
            return state
        except BaseException as error:
            self._fault = f"counted native tick failed: {type(error).__name__}: {error}"
            raise

    def pop_tick_records(self):
        records = self.tick_records[:]
        self.tick_records.clear()
        return records

    def clock(self):
        return float(self.raw.ledger.time)

    def remaining_ticks(self):
        return max(0, self.budget_ticks - int(self.raw.ledger.ticks))

    def terminal(self):
        return (self._closed or self._fault is not None or self.remaining_ticks() == 0
                or self._last_native_result["terminal_failure"])

    def close(self):
        if not self._closed:
            self._closed = True
            self.raw.close()


NativeBackend = TruthNativeBackend
Scenario = MovingBlockScenario
