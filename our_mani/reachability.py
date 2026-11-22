"""IK and a timed joint trajectory with terminal velocity and sampled collisions."""
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Pose, Reachability


class RobotModel(Protocol):
    name: str
    lower: np.ndarray
    upper: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray

    def inverse(self, pose: Pose, seed): ...
    def forward(self, joints) -> Pose: ...
    def jacobian(self, joints): ...
    def collision(self, joints, future_offset: float) -> bool: ...


def hermite(q0, v0, q1, v1, duration, samples, *, return_times=False):
    """Cubic position trajectory; velocities and accelerations are analytic."""
    a = 2*q0-2*q1+duration*(v0+v1)
    b = -3*q0+3*q1-duration*(2*v0+v1)
    c = duration*v0
    phases = list(np.linspace(0.0, 1.0, samples))
    for aa, bb, cc in zip(a, b, c):
        # Include all position/velocity extrema, not only uniform samples.
        coefficients = np.trim_zeros(np.array([3*aa, 2*bb, cc]), 'f')
        if len(coefficients) > 1:
            for root in np.roots(coefficients):
                if abs(root.imag) < 1e-10 and 0 < root.real < 1:
                    phases.append(float(root.real))
        if abs(aa) > 1e-12 and 0 < -bb/(3*aa) < 1:
            phases.append(float(-bb/(3*aa)))
    s = np.array(sorted(set(phases)))[:, None]
    q = (2*s**3-3*s**2+1)*q0 + (s**3-2*s**2+s)*duration*v0
    q += (-2*s**3+3*s**2)*q1 + (s**3-s**2)*duration*v1
    v = (6*s**2-6*s)/duration*q0 + (3*s**2-4*s+1)*v0
    v += (-6*s**2+6*s)/duration*q1 + (3*s**2-2*s)*v1
    a = (12*s-6)/duration**2*q0 + (6*s-4)/duration*v0
    a += (-12*s+6)/duration**2*q1 + (6*s-2)/duration*v1
    return (q, v, a, s[:, 0]*duration) if return_times else (q, v, a)


class TimedReachability:
    def __init__(self, model: RobotModel, *, max_joint_step=0.03, minimum_samples=21,
                 twist_tolerance=0.04, position_tolerance=0.008, orientation_tolerance=0.08):
        self.model = model
        self.max_joint_step, self.minimum_samples = max_joint_step, minimum_samples
        self.twist_tolerance = twist_tolerance
        self.position_tolerance, self.orientation_tolerance = position_tolerance, orientation_tolerance

    def check(self, robot, pose, terminal_velocity, terminal_angular_velocity, duration):
        if duration <= 0:
            return Reachability(False, "arrival_deadline_elapsed")
        model = self.model
        q = model.inverse(pose, robot.joints)
        if q is None:
            return Reachability(False, "ik_failed")
        q = np.asarray(q, dtype=float)
        if q.shape != robot.joints.shape or not np.isfinite(q).all():
            return Reachability(False, "invalid_ik_solution")
        actual = model.forward(q)
        if np.linalg.norm(actual.position-pose.position) > self.position_tolerance:
            return Reachability(False, "ik_position_residual")
        if Rotation.from_matrix(actual.rotation @ pose.rotation.T).magnitude() > self.orientation_tolerance:
            return Reachability(False, "ik_orientation_residual")
        jacobian = np.asarray(model.jacobian(q))
        target_twist = np.r_[terminal_velocity, terminal_angular_velocity]
        v = np.linalg.lstsq(jacobian, target_twist, rcond=1e-5)[0]
        if np.linalg.norm(jacobian@v-target_twist) > self.twist_tolerance:
            return Reachability(False, "terminal_twist_unreachable")
        if np.any(np.abs(v) > model.velocity_limits):
            return Reachability(False, "terminal_joint_speed_limit")
        return self.check_joints(robot, q, v, duration)

    def check_joints(self, robot, q, v, duration):
        """Validate an explicit joint target without re-solving its IK branch."""
        model = self.model
        q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        if duration <= 0:
            return Reachability(False, 'arrival_deadline_elapsed')
        if q.shape != robot.joints.shape or v.shape != q.shape or not np.isfinite(np.r_[q,v]).all():
            return Reachability(False, 'invalid_joint_target')
        if np.any(np.abs(v) > model.velocity_limits):
            return Reachability(False, 'terminal_joint_speed_limit')
        lower_bound = float(np.max(np.abs(q-robot.joints)/model.velocity_limits))
        if duration < lower_bound:
            return Reachability(False, "insufficient_arrival_time", lower_bound)
        samples = max(self.minimum_samples, int(np.ceil(3*np.max(np.abs(q-robot.joints))/self.max_joint_step))+1)
        path, speeds, accelerations, offsets = hermite(robot.joints, robot.joint_velocities, q, v, duration, samples, return_times=True)
        while np.max(np.abs(np.diff(path, axis=0))) > self.max_joint_step and samples < 10000:
            samples *= 2
            path, speeds, accelerations, offsets = hermite(robot.joints, robot.joint_velocities, q, v, duration, samples, return_times=True)
        if np.max(np.abs(np.diff(path, axis=0))) > self.max_joint_step:
            return Reachability(False, 'collision_sampling_budget_exceeded')
        evidence = {"robot_model": model.name, "trajectory": "cubic_hermite",
                    "collision_samples": len(path), "collision_scene": getattr(model, "collision_scene", "provided_backend"),
                    "maximum_joint_step_rad": float(np.max(np.abs(np.diff(path, axis=0)))),
                    "maximum_speed_fraction": float(np.max(np.abs(speeds)/model.velocity_limits)),
                    "maximum_acceleration_fraction": float(np.max(np.abs(accelerations)/model.acceleration_limits))}
        if np.any(path < model.lower-1e-7) or np.any(path > model.upper+1e-7):
            return Reachability(False, "joint_position_limit", lower_bound, evidence=evidence)
        if np.any(np.abs(speeds) > model.velocity_limits+1e-7):
            return Reachability(False, "trajectory_joint_speed_limit", lower_bound, evidence=evidence)
        if np.any(np.abs(accelerations) > model.acceleration_limits+1e-7):
            return Reachability(False, "trajectory_joint_acceleration_limit", lower_bound, evidence=evidence)
        for joint_positions, offset in zip(path, offsets):
            if model.collision(joint_positions, float(offset)):
                return Reachability(False, "trajectory_collision", lower_bound, evidence=evidence)
        return Reachability(True, "feasible_sampled_trajectory", lower_bound, q, v, evidence)


class PyRepRobotModel:
    """Operate only on an owned simulator. Probes restore joints and IK target.

    Current-scene collisions are sampled. Moving obstacle forecasts can be
    supplied through collision_query; without them this is a static-scene check.
    """
    name = "pyrep_panda"

    def __init__(self, arm, *, acceleration_limits=12.0, collision_query=None):
        self.arm = arm
        cyclic, intervals = arm.get_joint_intervals()
        intervals = np.asarray(intervals)
        # PyRep returns [minimum, RANGE], not [minimum, maximum].
        self.lower = np.where(cyclic, -np.inf, intervals[:, 0])
        self.upper = np.where(cyclic, np.inf, intervals[:, 0]+intervals[:, 1])
        self.velocity_limits = np.asarray(arm.get_joint_upper_velocity_limits(), dtype=float)
        self.acceleration_limits = np.broadcast_to(acceleration_limits, self.velocity_limits.shape).copy()
        self.collision_query = collision_query
        self.collision_scene = "forecast_backend" if collision_query else "current_scene_static_only"

    def _probe(self, joints, fn):
        before = self.arm.get_joint_positions()
        target = self.arm._ik_target.get_pose()
        try:
            self.arm.set_joint_positions(np.asarray(joints).tolist())
            return fn()
        finally:
            self.arm.set_joint_positions(before)
            self.arm._ik_target.set_pose(target)

    def inverse(self, pose, seed):
        from pyrep.errors import ConfigurationError, IKError
        def solve():
            try:
                return np.asarray(self.arm.solve_ik_via_jacobian(pose.position.tolist(), quaternion=pose.xyzw()[3:].tolist()))
            except IKError:
                try:
                    return self.arm.solve_ik_via_sampling(pose.position.tolist(), quaternion=pose.xyzw()[3:].tolist(),
                                                          ignore_collisions=False, trials=8, max_time_ms=10)[0]
                except ConfigurationError:
                    return None
        return self._probe(seed, solve)

    def forward(self, joints):
        return self._probe(joints, lambda: Pose.from_xyzw(self.arm.get_tip().get_pose()))

    def jacobian(self, joints):
        # Numerical differentiation avoids ambiguity in PyRep IK row/frame ordering.
        q = np.asarray(joints)
        initial = self.forward(q)
        j = np.zeros((6, len(q)))
        epsilon = 1e-4
        for column in range(len(q)):
            shifted = q.copy()
            shifted[column] += epsilon
            changed = self.forward(shifted)
            j[:3, column] = (changed.position-initial.position)/epsilon
            j[3:, column] = Rotation.from_matrix(changed.rotation @ initial.rotation.T).as_rotvec()/epsilon
        return j

    def collision(self, joints, future_offset):
        def query():
            return bool(self.collision_query(future_offset) if self.collision_query else self.arm.check_arm_collision())
        return self._probe(joints, query)
