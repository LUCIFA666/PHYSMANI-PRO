"""Latest-observation handover checks and local visual/kinematic MPC control."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .contracts import Pose


@dataclass(frozen=True)
class HandoverCheck:
    allowed: bool
    reason: str
    position_metric: float = float("inf")
    relative_speed: float = float("inf")
    orientation_error: float = float("inf")


class HandoverGate:
    def __init__(self, *, max_age=0.15, max_skew=0.035, min_confidence=0.5, time_tolerance=0.2):
        self.max_age, self.max_skew = max_age, max_skew
        self.min_confidence, self.time_tolerance = min_confidence, time_tolerance

    def check(self, goal, obj, robot, now):
        if not obj.visible or obj.confidence < self.min_confidence:
            return HandoverCheck(False, "target_unreliable")
        if now-obj.timestamp > self.max_age or now-robot.timestamp > self.max_age:
            return HandoverCheck(False, "stale_observation")
        if min(now-obj.timestamp, now-robot.timestamp) < -1e-6:
            return HandoverCheck(False, "future_observation")
        if abs(obj.timestamp-robot.timestamp) > self.max_skew:
            return HandoverCheck(False, "unsynchronized_robot_and_object")
        if abs(now-goal.arrival_time) > self.time_tolerance:
            return HandoverCheck(False, "outside_handover_time_window")
        pose, velocity = goal.region.in_world(obj.pose, obj.linear_velocity, obj.angular_velocity)
        error_local = pose.rotation.T@(robot.pose.position-pose.position)
        position_metric = float(np.linalg.norm(error_local/goal.region.radii))
        relative_speed = float(np.linalg.norm(robot.linear_velocity-velocity))
        orientation_error = float(Rotation.from_matrix(robot.pose.rotation@pose.rotation.T).magnitude())
        allowed = position_metric <= 1 and relative_speed <= goal.region.max_relative_speed and orientation_error <= goal.region.max_orientation_error
        return HandoverCheck(allowed, "handover_ready" if allowed else "outside_local_capture_region",
                             position_metric, relative_speed, orientation_error)


class VisualMPC:
    """Receding-horizon Cartesian integrator MPC, not a trained latent model.

    Optimizes translation velocities; orientation uses a bounded visual servo.
    Every output still requires robot IK/collision validation before execution.
    """
    name = "kinematic_visual_mpc"

    def __init__(self, *, dt=0.05, horizon=6, max_speed=0.4, max_acceleration=2.0,
                 max_angular_speed=1.5, position_weight=50., smoothness_weight=0.2):
        self.dt, self.horizon = dt, horizon
        self.max_speed, self.max_acceleration = max_speed, max_acceleration
        self.max_angular_speed = max_angular_speed
        self.position_weight, self.smoothness_weight = position_weight, smoothness_weight

    def command(self, robot, obj, local_grasp_pose):
        offsets = np.arange(1, self.horizon+1)*self.dt
        desired = []
        for offset in offsets:
            r = Rotation.from_rotvec(obj.angular_velocity*offset).as_matrix()@obj.pose.rotation
            future = Pose(obj.pose.position+obj.linear_velocity*offset, r)
            desired.append(future.apply(local_grasp_pose.position))
        desired = np.array(desired)
        # Starting at the target velocity can violate every acceleration
        # constraint during a fast handover. A constant current-velocity plan
        # is feasible whenever the measured state is within the speed limit.
        seed_velocity = np.asarray(robot.linear_velocity).copy()
        seed_velocity *= min(1., self.max_speed/max(np.linalg.norm(seed_velocity), 1e-9))
        initial_u = np.tile(seed_velocity, (self.horizon, 1))
        def objective(flat):
            u = flat.reshape(self.horizon, 3)
            positions = robot.pose.position + np.cumsum(u, axis=0)*self.dt
            du = np.diff(np.vstack([robot.linear_velocity, u]), axis=0)
            return self.position_weight*np.square(positions-desired).sum()+self.smoothness_weight*np.square(du).sum()
        def constraint(flat):
            u = flat.reshape(self.horizon, 3)
            du = np.diff(np.vstack([robot.linear_velocity, u]), axis=0)
            return np.r_[self.max_speed**2-np.square(u).sum(1),
                         (self.max_acceleration*self.dt)**2-np.square(du).sum(1)]
        result = minimize(objective, initial_u.ravel(), method="SLSQP",
                          bounds=[(-self.max_speed, self.max_speed)]*(3*self.horizon),
                          constraints=[{"type": "ineq", "fun": constraint}],
                          options={"maxiter": 120, "ftol": 1e-7})
        feasible = np.isfinite(result.fun) and np.min(constraint(result.x)) >= -1e-5
        if not feasible:
            raise RuntimeError(f"local MPC infeasible: {result.message}")
        v = result.x.reshape(self.horizon, 3)[0]
        desired_rotation = obj.pose.rotation@local_grasp_pose.rotation
        w = 4.*Rotation.from_matrix(desired_rotation@robot.pose.rotation.T).as_rotvec()+obj.angular_velocity
        w *= min(1., self.max_angular_speed/max(np.linalg.norm(w), 1e-9))
        next_pose = Pose(robot.pose.position+v*self.dt, Rotation.from_rotvec(w*self.dt).as_matrix()@robot.pose.rotation)
        return next_pose, v, w, {"controller": self.name, "cost": float(result.fun), "iterations": result.nit,
                               "optimizer_converged": bool(result.success)}
