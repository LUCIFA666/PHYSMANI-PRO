"""Validate each streamed action against the latest robot state before execution."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Pose


@dataclass(frozen=True)
class JointCommand:
    joints: np.ndarray
    terminal_joint_velocities: np.ndarray
    gripper_open: float
    evidence: dict


class CartesianActionLimiter:
    """Convert a neural waypoint into a bounded next tick, then validate in IK.

    This limits translation/rotation relative to measured current motion. It
    supplies no goal trajectory or substitute approach policy.
    """
    def __init__(self, *, dt=.05, speed=.6, acceleration=2., angular_speed=1.5, angular_acceleration=3.):
        self.dt, self.speed, self.acceleration = dt, speed, acceleration
        self.angular_speed, self.angular_acceleration = angular_speed, angular_acceleration

    @staticmethod
    def bound(vector, maximum):
        return vector*min(1., maximum/max(np.linalg.norm(vector), 1e-9))

    def filter(self, action, robot):
        action = np.asarray(action, dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all():
            raise ValueError('invalid neural waypoint')
        target = Pose.from_xyzw(action[:7])
        velocity = self.bound((target.position-robot.pose.position)/self.dt, self.speed)
        velocity = robot.linear_velocity+self.bound(velocity-robot.linear_velocity, self.acceleration*self.dt)
        angular = self.bound(Rotation.from_matrix(target.rotation@robot.pose.rotation.T).as_rotvec()/self.dt, self.angular_speed)
        angular = robot.angular_velocity+self.bound(angular-robot.angular_velocity, self.angular_acceleration*self.dt)
        pose = Pose(robot.pose.position+velocity*self.dt,
                    Rotation.from_rotvec(angular*self.dt).as_matrix()@robot.pose.rotation)
        return np.r_[pose.xyzw(), np.clip(action[7], 0., 1.)]


class FeasibleExecutor:
    def __init__(self, reachability, *, dt=0.05, max_robot_state_age=0.1):
        self.reachability, self.dt, self.max_age = reachability, dt, max_robot_state_age

    def prepare(self, action, robot, *, now, approach=True):
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('action must contain finite world xyz xyzw and gripper-open')
        if now-robot.timestamp > self.max_age or robot.timestamp > now+1e-6:
            return None, 'stale_robot_state'
        pose = Pose.from_xyzw(action[:7])
        v = (pose.position-robot.pose.position)/self.dt
        w = Rotation.from_matrix(pose.rotation@robot.pose.rotation.T).as_rotvec()/self.dt
        check = self.reachability.check(robot, pose, v, w, self.dt)
        if not check.feasible:
            return None, check.reason
        # Closing is an explicit state-machine action after a current visual
        # handover/grasp check. A VLA approach chunk cannot close the gripper.
        opening = 1. if approach else float(np.clip(action[7], 0., 1.))
        return JointCommand(check.joints, check.joint_velocities, opening, check.evidence), 'feasible'

    def prepare_joint_limited(self, action, robot, *, now, approach=True, acceleration_fraction=.5):
        """Bound a neural IK target in joint space, including near singularities.

        The next joint position integrates the mean of current and terminal
        velocity. Its Hermite interpolant has constant acceleration. All joint
        limits and forecast collision checks remain mandatory.
        """
        action = np.asarray(action, dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all():
            raise ValueError('invalid neural waypoint')
        if not 0 < acceleration_fraction <= 1:
            raise ValueError('acceleration fraction must be in (0,1]')
        if now-robot.timestamp > self.max_age or robot.timestamp > now+1e-6:
            return None, 'stale_robot_state'
        model = self.reachability.model
        target = model.inverse(Pose.from_xyzw(action[:7]), robot.joints)
        if target is None:
            return None, 'waypoint_ik_failed'
        target = np.asarray(target)
        if target.shape!=robot.joints.shape or not np.isfinite(target).all():
            return None, 'invalid_waypoint_ik'
        desired = np.clip((target-robot.joints)/self.dt, -model.velocity_limits, model.velocity_limits)
        maximum_change = model.acceleration_limits*acceleration_fraction*self.dt
        terminal = robot.joint_velocities+np.clip(desired-robot.joint_velocities, -maximum_change, maximum_change)
        joints = robot.joints+.5*(robot.joint_velocities+terminal)*self.dt
        check = self.reachability.check_joints(robot, joints, terminal, self.dt)
        if not check.feasible:
            return None, check.reason
        opening = 1. if approach else float(np.clip(action[7], 0., 1.))
        return JointCommand(check.joints, check.joint_velocities, opening,
                            {**check.evidence, 'neural_waypoint_joint_rate_limit':True,
                             'acceleration_fraction':acceleration_fraction}), 'feasible_joint_limited'


class PyRepExecutor:
    def __init__(self, arm, gripper, validator):
        self.arm, self.gripper, self.validator = arm, gripper, validator

    def apply(self, action, robot, *, now, approach=True):
        command, reason = self.validator.prepare(action, robot, now=now, approach=approach)
        if command is None:
            return False, reason
        self.arm.set_joint_target_positions(command.joints.tolist())
        self.gripper.actuate(command.gripper_open, velocity=.2)
        return True, reason  # caller advances its owned simulation once
