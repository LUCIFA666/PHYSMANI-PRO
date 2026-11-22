"""Versioned, observable-state local capture controller and rollout features."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Pose
from .control import VisualMPC


def local_features(robot, obj):
    """Object-relative quantities available from current tracking/proprioception."""
    return np.r_[robot.pose.position-obj.pose.position,
                 robot.linear_velocity-obj.linear_velocity, obj.linear_velocity,
                 obj.angular_velocity,
                 Rotation.from_matrix(robot.pose.rotation@Pose.from_xyzw([0, 0, 0, 0, 1, 0, 0]).rotation.T).as_rotvec(),
                 float(robot.gripper_open), float(robot.contact)].astype(np.float32)


@dataclass(frozen=True)
class LocalConfig:
    dt: float = .05
    max_speed: float = .8
    max_acceleration: float = 3.
    close_distance: float = .022
    close_relative_speed: float = .14
    grasp_offset_z: float = -.015
    lift_height: float = .10
    lift_speed: float = .25
    transport_speed: float = .35


class LocalCaptureController:
    name = 'rgbd_mpc_capture_and_visual_goal_v1'

    def __init__(self, goal_pose, config=None, *, learned_policy=None):
        self.config = config or LocalConfig()
        self.mpc = VisualMPC(dt=self.config.dt, max_speed=self.config.max_speed,
                             max_acceleration=self.config.max_acceleration)
        self.grasp = Pose.from_xyzw([0., 0., self.config.grasp_offset_z, 0., 1., 0., 0.])
        self.goal_pose, self.learned_policy = goal_pose, learned_policy
        self.stage, self.close_time, self.lift_goal = 'track', None, None

    def command(self, robot, obj):
        c = self.config
        if robot.contact and self.stage in ('track', 'close'):
            self.stage = 'lift'
            self.lift_goal = Pose(robot.pose.position+[0, 0, c.lift_height], robot.pose.rotation)
            # Control the carried object's visually observed center into the
            # marker. Early native contact need not coincide with nominal TCP
            # grasp offset; driving the TCP to the marker can miss the goal.
            carried_local = robot.pose.rotation.T@(obj.pose.position-robot.pose.position)
            self.goal_pose = Pose(self.goal_pose.position-self.goal_pose.rotation@carried_local,
                                  self.goal_pose.rotation)
        if not robot.contact and self.stage in ('lift', 'transport'):
            raise RuntimeError('grasp lost during lift/transport')
        if self.stage in ('lift', 'transport'):
            if self.stage == 'lift' and np.linalg.norm(self.lift_goal.position-robot.pose.position) < .015:
                self.stage = 'transport'
            target = self.lift_goal if self.stage == 'lift' else self.goal_pose
            velocity = 4.*(target.position-robot.pose.position)
            speed = c.lift_speed if self.stage == 'lift' else c.transport_speed
            velocity *= min(1., speed/max(np.linalg.norm(velocity), 1e-8))
            angular = 3.*Rotation.from_matrix(target.rotation@robot.pose.rotation.T).as_rotvec()
            angular *= min(1., 1.5/max(np.linalg.norm(angular), 1e-8))
            return Pose(robot.pose.position+velocity*c.dt,
                        Rotation.from_rotvec(angular*c.dt).as_matrix()@robot.pose.rotation), 0., velocity, angular, {'stage': self.stage}
        distance = np.linalg.norm(robot.pose.position-obj.pose.apply(self.grasp.position))
        relative_speed = np.linalg.norm(robot.linear_velocity-obj.linear_velocity)
        if self.close_time is None and distance < c.close_distance and relative_speed < c.close_relative_speed:
            self.close_time, self.stage = robot.timestamp, 'close'
        opening = float(self.close_time is None)
        if self.learned_policy is None:
            pose, velocity, angular, details = self.mpc.command(robot, obj, self.grasp)
        else:
            action = self.learned_policy(local_features(robot, obj))
            velocity, angular = np.asarray(action[:3]), np.asarray(action[3:6])
            velocity *= min(1., c.max_speed/max(np.linalg.norm(velocity), 1e-8))
            angular *= min(1., 1.5/max(np.linalg.norm(angular), 1e-8))
            # The learned continuous servo uses the same observable grasp gate
            # and downstream IK/joint-velocity limits as the teacher.
            pose = Pose(robot.pose.position+velocity*c.dt,
                        Rotation.from_rotvec(angular*c.dt).as_matrix()@robot.pose.rotation)
            details = {'controller': 'learned_local_servo_with_observable_grasp_gate'}
        return pose, opening, velocity, angular, {**details, 'stage': self.stage,
                'distance_m': float(distance), 'relative_speed_mps': float(relative_speed)}
