"""Owned PhysMani/Panda simulation for training-only local control experiments.

Target kinematics for resets may come from TRAIN labels. Policies see only
current RGB-D and robot proprioception. Simulator target poses are metrics and
labels, never perception inputs. No connection to existing evaluation servers.
"""
from pathlib import Path
import importlib
import math

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Pose, RobotState
from .perception import CameraCalibration
from .reachability import PyRepRobotModel


class LocalSimulation:
    def __init__(self, task_name, calibration_episode, *, seed=0, resolution=256, servo_lead=2.):
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode
        np.random.seed(seed)
        config = ObservationConfig()
        config.set_all(False)
        config.joint_positions = True
        config.gripper_pose = True
        config.gripper_open = True
        # RLBench removes unused camera objects during launch.
        config.front_camera.rgb = True
        config.front_camera.depth = True
        config.front_camera.image_size = (resolution, resolution)
        action_mode = MoveArmThenGripper(JointPosition(), Discrete())
        self.environment = Environment(action_mode, obs_config=config, headless=True)
        self.environment.launch()
        module = importlib.import_module('rlbench.tasks.'+task_name)
        task_class = getattr(module, ''.join(word.capitalize() for word in task_name.split('_')))
        self.task_environment = self.environment.get_task(task_class)
        self.task_environment.set_variation(0)
        self.task_environment.reset()
        self.task = self.task_environment._task
        self.task.disable_expert_plan()
        self.scene = self.environment._scene
        self.pyrep = self.environment._pyrep
        self.pyrep.set_simulation_timestep(.05)
        self.arm, self.gripper = self.environment._robot.arm, self.environment._robot.gripper
        self.robot_model = PyRepRobotModel(self.arm)
        # Compensate the finite response of a one-physics-tick joint position
        # servo. Native joint velocity limits and physics remain active.
        self.servo_lead = float(servo_lead)
        self.cameras, self.calibrations = {}, {}
        with h5py.File(calibration_episode, 'r') as f:
            self.home_joints = f['low_dim_obs/joint_positions'][0].copy()
            for name in ('front', '8', '16', '36'):
                params = f[f'cam_params/{name}']
                camera = VisionSensor('cam_front') if name == 'front' else VisionSensor.create(
                    [resolution, resolution], explicit_handling=True, render_mode=RenderMode.OPENGL)
                camera.set_resolution([resolution, resolution])
                camera.set_explicit_handling(True)
                camera.set_matrix(params['extrinsics'][...])
                focal = abs(float(params['intrinsics'][0, 0]))*resolution/256
                camera.set_perspective_angle(math.degrees(2*math.atan(resolution/(2*focal))))
                camera.set_near_clipping_plane(float(params['near'][()]))
                camera.set_far_clipping_plane(float(params['far'][()]))
                self.cameras[name] = camera
                matrix = camera.get_matrix()
                self.calibrations[name] = CameraCalibration(camera.get_intrinsic_matrix(), Pose(matrix[:3, 3], matrix[:3, :3]))
        self.time = 0.
        self.previous_robot = None

    def reset(self, *, target_position, target_velocity, target_acceleration, relative_ee=(0., 0., .08), goal_position=None):
        self.task_environment.reset()
        self.task.disable_expert_plan()
        self.gripper.release()
        self.gripper.set_joint_positions([.04, .04], disable_dynamics=True)
        self.gripper.set_joint_target_positions([.04, .04])
        self.arm.set_joint_positions(self.home_joints.tolist(), disable_dynamics=True)
        self.arm.set_joint_target_positions(self.home_joints.tolist())
        self.task.target_state_list = [{'x': np.asarray(target_position).copy(),
                                        'v': np.asarray(target_velocity).copy(),
                                        'a': np.asarray(target_acceleration).copy(), 't0': 0.}]
        self.task.t, self.task.step_id = 0., 0
        self.task.target_block.set_position(np.asarray(target_position).tolist())
        self.task.target_block.set_orientation([0., 0., 0.])
        if goal_position is not None:
            # Reset specification only; policies localize this visible marker
            # through RGB-D and do not read this simulator object.
            self.task.success_detector.set_position(np.asarray(goal_position).tolist())
        desired = Pose.from_xyzw(np.r_[np.asarray(target_position)+relative_ee, [0., 1., 0., 0.]])
        joints = self.robot_model.inverse(desired, self.home_joints)
        invalid = joints is None or np.any(joints < self.robot_model.lower) or np.any(joints > self.robot_model.upper)
        if not invalid:
            invalid = self.robot_model.collision(joints, 0.)
        if invalid:
            from pyrep.errors import ConfigurationError
            try:
                joints = self.arm.solve_ik_via_sampling(desired.position.tolist(), quaternion=desired.xyzw()[3:].tolist(),
                                                        ignore_collisions=False, trials=30, max_time_ms=10)[0]
            except ConfigurationError:
                return False
        self.arm.set_joint_positions(joints.tolist(), disable_dynamics=True)
        self.arm.set_joint_target_positions(joints.tolist())
        for _ in range(3):
            self.pyrep.step()  # settle reset dynamics before starting target motion
        self.time, self.previous_robot = 0., None
        return True

    def observe(self):
        images, depths = {}, {}
        for name, camera in self.cameras.items():
            camera.handle_explicitly()
            images[name] = np.clip(np.rint(camera.capture_rgb()*255), 0, 255).astype(np.uint8)
            depths[name] = camera.capture_depth(in_meters=True)
        pose = Pose.from_xyzw(self.arm.get_tip().get_pose())
        velocity, angular = np.zeros(3), np.zeros(3)
        if self.previous_robot is not None and self.time > self.previous_robot.timestamp:
            dt = self.time-self.previous_robot.timestamp
            velocity = (pose.position-self.previous_robot.pose.position)/dt
            angular = Rotation.from_matrix(pose.rotation@self.previous_robot.pose.rotation.T).as_rotvec()/dt
        contact = self.task.target_block in self.gripper.get_grasped_objects()
        success, _ = self.task.success()
        robot = RobotState(self.time, pose, velocity, angular, self.arm.get_joint_positions(),
                           self.arm.get_joint_velocities(), bool(np.mean(self.gripper.get_open_amount()) > .5),
                           contact, bool(success))
        self.previous_robot = robot
        observation = {'timestamp': self.time, 'observation/images': images, 'observation/depths': depths,
                       'observation/calibrations': self.calibrations,
                       'observation/state': np.r_[pose.xyzw(), float(robot.gripper_open)].astype(np.float32)}
        return observation, robot

    def advance(self, pose, *, opening=1.):
        current_pose = Pose.from_xyzw(self.arm.get_tip().get_pose())
        rotation_delta = Rotation.from_matrix(pose.rotation@current_pose.rotation.T).as_rotvec()
        servo_pose = Pose(current_pose.position+self.servo_lead*(pose.position-current_pose.position),
                          Rotation.from_rotvec(self.servo_lead*rotation_delta).as_matrix()@current_pose.rotation)
        joints = self.robot_model.inverse(servo_pose, self.arm.get_joint_positions())
        if joints is None:
            self.scene.step()
            self.time += .05
            return False
        current = np.asarray(self.arm.get_joint_positions())
        if np.any(joints < self.robot_model.lower) or np.any(joints > self.robot_model.upper):
            self.scene.step()
            self.time += .05
            return False
        delta = np.clip(joints-current, -.05*self.robot_model.velocity_limits, .05*self.robot_model.velocity_limits)
        self.arm.set_joint_target_positions((current+delta).tolist())
        if opening > .5:
            self.gripper.release()
        else:
            self.gripper.grasp(self.task.target_block)
        self.gripper.actuate(float(opening > .5), velocity=.2)
        self.scene.step()
        self.time += .05
        return True

    def metric_target_position(self):
        return np.asarray(self.task.target_block.get_position())

    def close(self):
        self.environment.shutdown()
