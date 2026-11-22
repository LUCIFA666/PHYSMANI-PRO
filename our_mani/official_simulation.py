"""Read-only PhysMani TEST resets using the released reset sequence.

Unlike local-development resets, this restores the released variation, target
driver, distractor colors/poses and goal. Only current rendered RGB-D enters the
pipeline; privileged reset fields are retained solely as reset/audit evidence.
"""
import hashlib
import importlib
import math
from pathlib import Path
import pickle

import numpy as np

from .contracts import Pose
from .perception import CameraCalibration
from .pyrep_env import LocalSimulation
from .reachability import PyRepRobotModel


class OfficialTestSimulation(LocalSimulation):
    def __init__(self, task_name, calibration_episode, *, seed=0, servo_lead=1.4):
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
        config.joint_positions = config.gripper_pose = config.gripper_open = True
        config.front_camera.rgb = config.front_camera.depth = True
        config.front_camera.image_size = (256,256)
        self.environment = Environment(MoveArmThenGripper(JointPosition(),Discrete()),obs_config=config,headless=True)
        self.environment.launch()
        module = importlib.import_module('rlbench.tasks.'+task_name)
        task_class = getattr(module,''.join(x.capitalize() for x in task_name.split('_')))
        self.task_environment = self.environment.get_task(task_class)
        self.task = self.task_environment._task
        self.scene, self.pyrep = self.environment._scene, self.environment._pyrep
        self.pyrep.set_simulation_timestep(.05)
        self.arm, self.gripper = self.environment._robot.arm,self.environment._robot.gripper
        self.robot_model = PyRepRobotModel(self.arm)
        self.servo_lead = float(servo_lead)
        self.cameras, self.calibrations = {},{}
        path = Path(calibration_episode)
        # Frame-zero camera calibration is public sensor metadata, not a future
        # target observation. The demo itself is used only by reset_to_demo.
        demo = pickle.loads((path/'low_dim_obs.pkl').read_bytes())
        misc = demo[0].misc
        for name in ('front','8','16','36'):
            if name == 'front':
                extrinsic,intrinsic,near,far = [misc['front_camera_'+key] for key in ('extrinsics','intrinsics','near','far')]
                camera = VisionSensor('cam_front')
            else:
                params = pickle.loads((path/f'nerf_data/0/poses/{name}.pkl').read_bytes())
                extrinsic,intrinsic,near,far = [params[key] for key in ('extrinsic','intrinsic','near','far')]
                camera = VisionSensor.create([256,256],explicit_handling=True,render_mode=RenderMode.OPENGL)
            camera.set_resolution([256,256])
            camera.set_explicit_handling(True)
            camera.set_matrix(extrinsic)
            camera.set_perspective_angle(math.degrees(2*math.atan(256/(2*abs(float(intrinsic[0,0]))))))
            camera.set_near_clipping_plane(float(near));camera.set_far_clipping_plane(float(far))
            matrix = camera.get_matrix()
            self.cameras[name] = camera
            self.calibrations[name] = CameraCalibration(camera.get_intrinsic_matrix(),Pose(matrix[:3,3],matrix[:3,:3]))
        self.time,self.previous_robot = 0.,None

    def reset_official(self, episode):
        path = Path(episode['path'])
        variation = int(episode['variation'])
        with (path/'variation_number.pkl').open('rb') as stream:
            if int(pickle.load(stream)) != variation:
                raise ValueError('official variation mapping differs from dataset')
        self.task_environment.set_variation(variation)
        demo = pickle.loads((path/'low_dim_obs.pkl').read_bytes())
        demo.variation_number = variation
        descriptions,_ = self.task_environment.reset_to_demo(demo)
        self.task.var2target_state_list = pickle.loads((path/'target_state.pkl').read_bytes())
        self.task.init_episode(variation)
        self.pyrep.step()
        self.task.disable_expert_plan()
        self.time,self.previous_robot = 0.,None
        observation,robot = self.observe()
        # Auditing only: these fields are never placed in policy observations.
        self.reset_receipt = {
            'task':episode['task'],'raw_episode_id':episode['raw_episode_id'],'variation':variation,
            'official_episode_index':episode['official_episode_index'],
            'reset_sequence':'reset_to_demo -> load target_state -> init_episode(variation) -> pyrep.step -> disable_expert_plan',
            'target_state_sha256':hashlib.sha256((path/'target_state.pkl').read_bytes()).hexdigest(),
            'target_position_metric_only':np.asarray(self.task.target_block.get_position()).tolist(),
            'goal_pose_metric_only':np.asarray(self.task.success_detector.get_pose()).tolist(),
            'distractor_poses_metric_only':[np.asarray(x.get_pose()).tolist() for x in self.task.distractors],
            'robot_pose':robot.pose.xyzw().tolist(),'robot_joints':robot.joints.tolist(),
            'expert_actions_used':False,'future_observations_used':False,
        }
        self.previous_robot = None
        return descriptions,observation

    def hold_tick(self):
        self.arm.set_joint_target_positions(self.arm.get_joint_positions())
        self.scene.step()
        self.time += .05
