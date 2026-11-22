"""Shared R10 native runtime without moving-pick object assumptions.

Official TEST uses the author's family-specific archive replay. Development
sources provide only reproducible native generation seeds and camera metadata;
their HDF5 motion is not represented as a complete native scene replay.
Native task bodies, motion drivers and outcomes stay in metric-only receipts.
"""
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import random

import numpy as np
from scipy.spatial.transform import Rotation

from ..aligned.protocol import TickLedger
from ..contracts import Pose, RobotState
from ..perception import CameraCalibration
from ..sensor_alignment.author_rgbd import PROTOCOL as SENSOR_PROTOCOL, RGB_SPEC
from .maps import SharedMultiObjectMap, AlignedRobotModel, KnownMapForecastCollision
from .native_tasks import (NativeOutcomeLatch, metric_initial_snapshot, native_result,
                           official_reset_plan, validate_reset_metadata, verify_native_sources)
from .protocol import (EpisodeTerminatedError, MultitaskProtocolError,
                       MultitaskInfrastructureError, validate_protocol)
from .specs import get_task_spec

CAMERAS = ('front', '8', '16', '36')
OFFICIAL_SOURCE_FILES = (
    'low_dim_obs.pkl', 'variation_number.pkl', 'target_state.pkl', 'variation_descriptions.pkl',
    'front_rgb/0.png', 'front_depth/0.png', 'front_mask/0.png',
    *(f'nerf_data/0/{kind}/{name}.{extension}' for name in CAMERAS[1:]
      for kind, extension in (('images', 'png'), ('depths', 'png'), ('masks', 'png'), ('poses', 'pkl'))),
)


def physics_time():
    from pyrep.backend import sim
    return float(sim.simGetSimulationTime())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_official_source_files(episode, spec, native_repo):
    """Verify frozen reset/calibration/scene inputs before deserializing them."""
    path = Path(episode['path'])
    required = episode.get('source_required_files')
    if not isinstance(required, Mapping) or set(required) != set(OFFICIAL_SOURCE_FILES):
        raise MultitaskProtocolError('official manifest needs all 19 reset/calibration/first-frame hashes')
    checked = {}
    for relative in OFFICIAL_SOURCE_FILES:
        item, expected_path = required[relative], path / relative
        if (not isinstance(item, Mapping) or 'path' not in item or 'sha256' not in item or 'bytes' not in item
                or Path(item['path']).resolve() != expected_path.resolve()
                or not expected_path.is_file() or expected_path.stat().st_size != item['bytes']
                or file_sha256(expected_path) != item['sha256']):
            raise MultitaskProtocolError('official input missing or changed: ' + relative)
        checked[relative] = item['sha256']
    source_path = Path(native_repo) / spec.native_source_relative_path
    scene_path = Path(native_repo) / 'third_party/RLBench/rlbench/task_ttms' / (spec.task_name + '.ttm')
    for key, expected_path in (('task_source', source_path), ('task_scene', scene_path)):
        if (key + '_path' not in episode or key + '_sha256' not in episode
                or Path(episode[key + '_path']).resolve() != expected_path.resolve()
                or not expected_path.is_file() or file_sha256(expected_path) != episode[key + '_sha256']):
            raise MultitaskProtocolError('official native source or scene changed: ' + key)
    if (episode['task_source_sha256'] != spec.native_source_sha256
            or episode.get('target_state_sha256') != checked['target_state.pkl']):
        raise MultitaskProtocolError('official source hashes conflict with declared native task/reset')
    return dict(source_required_sha256=checked, task_scene_sha256=episode['task_scene_sha256'],
                task_source_sha256=episode['task_source_sha256'])


def development_seed(task, source, split, *, seed=0, replica=0):
    if split not in ('train', 'validation') or not source.get('raw_episode_id') or source.get('task') != task:
        raise MultitaskProtocolError('native development generation needs a matching TRAIN/VAL source identity')
    identity = dict(task=task, raw_episode_id=source['raw_episode_id'], split=split,
                    base_seed=int(seed), replica=int(replica), version='r10_native_generated_development_v1')
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return int(digest[:8], 16), identity


def sensor_conditions(env):
    return dict(version=SENSOR_PROTOCOL,
                camera_render_modes={name: camera.get_render_mode().name for name, camera in env.cameras.items()},
                rgb_processing=RGB_SPEC)


def public_initial_conditions(env, observation, robot):
    digest, cameras = hashlib.sha256(), {}
    sensor = sensor_conditions(env)
    for name in sorted(CAMERAS):
        digest.update(name.encode())
        digest.update(np.asarray(observation['observation/images'][name]).tobytes())
        digest.update(np.asarray(observation['observation/depths'][name]).tobytes())
        calibration = observation['observation/calibrations'][name]
        cameras[name] = dict(intrinsics=calibration.intrinsics.tolist(),
            camera_to_world=calibration.camera_to_world.xyzw().tolist(),
            render_mode=sensor['camera_render_modes'][name],
            raw_rgb_255_processing=RGB_SPEC['observation/raw_rgb_255'][name])
    return dict(robot_joints=robot.joints.tolist(), robot_pose=robot.pose.xyzw().tolist(),
                cameras=cameras, sensor_protocol=sensor, rgbd_sha256=digest.hexdigest(),
                static_map=deepcopy(env.map_service.map), information_setting=env.protocol['information_setting'])


class MultitaskSimulation:
    def __init__(self, task_name, calibration_episode, *, protocol, seed=0, servo_lead=1.4):
        validate_protocol(protocol)
        self.protocol, self.spec = dict(protocol), get_task_spec(task_name)
        self.seed, self.servo_lead = int(seed), float(servo_lead)
        self.native_repo = Path(protocol['baseline_repo'])
        self.native_source_hashes = verify_native_sources(self.native_repo, self.spec)
        self.calibration_source = Path(calibration_episode)
        self._closed, self._aligned = False, False
        self.time, self.previous_robot = 0., None
        random.seed(self.seed)
        np.random.seed(self.seed)
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        config = ObservationConfig()
        config.set_all(False)
        config.joint_positions = config.gripper_pose = config.gripper_open = True
        config.front_camera.rgb = config.front_camera.depth = True
        config.front_camera.image_size = (256, 256)
        self.environment = Environment(MoveArmThenGripper(JointPosition(), Discrete()),
                                       obs_config=config, headless=True)
        try:
            self.environment.launch()
            module = importlib.import_module('rlbench.tasks.' + task_name)
            task_class = getattr(module, ''.join(word.capitalize() for word in task_name.split('_')))
            self.task_environment = self.environment.get_task(task_class)
            self.task = self.task_environment._task
            self.scene, self.pyrep = self.environment._scene, self.environment._pyrep
            self.arm, self.gripper = self.environment._robot.arm, self.environment._robot.gripper
            self.pyrep.set_simulation_timestep(self.protocol['dt'])
            self.cameras, self.calibrations, self.mask_cameras = {}, {}, {}
            self._configure_cameras()
        except Exception:
            self.environment.shutdown()
            self._closed = True
            raise

    @staticmethod
    def physics_time():
        return physics_time()

    @staticmethod
    def resolve_scene_object(name):
        from pyrep.objects.object import Object
        return Object.get_object(name)

    def _clock_unchanged(self, before, label):
        if abs(self.physics_time() - before[0]) > 1e-6 or float(self.task.t) != before[1]:
            raise MultitaskProtocolError(label + ' advanced physics/task clock')

    def _configure_cameras(self):
        from pyrep.const import RenderMode
        from pyrep.objects.vision_sensor import VisionSensor
        path = self.calibration_source
        if path.is_dir():
            demo = pickle.loads((path / 'low_dim_obs.pkl').read_bytes())
            misc = demo[0].misc
            parameters = {'front': dict(extrinsics=misc['front_camera_extrinsics'],
                intrinsics=misc['front_camera_intrinsics'], near=misc['front_camera_near'], far=misc['front_camera_far'])}
            for name in CAMERAS[1:]:
                item = pickle.loads((path / f'nerf_data/0/poses/{name}.pkl').read_bytes())
                parameters[name] = dict(extrinsics=item['extrinsic'], intrinsics=item['intrinsic'],
                                        near=item['near'], far=item['far'])
        else:
            import h5py
            with h5py.File(path, 'r') as source:
                parameters = {name: {key: source[f'cam_params/{name}/{key}'][...]
                                     for key in ('extrinsics', 'intrinsics', 'near', 'far')} for name in CAMERAS}
        for name in CAMERAS:
            item = parameters[name]
            camera = VisionSensor('cam_front') if name == 'front' else VisionSensor.create(
                [256, 256], explicit_handling=True, render_mode=RenderMode.OPENGL)
            camera.set_resolution([256, 256])
            camera.set_explicit_handling(True)
            camera.set_render_mode(RenderMode.OPENGL)
            camera.set_matrix(np.asarray(item['extrinsics']))
            focal = abs(float(np.asarray(item['intrinsics'])[0, 0]))
            camera.set_perspective_angle(math.degrees(2 * math.atan(256 / (2 * focal))))
            camera.set_near_clipping_plane(float(item['near']))
            camera.set_far_clipping_plane(float(item['far']))
            matrix = camera.get_matrix()
            self.cameras[name] = camera
            self.calibrations[name] = CameraCalibration(camera.get_intrinsic_matrix(),
                                                        Pose(matrix[:3, 3], matrix[:3, :3]))

    def _start_alignment(self):
        from pyrep.const import RenderMode
        from pyrep.objects.vision_sensor import VisionSensor
        before = self.physics_time(), float(self.task.t)
        self._aligned = False
        if hasattr(self, 'map_service'):
            self.map_service.close()
        self.ledger = TickLedger(dt=self.protocol['dt'], budget=self.protocol['simulation_seconds'])
        self.time, self.previous_robot = 0., None
        self._physics_origin, self._task_origin = before
        self._requested_opening = float(np.mean(self.gripper.get_open_amount()) > .5)
        self._outcome = NativeOutcomeLatch()
        self._terminal_tick = None
        self._native_success_latched = self._native_failure_latched = self._native_terminal_latched = False
        self.map_service = SharedMultiObjectMap(self)
        self.robot_model = AlignedRobotModel(self.arm, self.map_service)
        if 'global_joint_acceleration' in self.protocol:
            self.robot_model.acceleration_limits[:] = self.protocol['global_joint_acceleration']
        for name, source in self.cameras.items():
            if name not in self.mask_cameras:
                self.mask_cameras[name] = VisionSensor.create([256, 256], explicit_handling=True,
                                                              render_mode=RenderMode.OPENGL_COLOR_CODED)
            camera = self.mask_cameras[name]
            camera.set_matrix(source.get_matrix())
            camera.set_perspective_angle(source.get_perspective_angle())
            camera.set_near_clipping_plane(source.get_near_clipping_plane())
            camera.set_far_clipping_plane(source.get_far_clipping_plane())
        self._latch_native_result()
        self._clock_unchanged(before, 'alignment setup')
        self._aligned = True

    def _latch_native_result(self):
        try:
            value = self._outcome.update(native_result(self.task))
        except Exception as exc:
            raise MultitaskInfrastructureError('invalid native task outcome') from exc
        self._native_success_latched = value.success
        self._native_failure_latched = value.terminal_failure
        self._native_terminal_latched = value.terminate
        if value.terminate and self._terminal_tick is None:
            self._terminal_tick = self.ledger.ticks
        return value

    @property
    def native_outcome(self):
        return self._outcome.outcome

    def _before_action(self):
        if not self._aligned:
            raise MultitaskProtocolError('reset must complete before observations/actions')
        if self._native_terminal_latched:
            raise EpisodeTerminatedError('native terminal result already latched')
        try:
            self.ledger.before()
        except Exception as exc:
            raise MultitaskProtocolError('counted episode budget exhausted') from exc
        current = self.physics_time()
        tolerance = max(2e-5, 2 * float(np.spacing(np.float32(max(1., current)))))
        if (abs(self.time - self.ledger.time) > 1e-9
                or abs(current - self._physics_origin - self.ledger.time) > tolerance
                or abs(float(self.task.t) - self._task_origin - self.ledger.time) > 2e-5):
            raise MultitaskProtocolError('physics/task advanced outside the common counted tick')

    def _tick(self, phase):
        self._before_action()
        physics_before, task_before = self.physics_time(), float(self.task.t)
        tolerance = max(2e-5, 2 * float(np.spacing(np.float32(max(1., physics_before)))))
        if (abs(physics_before - self._physics_origin - self.ledger.time) > tolerance or
                abs(task_before - self._task_origin - self.ledger.time) > 2e-5):
            raise MultitaskProtocolError('physics/task advanced outside the common counted tick')
        try:
            self.scene.step()
        except Exception as exc:
            raise MultitaskInfrastructureError('native counted scene step failed') from exc
        try:
            self.ledger.after(self.physics_time() - physics_before, float(self.task.t) - task_before,
                              phase, physics_tolerance=tolerance)
        except Exception as exc:
            raise MultitaskProtocolError('counted physics/task tick invariant failed') from exc
        self.time = self.ledger.time
        self._latch_native_result()

    def observe(self):
        try:
            return self._observe_current()
        except (MultitaskProtocolError, MultitaskInfrastructureError):
            raise
        except Exception as exc:
            raise MultitaskInfrastructureError('current native sensor/proprioception read failed') from exc

    def _observe_current(self):
        if not self._aligned:
            raise MultitaskProtocolError('reset must complete before observations/actions')
        before = self.physics_time(), float(self.task.t)
        images, raw, depths, masks = {}, {}, {}, {}
        for name, camera in self.cameras.items():
            camera.handle_explicitly()
            values = np.asarray(camera.capture_rgb(), dtype=np.float32) * 255.
            depth = np.asarray(camera.capture_depth(in_meters=True))
            if (values.shape != (256, 256, 3) or depth.shape != (256, 256)
                    or not np.isfinite(values).all() or not np.isfinite(depth).all()
                    or np.any(values < 0) or np.any(values > 255) or np.any(depth < 0)):
                raise MultitaskProtocolError('invalid common RGB-D frame')
            images[name] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
            raw[name] = np.clip(values.astype(np.uint8), 0, 255) if name == 'front' else values
            depths[name] = depth
            self.mask_cameras[name].handle_explicitly()
            masks[name] = np.asarray(self.mask_cameras[name].capture_rgb(), dtype=np.float32)
            if masks[name].shape != values.shape or not np.isfinite(masks[name]).all():
                raise MultitaskProtocolError('invalid current instance mask frame')
        pose = Pose.from_xyzw(self.arm.get_tip().get_pose())
        velocity = angular = np.zeros(3)
        previous = self.previous_robot
        if previous is not None:
            if self.time > previous.timestamp:
                elapsed = self.time - previous.timestamp
                velocity = (pose.position - previous.pose.position) / elapsed
                angular = Rotation.from_matrix(pose.rotation @ previous.pose.rotation.T).as_rotvec() / elapsed
            elif self.time == previous.timestamp:
                velocity, angular = previous.linear_velocity, previous.angular_velocity
            else:
                raise MultitaskProtocolError('robot observation time moved backwards')
        robot = RobotState(self.time, pose, velocity, angular, self.arm.get_joint_positions(),
            self.arm.get_joint_velocities(), bool(np.mean(self.gripper.get_open_amount()) > .5),
            bool(self.gripper.get_grasped_objects()), False)
        self.previous_robot = robot
        observation = {'timestamp': self.time, 'observation/images': images, 'observation/raw_rgb_255': raw,
            'observation/depths': depths, 'observation/instance_masks': masks,
            'observation/calibrations': self.calibrations,
            'observation/state': np.r_[pose.xyzw(), float(robot.gripper_open)].astype(np.float32),
            'observation/sensor_protocol': sensor_conditions(self)}
        self._clock_unchanged(before, 'current sensor/proprioception observation')
        return observation, robot

    def command_joints(self, joints, *, opening=None, phase='arm'):
        self._before_action()
        current, requested = np.asarray(self.arm.get_joint_positions()), np.asarray(joints, dtype=float)
        if requested.shape != current.shape or not np.isfinite(requested).all():
            raise MultitaskProtocolError('invalid joint reference')
        target = current + np.clip(requested - current, -self.robot_model.velocity_limits * self.ledger.dt,
                                   self.robot_model.velocity_limits * self.ledger.dt)
        target = np.clip(target, self.robot_model.lower, self.robot_model.upper)
        try:
            self.arm.set_joint_target_positions(target.tolist())
            if opening is not None:
                requested_opening = float(opening >= .5)
                if requested_opening != self._requested_opening:
                    if requested_opening:
                        self.gripper.release()
                        self.ledger.release_requests += 1
                    else:
                        for obj in self.task.get_graspable_objects():
                            self.gripper.grasp(obj)
                        self.ledger.grasp_requests += 1
                    self._requested_opening = requested_opening
                self.gripper.actuate(requested_opening, velocity=self.protocol['gripper_velocity'])
        except Exception as exc:
            raise MultitaskInfrastructureError('native arm/gripper reference application failed') from exc
        self._tick(phase)
        return True

    def advance(self, pose, *, opening=1.):
        self._before_action()
        current = np.asarray(self.arm.get_joint_positions())
        joints = self.robot_model.inverse(pose, current)
        if joints is None:
            self.command_joints(current, opening=opening, phase='ik_rejected')
            return False
        target = current + self.servo_lead * (np.asarray(joints) - current)
        return self.command_joints(target, opening=opening, phase='local_interaction')

    def hold_tick(self):
        return self.command_joints(self.arm.get_joint_positions(), opening=self._requested_opening, phase='hold')

    def alignment_receipt(self):
        return dict(clock=self.ledger.receipt(), map=self.map_service.receipt(),
            actual_physics_seconds=self.physics_time() - self._physics_origin,
            actual_task_seconds=float(self.task.t) - self._task_origin,
            native_success_latched=self._native_success_latched,
            native_terminal_failure_latched=self._native_failure_latched,
            native_terminal_latched=self._native_terminal_latched, native_terminal_tick=self._terminal_tick,
            native_outcome=asdict(self.native_outcome),
            outcomes_are_policy_observations=False,
            native_task_source_sha256=self.spec.native_source_sha256)

    def _reset_metric_receipt(self, metadata):
        snapshot = metric_initial_snapshot(self.task, self.spec,
            scene_object_resolver=self.resolve_scene_object, physics_time=self.physics_time)
        self.reset_receipt = dict(metadata, metric_only=True, policy_use_allowed=False,
            native_initial_snapshot=snapshot.to_report(), native_source_sha256=self.native_source_hashes,
            expert_actions_used=False, future_observations_used=False,
            full_native_physics_state_claimed=False)

    def close(self):
        if self._closed:
            return
        errors = []
        if hasattr(self, 'map_service'):
            try:
                self.map_service.close()
            except Exception as exc:
                errors.append(repr(exc))
        try:
            self.environment.shutdown()
        except Exception as exc:
            errors.append(repr(exc))
        self._closed, self._aligned = True, False
        if errors:
            raise MultitaskInfrastructureError('native runtime cleanup failed: ' + '; '.join(errors))


class AlignedOfficialSimulation(MultitaskSimulation):
    def reset_official(self, episode):
        if hasattr(self, 'ledger'):
            raise MultitaskProtocolError('official R10 cases require a fresh simulator process')
        path = Path(episode['path'])
        verified_inputs = verify_official_source_files(episode, self.spec, self.native_repo)
        variation = pickle.loads((path / 'variation_number.pkl').read_bytes())
        variation = int(variation)
        states = pickle.loads((path / 'target_state.pkl').read_bytes())
        validate_reset_metadata(self.spec, episode, archived_variation=variation, archived_states=states)
        alias = Path(episode.get('official_alias_path', ''))
        if (episode.get('mapping_status') != 'verified' or not alias.is_dir()
                or alias.resolve() != path.resolve()
                or alias.name != 'episode' + str(episode['official_episode_index'])
                or alias.parent.parent.name != 'variation' + str(variation)):
            raise MultitaskProtocolError('official source must use the verified variation-local alias mapping')
        if self.calibration_source.resolve() != path.resolve():
            raise MultitaskProtocolError('official sensors must be calibrated from this same reset source')
        demo = pickle.loads((path / 'low_dim_obs.pkl').read_bytes())
        demo.variation_number = variation
        plan = official_reset_plan(self.spec, variation)
        descriptions = None
        for step in plan.steps:
            if step.action == 'set_variation':
                self.task_environment.set_variation(*step.args)
            elif step.action.startswith('load_archived_motion_state'):
                self.task.var2target_state_list = deepcopy(states)
            elif step.action == 'reset_to_demo':
                descriptions, _ = self.task_environment.reset_to_demo(demo)
            elif step.action == 'initialize_rotation_driver_from_archived_variation':
                selected = ((states[variation] if variation in states else states[str(variation)])
                            if isinstance(states, Mapping) else states[variation])
                self.task.target_state_list = [dict(yaw_speed=deepcopy(selected['yaw_speed']), t0=0)]
            elif step.action == 'init_episode':
                self.task.init_episode(*step.args, **dict(step.kwargs))
            elif step.action == 'pyrep_step_once':
                self.pyrep.step()
            elif step.action == 'get_observation':
                self.task_environment.get_observation()
            elif step.action == 'disable_expert_plan':
                self.task.disable_expert_plan()
            else:
                raise MultitaskProtocolError('unimplemented declared official reset step: ' + step.action)
        self._start_alignment()
        self._reset_metric_receipt(dict(task=episode['task'], raw_episode_id=episode['raw_episode_id'],
            variation=variation, official_episode_index=episode['official_episode_index'],
            reset_kind='released_official_TEST_family_dispatch', original_source_replayed=True,
            reset_sequence=[asdict(step) for step in plan.steps],
            verified_source_inputs=verified_inputs,
            target_state_sha256=file_sha256(path / 'target_state.pkl')))
        observation, _ = self.observe()
        return descriptions, observation


class AlignedDevelopmentSimulation(MultitaskSimulation):
    def __init__(self, task_name, calibration_episode, *, source, split, protocol,
                 seed=0, replica=0, servo_lead=1.4):
        derived, identity = development_seed(task_name, source, split, seed=seed, replica=replica)
        if Path(calibration_episode).resolve() != Path(source['path']).resolve():
            raise MultitaskProtocolError('development calibration must match its HDF5 source')
        if Path(calibration_episode).is_dir():
            raise MultitaskProtocolError('declare native archive replay separately; development source is HDF5')
        self.development_source, self.development_split = dict(source), split
        self.development_seed_identity, self.replica = identity, int(replica)
        super().__init__(task_name, calibration_episode, protocol=protocol, seed=derived, servo_lead=servo_lead)

    def reset_development(self, *, variation=None):
        if hasattr(self, 'ledger'):
            raise MultitaskProtocolError('development cases also require a fresh simulator process')
        random.seed(self.seed)
        np.random.seed(self.seed)
        variation = (self.development_source.get('variation', self.spec.supported_variations[self.seed % self.spec.variation_count])
                     if variation is None else variation)
        if type(variation) is not int or variation not in self.spec.supported_variations:
            raise MultitaskProtocolError('unsupported development variation')
        self.task_environment.set_variation(variation)
        descriptions, _ = self.task_environment.reset()
        self.pyrep.step()
        self.task_environment.get_observation()
        self.task.disable_expert_plan()
        self._start_alignment()
        self._reset_metric_receipt(dict(task=self.spec.task_name,
            raw_episode_id=self.development_source['raw_episode_id'], variation=variation,
            source_split=self.development_split, source_path=str(self.calibration_source),
            source_hdf5_sha256=file_sha256(self.calibration_source), seed=self.seed,
            seed_identity=self.development_seed_identity, replica=self.replica,
            reset_kind='native_generated_synthetic_development_augmentation',
            original_source_replayed=False, official_TEST=False,
            source_fields_used='source identity for seed and frame-zero camera calibration only',
            source_motion_or_expert_trajectory_replayed=False))
        observation, _ = self.observe()
        return descriptions, observation


AlignedLocalSimulation = AlignedDevelopmentSimulation
