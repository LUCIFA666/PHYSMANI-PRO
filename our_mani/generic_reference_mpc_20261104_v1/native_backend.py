"""Task-generic Native mechanics; object/held/success never enter the policy.

Only explicit Root execution constructs this backend. Generated development
reset is implemented for all16 tasks. Official archive-reset evaluation is a
separate, currently unbound admission; generated reset is never called TEST.
"""
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import importlib
import math
from pathlib import Path
import random
import numpy as np

from ..local_mpc_20261102.native import TruthNativeBackend
from ..local_mpc_20261102.executor import JointLimits, ExecutionConfig, ShortHorizonExecutor
from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from ..multitask_execution_pipeline_20261104_v1.registry import inventory, TASKS
from ..multitask_execution_pipeline_20261104_v1.native_adapter import NativeOutcomeAdapter
from .contracts import vector
from .public_observation import CAMERAS, pack_public_observation

ROOT = Path(__file__).resolve().parents[2]
DT = .05


class GenericTaskNativeBackend:
    # Exact robot-only routines; do not reuse Cube-specific constructor/read_state.
    _read_limits = TruthNativeBackend._read_limits
    get_joint_positions = TruthNativeBackend.get_joint_positions
    get_joint_velocities = TruthNativeBackend.get_joint_velocities
    _restoration_errors = TruthNativeBackend._restoration_errors
    solve_ik = TruthNativeBackend.solve_ik
    set_joint_target_positions = TruthNativeBackend.set_joint_target_positions

    def __init__(self, task, calibration_episode, *, seed, variation, budget_ticks,
                 execute_native=False, gripper_velocity=.2, dataset_use='development'):
        if execute_native is not True:
            raise ValueError('Root explicit Native execution required before imports/launch')
        if task not in TASKS or type(seed) is not int or type(variation) is not int:
            raise ValueError('Explicit real task/seed/variation required')
        if type(budget_ticks) is not int or budget_ticks < 2 or budget_ticks > 200:
            raise ValueError('Predeclared physical budget2..200ticks; no implicit budget extension')
        if dataset_use not in ('development','train'):
            raise ValueError('This generated-reset adapter does not admit official VAL/TEST')
        calibration_episode = Path(calibration_episode).resolve(strict=True)
        if 'train_package_compressed' not in calibration_episode.parts or calibration_episode.name != 'data.h5':
            raise ValueError('Public frame-zero camera calibration from original TRAIN HDF only')
        contract = next(r for r in inventory(ROOT)['tasks'] if r['task'] == task)
        self._stack = ExitStack(); self._closed = False; self._fault = None; self._ticks = 0
        self.task_name, self.budget_ticks, self.gripper_velocity = task, budget_ticks, float(gripper_velocity)
        self.requested_opening = None; self.last_collision_probe = None; self.last_ik_receipt = None
        self.environment = None
        self.dataset_use=dataset_use;self._train_recorder=None;self._train_bodies=None
        self._last_tick_observation=None;self._issued_training_reference=None
        try:
            for name in ('fixed_TRAIN104_geometry_native.lock', 'TRAIN_postcomplete_diagnostics_v1106.lock'):
                path = ROOT / 'runtime/framework' / name
                stream = self._stack.enter_context(path.open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            random.seed(seed); np.random.seed(seed)
            from rlbench.environment import Environment
            from rlbench.observation_config import ObservationConfig
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import JointPosition
            from rlbench.action_modes.gripper_action_modes import Discrete
            from pyrep.errors import ConfigurationError, IKError
            from pyrep.backend import sim
            config = ObservationConfig(); config.set_all(False)
            config.joint_positions = config.gripper_pose = True
            # Never request native conditioned gripper_open or instance masks.
            config.front_camera.rgb = config.front_camera.depth = True
            config.front_camera.image_size = (256, 256)
            self.environment = Environment(MoveArmThenGripper(JointPosition(), Discrete()), obs_config=config, headless=True)
            self.environment.launch()
            module = importlib.import_module(contract['native_module'])
            self.task_environment = self.environment.get_task(getattr(module, contract['native_class']))
            self.task_environment.set_variation(variation)
            self.descriptions, _ = self.task_environment.reset()
            self.task = self.task_environment._task
            self.task.disable_expert_plan()
            self.scene, self.pyrep = self.environment._scene, self.environment._pyrep
            self.arm, self.gripper = self.environment._robot.arm, self.environment._robot.gripper
            self.pyrep.set_simulation_timestep(DT)
            self._physics_clock = sim.simGetSimulationTime
            self._ik_errors = (ConfigurationError, IKError)
            self._configure_cameras(calibration_episode)
            self._limits = self._read_limits()
            self._origin = (float(self._physics_clock()), float(self.task.t))
            self._metric = NativeOutcomeAdapter(self.task, contract)
            self._metric.observe_after_physics(0.)
            self.requested_opening = float(np.mean(self.gripper.get_open_amount()) >= .5)
            self.provenance = dict(task=task, seed=seed, variation=variation,
                reset_kind='RLBench_TaskEnvironment_set_variation_then_reset_generated_development',
                original_episode_motion_replayed=False, official_TEST=False,
                calibration_use='TRAIN_framezero_cameras_only_not_object_motion_or_expert',
                Native_held_object_pose_success_or_instance_masks_in_policy=False,
                gripper_scalar_in_policy='fixed0_maskFalse; no HDF gripper_open',
                Native_graspable_dispatch='task.get_graspable_objects private legal Native grasp calls',
                renderer_profile='RLBench_default_frontOPENGL3_PhysMani_sideOPENGL',
                physics_dt=DT, requested_budget_ticks=budget_ticks,
                first_native_terminal_ends_episode=True, real_time_guaranteed=False)
        except BaseException:
            self.close(); raise

    def _configure_cameras(self, calibration):
        import h5py
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode
        self.cameras = {}
        with h5py.File(calibration, 'r') as source:
            for name in CAMERAS:
                params = source['cam_params/' + name]
                K = np.asarray(params['intrinsics']); T = np.asarray(params['extrinsics'])
                camera = VisionSensor('cam_front') if name == 'front' else VisionSensor.create(
                    [256,256], explicit_handling=True, render_mode=RenderMode.OPENGL)
                camera.set_resolution([256,256]); camera.set_explicit_handling(True)
                camera.set_render_mode(RenderMode.OPENGL3 if name == 'front' else RenderMode.OPENGL)
                camera.set_matrix(T[:3])
                camera.set_perspective_angle(math.degrees(2 * math.atan(256 / (2 * abs(float(K[0,0]))))))
                camera.set_near_clipping_plane(float(params['near'][()]))
                camera.set_far_clipping_plane(float(params['far'][()]))
                self.cameras[name] = camera

    def clock(self):
        # Measured relative simulator time, not a synthesized tick*DT timestamp.
        return float(self._physics_clock())-self._origin[0]
    def remaining_ticks(self): return max(0, self.budget_ticks - self._ticks)
    def _clock_tuple(self): return float(self._physics_clock()), float(self.task.t), self._ticks

    def _check_live_clock(self):
        if self._closed or self._fault: raise RuntimeError('Closed/faulted Native backend')
        p, t, _ = self._clock_tuple()
        if abs(p-self._origin[0]-self._ticks*DT)>2e-5 or abs(t-self._origin[1]-self.clock())>2e-5:
            raise RuntimeError('Uncounted Native physics/task step')

    def _assert_unchanged(self, before, operation):
        if self._clock_tuple() != before: raise RuntimeError(operation + ' changed Native clock')

    def terminal(self):
        return (self._closed or self._fault is not None or self.remaining_ticks() == 0
            or self._metric.metric_receipt()['native_first_terminal'] is not None)

    def _can_command(self):
        self._check_live_clock()
        if self.terminal(): raise RuntimeError('Terminal Native episode cannot receive actions')

    def joint_limits(self): return deepcopy(self._limits)

    def mechanical_state(self, actuator_state):
        """Internal measured robot-only controller state; never task body/held."""
        self._check_live_clock(); before = self._clock_tuple()
        result = dict(time=self.clock(), q=self.get_joint_positions(), dq=self.get_joint_velocities(),
            tcp_pose=vector(self.arm.get_tip().get_pose(), 7, 'actual TCP'),
            finger_open_amounts=vector(self.gripper.get_open_amount(), 2, 'actual two finger coordinates'),
            actuator_state=deepcopy(actuator_state), requested_opening=self.requested_opening)
        self._assert_unchanged(before, 'robot-only proprioception read')
        return result

    def public_sample(self):
        self._check_live_clock(); before = self._clock_tuple()
        rgbs, depths, Ks, Ts, clips = [], [], [], [], []
        for camera in self.cameras.values():
            camera.handle_explicitly()
            rgbs.append((np.clip(camera.capture_rgb(), 0, 1) * 255).astype(np.uint8))
            depths.append(np.asarray(camera.capture_depth(in_meters=True), dtype=np.float32))
            Ks.append(np.asarray(camera.get_intrinsic_matrix(), float))
            T = np.eye(4); T[:3] = camera.get_matrix(); Ts.append(T)
            clips.append([camera.get_near_clipping_plane(),camera.get_far_clipping_plane()])
        packet = pack_public_observation(timestamp=self.clock(), rgb=np.stack(rgbs), depth_m=np.stack(depths),
            K=np.stack(Ks), T=np.stack(Ts),camera_clips=np.asarray(clips),
            q=self.arm.get_joint_positions(), tcp_pose=self.arm.get_tip().get_pose())
        self._assert_unchanged(before, 'same-frame public RGBD/proprioception read')
        return packet

    def actuate_gripper(self, opening):
        self._can_command(); before = self._clock_tuple()
        if not np.isfinite(opening) or not 0 <= opening <= 1: raise ValueError('Finite binary opening intent required')
        requested = float(opening >= .5)
        if requested != self.requested_opening:
            if requested: self.gripper.release()
            else:
                for obj in self.task.get_graspable_objects(): self.gripper.grasp(obj)
            self.requested_opening = requested
        done = bool(self.gripper.actuate(requested, velocity=self.gripper_velocity))
        self._assert_unchanged(before, 'original nonblocking Native gripper actuation')
        return done

    def tick(self, phase='generic_reference_tracking'):
        self._can_command(); before = self._clock_tuple()
        self.scene.step(); self._ticks += 1
        after = self._clock_tuple()
        if abs(after[0] - before[0] - DT) > 2e-5 or abs(after[1] - before[1] - DT) > 2e-5:
            self._fault = 'Native tick not exactly one50ms physics/task step'; raise RuntimeError(self._fault)
        self._last_tick_observation=dict(tick_before=before[2],tick_after=after[2],
            physics_time_before=before[0],physics_time_after=after[0],
            task_time_before=before[1],task_time_after=after[1],phase=str(phase),
            physics_dt_observed=after[0]-before[0],task_dt_observed=after[1]-before[1])
        self._metric.observe_after_physics(self.clock())

    def install_train_label_recorder(self, sink):
        """Explicit separate TRAIN label stream, before initial row/first tick."""
        from .train_labels import TRAINLabelRecorder, resolve_training_bodies
        if self.dataset_use != 'train' or self._ticks != 0 or self._train_recorder is not None:
            raise ValueError('One TRAIN recorder must be installed before the first real tick')
        from pyrep.objects.object import Object
        self._train_bodies,self._train_role_metadata=resolve_training_bodies(
            self.task,self.task_name,Object.get_object)
        self._train_recorder=TRAINLabelRecorder(self,sink)
        return self._train_recorder

    def _bind_issued_training_reference(self, applied):
        if self.dataset_use != 'train' or self._train_recorder is None:
            raise ValueError('Actual reference issue recording is TRAIN-only')
        self._check_live_clock()
        old=self._issued_training_reference
        if old is not None and old['physics_tick_before']==self._ticks:
            raise ValueError('One issued robot reference per physical tick')
        self._issued_training_reference=deepcopy(dict(applied,physics_tick_before=self._ticks,
            actual_clock_before=self.clock()))

    def _training_label_snapshot(self):
        """Privileged TRAIN pose/attachment labels; never called by public_sample."""
        if self.dataset_use != 'train' or self._train_recorder is None or self._train_bodies is None:
            raise ValueError('Separate TRAIN label recorder has not been installed')
        from scipy.spatial.transform import Rotation
        self._check_live_clock();before=self._clock_tuple();bodies={}
        for role,obj in self._train_bodies.items():
            pose=vector(obj.get_pose(),7,'TRAIN body pose')
            if not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-5,rtol=0):
                raise ValueError('Actual TRAIN body quaternion must be unit')
            bodies[role]=dict(native_handle=int(obj.get_handle()),pose_world_xyzw=pose,
                rotation_world=Rotation.from_quat(pose[3:]).as_matrix())
        row=dict(task_name_label_only=self.task_name,body_roles=deepcopy(self._train_role_metadata),
            bodies=bodies,physics_tick=before[2],physics_time_absolute=before[0],
            task_time_absolute=before[1],physics_time_origin=self._origin[0],task_time_origin=self._origin[1],
            actual_relative_physics_timestamp=before[0]-self._origin[0],
            actual_relative_task_timestamp=before[1]-self._origin[1],
            nominal_tick_dt_diagnostic=DT,tick_times_DT_diagnostic_only=self._ticks*DT,
            q=self.get_joint_positions(),dq=self.get_joint_velocities(),
            tcp_pose=vector(self.arm.get_tip().get_pose(),7,'TRAIN actual TCP'),
            finger_open_amounts=vector(self.gripper.get_open_amount(),2,'TRAIN actual two fingers'),
            finger_joint_positions=vector(self.gripper.get_joint_positions(),2,'TRAIN actual finger joints'),
            finger_joint_velocities=vector(self.gripper.get_joint_velocities(),2,'TRAIN actual finger joint velocities'),
            Native_grasped_handles_label_only=[int(o.get_handle()) for o in self.gripper.get_grasped_objects()],
            last_physical_tick=deepcopy(self._last_tick_observation),
            issued_reference=deepcopy(self._issued_training_reference))
        self._assert_unchanged(before,'TRAIN-only true-clock/body label read')
        return row

    def probe_joint_collision(self, q):
        """Full current scene, no dynamic-target blanket exclusion; zero steps.

        Task-object pose/velocity snapshots below are restoration diagnostics
        ONLY. They never enter mechanical_state/public_sample/model inputs.
        """
        self._can_command(); q = vector(q, 7, 'collision q')
        if np.any(q < self._limits['lower']) or np.any(q > self._limits['upper']): return True
        before = self._clock_tuple(); actual = self.get_joint_positions(); dq = self.get_joint_velocities()
        servo = np.asarray(self.arm.get_joint_target_positions(), float); ik = self.arm._ik_target.get_pose()
        objects = {o.get_handle(): o for o in self.task.get_base().get_objects_in_tree(exclude_base=False)}
        objects.update({o.get_handle(): o for o in self.task.get_graspable_objects()})
        saved = {k: (np.asarray(o.get_pose(), float), tuple(np.asarray(v, float) for v in o.get_velocity())) for k, o in objects.items()}
        collision = None
        try:
            self.arm.set_joint_positions(q.tolist(), disable_dynamics=False)
            collision = bool(self.arm.check_arm_collision())
            return collision
        finally:
            self.arm.set_joint_positions(actual.tolist(), disable_dynamics=False)
            self.arm.set_joint_target_positions(servo.tolist()); self.arm._ik_target.set_pose(ik)
            self._assert_unchanged(before, 'full-scene sampled collision restoration')
            _, _, restored = self._restoration_errors(actual, servo, ik)
            restored &= np.allclose(self.arm.get_joint_velocities(), dq, atol=1e-9, rtol=0)
            for key, (pose, velocity) in saved.items():
                obj = objects[key]
                restored &= np.allclose(obj.get_pose(), pose, atol=1e-6, rtol=0)
                restored &= all(np.allclose(a, b, atol=1e-9, rtol=0) for a, b in zip(obj.get_velocity(), velocity))
            self.last_collision_probe = dict(collision=collision, physics_steps=0,
                actual_robot_and_task_objects_preserved=bool(restored),
                scope='arm_vs_full_current_scene; no task-wide collision exclusion',
                full_swept_path_or_contact_success_certified=False)
            if not restored:
                self._fault = 'Collision query changed robot/task state'; raise RuntimeError(self._fault)

    def metric_receipt(self): return self._metric.metric_receipt()

    def close(self):
        if not self._closed:
            try:
                if self.environment is not None: self.environment.shutdown()
            finally:
                self._closed = True; self._stack.close()


def make_executor(backend, *, reference_acceleration=12., reference_jerk=1000., train_label_sink=None):
    """Same100ms/two50ms executor and completion-latch Native motor protocol."""
    completed = CompletedGripperBackend(backend)
    native = backend.joint_limits()
    limits = JointLimits(lower=native['lower'], upper=native['upper'], velocity=native['velocity'],
        acceleration=np.full(7, reference_acceleration), jerk=np.full(7, reference_jerk))
    cls=ShortHorizonExecutor;extra={}
    if train_label_sink is not None:
        from .train_labels import TRAINRecordingExecutor
        cls=TRAINRecordingExecutor;extra['recorder']=backend.install_train_label_recorder(train_label_sink)
    executor = cls(completed, limits, ExecutionConfig(), initial_gripper_opening=backend.requested_opening,**extra)
    if extra:extra['recorder'].record_initial(executor.state())
    return executor, completed
