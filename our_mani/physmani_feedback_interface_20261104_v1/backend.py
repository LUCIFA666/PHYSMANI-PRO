"""New16-task measured-feedback backend; old robot execution math is reused.

Only Root explicit execution launches Native. Generated TRAIN/development reset
is retained. A reset call may include native placement retries; this adapter
reports their count as unknown rather than calling the reset one attempt.
"""
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import importlib
import math
from pathlib import Path
import random
import numpy as np

from ..generic_reference_mpc_20261104_v1.native_backend import GenericTaskNativeBackend,ROOT,DT
from ..multitask_execution_pipeline_20261104_v1.registry import inventory,TASKS
from ..multitask_execution_pipeline_20261104_v1.native_adapter import NativeOutcomeAdapter
from .observation import CAMERAS,pack_feedback_snapshot,legacy_policy_packet


class PhysManiFeedbackBackendV1(GenericTaskNativeBackend):
    def __init__(self,task,calibration_episode,*,seed,variation,budget_ticks,
                 execute_native=False,gripper_velocity=.2,dataset_use='development',
                 renderer_profile='legacy_front_GL3_side_GL'):
        if execute_native is not True:raise ValueError('Root explicit Native execution required before any launch/path access')
        if task not in TASKS or type(seed)is not int or type(variation)is not int or variation<0:
            raise ValueError('Explicit one of16 tasks/seed/variation required')
        if type(budget_ticks)is not int or not 2<=budget_ticks<=200:raise ValueError('Fixed2..200physical tick budget required')
        if dataset_use not in ('development','train'):raise ValueError('Generated TRAIN/development only; no archived VAL/TEST claim')
        if isinstance(gripper_velocity,bool)or not np.isfinite(gripper_velocity)or gripper_velocity<=0:
            raise ValueError('Positive original mechanical gripper velocity required')
        if renderer_profile not in ('legacy_front_GL3_side_GL','official_all_GL'):
            raise ValueError('Explicit actual sensor renderer protocol required')
        calibration_episode=Path(calibration_episode).resolve(strict=True)
        if 'train_package_compressed'not in calibration_episode.parts or calibration_episode.name!='data.h5':
            raise ValueError('Original TRAIN public calibration only')
        contract=next(row for row in inventory(ROOT)['tasks']if row['task']==task)
        self._stack=ExitStack();self._closed=False;self._fault=None;self._ticks=0
        self.task_name,self.budget_ticks,self.gripper_velocity=task,budget_ticks,float(gripper_velocity)
        self.dataset_use=dataset_use;self.renderer_profile=renderer_profile
        self.requested_opening=None;self.last_collision_probe=None;self.last_ik_receipt=None;self.environment=None
        self._train_recorder=None;self._train_bodies=None;self._last_tick_observation=None;self._issued_training_reference=None
        try:
            for name in ('fixed_TRAIN104_geometry_native.lock','TRAIN_postcomplete_diagnostics_v1106.lock'):
                stream=self._stack.enter_context((ROOT/'runtime/framework'/name).open('a'))
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            random.seed(seed);np.random.seed(seed)
            from rlbench.environment import Environment
            from rlbench.observation_config import ObservationConfig
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import JointPosition
            from rlbench.action_modes.gripper_action_modes import Discrete
            from pyrep.errors import ConfigurationError,IKError
            from pyrep.backend import sim
            from rlbench.task_environment import _MAX_RESET_ATTEMPTS
            from rlbench.backend.task import Task as NativeTask
            config=ObservationConfig();config.set_all(False)
            config.joint_positions=config.joint_velocities=config.gripper_pose=config.gripper_joint_positions=True
            config.front_camera.rgb=config.front_camera.depth=True;config.front_camera.image_size=(256,256)
            # No native conditioned scalar opening, lowdim target or ID masks.
            self.environment=Environment(MoveArmThenGripper(JointPosition(),Discrete()),obs_config=config,headless=True)
            self.environment.launch();self.pyrep=self.environment._pyrep
            self.pyrep.set_simulation_timestep(DT)  # before task construction/reset
            observed_dt=float(self.pyrep.get_simulation_timestep())
            if not math.isfinite(observed_dt)or abs(observed_dt-DT)>1e-7:
                raise RuntimeError('Native timestep did not become original50ms before task construction')
            self._physics_clock=sim.simGetSimulationTime;self._ik_errors=(ConfigurationError,IKError)
            module=importlib.import_module(contract['native_module'])
            self.task_environment=self.environment.get_task(getattr(module,contract['native_class']))
            self.scene=self.environment._scene;self.task=self.task_environment._task
            self.arm,self.gripper=self.environment._robot.arm,self.environment._robot.gripper
            # init_task sets _bool_expert=True in all16 native classes. Initialize
            # once, then disable before reset; init_episode/cleanup retainFalse.
            self.scene.init_task();self.task.disable_expert_plan()
            self.task_environment.set_variation(variation)
            reset_before=(float(self._physics_clock()),float(self.task.t))
            self.descriptions,_=self.task_environment.reset()
            self.task.disable_expert_plan()
            reset_after=(float(self._physics_clock()),float(self.task.t))
            self.reset_receipt=dict(protocol='original_TaskEnvironment_generated_reset',external_reset_calls=1,
                internal_placement_attempts=None,internal_attempts_observed=False,official_maximum_placement_attempts=int(_MAX_RESET_ATTEMPTS),
                physics_before=reset_before[0],physics_after=reset_after[0],
                task_time_before=reset_before[1],task_time_after=reset_after[1],
                actual_physics_dt_before_reset=float(self.pyrep.get_simulation_timestep()),
                expert_plan_disabled_before_reset=True,descriptions_cached_from_reset=True,
                snapshot_origin_is_postreset=True,reset_is_not_zero_physics=True,
                generated_new_seed=seed,original_HDF_scene_replayed=False)
            self._configure_cameras(calibration_episode);self._limits=self._read_limits()
            self._origin=(float(self._physics_clock()),float(self.task.t))
            if getattr(self.task.success,'__func__',None)is not NativeTask.success:
                raise TypeError('Original registered Task.success required before metric binding')
            self._metric=NativeOutcomeAdapter(self.task,contract);self._metric.observe_after_physics(0.)
            self.requested_opening=float(np.mean(self.gripper.get_open_amount())>=.5)
            self.provenance=dict(task=task,seed=seed,variation=variation,dataset_use=dataset_use,
                reset=deepcopy(self.reset_receipt),calibration_use='TRAIN_camera_parameters_only',
                measured_feedback_schema='physmani_public_robot_RGBD_feedback_20261104_v1',
                renderer_profile=renderer_profile,physics_dt=DT,requested_budget_ticks=budget_ticks,
                measured_fingers_are_not_contact_or_held=True,
                legacy_NN_gripper_channel='0_maskFalse_original_ABI',
                Native_held_body_pose_success_or_instance_masks_in_policy=False,
                official_VAL_or_TEST=False,formal_task_success_rate=False,real_time_guaranteed=False)
        except BaseException:
            self.close();raise

    def _configure_cameras(self,calibration):
        import h5py
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode
        self.cameras={};self.camera_setup_evidence={}
        with h5py.File(calibration,'r')as source:
            for name in CAMERAS:
                params=source['cam_params/'+name]
                K=np.asarray(params['intrinsics'],dtype=float);T=np.asarray(params['extrinsics'],dtype=float)
                near=float(params['near'][()]);far=float(params['far'][()])
                if K.shape!=(3,3)or T.shape!=(4,4)or not np.isfinite(K).all()or not np.isfinite(T).all()or not 0<near<far:
                    raise ValueError('Original finite camera K3/T4/actualclips required')
                camera=VisionSensor('cam_front')if name=='front'else VisionSensor.create([256,256],explicit_handling=True,render_mode=RenderMode.OPENGL)
                camera.set_resolution([256,256]);camera.set_explicit_handling(True)
                mode=RenderMode.OPENGL3 if name=='front'and self.renderer_profile=='legacy_front_GL3_side_GL'else RenderMode.OPENGL
                camera.set_render_mode(mode);camera.set_matrix(T.copy())
                camera.set_perspective_angle(math.degrees(2*math.atan(256/(2*abs(float(K[0,0]))))))
                camera.set_near_clipping_plane(near);camera.set_far_clipping_plane(far)
                self.cameras[name]=camera
                self.camera_setup_evidence[name]=dict(actual_render_mode=camera.get_render_mode().name,
                    requested_K=K.tolist(),actual_K=np.asarray(camera.get_intrinsic_matrix()).tolist(),
                    actual_T=np.asarray(camera.get_matrix()).tolist(),
                    actual_clips=[camera.get_near_clipping_plane(),camera.get_far_clipping_plane()])

    def read_snapshot(self):
        self._check_live_clock();before=self._clock_tuple()
        rgb=[];depth=[];K=[];T=[];clips=[];render_modes=[]
        for camera in self.cameras.values():
            camera.handle_explicitly()
            rgb.append((np.clip(camera.capture_rgb(),0,1)*255).astype(np.uint8))
            depth.append(np.asarray(camera.capture_depth(in_meters=True),dtype=np.float32))
            K.append(np.asarray(camera.get_intrinsic_matrix(),dtype=float))
            matrix=np.asarray(camera.get_matrix(),dtype=float)
            if matrix.shape!=(4,4):raise ValueError('Actual PyRep get_matrix must return original4x4 T')
            T.append(matrix.copy())  # Never assign4x4 into a3x4 slice.
            clips.append([camera.get_near_clipping_plane(),camera.get_far_clipping_plane()])
            render_modes.append(camera.get_render_mode().name)
        packet=pack_feedback_snapshot(physics_timestamp=before[0],task_timestamp=before[1],
            physics_time_origin=self._origin[0],task_time_origin=self._origin[1],rgb=np.stack(rgb),depth_m=np.stack(depth),
            K=np.stack(K),T=np.stack(T),camera_clips=np.asarray(clips),render_modes=render_modes,
            q=self.arm.get_joint_positions(),dq=self.arm.get_joint_velocities(),tcp_pose=self.arm.get_tip().get_pose(),
            finger_joint_positions=self.gripper.get_joint_positions(),finger_joint_velocities=self.gripper.get_joint_velocities(),
            finger_open_amounts=self.gripper.get_open_amount())
        self._assert_unchanged(before,'complete current measured RGBD/robot feedback')
        return packet

    def public_sample(self):return legacy_policy_packet(self.read_snapshot())

    def mechanical_state(self,actuator_state):
        result=super().mechanical_state(actuator_state);before=self._clock_tuple()
        result.update(finger_joint_positions=np.asarray(self.gripper.get_joint_positions(),float),
                      finger_joint_velocities=np.asarray(self.gripper.get_joint_velocities(),float))
        self._assert_unchanged(before,'actual finger mechanical feedback')
        return result


    def install_train_label_recorder(self,sink):
        """Original real-h recorder with complete private role identities."""
        from dataclasses import replace
        from ..generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder
        from ..multitask.specs import get_task_spec
        from ..multitask.native_tasks import resolve_native_roles
        if self.dataset_use!='train'or self._ticks!=0 or self._train_recorder is not None:
            raise ValueError('One TRAIN-only separate recorder before first control tick required')
        from pyrep.objects.object import Object
        self._train_joint_roles={}
        if self.task_name.removesuffix('_high_speed')=='pick_moving_target_on_the_table':
            self._train_bodies={'target_block':self.task.target_block}
            self._train_role_metadata=dict(dynamic_roots=['target_block'],manipulated_roles=['target_block'],
                all_body_role_names=['target_block'],selected_role_observed=False)
        else:
            spec=get_task_spec(self.task_name)
            body=replace(spec,roles=tuple(r for r in spec.roles if r.kind=='body'))
            joints=replace(spec,roles=tuple(r for r in spec.roles if r.kind=='joint'))
            self._train_bodies=resolve_native_roles(self.task,body,scene_object_resolver=Object.get_object)
            self._train_joint_roles=resolve_native_roles(self.task,joints,scene_object_resolver=Object.get_object)
            self._train_role_metadata=dict(dynamic_roots=list(spec.dynamic_roots),
                manipulated_roles=list(spec.primary_manipulated_roles),all_body_role_names=list(self._train_bodies),
                joint_role_names=list(self._train_joint_roles),selected_role_observed=False)
        self._train_recorder=TRAINLabelRecorder(self,sink)
        return self._train_recorder

    def _training_label_snapshot(self):
        row=super()._training_label_snapshot();before=self._clock_tuple()
        row['joint_role_state_private']={name:dict(native_handle=int(obj.get_handle()),
            position=float(obj.get_joint_position()),velocity=float(obj.get_joint_velocity()))
            for name,obj in getattr(self,'_train_joint_roles',{}).items()}
        self._assert_unchanged(before,'TRAIN-only current joint-role labels')
        return row
