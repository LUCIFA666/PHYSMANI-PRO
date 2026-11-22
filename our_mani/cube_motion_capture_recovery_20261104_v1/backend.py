"""Original Cube task mechanics with measured clocks and separate label hooks."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from ..cube_v72_live_20261104_v1.backend import CubeTaskNativeBackendV1
from ..physmani_feedback_interface_20261104_v1.observation import pack_feedback_snapshot


class CubeMotionObservationBackendV1(CubeTaskNativeBackendV1):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dataset_use='train'
        self._capture_cache=None;self._capture_cache_clock=None
        self._issued_training_reference=None;self._last_tick_observation=None
        self.provenance.update(actual_physics_clock=True,
            time_basis='simGetSimulationTime-minus-actual-postreset-origin',
            commanded_motion='initial_measured_q_and_OPEN_only',
            private_labels_not_policy_inputs=True,contact_supervision_ready=False)

    @property
    def _ticks(self):return int(self.raw.ledger.ticks)

    def clock(self):
        if not hasattr(self,'_physics_origin'):
            raise RuntimeError('Physical clock origin has not been established by reset')
        return float(self._physics_clock())-self._physics_origin

    def _capture(self):
        self._check_live_clock();before=self._clock_tuple();now=self.clock()
        if self._capture_cache_clock==now:return self._capture_cache
        # Exactly one original Cube capture, including its original uint8 rint
        # conversion. The measured packet and perceptor consume the same arrays.
        observation=super().observe_rgbd()
        observation=dict(observation,timestamp=now)
        names=('front','8','16','36');cameras=self.raw.cameras
        matrices=[np.asarray(cameras[n].get_matrix(),float)for n in names]
        if any(v.shape!=(4,4)for v in matrices):raise ValueError('Actual complete4x4 camera matrices required')
        packet=pack_feedback_snapshot(physics_timestamp=before[0],task_timestamp=before[1],
            physics_time_origin=self._physics_origin,task_time_origin=self._task_origin,
            rgb=np.stack([observation['observation/images'][n]for n in names]),
            depth_m=np.stack([np.asarray(observation['observation/depths'][n],np.float32)for n in names]),
            K=np.stack([cameras[n].get_intrinsic_matrix()for n in names]),T=np.stack(matrices),
            camera_clips=np.asarray([[cameras[n].get_near_clipping_plane(),cameras[n].get_far_clipping_plane()]for n in names]),
            render_modes=[cameras[n].get_render_mode().name for n in names],
            q=self.get_joint_positions(),dq=self.get_joint_velocities(),tcp_pose=self.arm.get_tip().get_pose(),
            finger_joint_positions=self.gripper.get_joint_positions(),
            finger_joint_velocities=self.gripper.get_joint_velocities(),finger_open_amounts=self.gripper.get_open_amount())
        self._assert_unchanged(before,'one-frame public RGBD/proprioception capture')
        if packet['timestamp']!=now:raise ValueError('Public frame and actual physical clock differ')
        self._capture_cache=(observation,packet);self._capture_cache_clock=now
        return self._capture_cache

    def observe_rgbd(self):return self._capture()[0]
    def read_snapshot(self):return self._capture()[1]

    def tick(self,phase='TRAIN_keep_open_motion_observation'):
        before=self._clock_tuple();result=super().tick(phase);after=self._clock_tuple()
        self._last_tick_observation=dict(tick_before=int(before[3]),tick_after=int(after[3]),
            physics_time_before=before[0],physics_time_after=after[0],
            task_time_before=before[1],task_time_after=after[1],phase=phase,
            physics_dt_observed=after[0]-before[0],task_dt_observed=after[1]-before[1])
        return result

    def _bind_issued_training_reference(self,issued):
        if self._issued_training_reference is not None and self._issued_training_reference['physics_tick_before']==self._ticks:
            raise ValueError('One actually issued reference per real physical tick')
        self._issued_training_reference=deepcopy(dict(issued,physics_tick_before=self._ticks,actual_clock_before=self.clock()))

    def _training_label_snapshot(self):
        """Called only by the separate TRAIN recorder after public inference."""
        self._check_live_clock();before=self._clock_tuple();state=self.read_state()
        p=state['object_position'];q=state['object_quaternion'];tcp=np.r_[state['tcp_position'],state['tcp_quaternion']]
        result=dict(physics_tick=self._ticks,physics_time_absolute=before[0],task_time_absolute=before[1],
            physics_time_origin=self._physics_origin,task_time_origin=self._task_origin,
            actual_relative_physics_timestamp=self.clock(),actual_relative_task_timestamp=before[1]-self._task_origin,
            last_physical_tick=deepcopy(self._last_tick_observation),issued_reference=deepcopy(self._issued_training_reference),
            target_position_world=np.asarray(p).tolist(),target_quaternion_world_xyzw=np.asarray(q).tolist(),
            target_rotation_world=Rotation.from_quat(q).as_matrix().tolist(),
            target_velocity_world=state['object_causal_fd_velocity'].tolist()if state['object_causal_twist_valid']else None,
            target_velocity_observed=bool(state['object_causal_twist_valid']),
            target_native_velocity_world=state['object_native_linear_velocity'].tolist(),
            target_native_velocity_is_true_kinematic_velocity_certificate=False,
            tcp_pose_world_xyzw=tcp.tolist(),q=state['q'].tolist(),dq=state['dq'].tolist(),
            finger_joint_positions=state['finger_positions'].tolist(),finger_joint_velocities=state['finger_velocities'].tolist(),
            finger_open_amounts=state['finger_open_amounts'].tolist(),servo_target_positions=state['servo_target_positions'].tolist(),
            requested_opening=float(state['requested_opening']),held=bool(state['held']),
            native_success=bool(state['native_success']),native_result=deepcopy(state['native_result']),
            task_name_label_only=self.task_name,contact_force_observed=False,
            native_held_is_not_force_contact=True,teacher_contact_supervision_available=False)
        self._assert_unchanged(before,'private TRAIN labels only')
        return result
