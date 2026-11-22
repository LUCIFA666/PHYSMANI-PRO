"""Recorded reset parity, without PyRep, native physics, GPU or TEST outcomes."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.aligned.protocol import TickLedger
from our_mani.contracts import Pose, RobotState
from our_mani.perception import CameraCalibration
from our_mani.multitask.native_tasks import metric_initial_snapshot, verify_native_sources
from our_mani.multitask.specs import get_task_spec, NATIVE_COMMON_SOURCE_SHA256
from our_mani.multitask.simulation import OFFICIAL_SOURCE_FILES
from our_mani.multitask.reset_parity import (capture_recorded_initial_conditions, pending_tolerance_profile,
    compare_initial_conditions, compare_five_way, digest, METHODS, RecordedConditionError)


LEGACY = dict(robot_position_m=.001, robot_orientation_rad=.002, robot_joints_rad=.0001,
              target_position_m=.0001, fixture_pose_component=.0001, camera_calibration_component=.00001)


class Body:
    def __init__(self, name, handle): self.name, self.handle = name, handle
    def get_pose(self): return [0., 0., .8, 0., 0., 0., 1.]
    def get_name(self): return self.name
    def get_handle(self): return self.handle
    def get_velocity(self): return [0., .2, 0.], [0., 0., .5]
    def get_objects_in_tree(self): return []


def record():
    spec = get_task_spec('moving_basketball_in_hoop')
    tip = Body('tip', 1)
    arm = SimpleNamespace(get_tip=lambda: tip, get_joint_positions=lambda: [0.] * 7,
        get_joint_velocities=lambda: [0.] * 7, get_joint_intervals=lambda: ([False]*7, [[-3., 6.]]*7),
        get_joint_upper_velocity_limits=lambda: [2.] * 7)
    gripper = SimpleNamespace(get_open_amount=lambda: [1., 1.], get_grasped_objects=lambda: [])
    driver = dict(x=[0., 0., .8], v=[0., .2, 0.], a=[0., 0., 0.], t0=0.)
    task = SimpleNamespace(ball=Body('ball', 2), hoop=Body('hoop', 3), t=0., step_id=0,
        var_index=0, stage='wp0', _has_been_picked=False, target_state_list=[[driver, driver]],
        var2target_state_list={0: [driver, driver]}, success=lambda: (False, False),
        robot=SimpleNamespace(arm=arm, gripper=gripper))
    snapshot = metric_initial_snapshot(task, spec, scene_object_resolver=lambda name: Body(name, 4),
                                       physics_time=lambda: 1.).to_report()
    sources = {spec.native_source_relative_path: spec.native_source_sha256, **NATIVE_COMMON_SOURCE_SHA256}
    reset = dict(task=spec.task_name, raw_episode_id='episode5', variation=0, official_episode_index=0,
        reset_kind='released_official_TEST_family_dispatch', metric_only=True, policy_use_allowed=False,
        native_source_sha256=sources, native_initial_snapshot=snapshot, original_source_replayed=True,
        expert_actions_used=False, future_observations_used=False, target_state_sha256='0'*64, reset_sequence=[],
        verified_source_inputs=dict(task_source_sha256=spec.native_source_sha256, task_scene_sha256='1'*64,
            source_required_sha256={name: '0'*64 for name in OFFICIAL_SOURCE_FILES}))
    calibration = CameraCalibration(np.eye(3), Pose(np.zeros(3)))
    cameras = {name: SimpleNamespace(get_render_mode=lambda: SimpleNamespace(name='OPENGL'))
               for name in ('front', '8', '16', '36')}
    observation = {'timestamp': 0., 'observation/images': {name: np.zeros((2, 2, 3), np.uint8) for name in cameras},
        'observation/depths': {name: np.ones((2, 2)) for name in cameras},
        'observation/calibrations': {name: calibration for name in cameras}}
    robot = RobotState(0., Pose.from_xyzw(tip.get_pose()), np.zeros(3), np.zeros(3), [0.]*7, [0.]*7)
    env = SimpleNamespace(_aligned=True, ledger=TickLedger(), time=0., physics_time=lambda: 1., task=task,
        arm=arm, pyrep=SimpleNamespace(get_simulation_timestep=lambda: .05), reset_receipt=reset, cameras=cameras,
        protocol=dict(dt=.05, simulation_seconds=10., gripper_velocity=.2, current_instance_masks=True,
                      information_setting='shared_public_RGBD_and_declared_priors'),
        map_service=SimpleNamespace(map=[dict(name='table', pose=[0., 0., 0., 0., 0., 0., 1.],
                                            bounding_box=[-1., 1.] * 3, mesh_sha256='2'*64)]))
    return capture_recorded_initial_conditions(env, observation, robot), env


class ParityTests(unittest.TestCase):
    def test_pending_profile_never_passes_formal_gate_even_for_identical_record(self):
        a, _ = record();profile = pending_tolerance_profile(LEGACY)
        checked = compare_initial_conditions(a, deepcopy(a), profile)
        assert checked['recorded_conditions_match'] is True
        assert checked['formal_acceptance_eligible'] is False and checked['accepted'] is False
        assert 'new_tolerances_not_TRAIN_validated_and_frozen' in checked['problems']
        with self.assertRaises(RecordedConditionError):
            pending_tolerance_profile(dict(LEGACY, target_position_m=.01))

    def test_quaternion_sign_is_equivalent_for_robot_body_camera_and_static_map(self):
        a, _ = record();b = deepcopy(a)
        n = b['reset_receipt']['native_initial_snapshot']
        for body in n['bodies'].values(): body['pose_xyzw'][3:] = [-v for v in body['pose_xyzw'][3:]]
        n['robot']['tip_pose_xyzw'][3:] = [-v for v in n['robot']['tip_pose_xyzw'][3:]]
        p = b['public_initial_conditions']
        p['robot_pose'][3:] = [-v for v in p['robot_pose'][3:]]
        for camera in p['cameras'].values(): camera['camera_to_world'][3:] = [-v for v in camera['camera_to_world'][3:]]
        p['static_map'][0]['pose'][3:] = [-v for v in p['static_map'][0]['pose'][3:]]
        checked = compare_initial_conditions(a, b, pending_tolerance_profile(LEGACY), require_frozen=False)
        assert checked['accepted'] is True and not checked['formal_acceptance_eligible']
        assert max(value for key, value in checked['measured_deltas'].items() if 'orientation_rad' in key) == 0.

    def test_actual_rotation_angle_uses_so3_angle_and_rejects_real_rotation(self):
        a, _ = record();b = deepcopy(a);profile = pending_tolerance_profile(LEGACY)
        profile['thresholds']['body_orientation_rad'] = .005
        b['reset_receipt']['native_initial_snapshot']['bodies']['ball']['pose_xyzw'][3:] = Rotation.from_rotvec([0., 0., .01]).as_quat().tolist()
        checked = compare_initial_conditions(a, b, profile, require_frozen=False)
        assert checked['accepted'] is False
        self.assertAlmostEqual(checked['measured_deltas']['native.bodies.ball.orientation_rad'], .01)

    def test_missing_native_velocity_field_is_fail_closed(self):
        a, _ = record();b = deepcopy(a)
        del b['reset_receipt']['native_initial_snapshot']['bodies']['hoop']['linear_velocity_m_s']
        checked = compare_initial_conditions(a, b, pending_tolerance_profile(LEGACY), require_frozen=False)
        assert checked['accepted'] is False and any('missing_field' in p for p in checked['problems'])

    def test_real_schedule_change_is_rejected_even_if_schedule_hash_is_updated(self):
        a, _ = record();b = deepcopy(a)
        n = b['reset_receipt']['native_initial_snapshot']
        n['motion_schedule_history'][0][0]['v'][1] += .01
        n['active_motion_schedule'] = deepcopy(n['motion_schedule_history'][-1])
        n['motion_schedule_sha256'] = digest(n['motion_schedule_history'])
        checked = compare_initial_conditions(a, b, pending_tolerance_profile(LEGACY), require_frozen=False)
        assert checked['accepted'] is False and any('motion_schedule' in p for p in checked['problems'])

    def test_handles_pid_and_wall_time_are_not_physical_conditions(self):
        a, _ = record();b = deepcopy(a)
        for body in b['reset_receipt']['native_initial_snapshot']['bodies'].values(): body['native_handle_metric_only'] += 100
        b.update(evaluation_process_pid=999, compute_seconds=100., wall_seconds=200.)
        assert compare_initial_conditions(a, b, pending_tolerance_profile(LEGACY), require_frozen=False)['accepted']

    def test_complete_five_way_means_all_ten_pairs_and_missing_method_rejected(self):
        a, _ = record();values = {method: deepcopy(a) for method in METHODS}
        checked = compare_five_way(values, pending_tolerance_profile(LEGACY), require_frozen=False)
        assert checked['accepted'] is True and len(checked['checks']) == 10
        del values['physmani']
        assert compare_five_way(values, pending_tolerance_profile(LEGACY))['complete'] is False

    def test_metric_capture_rejects_post_action_time(self):
        _, env = record();env.ledger.ticks = 1
        with self.assertRaises(RecordedConditionError): capture_recorded_initial_conditions(env)


if __name__ == '__main__':
    unittest.main()
