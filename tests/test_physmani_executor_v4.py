"""Mock protocol checks; no simulator, released weights or GPU execution."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.ablations import physmani_executor_v4 as executor


class FakeArm:
    def __init__(self):
        self.q = np.zeros(7)
        self.targets = np.full(7, .4)
        self._ik_target = SimpleNamespace(get_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
                                          set_pose=lambda value: None)

    def get_joint_count(self):
        return 7

    def get_joint_positions(self):
        return self.q.tolist()

    def set_joint_positions(self, value):
        self.q = np.asarray(value).copy()

    def get_joint_target_positions(self):
        return self.targets.tolist()

    def set_joint_target_positions(self, value):
        self.targets = np.asarray(value).copy()

    def get_tip(self):
        return SimpleNamespace(get_pose=lambda: [*self.q[:3], 0., 0., 0., 1.])


class FakePath:
    def __init__(self, arm, *, points=100, step_error=None):
        self._arm = arm
        self._path_points = np.linspace(np.zeros(7), np.full(7, .5), points).reshape(-1)
        self._rml_handle = None
        self._path_done = False
        self._joint_position_action = None
        self.step_error = step_error
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        if self.step_error:
            raise self.step_error
        self._joint_position_action = np.ones(7)
        self._arm.set_joint_target_positions(self._joint_position_action)
        return False


class FakeEnv:
    def __init__(self, *, path_error=None, step_error=None, points=100):
        self.arm = FakeArm()
        self.path = FakePath(self.arm, points=points, step_error=step_error)
        self.plan_calls = 0
        self.path_error = path_error
        self.map_service = SimpleNamespace(plan_path=self.plan)
        self.scene = SimpleNamespace(check_target_in_workspace=lambda value: True)
        self.ledger = SimpleNamespace(remaining=10., dt=.05)
        self.robot_model = SimpleNamespace(velocity_limits=np.full(7, 2.),
                                           lower=np.full(7, -3.), upper=np.full(7, 3.))
        self._native_success_latched = False
        self._requested_opening = 1.
        self.time = 0.
        self.commands = []

    def plan(self, *args, **kwargs):
        self.plan_calls += 1
        self.plan_kwargs = kwargs
        if self.path_error:
            raise self.path_error
        return self.path

    def command_joints(self, joints, *, opening, phase):
        current = self.arm.q.copy()
        requested = np.asarray(joints)
        bounded = np.clip(current + np.clip(
            requested-current, -self.robot_model.velocity_limits*self.ledger.dt,
            self.robot_model.velocity_limits*self.ledger.dt),
            self.robot_model.lower, self.robot_model.upper)
        self.arm.set_joint_target_positions(bounded)
        self.arm.set_joint_positions(bounded)
        self._requested_opening = float(opening >= .5)
        self.commands.append((requested.copy(), self._requested_opening, phase))
        self.time += self.ledger.dt
        self.ledger.remaining -= self.ledger.dt

    def observe(self):
        return {}, SimpleNamespace(timestamp=self.time)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        constants = ModuleType('pyrep.const')
        constants.ConfigurationPathAlgorithms = SimpleNamespace(RRTConnect=1)
        errors = ModuleType('pyrep.errors')
        errors.ConfigurationPathError = type('ConfigurationPathError', (Exception,), {})
        errors.IKError = type('IKError', (Exception,), {})
        self.modules = patch.dict(sys.modules, {'pyrep.const': constants, 'pyrep.errors': errors})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_executor(self, env, action=None):
        adapter = object.__new__(executor.PhysManiAdapter)
        adapter.config = {'baseline_decision_ticks': 10}
        adapter.observe = Mock()
        if action is None:
            action = [0., 0., 1., 0., 0., 0., 1., 0.]
        with patch.object(executor, 'physics_time', side_effect=lambda: env.time):
            return adapter.execute(env, action)

    def test_extraction_matches_author_target_reset_without_physics(self):
        env = FakeEnv()
        with patch.object(executor, 'physics_time', return_value=0.):
            poses, joints = executor.path_poses_without_physics(env.path)
        np.testing.assert_allclose(env.arm.q, 0.)
        np.testing.assert_allclose(env.arm.targets, 0.)
        self.assertEqual(len(poses), 99)
        np.testing.assert_array_equal(joints, env.path._path_points.reshape(-1, 7)[1:])

    def test_waypoint_rule_discards_ten_and_uses_strict_spacing(self):
        poses = np.zeros((14, 7))
        poses[11, 0] = .01
        poses[12, 0] = .02
        poses[13, 0] = .025
        self.assertEqual(executor.released_waypoint_indices(poses), [10, 12])

    def test_recoverable_planning_error_rests_and_still_closes_gripper(self):
        env = FakeEnv(path_error=RuntimeError('native planner rejected action'))
        report = self.run_executor(env)
        self.assertEqual(report['ticks'], 10)
        self.assertFalse(report['path_found'])
        self.assertEqual([command[1] for command in env.commands], [1.]*9+[0.])
        self.assertEqual(env.commands[-1][2], 'physmani_gripper')
        self.assertAlmostEqual(env.time, .5)

    def test_recoverable_rml_error_rests_and_still_closes_gripper(self):
        env = FakeEnv(step_error=ValueError('native RML rejected action'))
        report = self.run_executor(env)
        self.assertEqual(env.path.step_calls, 1)
        self.assertIn('native RML', report['arm_error'])
        self.assertEqual(report['ticks'], 10)
        self.assertEqual(env.commands[-1][1], 0.)

    def test_short_modified_path_resets_old_servo_target(self):
        env = FakeEnv(points=8)
        report = self.run_executor(env)
        self.assertEqual(report['error'], 'released_modified_path_too_short')
        self.assertEqual(env.path.step_calls, 0)
        np.testing.assert_allclose(env.arm.q, 0.)
        np.testing.assert_allclose(env.arm.targets, 0.)
        self.assertEqual(report['ticks'], 10)

    def test_quaternion_uses_released_numpy_default_tolerance(self):
        for quaternion_w in (1.000001, 1.0005):
            env = FakeEnv()
            report = self.run_executor(env, [0., 0., 1., 0., 0., 0., quaternion_w, 0.])
            valid = bool(np.isclose(quaternion_w, 1.))
            self.assertEqual(env.plan_calls, int(valid))
            self.assertEqual(report['error'] == 'nonunit_action_quaternion', not valid)
            self.assertEqual(report['ticks'], 10)

    def test_velocity_and_position_clipping_are_reported_each_tick(self):
        env = FakeEnv()
        env.robot_model.velocity_limits[:] = .1
        env.robot_model.upper[:] = .002
        report = self.run_executor(env)
        clipping = report['common_joint_command_clipping']
        self.assertEqual(env.path.step_calls, 9)
        self.assertTrue(env.plan_kwargs['ignore_collisions'])
        self.assertEqual(clipping['command_ticks'], 10)
        self.assertEqual(clipping['velocity_clip_ticks'], 10)
        self.assertEqual(clipping['position_clip_ticks'], 10)
        self.assertEqual(clipping['any_clip_ticks'], 10)
        self.assertAlmostEqual(clipping['maximum_velocity_clip_rad'], .995)
        self.assertAlmostEqual(clipping['maximum_position_clip_rad'], .005)
        self.assertAlmostEqual(clipping['maximum_requested_minus_applied_rad'], .998)

    def test_oom_is_not_converted_to_recoverable_arm_error(self):
        for message in ('CUDA out of memory', 'simulator connection lost', 'invalid object handle'):
            env = FakeEnv(path_error=RuntimeError(message))
            with self.assertRaises(executor.InfrastructureExecutionError):
                self.run_executor(env)
            self.assertEqual(env.commands, [])

    def test_planning_clock_violation_escapes_policy_failure(self):
        env = FakeEnv()
        original_plan = env.plan
        def advancing_plan(*args, **kwargs):
            result = original_plan(*args, **kwargs)
            env.time += .05
            return result
        env.map_service.plan_path = advancing_plan
        with self.assertRaises(executor.ProtocolInvariantError):
            self.run_executor(env)
        self.assertEqual(env.commands, [])
        self.assertFalse(issubclass(executor.ProtocolInvariantError, RuntimeError))
        self.assertFalse(issubclass(executor.ProtocolInvariantError, ValueError))

    def test_restore_corruption_escapes_arm_recovery(self):
        env = FakeEnv()
        env.arm.set_joint_target_positions = lambda value: None
        with self.assertRaises(executor.ProtocolInvariantError):
            self.run_executor(env)
        self.assertEqual(env.commands, [])

    def test_common_actuator_clock_error_is_not_a_policy_failure(self):
        env = FakeEnv()
        env.command_joints = Mock(side_effect=RuntimeError('task/physics clocks diverged: .1'))
        with self.assertRaises(executor.ProtocolInvariantError):
            self.run_executor(env)

    def test_baseline_config_changes_only_executor_description(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root/'config/framework/aligned_protocol_r06.json').read_text())
        baseline = json.loads((root/'config/framework/aligned_protocol_r06_baseline_v4.json').read_text())
        self.assertEqual(set(original), set(baseline))
        self.assertEqual([key for key in original if original[key] != baseline[key]], ['baseline_executor'])
        self.assertEqual(baseline['baseline_executor'], 'released_waypoints_no_hidden_physics_v4')


if __name__ == '__main__':
    unittest.main()
