import math
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.aligned.protocol import TickLedger
from our_mani.local_mpc_20261102.native import (
    DT, MovingBlockScenario, NativeBackendError, TruthNativeBackend,
)


class FakeIKError(Exception):
    pass


class Body:
    def __init__(self, position=(0., 0., .8)):
        self.pose = np.r_[position, [0., 0., 0., 1.]]
        self.velocity, self.angular = np.zeros(3), np.zeros(3)
        self.collidable = True

    def get_pose(self): return self.pose.copy()
    def set_pose(self, value, **kwargs): self.pose = np.asarray(value, dtype=float).copy()
    def get_position(self): return self.pose[:3].copy()
    def set_position(self, value): self.pose[:3] = value
    def get_velocity(self): return self.velocity.copy(), self.angular.copy()
    def is_collidable(self): return self.collidable
    def set_collidable(self, value): self.collidable = value
    def get_bounding_box(self): return [-.025, .025, -.025, .025, -.025, .025]


class FakeArm:
    def __init__(self, raw):
        self.raw = raw
        self.q, self.dq, self.targets = np.zeros(7), np.zeros(7), np.zeros(7)
        self._ik_target, self.tip = Body((.3, 0., .9)), Body((.3, 0., .9))
        self.jacobian_mode, self.sampling_mode = "ok", "ok"
        self.disable_dynamics_calls = []

    def get_joint_intervals(self): return [False]*7, [[-2., 4.]]*7
    def get_joint_upper_velocity_limits(self): return [2.]*7
    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.dq.copy()
    def get_joint_target_positions(self): return self.targets.copy()
    def set_joint_target_positions(self, value): self.targets = np.asarray(value, dtype=float).copy()
    def get_tip(self): return self.tip
    def check_arm_collision(self): return self.raw.collision_check()

    def set_joint_positions(self, value, disable_dynamics=False):
        self.disable_dynamics_calls.append(disable_dynamics)
        self.q = np.asarray(value, dtype=float).copy()
        if disable_dynamics:
            self.raw.physics += 2*DT
            self.targets = self.q.copy()

    def solve_ik_via_jacobian(self, position, quaternion):
        self.q += .3; self.targets += .4
        self._ik_target.set_pose(np.r_[position, quaternion])
        if self.jacobian_mode == "clock_leak": self.raw.physics += DT
        if self.jacobian_mode == "fail": raise FakeIKError("no local solution")
        if self.jacobian_mode == "unexpected": raise RuntimeError("backend broke")
        return [.1]*7

    def solve_ik_via_sampling(self, position, quaternion, **kwargs):
        self.q += .2; self.targets += .2
        if self.sampling_mode == "fail": raise FakeIKError("no sampled solution")
        return np.full((1, 7), .2 if self.sampling_mode == "ok" else 3.)


class FakeGripper:
    def __init__(self, raw):
        self.raw, self.held = raw, []
        self.amounts, self.requested = np.ones(2), 1.
        self.attempts, self.releases = 0, 0
        self.detected = True

    def release(self): self.held.clear(); self.releases += 1
    def grasp(self, obj):
        self.attempts += 1
        if self.detected and obj not in self.held: self.held.append(obj)
        return self.detected
    def actuate(self, opening, velocity): self.requested = opening; return bool(np.all(self.amounts == opening))
    def get_grasped_objects(self): return self.held[:]
    def get_open_amount(self): return self.amounts.copy()
    def get_joint_positions(self): return self.amounts * .04
    def get_joint_velocities(self): return np.zeros(2)
    def get_model_bounding_box(self): return [-.04, .04, -.05, .05, -.1, .1]
    def set_joint_target_positions(self, values): pass
    def set_joint_positions(self, values, disable_dynamics=False):
        self.amounts[:] = 1
        if disable_dynamics: self.raw.physics += 2*DT


class FakeTask:
    def __init__(self, raw):
        self.raw, self.t, self.step_id = raw, 0., 0
        self.target_block, self.success_detector = Body(), Body((.25, -.15, .95))
        self.target_state_list = []
        self.area = [.05, -.45, .8, .35, .45, .8]
        self.forced_success = False
        self.forced_failure = False
        self.expert_disabled = False

    def disable_expert_plan(self): self.expert_disabled = True
    def get_graspable_objects(self): return [self.target_block]
    def success(self): return self.forced_success, self.forced_success or self.forced_failure


class FakeRaw:
    def __init__(self):
        self.physics, self.time, self.closed = 0., 0., False
        self.arm, self.gripper = FakeArm(self), FakeGripper(self)
        self.task = FakeTask(self)
        self.home_joints = np.zeros(7)
        self.ledger = TickLedger()
        self.task_environment = SimpleNamespace(reset=self.reset)
        self.pyrep = SimpleNamespace(step=self.raw_step)
        self.scene = SimpleNamespace(step=self.scene_step)
        self.observe_leaks_clock = False
        self.collision_result = False
        self.collision_mode = 'ok'
        self.collision_samples = []
        self.map_service = SimpleNamespace(check=self.collision_check, static=[], ghosts=[],
            target=self.task.target_block, env=self, queries=0)

    def collision_check(self):
        self.collision_samples.append(self.arm.q.copy())
        self.arm.targets += .2
        self.arm._ik_target.pose[:3] += .1
        if self.collision_mode == 'raise': raise RuntimeError('native collision backend failed')
        if self.collision_mode == 'clock_leak': self.physics += DT
        if self.collision_mode == 'velocity_leak': self.task.target_block.velocity[0] += 1
        return self.collision_result

    def reset(self): self.physics += 2*DT; self.task.t = 0.; self.task.step_id = 0
    def raw_step(self): self.physics += DT
    def _start_alignment(self):
        self.ledger = TickLedger()
        self.time, self._requested_opening, self._native_success_latched = 0., 1., False

    def scene_step(self):
        self.physics += DT
        delta = np.clip(self.arm.targets-self.arm.q, -.01, .01)
        self.arm.q += delta; self.arm.dq = delta/DT
        self.arm.tip.pose[:3] += delta[:3]*.1
        self.gripper.amounts += np.clip(self.gripper.requested-self.gripper.amounts, -.25, .25)
        if self.task.target_block not in self.gripper.held:
            segment = self.task.target_state_list[-1]
            position = segment['x'] + segment['v']*(self.task.t-segment['t0'])
            self.task.target_block.pose[:3] = position
            # Deliberately emulate the unreliable zero native velocity of a
            # kinematically moved object. The backend must use observed FD.
            self.task.target_block.velocity[:] = 0
        self.task.t += DT; self.task.step_id += 1

    def observe(self):
        if self.observe_leaks_clock: self.physics += DT
        return dict(timestamp=self.time, **{'observation/images': {}, 'observation/depths': {},
                                            'observation/calibrations': {}}), None

    def close(self): self.closed = True


class TruthNativeTests(unittest.TestCase):
    def backend(self, *, scenario=None, budget_ticks=200):
        raw = FakeRaw()
        scenario = scenario or MovingBlockScenario('uniform', (.25, -.15, .8), (0., .02, 0.))
        backend = TruthNativeBackend('CPU_fixture_NO_HDF_opened', scenario, budget_ticks=budget_ticks,
                                    _raw=raw, _physics_clock=lambda: raw.physics, _ik_errors=(FakeIKError,))
        self.addCleanup(backend.close)
        return backend, raw

    def test_home_reset_hidden_staging_is_separate_and_reads_do_not_step(self):
        backend, raw = self.backend()
        self.assertEqual(backend.clock(), 0.)
        self.assertEqual(backend.remaining_ticks(), 200)
        self.assertAlmostEqual(backend.staging_receipt['raw_physics_seconds'], .45)
        np.testing.assert_array_equal(raw.arm.q, raw.home_joints)
        before = backend._clock_tuple()
        first = backend.read_state()
        backend.observe_truth(); backend.observe_rgbd(); backend.get_joint_positions(); backend.get_joint_velocities()
        backend.finish_home_reset()
        self.assertEqual(before, backend._clock_tuple())
        self.assertTrue(first['truth_diagnostic_only'])
        self.assertFalse(first['official_public_input_evaluation'])
        self.assertFalse(first['object_causal_twist_valid'])
        np.testing.assert_array_equal(first['object_linear_velocity'], np.zeros(3))
        self.assertEqual(backend.initial_snapshot['native_bounds']['target_dimensions'], [.05]*3)
        np.testing.assert_array_equal(first['lift_goal_position'], [.25, -.15, .95])

    def test_arm_and_gripper_share_one_tick_and_true_native_success_does_not_stop_hold(self):
        backend, raw = self.backend(budget_ticks=12)
        backend.set_joint_target_positions(np.full(7, .1))
        backend.actuate_gripper(0.)
        backend.actuate_gripper(0.)
        self.assertEqual(raw.gripper.attempts, 1)
        self.assertEqual(backend.clock(), 0.)
        raw.task.forced_success = True
        for _ in range(10):
            backend.actuate_gripper(0.)
            state = backend.tick('hold_diagnostic')
            self.assertTrue(state['held'])
            self.assertTrue(state['native_success'])
        self.assertAlmostEqual(backend.clock(), .5)
        self.assertFalse(backend.terminal())
        self.assertGreater(float(state['q'][0]), 0)
        self.assertLess(state['opening'], 1.)
        records = backend.pop_tick_records()
        self.assertEqual([r['counted_tick'] for r in records], list(range(1, 11)))
        self.assertEqual(backend.tick_records, [])
        backend.tick(); backend.tick()
        self.assertTrue(backend.terminal())
        with self.assertRaises(NativeBackendError): backend.tick()
        with self.assertRaises(NativeBackendError): backend.finish_home_reset()

    def test_uniform_and_single_switch_have_continuous_actual_pose_velocity(self):
        scenario = MovingBlockScenario('switch', (.25, -.15, .8), (0., .02, 0.), 3, (0., .04, 0.))
        backend, raw = self.backend(scenario=scenario)
        for tick in range(1, 7):
            state = backend.tick()
            speed = .02 if tick <= 3 else .04
            expected_y = -.15 + .02 * min(tick, 3)*DT + .04 * max(tick-3, 0)*DT
            self.assertAlmostEqual(state['object_position'][1], expected_y)
            np.testing.assert_allclose(state['object_linear_velocity'], [0., speed, 0.], atol=1e-12)
            np.testing.assert_array_equal(state['object_native_linear_velocity'], [0., 0., 0.])
            self.assertTrue(state['object_causal_twist_valid'])
            self.assertIn('causal_world_SE3_difference', state['velocity_source'])
        self.assertEqual(len(backend.motion_events), 1)
        self.assertEqual(backend.motion_events[0]['counted_tick'], 3)
        self.assertEqual(backend.motion_events[0]['first_affected_tick'], 4)
        self.assertEqual(len(raw.task.target_state_list), 2)

    def test_switch_does_not_teleport_an_already_native_held_target(self):
        scenario = MovingBlockScenario('held_switch', (.25, -.15, .8), (0., .02, 0.), 1, (0., .04, 0.))
        backend, raw = self.backend(scenario=scenario)
        backend.actuate_gripper(0.)
        backend.tick(); before = raw.task.target_block.get_position()
        backend.tick()
        np.testing.assert_array_equal(raw.task.target_block.get_position(), before)
        self.assertFalse(backend.motion_events[0]['applied'])
        self.assertEqual(backend.motion_events[0]['reason'], 'native_target_already_held')

    def test_ik_restores_actual_joints_servo_and_ik_target_without_physics(self):
        backend, raw = self.backend()
        raw.arm.q = np.full(7, -.1); raw.arm.targets = np.full(7, .15)
        before = backend._clock_tuple(), raw.arm.q.copy(), raw.arm.targets.copy(), raw.arm._ik_target.get_pose()
        solution = backend.solve_ik([.25, -.15, .9], [0, 1, 0, 0])
        np.testing.assert_array_equal(solution, np.full(7, .1))
        self.assertEqual(backend._clock_tuple(), before[0])
        np.testing.assert_array_equal(raw.arm.q, before[1]); np.testing.assert_array_equal(raw.arm.targets, before[2])
        np.testing.assert_array_equal(raw.arm._ik_target.get_pose(), before[3])
        self.assertFalse(raw.arm.disable_dynamics_calls[-1])
        self.assertEqual(backend.last_ik_receipt['physics_steps'], 0)
        raw.arm.jacobian_mode = 'fail'
        raw.arm.sampling_mode = 'fail'
        self.assertIsNone(backend.solve_ik([.25, -.15, .9], [0, 1, 0, 0]))
        self.assertFalse(backend.last_ik_receipt['success'])
        np.testing.assert_array_equal(raw.arm.q, before[1])
        self.assertEqual(backend.clock(), 0.)

    def test_ik_unexpected_failure_still_restores_and_clock_leak_is_not_hidden(self):
        backend, raw = self.backend()
        raw.arm.jacobian_mode = 'unexpected'
        q, targets = raw.arm.q.copy(), raw.arm.targets.copy()
        with self.assertRaises(RuntimeError): backend.solve_ik([0, 0, .9], [0, 0, 0, 1])
        np.testing.assert_array_equal(raw.arm.q, q); np.testing.assert_array_equal(raw.arm.targets, targets)
        raw.arm.jacobian_mode = 'clock_leak'
        with self.assertRaises(NativeBackendError): backend.solve_ik([0, 0, .9], [0, 0, 0, 1])
        self.assertTrue(backend.terminal())

    def test_native_limits_are_minimum_plus_range_and_not_controller_reference_limits(self):
        backend, _ = self.backend()
        limits = backend.joint_limits()
        np.testing.assert_array_equal(limits['lower'], [-2.]*7)
        np.testing.assert_array_equal(limits['upper'], [2.]*7)
        np.testing.assert_array_equal(limits['velocity'], [2.]*7)
        self.assertTrue(limits['reference_parameters_are_not_native_physical_limits'])
        limits['lower'][0] = -999
        self.assertEqual(backend.joint_limits()['lower'][0], -2.)
        with self.assertRaises(ValueError): backend.set_joint_target_positions(np.full(7, 3.))
        with self.assertRaises(ValueError): backend.solve_ik([0, 0, .9], [0, 0, 0, 0])

    def test_rgbd_clock_leak_is_rejected_and_reset_clears_control_records(self):
        backend, raw = self.backend()
        backend.tick()
        backend.reset_scenario()
        self.assertEqual(backend.clock(), 0.)
        self.assertEqual(backend.tick_records, [])
        raw.observe_leaks_clock = True
        with self.assertRaises(NativeBackendError): backend.observe_rgbd()

    def test_static_collision_probe_restores_joint_servo_ik_and_preserves_native_velocity(self):
        backend, raw = self.backend()
        raw.arm.q[:] = -.1; raw.arm.dq[:] = .3; raw.arm.targets[:] = .2
        raw.task.target_block.velocity[:] = [.01, .02, 0]
        before = backend._clock_tuple(), raw.arm.q.copy(), raw.arm.targets.copy(), raw.arm._ik_target.get_pose()
        requested = np.full(7, .4)
        self.assertFalse(backend.probe_joint_collision(requested))
        np.testing.assert_array_equal(raw.collision_samples[-1], requested)
        np.testing.assert_array_equal(raw.arm.q, before[1]); np.testing.assert_array_equal(raw.arm.targets, before[2])
        np.testing.assert_array_equal(raw.arm._ik_target.get_pose(), before[3])
        np.testing.assert_array_equal(raw.arm.dq, [.3]*7)
        np.testing.assert_array_equal(raw.task.target_block.velocity, [.01, .02, 0])
        self.assertEqual(backend._clock_tuple(), before[0])
        self.assertTrue(backend.last_collision_probe['actual_q_qdot_servo_ik_and_target_native_state_preserved'])
        self.assertIn('held_geometry_excluded', backend.last_collision_probe['scope'])
        raw.collision_result = True
        self.assertTrue(backend.probe_joint_collision(requested))
        count = len(raw.collision_samples)
        self.assertTrue(backend.probe_joint_collision(np.full(7, 3.)))
        self.assertEqual(len(raw.collision_samples), count)

    def test_collision_query_errors_restore_and_state_mutation_fails_closed(self):
        backend, raw = self.backend()
        q, targets, ik = raw.arm.q.copy(), raw.arm.targets.copy(), raw.arm._ik_target.get_pose()
        raw.collision_mode = 'raise'
        with self.assertRaises(RuntimeError): backend.probe_joint_collision(np.full(7, .2))
        np.testing.assert_array_equal(raw.arm.q, q); np.testing.assert_array_equal(raw.arm.targets, targets)
        np.testing.assert_array_equal(raw.arm._ik_target.get_pose(), ik)
        raw.collision_mode = 'velocity_leak'
        with self.assertRaises(NativeBackendError): backend.probe_joint_collision(np.full(7, .2))
        self.assertTrue(backend.terminal())

    def test_real_native_terminal_failure_stops_but_success_keeps_hold_available(self):
        backend, raw = self.backend()
        raw.task.forced_failure = True
        state = backend.tick()
        self.assertTrue(state['native_result']['terminal_failure'])
        self.assertTrue(backend.terminal())
        with self.assertRaises(NativeBackendError): backend.tick()


if __name__ == '__main__':
    unittest.main()
