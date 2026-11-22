import unittest
import numpy as np
from our_mani.contracts import Pose, ObjectState, RobotState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.block_clip_brake_v315 import SmoothBlockEnvironment
from test_block_reference_v186 import Native


class Tests(unittest.TestCase):
    def fixture(self):
        env = Native()
        b = SmoothBlockEnvironment(env)
        b._initialize_reference()
        b.reference_q = np.zeros(7)
        b.reference_v = np.ones(7)
        b.public_robot = RobotState(env.time, Pose([0., 0., 0.]), np.zeros(3), np.zeros(3),
            np.zeros(7), np.ones(7), gripper_open=True, contact=False)
        b.public_body = ObjectState('target', env.time, Pose([.5, 0., 0.]),
            np.array([[.5, 0, 0], [.51, 0, 0], [.5, .01, 0]]), np.zeros(3), np.zeros(3))
        b.local_dynamics = ConstantVelocityDynamics()
        return env, b

    def test_unexecutable_moving_reference_uses_checked_continuous_brake(self):
        env, b = self.fixture()
        # .2rad cannot be applied in one 50ms tick at the original 2rad/s.
        # A stop continuing the already-confirmed 1rad/s boundary can be.
        b._issue(np.full(7, .2), np.ones(7), 1., {})
        self.assertEqual(env.actions, 1)
        self.assertTrue(np.all(env.target > 0.))
        self.assertTrue(np.all(env.target < .1))
        e = b.reference_events[-1]['evidence']
        self.assertEqual(e['kind'], 'original_collision_checked_continuous_brake')
        self.assertEqual(e['actual_gate'], 'feasible_sampled_trajectory')
        self.assertEqual(e['reference_gate'], 'feasible_sampled_trajectory')
        self.assertTrue(e['exact_next_tick_original_collision_gate_passed'])
        self.assertFalse(b.clip_brake_active)

    def test_collision_rejects_brake_and_keeps_failure_explicit(self):
        env, b = self.fixture()
        env.robot_model.block = True
        b._issue(np.full(7, .2), np.ones(7), 1., {})
        self.assertEqual(env.actions, 1)
        np.testing.assert_array_equal(env.target, np.zeros(7))
        self.assertTrue(b.reference_events[-1]['actual_stopped_not_claimed'])
        self.assertFalse(b.clip_brake_active)

    def test_backend_error_after_tick_never_causes_a_second_dispatch(self):
        env, b = self.fixture()
        original = env.command_joints
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise ValueError('native failed after tick')
        env.command_joints = fail
        with self.assertRaisesRegex(ValueError, 'after tick'):
            b._issue(np.full(7, .2), np.ones(7), 1., {})
        self.assertEqual(env.actions, 1)
        self.assertFalse(b.clip_brake_active)


if __name__ == '__main__':
    unittest.main()
