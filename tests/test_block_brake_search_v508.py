from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.contracts import Pose, RobotState
from our_mani.reachability import TimedReachability
from our_mani.multitask.block_brake_search_v508 import (
    SmoothBlockEnvironment, find_stop, endpoint_interval, public_decision_scope)
from our_mani.multitask.block_clip_brake_v315 import public_decision_scope as original_global_scope
from our_mani.multitask.block_fresh_tail_v448 import SmoothBlockEnvironment as OriginalBoundary


class Model:
    name = 'test_original_gate_model'
    def __init__(self, acceleration, collision):
        self.lower = np.full(7, -2.)
        self.upper = np.full(7, 2.)
        self.velocity_limits = np.full(7, 2.)
        self.acceleration_limits = np.full(7, acceleration)
        self.collision = collision


class Tests(unittest.TestCase):
    def fixture(self, collision):
        q, v = np.zeros(7), np.r_[.2, np.zeros(6)]
        robot = RobotState(timestamp=1., pose=Pose(np.zeros(3), np.eye(3)),
            linear_velocity=np.zeros(3), angular_velocity=np.zeros(3),
            joints=q, joint_velocities=v, gripper_open=True, contact=False)
        return (TimedReachability(Model(6., collision)),
                TimedReachability(Model(4., collision)), robot, q, v)

    def search(self, args, **kw):
        return find_stop(*args, (.3,), dt=.05, deadline=10.,
                         command_velocity_limits=np.full(7, 2.), **kw)

    def test_original_control_and_global_scope_remain_the_same(self):
        self.assertIs(public_decision_scope, original_global_scope)
        self.assertIs(SmoothBlockEnvironment.advance, OriginalBoundary.advance)
        self.assertIs(SmoothBlockEnvironment._brake, OriginalBoundary._brake)
        self.assertIs(SmoothBlockEnvironment._issue, OriginalBoundary._issue)

    def test_additional_endpoint_avoids_obstacle_and_passes_complete_original_gates(self):
        def collision(q, offset):
            return bool(np.linalg.norm(q[:2]-[.03, 0.]) < .005)
        actual, reference, robot, q, v = self.fixture(collision)
        nominal = q+.5*v*.3
        self.assertFalse(reference.check_joints(robot, nominal, np.zeros(7), .3).feasible)
        plan, proof = self.search((actual, reference, robot, q, v))
        self.assertIsNotNone(plan, proof)
        self.assertTrue(actual.check_joints(robot, plan.q1, plan.v1, .3).feasible)
        self.assertTrue(reference.check_joints(robot, plan.q1, plan.v1, .3).feasible)
        self.assertEqual(plan.end, 1.3)
        np.testing.assert_array_equal(plan.v1, np.zeros(7))
        self.assertTrue(proof['actual_reaches_next_reference_target_not_assumed'])

    def test_no_collision_bypass_or_deadline_extension(self):
        args = self.fixture(lambda q, offset: True)
        plan, proof = self.search(args)
        self.assertIsNone(plan)
        self.assertTrue(proof['no_full_two_branch_collision_free_stop_found'])
        actual, reference, robot, q, v = self.fixture(lambda q, offset: False)
        robot = replace(robot, timestamp=9.8)
        plan, proof = self.search((actual, reference, robot, q, v))
        self.assertIsNone(plan)
        self.assertFalse(proof['trials'])

    def test_endpoint_interval_and_unclipped_native_dispatch_bounds(self):
        low, high = endpoint_interval(np.zeros(7), np.r_[.2, np.zeros(6)], np.full(7, 4.), .3)
        self.assertTrue(np.all(low<=high))
        self.assertAlmostEqual(low[0], -.02)
        self.assertAlmostEqual(high[0], .08)
        actual, reference, robot, q, v = self.fixture(lambda q, offset: False)
        # A far reference is never emitted if the old actuator would clip it.
        q = q+.5
        plan, proof = self.search((actual, reference, robot, q, v))
        self.assertIsNone(plan)
        with self.assertRaises(ValueError):
            find_stop(actual, reference, robot, q, v, (.3,), dt=.1, deadline=10.,
                      command_velocity_limits=np.full(7, 2.))


if __name__ == '__main__':
    unittest.main()
