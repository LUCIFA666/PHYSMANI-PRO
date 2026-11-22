import unittest

import numpy as np

from our_mani.multitask.train_teacher import TeacherPathPlanningFailure
from tests import test_multitask_teacher_periodic_replan as fixtures


class PeriodicBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PeriodicFollowerTests()

    def test_periodic_fresh_query_that_fails_keeps_two_independent_counters(self):
        follower, robot, paths, _ = self.fixture.fixture()
        self.fixture.query(follower, robot, 0.)
        for tick in range(1, 11):
            self.fixture.tick(follower, tick * .05)
            if tick < 10:
                self.fixture.query(follower, robot, tick * .05)

        def fail(goal):
            raise TeacherPathPlanningFailure('new query path fails')

        follower.planner = fail
        self.fixture.query(follower, robot, .5)
        self.assertEqual(follower.periodic_requery_count, 1)
        self.assertEqual(follower.requery_count, 0)
        self.assertIn('planning_failure', follower.commands[1])
        self.assertEqual(follower.commands[1]['fresh_query_after_periodic_interrupted_command_id'], 0)
        for tick in range(11, 21):
            self.fixture.tick(follower, tick * .05)
            if tick < 20:
                self.fixture.query(follower, robot, tick * .05)
        self.fixture.query(follower, robot, 1.)
        self.assertEqual(follower.periodic_requery_count, 1)
        self.assertEqual(follower.requery_count, 1)
        self.assertEqual(follower.commands[2]['fresh_query_after_failed_command_id'], 1)
        self.assertNotIn('fresh_query_after_periodic_interrupted_command_id', follower.commands[2])
        self.assertEqual(paths[0].closed, 1)
        self.assertTrue(all('command_completed_at' not in row for row in follower.commands))

    def test_public_recovery_after_cut_delays_actual_query_without_reexecuting_path(self):
        follower, robot, paths, _ = self.fixture.fixture()
        self.fixture.query(follower, robot, 0.)
        for tick in range(1, 11):
            self.fixture.tick(follower, tick * .05)
            if tick < 10:
                self.fixture.query(follower, robot, tick * .05)
        robot.joints = np.full(7, 1.01)
        result = self.fixture.query(follower, robot, .5)
        self.assertTrue(result['joint_limit_recovery_action'])
        self.assertEqual(follower.command_id, 0)
        self.fixture.tick(follower, .55, recovery=True)
        self.assertEqual(paths[0].closed, 1)
        self.assertEqual(paths[0].steps, 10)
        self.assertEqual(follower.commands[0]['periodic_interruption']['actual_feedback_ticks'], 10)
        self.assertEqual(follower.commands[0]['post_query_actual_feedback_ticks'], 11)
        self.assertEqual(follower.periodic_requery_count, 0)
        robot.joints = np.zeros(7)
        self.fixture.query(follower, robot, .55)
        self.assertEqual(follower.periodic_requery_count, 1)
        self.assertEqual(follower.commands[0]['periodic_interruption']['fresh_generated_at'], .55)

    def test_public_recovery_at_due_boundary_has_a_truthful_deferred_cut(self):
        follower, robot, paths, _ = self.fixture.fixture()
        self.fixture.query(follower, robot, 0.)
        for tick in range(1, 10):
            self.fixture.tick(follower, tick * .05)
            self.fixture.query(follower, robot, tick * .05)
        self.fixture.tick(follower, .5, recovery=True)
        self.assertEqual(paths[0].closed, 0)
        self.assertEqual(follower.commands[0]['periodic_replan_deferred_reason'], 'actual_public_joint_limit_recovery')
        self.fixture.query(follower, robot, .5)
        self.fixture.tick(follower, .55)
        self.assertEqual(paths[0].closed, 1)
        self.assertEqual(follower.commands[0]['periodic_interruption']['actual_feedback_ticks'], 11)
        self.assertEqual(follower.periodic_requery_count, 0)
        self.assertNotIn('command_completed_at', follower.commands[0])


if __name__ == '__main__':
    unittest.main()
