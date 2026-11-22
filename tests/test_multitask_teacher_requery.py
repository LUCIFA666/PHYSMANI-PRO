import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask.train_teacher import FollowerConfig, TeacherPathPlanningFailure
from our_mani.multitask.train_teacher_requery import (
    CountedPlanningRequeryFollower, PlanningRequeryConfig)


class Robot:
    timestamp = 0.
    joints = np.zeros(7)
    pose = Pose(np.zeros(3))


def goal():
    return dict(tcp_xyzw=Pose(np.ones(3)).xyzw(), opening=1., stage='wp3',
                phase='transport', reference_body='spoke0', manipulated_body='cup0',
                reference_role='selected_spoke', reference_kind='observed_se3',
                anchor_id='cup:transport', manipulated_role='cup0')


def feedback(t):
    return dict(timestamp=t, gripper_actuation_done=False,
                measured_tcp_xyzw=Pose(np.zeros(3)).xyzw(), joint_limit_recovery_action=False)


class RequeryTest(unittest.TestCase):
    def make(self, planner=None):
        self.calls = 0
        def teacher():
            self.calls += 1
            return goal()
        def failed(_):
            raise TeacherPathPlanningFailure('actual planning failure fixture')
        return CountedPlanningRequeryFollower(
            teacher, planner or failed, FollowerConfig(.02, .1, .005, 3),
            initial_opening=1., requery_configuration=PlanningRequeryConfig())

    def command(self, follower, time):
        robot = Robot()
        robot.timestamp = time
        return follower.command(robot, np.full(7, -1.), np.full(7, 1.))

    def test_wait_counts_real_feedback_and_next_query_keeps_failure(self):
        follower = self.make()
        first = self.command(follower, 0.)
        self.assertFalse(first['teacher_execution_valid'])
        for tick in range(1, 10):
            self.assertFalse(follower.after_tick(feedback(tick * .05)))
            self.command(follower, tick * .05)
            self.assertEqual(self.calls, 1)
        self.assertFalse(follower.after_tick(feedback(.5)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(follower.requery_count, 0)
        old = dict(follower.commands[0])
        self.assertIn('planning_failure', old)
        self.assertNotIn('command_completed_at', old)
        self.command(follower, .5)
        self.assertEqual(self.calls, 2)
        self.assertEqual(follower.requery_count, 1)
        self.assertEqual(follower.commands[1]['actual_planning_requery_index'], 1)
        self.assertEqual(follower.commands[0], old)
        self.assertEqual(follower.commands[1]['fresh_query_after_failed_command_id'], 0)
        self.assertTrue(follower.commands[1]['same_episode_after_counted_wait'])

    def test_duplicate_or_skipped_feedback_is_rejected(self):
        for wrong in (.05, .2):
            follower = self.make()
            self.command(follower, 0.)
            follower.after_tick(feedback(.05))
            with self.assertRaises(ValueError):
                follower.after_tick(feedback(wrong))

    def test_feedback_clock_must_start_after_actual_query(self):
        follower = self.make()
        self.command(follower, 1.)
        with self.assertRaises(ValueError):
            follower.after_tick(feedback(.05))
        with self.assertRaises(ValueError):
            self.command(self.make(), float('nan'))

    def test_finished_budget_does_not_trigger_another_teacher_query(self):
        follower = self.make()
        self.command(follower, 0.)
        for tick in range(1, 11):
            follower.after_tick(feedback(tick * .05))
        follower.finish('fixed_tick_budget', .5)
        self.assertEqual(self.calls, 1)
        self.assertEqual(follower.requery_count, 0)
        self.assertEqual(len(follower.commands), 1)
        self.assertNotIn('command_completed_at', follower.commands[0])

    def test_infrastructure_error_is_not_caught_as_planning_failure(self):
        def broken(_):
            raise RuntimeError('infrastructure failure')
        follower = self.make(broken)
        with self.assertRaisesRegex(RuntimeError, 'infrastructure failure'):
            self.command(follower, 0.)
        self.assertIsNone(follower.failed)

    def test_budget_limit_holds_instead_of_requerying(self):
        follower = self.make()
        follower.requery_count = 20
        self.command(follower, 0.)
        for tick in range(1, 13):
            follower.after_tick(feedback(tick * .05))
            self.command(follower, tick * .05)
        self.assertEqual(self.calls, 1)
        self.assertEqual(follower.requery_count, 20)
        self.assertIsNotNone(follower.failed)


if __name__ == '__main__':
    unittest.main()
