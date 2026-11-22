from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask.train_teacher import FollowerConfig, TeacherPathPlanningFailure
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_periodic_replan import (
    PeriodicFeedbackReplanningFollower, PeriodicReplanningConfig)


class Path:
    def __init__(self, done=False):
        self.done = done
        self.closed = 0
        self.steps = 0

    def step(self):
        self.steps += 1
        return self.done

    def get_executed_joint_position_action(self):
        return np.zeros(7)

    def close(self):
        self.closed += 1


class PeriodicFollowerTests(unittest.TestCase):
    def fixture(self, *, done=False, opening=0., initial_opening=0., settle=0, fail=False):
        paths = []
        goals = []

        def teacher():
            goals.append(len(goals))
            return dict(tcp_xyzw=[0., 0., 0., 0., 0., 0., 1.], opening=opening,
                        stage='wp3', phase='transport', reference_body='spoke0', manipulated_body='cup0',
                        reference_role='selected_spoke', reference_kind='observed_se3',
                        anchor_id='test', manipulated_role='cup0')

        def planner(goal):
            if fail:
                raise TeacherPathPlanningFailure('fixed failed path')
            path = Path(done)
            paths.append(path)
            return path

        follower = PeriodicFeedbackReplanningFollower(teacher, planner,
            FollowerConfig(.02, .1, .005, settle), initial_opening=initial_opening,
            periodic_configuration=PeriodicReplanningConfig(), failure_configuration=PlanningRequeryConfig())
        robot = SimpleNamespace(timestamp=0., joints=np.zeros(7), pose=Pose(np.zeros(3), np.eye(3)))
        return follower, robot, paths, goals

    def query(self, follower, robot, time):
        robot.timestamp = time
        return follower.command(robot, -np.ones(7), np.ones(7))

    def tick(self, follower, time, *, gripper_done=False, recovery=False):
        return follower.after_tick(dict(timestamp=time, gripper_actuation_done=gripper_done,
            joint_limit_recovery_action=recovery, measured_tcp_xyzw=[0., 0., 0., 0., 0., 0., 1.]))

    def test_ten_real_feedbacks_close_and_censor_before_actual_fresh_query(self):
        follower, robot, paths, goals = self.fixture()
        self.query(follower, robot, 0.)
        for tick in range(1, 10):
            self.tick(follower, tick * .05)
            self.assertEqual(follower.state, 'follow_path')
            self.assertEqual(paths[0].closed, 0)
            self.query(follower, robot, tick * .05)
        self.tick(follower, .5)
        old = follower.commands[0]
        self.assertEqual(paths[0].closed, 1)
        self.assertEqual(follower.periodic_requery_count, 0)
        self.assertFalse(old['periodic_interruption']['actual_fresh_query_performed'])
        self.assertNotIn('command_completed_at', old)
        self.assertEqual(old['executed_tick_count'], 10)
        self.query(follower, robot, .5)
        self.assertEqual(follower.periodic_requery_count, 1)
        self.assertEqual(len(goals), 2)
        self.assertTrue(old['periodic_interruption']['actual_fresh_query_performed'])
        self.assertEqual(follower.commands[1]['fresh_query_after_periodic_interrupted_command_id'], 0)
        self.assertNotIn('fresh_query_after_failed_command_id', follower.commands[1])
        self.assertFalse(follower.completed)

    def test_actual_gripper_actuation_is_never_interrupted_at_ten(self):
        follower, robot, paths, _ = self.fixture(done=True, opening=0., initial_opening=1.)
        self.query(follower, robot, 0.)
        for tick in range(1, 13):
            self.tick(follower, tick * .05)
            self.assertEqual(follower.state, 'actuate')
            self.assertEqual(paths[0].closed, 0)
            self.query(follower, robot, tick * .05)
        self.assertIn('periodic_replan_deferred_reason', follower.commands[0])
        self.assertTrue(self.tick(follower, .65, gripper_done=True))
        self.assertIn('command_completed_at', follower.commands[0])
        self.query(follower, robot, .65)
        self.assertEqual(follower.periodic_requery_count, 0)
        self.assertNotIn('periodic_interruption', follower.commands[0])

    def test_release_settle_is_preserved_and_does_not_create_periodic_completion(self):
        follower, robot, paths, _ = self.fixture(done=True, opening=1., initial_opening=0., settle=10)
        self.query(follower, robot, 0.)
        self.tick(follower, .05, gripper_done=True)
        for tick in range(2, 11):
            self.query(follower, robot, (tick - 1) * .05)
            self.assertFalse(self.tick(follower, tick * .05, gripper_done=True))
            self.assertEqual(paths[0].closed, 0)
        self.assertTrue(self.tick(follower, .55, gripper_done=True))
        self.assertNotIn('periodic_interruption', follower.commands[0])
        self.assertEqual(follower.periodic_requery_count, 0)

    def test_failure_only_wait_keeps_separate_actual_query_count_and_failure(self):
        follower, robot, paths, _ = self.fixture(fail=True)
        self.query(follower, robot, 0.)
        for tick in range(1, 11):
            self.tick(follower, tick * .05)
            if tick < 10:
                self.query(follower, robot, tick * .05)
        self.assertEqual(follower.requery_count, 0)
        self.query(follower, robot, .5)
        self.assertEqual(follower.requery_count, 1)
        self.assertEqual(follower.periodic_requery_count, 0)
        self.assertEqual(follower.commands[0]['executed_tick_count'], 0)
        self.assertNotIn('periodic_interruption', follower.commands[0])
        self.assertNotIn('command_completed_at', follower.commands[0])
        self.assertEqual(follower.commands[1]['fresh_query_after_failed_command_id'], 0)

    def test_pending_at_final_budget_is_not_counted_as_actual_requery(self):
        follower, robot, paths, _ = self.fixture()
        self.query(follower, robot, 0.)
        for tick in range(1, 11):
            self.tick(follower, tick * .05)
            if tick < 10:
                self.query(follower, robot, tick * .05)
        follower.finish('fixed_tick_budget', .5)
        self.assertEqual(follower.periodic_requery_count, 0)
        self.assertFalse(follower.commands[0]['periodic_interruption']['actual_fresh_query_performed'])
        self.assertEqual(follower.commands[0]['cutoff_reason'], 'fixed_tick_budget')
        self.assertNotIn('command_completed_at', follower.commands[0])
        self.assertEqual(paths[0].closed, 1)

    def test_duplicate_gap_nan_feedback_and_wrong_fixed_configuration_reject(self):
        for bad in (.05, .2, float('nan')):
            follower, robot, _, _ = self.fixture()
            self.query(follower, robot, 0.)
            self.tick(follower, .05)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.tick(follower, bad)
            self.assertEqual(follower.periodic_requery_count, 0)
        for kwargs in (dict(feedback_ticks=9), dict(maximum_periodic_requeries=21), dict(dt=.1)):
            with self.assertRaises(ValueError):
                PeriodicReplanningConfig(**kwargs)

    def test_wrong_first_post_query_clock_rejects_before_mutating_ledger(self):
        follower, robot, _, _ = self.fixture()
        self.query(follower, robot, 0.)
        with self.assertRaises(ValueError):
            self.tick(follower, .1)
        self.assertEqual(follower.commands[0]['executed_tick_count'], 0)
        self.assertEqual(follower.post_query_feedback_ticks, 0)


if __name__ == '__main__':
    unittest.main()
