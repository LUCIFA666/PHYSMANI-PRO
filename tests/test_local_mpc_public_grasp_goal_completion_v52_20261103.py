"""Synthetic current public frames only; no Native, checkpoint or fit."""
import json
import unittest

import numpy as np

from our_mani.contracts import ObjectState, Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import (
    PublicMotionEvidence, PublicRobotSensors,
)
from our_mani.local_mpc_control_20261103.public_grasp_goal_completion_v52 import (
    PublicGraspGoalCompletion, PublicGraspGoalCompletionConfig,
)


GOAL = Pose([.2, -.1, .9])


def frame(time, *, offset=(0., 0., 0.), attempt=1):
    body = GOAL.position + offset
    robot = PublicRobotSensors(time, body + [0., 0., .01], [0., 0., 0., 1.],
                               np.zeros(7), np.zeros(7), [.33, .99])
    target = ObjectState('target', time, Pose(body), body + [[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]],
                         np.zeros(3), np.zeros(3))
    evidence = PublicMotionEvidence('attachment_confirmed', time,
        ('rgbd_relative_pose_tracking', 'public_robot_pose', 'public_gripper_opening'),
        Pose(robot.tcp_pose.inverse_apply(body)), orientation_observable=False)
    result = dict(timestamp=time, mode='ATTACHED_PUBLIC', object_observation=target,
                  available=True, controlled_mode_available=True, Native_inputs_used=False,
                  future_observations_used=False, observations_fabricated=False,
                  attachment_estimate=dict(timestamp=time, mode='ATTACHED_PUBLIC', evidence=evidence,
                                           attempt_active=True, attempt_id=attempt))
    hold = dict(latest=dict(timestamp=time, public_hold_seconds=.5,
        public_lift_hold_observed=True, current_visual_target=True,
        current_public_attachment_confirmed=True, actual_preclose_visual_baseline_available=True,
        controlled_pose_used_as_visual_label=False, Native_labels_used=False))
    return result, robot, hold


def observe(metric, time, **kwargs):
    result, robot, hold = frame(time, **kwargs)
    return metric.observe(result, robot, hold, close_requested=True)


class PublicGraspGoalCompletionTests(unittest.TestCase):
    def test_near_public_goal_requires_three_real_frames_and_separates_earlier_hold(self):
        metric = PublicGraspGoalCompletion(GOAL)
        for time in [0., .05, .1, .15]:
            row = observe(metric, time, offset=(.03, 0., 0.))
            self.assertTrue(row['public_lift_hold_passed'])
            self.assertFalse(metric.passed)
            self.assertEqual(row['consecutive_goal_frames'], 0)
        for time in [.2, .25]:
            self.assertFalse(observe(metric, time)['public_grasp_goal_complete'])
        row = observe(metric, .3)
        self.assertTrue(metric.passed)
        self.assertEqual(row['consecutive_goal_frames'], 3)
        self.assertAlmostEqual(row['goal_dwell_seconds'], .1)
        self.assertFalse(row['Native_task_judge_replaced'])
        self.assertFalse(row['object_orientation_qualified'])
        json.dumps(metric.receipt())

    def test_controlled_model_or_historical_point_never_replaces_current_visual_body(self):
        for fault in ['missing', 'controlled_only', 'wrong_type', 'invisible', 'low_confidence', 'wrong_id', 'future_body']:
            with self.subTest(fault=fault):
                metric = PublicGraspGoalCompletion(GOAL)
                observe(metric, 0.); observe(metric, .05)
                result, robot, hold = frame(.1)
                original = result['object_observation']
                if fault in ['missing', 'controlled_only']:
                    result['object_observation'] = None
                    result['controlled'] = dict(object_pose=GOAL, height=.9, held=True)
                    result['forecast'] = dict(positions=[GOAL.position.tolist()])
                elif fault == 'wrong_type': result['object_observation'] = dict(position=GOAL.position)
                else:
                    result['object_observation'] = ObjectState(
                        'other' if fault == 'wrong_id' else 'target',
                        .15 if fault == 'future_body' else .1, original.pose, original.points_world,
                        np.zeros(3), np.zeros(3), .49 if fault == 'low_confidence' else 1.,
                        fault != 'invisible')
                row = metric.observe(result, robot, hold, close_requested=True)
                self.assertFalse(metric.passed)
                self.assertEqual(row['consecutive_goal_frames'], 0)

    def test_fresh_typed_attachment_both_modes_and_all_timestamps_required(self):
        for fault in ['dict_evidence', 'past_evidence', 'future_evidence', 'future_result',
                      'future_attachment', 'bridge_unknown', 'attachment_unknown', 'release']:
            with self.subTest(fault=fault):
                metric = PublicGraspGoalCompletion(GOAL)
                observe(metric, 0.); observe(metric, .05)
                result, robot, hold = frame(.1)
                att = result['attachment_estimate']
                if fault == 'dict_evidence': att['evidence'] = dict(event='attachment_confirmed', timestamp=.1)
                elif fault in ['past_evidence', 'future_evidence']:
                    att['evidence'] = PublicMotionEvidence('attachment_confirmed',
                        .05 if fault == 'past_evidence' else .15,
                        ('rgbd_relative_pose_tracking', 'public_robot_pose'), Pose([0., 0., -.01]))
                elif fault == 'future_result': result['timestamp'] = .15
                elif fault == 'future_attachment': att['timestamp'] = .15
                elif fault == 'bridge_unknown': result['mode'] = 'UNRESOLVED'
                elif fault == 'attachment_unknown': att['mode'] = 'UNRESOLVED'
                else: att['evidence'] = PublicMotionEvidence('release_confirmed', .1,
                          ('rgbd_object_release', 'public_gripper_opening'))
                row = metric.observe(result, robot, hold, close_requested=True)
                self.assertFalse(row['public_grasp_goal_complete'])
                self.assertEqual(row['consecutive_goal_frames'], 0)

    def test_open_or_inactive_attempt_and_changed_attempt_clear_dwell(self):
        for fault in ['open', 'inactive', 'new_attempt']:
            with self.subTest(fault=fault):
                metric = PublicGraspGoalCompletion(GOAL)
                observe(metric, 0.); observe(metric, .05)
                result, robot, hold = frame(.1, attempt=2 if fault == 'new_attempt' else 1)
                if fault == 'inactive': result['attachment_estimate']['attempt_active'] = False
                row = metric.observe(result, robot, hold, close_requested=fault != 'open')
                self.assertFalse(metric.passed)
                self.assertEqual(row['consecutive_goal_frames'], 1 if fault == 'new_attempt' else 0)
                self.assertFalse(observe(metric, .15, attempt=2 if fault == 'new_attempt' else 1)['public_grasp_goal_complete'])

    def test_unavailable_or_nonpublic_bridge_scope_cannot_claim_typed_attachment(self):
        for key, bad in [('available', False), ('controlled_mode_available', False),
                         ('Native_inputs_used', True), ('future_observations_used', True),
                         ('observations_fabricated', True), ('Native_inputs_used', None)]:
            with self.subTest(key=key, bad=bad):
                metric = PublicGraspGoalCompletion(GOAL)
                observe(metric, 0.); observe(metric, .05)
                result, robot, hold = frame(.1)
                result[key] = bad
                row = metric.observe(result, robot, hold, close_requested=True)
                self.assertFalse(metric.passed)
                self.assertFalse(row['gates']['current_available_public_observation_scope'])
                self.assertEqual(row['consecutive_goal_frames'], 0)

    def test_gap_earlier_hold_loss_and_stale_receipt_cannot_bridge_dwell(self):
        for fault in ['gap', 'false', 'short', 'stale', 'future', 'model', 'Native']:
            with self.subTest(fault=fault):
                metric = PublicGraspGoalCompletion(GOAL)
                observe(metric, 0.); observe(metric, .05)
                time = .2 if fault == 'gap' else .1
                result, robot, hold = frame(time)
                latest = hold['latest']
                if fault == 'false': latest['public_lift_hold_observed'] = False
                elif fault == 'short': latest['public_hold_seconds'] = .49
                elif fault == 'stale': latest['timestamp'] = .05
                elif fault == 'future': latest['timestamp'] = .15
                elif fault == 'model': latest['controlled_pose_used_as_visual_label'] = True
                elif fault == 'Native': latest['Native_labels_used'] = True
                row = metric.observe(result, robot, hold, close_requested=True)
                self.assertFalse(metric.passed)
                self.assertEqual(row['consecutive_goal_frames'], 0)
                self.assertFalse(observe(metric, time + .05)['public_grasp_goal_complete'])
                self.assertFalse(observe(metric, time + .1)['public_grasp_goal_complete'])
                self.assertTrue(observe(metric, time + .15)['public_grasp_goal_complete'])

    def test_past_completion_is_not_current_completion_after_unknown_or_reset(self):
        metric = PublicGraspGoalCompletion(GOAL)
        for time in [0., .05, .1]: observe(metric, time)
        self.assertTrue(metric.passed)
        result, robot, hold = frame(.15)
        result['object_observation'] = None
        row = metric.observe(result, robot, hold, close_requested=True)
        self.assertFalse(metric.passed)
        self.assertTrue(row['ever_public_grasp_goal_complete'])
        metric.reset()
        self.assertFalse(metric.passed)
        self.assertFalse(observe(metric, .2)['ever_public_grasp_goal_complete'])

    def test_boolean_current_passed_form_and_exact_position_boundary(self):
        metric = PublicGraspGoalCompletion(GOAL)
        for time in [0., .05, .1]:
            result, robot, _ = frame(time, offset=(.02, 0., 0.))
            row = metric.observe(result, robot, True, close_requested=True)
        self.assertTrue(row['public_grasp_goal_complete'])
        self.assertIn('bool', row['hold_input_scope'])
        result, robot, _ = frame(.15)
        self.assertFalse(metric.observe(result, robot, False, close_requested=True)['public_grasp_goal_complete'])

    def test_real_public_hold_metric_receipt_composes_without_model_lift(self):
        from our_mani.local_mpc_control_20261103.public_grasp import PublicLiftHoldMetric
        hold_metric = PublicLiftHoldMetric()
        completion = PublicGraspGoalCompletion(GOAL)
        baseline_body = GOAL.position - [0., 0., .05]
        baseline = ObjectState('target', 0., Pose(baseline_body),
            baseline_body + [[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]],
            np.zeros(3), np.zeros(3))
        for tick in range(1, 14):
            result, robot, _ = frame(tick * .05)
            result['controlled_mode_available'] = True
            hold_metric.observe(result, robot, result['attachment_estimate'], baseline)
            row = completion.observe(result, robot, hold_metric.receipt(), close_requested=True)
            self.assertEqual(row['public_lift_hold_passed'], tick >= 11)
            self.assertEqual(row['public_grasp_goal_complete'], tick >= 13)
        self.assertAlmostEqual(row['public_hold_seconds'], .6)
        self.assertAlmostEqual(row['goal_dwell_seconds'], .1)

    def test_strict_sensor_type_increasing_time_and_known_intent(self):
        metric = PublicGraspGoalCompletion(GOAL)
        result, robot, hold = frame(0.)
        with self.assertRaises(TypeError): metric.observe(result, dict(timestamp=0.), hold, close_requested=True)
        with self.assertRaises(TypeError): metric.observe(result, robot, hold, close_requested=1)
        metric.observe(result, robot, hold, close_requested=True)
        with self.assertRaises(ValueError): metric.observe(result, robot, hold, close_requested=True)
        self.assertFalse(metric.passed)

    def test_public_goal_is_copied_and_unknown_extra_fields_are_not_read(self):
        class Poison:
            def __bool__(self): raise AssertionError('private field read')
            def __float__(self): raise AssertionError('private field read')
        goal = Pose(GOAL.position)
        metric = PublicGraspGoalCompletion(goal)
        self.assertIsNot(metric.goal_pose, goal)
        for time in [0., .05, .1]:
            result, robot, hold = frame(time)
            result.update(Native_held=Poison(), Native_success=Poison(),
                          controlled=Poison(), forecast=Poison(), goal_pose=Poison())
            row = metric.observe(result, robot, hold, close_requested=True)
        self.assertTrue(row['public_grasp_goal_complete'])

    def test_config_cannot_relax_declared_v52_starts(self):
        for kwargs in [dict(sensor_dt=.1), dict(minimum_confidence=.49),
                       dict(maximum_goal_distance_m=.021), dict(goal_dwell_seconds=.09),
                       dict(minimum_public_hold_seconds=.49), dict(sensor_dt=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PublicGraspGoalCompletionConfig(**kwargs)
        with self.assertRaises(TypeError): PublicGraspGoalCompletion(dict(position=GOAL.position))


if __name__ == '__main__':
    unittest.main()
