"""Independent v56 causal producer regressions; CPU sensor fixtures only.

Frozen real GRU, cache, tracker and goal objects are constructed normally.
Only their observation methods return declared synthetic public measurements.
No Native environment, fitting or case-result selection is used here.
"""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

import our_mani.local_mpc_control_20261103.public_candidate_goal_maintenance_v56 as v56
from our_mani.contracts import Pose
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
from our_mani.local_mpc_control_20261103.public_grasp import PublicLiftHoldMetric
from our_mani.local_mpc_control_20261103.public_grasp_v41 import ConfirmedAttemptRecovery
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionEvidence
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame, actual_h
from tests.test_local_mpc_goal_aware_binding_v42_20261103 import raw_metadata
from tests.test_local_mpc_public_motion_bridge_20261103 import observation


class V56IndependentProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = v56.PublicCandidateBridgeV56(postconfirmed_maintenance=True,
            active_config=ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))

    def setUp(self):
        self.wrapper.reset()

    def feed(self, index, *, height=0., shift=(0., 0., 0.), close=True,
             visible=True, opening=None, inner_result=None):
        target, robot = sensor_frame(index, height=height, relative_shift=shift)
        if opening is not None:
            robot = replace(robot, finger_open_amounts=np.full(2, opening))
        def track(_):
            if not visible:
                raise ValueError('explicit synthetic public occlusion')
            return target, raw_metadata()
        real_observe = self.wrapper.bridge.observe
        def observe(*args, **kwargs):
            result = real_observe(*args, **kwargs)
            if inner_result is not None:
                inner_result(result)
            return result
        with patch.object(self.wrapper.tracker, 'observe', side_effect=track), \
             patch.object(self.wrapper.bridge.goal_marker, 'observe', return_value=(Pose([.3, .1, 1.]), {})), \
             patch.object(self.wrapper.bridge, 'observe', side_effect=observe):
            result = self.wrapper.observe(observation(robot.timestamp), robot, close_requested=close)
        return result, robot, target

    def confirm(self, assembler=None):
        result, robot, _ = self.feed(0, close=False, opening=1.)
        if assembler is not None:
            assembler.observe_robot(robot)
        result, robot, target = self.feed(1)
        if assembler is not None:
            assembler.observe_robot(robot)
        self.wrapper.active.begin_probe(target, robot, actual_h(robot))
        for index in range(2, 7):
            result, robot, _ = self.feed(index, height=.012 * (index - 1) / 5)
            if assembler is not None:
                assembler.observe_robot(robot)
        self.assertEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertTrue(self.wrapper.maintenance.started)
        return result, robot

    def test_disabled_feature_calls_exact_delegate_and_cannot_seed_maintenance(self):
        wrapper = v56.PublicCandidateBridgeV56(postconfirmed_maintenance=False)
        target, robot = sensor_frame(0)
        with patch.object(wrapper.tracker, 'observe', return_value=(target, raw_metadata())), \
             patch.object(wrapper.bridge.goal_marker, 'observe', return_value=(Pose([.3, .1, 1.]), {})), \
             patch.object(wrapper.delegate, 'observe', wraps=wrapper.delegate.observe) as delegated, \
             patch.object(wrapper.maintenance, 'start', side_effect=AssertionError('disabled helper must not run')):
            result = wrapper.observe(observation(0.), robot, close_requested=True)
        self.assertEqual(delegated.call_count, 1)
        self.assertIs(result, wrapper.delegate.latest_result)
        self.assertIs(result['attachment_estimate'], wrapper.active.latest)
        self.assertFalse(wrapper.maintenance.started)
        self.assertFalse(result['postconfirmed_position_maintenance_v56'])
        self.assertEqual(result['mode'], 'UNRESOLVED')

    def test_legacy_positive_never_admits_or_feeds_unknown_contact_to_GRU(self):
        self.feed(0, close=False, opening=1.)
        with patch.object(self.wrapper.estimator, 'observe', side_effect=lambda target, robot, **kw: dict(
                mode='ATTACHED_PUBLIC', evidence=PublicMotionEvidence('attachment_confirmed', robot.timestamp,
                    ('rgbd_relative_pose_tracking', 'public_robot_pose'), Pose([0., 0., -.015])))), \
             patch.object(self.wrapper.bridge.forecaster, 'observe', wraps=self.wrapper.bridge.forecaster.observe) as gru:
            for index in range(1, 8):
                result, _, _ = self.feed(index, height=.012)
        self.assertEqual(gru.call_count, 0)
        self.assertEqual(result['legacy_attachment_diagnostic']['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(result['mode'], 'UNRESOLVED')
        self.assertFalse(result['available'])
        self.assertFalse(result['controlled_mode_available'])
        self.assertFalse(self.wrapper.maintenance.started)
        self.assertIsNone(self.wrapper.active.spec)

    def test_current_visual_maintenance_enters_strict_assembler_without_old_probe_approval(self):
        assembler = v56.PublicCandidateStateAssemblerV56(self.wrapper.binding)
        self.confirm(assembler)
        result, robot, target = self.feed(7, height=.05, shift=(.009, 0., 0.))
        assembler.observe_robot(robot)
        self.assertTrue(self.wrapper.active.failed)
        self.assertEqual(result['original_probe_estimator_diagnostic']['mode'], 'UNRESOLVED')
        self.assertEqual(result['attachment_estimate']['protocol'], v56.MAINTENANCE_PROTOCOL)
        state = assembler.build(result, robot, actual_h(robot))
        self.assertTrue(state['held'])
        self.assertFalse(state['free_motion_asserted'])
        self.assertFalse(state['Native_held_used'])
        self.assertFalse(state['object_orientation_observable'])
        np.testing.assert_allclose(state['object_position'], target.pose.position, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result['attachment_estimate']['evidence'].tcp_relative_pose.position,
            robot.tcp_pose.inverse_apply(target.pose.position), atol=1e-12, rtol=0)
        stale = actual_h(robot); stale['reference_clock'] -= .05
        with self.assertRaisesRegex(ValueError, 'clock'):
            assembler.build(result, robot, stale)
        with self.assertRaisesRegex(ValueError, 'current actual bridge result'):
            assembler.build(dict(result), robot, actual_h(robot))

    def test_visual_loss_latches_unknown_even_with_model_height_and_past_old_deadline(self):
        result, robot = self.confirm()
        recovery = ConfirmedAttemptRecovery(); recovery.observe(result, robot)
        metric = PublicLiftHoldMetric()
        result, robot, _ = self.feed(7, height=.05, visible=False)
        # A model-only pose and genuine earlier proof never replace current RGBD.
        result['controlled'] = dict(estimated_world_pose=Pose([.2, -.15, 9.]))
        receipt = metric.observe(result, robot, self.wrapper.latest_attachment, self.wrapper.close_visual_baseline)
        self.assertFalse(receipt['current_visual_target'])
        self.assertFalse(receipt['public_lift_hold_observed'])
        self.assertTrue(self.wrapper.maintenance.failed)
        attempt = self.wrapper.active.attempt_id
        recovery.observe(result, robot)
        with patch.object(self.wrapper.bridge.forecaster, 'observe', wraps=self.wrapper.bridge.forecaster.observe) as gru:
            for index in range(8, 22):
                result, robot, _ = self.feed(index, height=.05)
                recovery.observe(result, robot)
        self.assertGreater(robot.timestamp, self.wrapper.active.spec['deadline'])
        self.assertTrue(recovery.preserves_closed_attempt(attempt))
        self.assertEqual(result['mode'], 'UNRESOLVED')
        self.assertFalse(result['free_forecast_available'])
        self.assertFalse(result['controlled_mode_available'])
        self.assertEqual(gru.call_count, 0)

    def test_only_real_typed_release_retires_failed_helper_then_new_attempt_needs_new_probe(self):
        self.confirm()
        original_helper = self.wrapper.maintenance
        for index in range(7, 10):
            result, _, _ = self.feed(index, height=.05, shift=(.1, 0., 0.), close=False, opening=1.)
            self.assertIs(self.wrapper.maintenance, original_helper)
            self.assertTrue(self.wrapper.maintenance.failed)
            self.assertNotEqual(result['mode'], 'FREE')
        result, _, _ = self.feed(10, height=.05, shift=(.1, 0., 0.), close=False, opening=1.)
        self.assertEqual(result['mode'], 'FREE')
        self.assertEqual(result['attachment_estimate']['evidence'].event, 'release_confirmed')
        self.assertIsNot(self.wrapper.maintenance, original_helper)
        self.assertFalse(self.wrapper.maintenance.started)
        self.assertEqual(len(self.wrapper.retired_maintenance), 1)
        result, _, _ = self.feed(11, height=.05)
        self.assertEqual(self.wrapper.active.attempt_id, 2)
        self.assertEqual(result['mode'], 'UNRESOLVED')
        self.assertIsNone(self.wrapper.active.spec)
        self.assertFalse(self.wrapper.maintenance.started)
        self.assertIsNone(self.wrapper.last_initial_admission)

    def test_unavailable_actual_inner_bridge_cannot_be_promoted_to_admission(self):
        self.feed(0, close=False, opening=1.)
        _, robot, target = self.feed(1)
        self.wrapper.active.begin_probe(target, robot, actual_h(robot))
        for index in range(2, 6):
            self.feed(index, height=.012 * (index - 1) / 5)
        with self.assertRaisesRegex(ValueError, 'fresh unchanged original public probe proof'):
            self.feed(6, height=.012, inner_result=lambda result: result.update(available=False))
        self.assertFalse(self.wrapper.maintenance.started)
        self.assertIsNone(self.wrapper.last_initial_admission)


if __name__ == '__main__':
    unittest.main()
