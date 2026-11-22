"""Independent synthetic public-sensor counterexamples; no Native or fitting.

These test the v4 evidence contract, not perception accuracy or physical contact.
The real actuator completion bit permits a probe only; no hidden held label is
present in any input.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import ObjectState, Pose
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors


TCP0 = np.array([.20, -.15, .790])
BODY0 = TCP0 + np.array([0., 0., -.015])


def frame(i, *, tcp_delta=(0., 0., 0.), body_delta=(0., 0., 0.),
          fingers=(.20, .65), confidence=1., identity='target',
          velocity=(0., 0., 0.), timestamp=None, quaternion=(0., 0., 0., 1.)):
    """Distinct measured TCP/body poses, with deliberately asymmetric fingers."""
    t = i * .05 if timestamp is None else float(timestamp)
    tcp = TCP0 + np.asarray(tcp_delta, float)
    body = BODY0 + np.asarray(body_delta, float)
    robot = PublicRobotSensors(t, tcp, quaternion, np.zeros(7), np.zeros(7), fingers)
    points = body + np.array([[0., 0., 0.], [.005, 0., 0.], [0., .005, 0.]])
    target = ObjectState(identity, t, Pose(body), points, velocity, np.zeros(3), confidence)
    return target, robot


def actuator(robot, **overrides):
    h = dict(reference_valid=True, reference_positions=robot.q.copy(),
             reference_velocities=np.zeros(7), reference_accelerations=np.zeros(7),
             target_joints=robot.q.copy(), reference_clock=robot.timestamp,
             gripper_target=0., gripper_done=True, reference_is_measured_state=False)
    h.update(overrides)
    return h


def admitted(*, fingers=(.20, .65)):
    model = ActiveVisualAttachment()
    target, robot = frame(0, fingers=fingers)
    model.observe(target, robot, close_requested=True)
    spec = model.begin_probe(target, robot, actuator(robot))
    if spec is None:
        raise AssertionError('fresh actual done-close and public target must admit the test')
    return model, spec


class IndependentActiveVisualAttachmentTests(unittest.TestCase):
    def assert_unknown(self, result):
        self.assertNotEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertNotEqual(getattr(result.get('evidence'), 'event', None), 'attachment_confirmed')

    def test_empty_done_gripper_and_measured_TCP_lift_do_not_lift_a_static_body(self):
        model, _ = admitted(fingers=(0., 0.))
        for i in range(1, 7):
            target, robot = frame(i, tcp_delta=(0., 0., min(i, 5) * .012 / 5), fingers=(0., 0.))
            self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_horizontal_joint_motion_without_world_Z_rise_is_not_attachment(self):
        model, _ = admitted()
        for i in range(1, 7):
            d = (min(i, 5) * .012 / 5, 0., 0.)
            target, robot = frame(i, tcp_delta=d, body_delta=d)
            self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_common_downward_motion_does_not_satisfy_signed_rise(self):
        model, _ = admitted()
        for i in range(1, 7):
            d = (0., 0., -min(i, 5) * .012 / 5)
            target, robot = frame(i, tcp_delta=d, body_delta=d)
            self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_visual_rise_without_real_TCP_rise_cannot_confirm(self):
        model, _ = admitted()
        for i in range(1, 7):
            target, robot = frame(i, body_delta=(0., 0., min(i, 5) * .009 / 5))
            self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_reported_object_velocity_never_substitutes_for_measured_height(self):
        model, _ = admitted()
        for i in range(1, 7):
            target, robot = frame(i, tcp_delta=(0., 0., min(i, 5) * .012 / 5), velocity=(0., 0., 50.))
            self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_six_actual_frames_and_quarter_second_are_required_even_if_motion_completed_early(self):
        model, _ = admitted()
        for i in range(1, 5):
            target, robot = frame(i, tcp_delta=(0., 0., .010), body_delta=(0., 0., .008))
            self.assert_unknown(model.observe(target, robot, close_requested=True))
        target, robot = frame(5, tcp_delta=(0., 0., .010), body_delta=(0., 0., .008))
        result = model.observe(target, robot, close_requested=True)
        self.assertEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(result['diagnostic']['real_probe_frames'], 6)
        self.assertAlmostEqual(result['diagnostic']['duration_seconds'], .25)
        # Asymmetry is not secretly restored as a required contact gate.
        self.assertGreater(abs(robot.finger_open_amounts[0] - robot.finger_open_amounts[1]), .12)
        self.assertFalse(result['force_or_frictional_grasp_proof'])

    def test_full_trace_excursion_cannot_be_erased_by_later_return_or_rolling_window(self):
        model, _ = admitted()
        for i in range(1, 10):
            z = min(i, 5) * .012 / 5
            x = .007 if i == 2 else 0.
            target, robot = frame(i, tcp_delta=(0., 0., z), body_delta=(x, 0., z))
            result = model.observe(target, robot, close_requested=True)
            self.assert_unknown(result)
        self.assertTrue(result['probe_failed'])
        self.assertIsNone(model.begin_probe(target, robot, actuator(robot)))

    def test_missing_frame_or_invalid_identity_latches_unknown_without_auto_retry(self):
        for fault in ('gap', 'missing', 'lowconfidence', 'wrong_identity'):
            with self.subTest(fault=fault):
                model, _ = admitted()
                target, robot = frame(1, tcp_delta=(0., 0., .003), body_delta=(0., 0., .003))
                model.observe(target, robot, close_requested=True)
                target, robot = frame(2, tcp_delta=(0., 0., .006), body_delta=(0., 0., .006),
                                      timestamp=.15 if fault == 'gap' else None,
                                      confidence=.49 if fault == 'lowconfidence' else 1.,
                                      identity='another_red_object' if fault == 'wrong_identity' else 'target')
                result = model.observe(None if fault == 'missing' else target, robot, close_requested=True)
                self.assert_unknown(result)
                # Return to fresh, apparently perfect co-motion after the fault.
                for j in range(1, 7):
                    target, robot = frame(0, timestamp=robot.timestamp + .05,
                                          tcp_delta=(0., 0., .012), body_delta=(0., 0., .012))
                    self.assert_unknown(model.observe(target, robot, close_requested=True))
                self.assertIsNone(model.begin_probe(target, robot, actuator(robot)))

    def test_actual_displacement_violation_wins_on_the_potential_confirmation_frame(self):
        model, _ = admitted()
        for i in range(1, 5):
            target, robot = frame(i, tcp_delta=(0., 0., .002 * i), body_delta=(0., 0., .002 * i))
            model.observe(target, robot, close_requested=True)
        target, robot = frame(5, tcp_delta=(0., 0., .016), body_delta=(0., 0., .016))
        result = model.observe(target, robot, close_requested=True)
        self.assert_unknown(result)
        self.assertTrue(result['probe_failed'])
        self.assertIn('bound', result['reason'])

    def test_gap_and_unobserved_attitude_violation_win_over_same_frame_rise_witness(self):
        for fault in ('100ms_gap', 'rotation'):
            with self.subTest(fault=fault):
                model, _ = admitted()
                for i in range(1, 5):
                    target, robot = frame(i, tcp_delta=(0., 0., .002 * i), body_delta=(0., 0., .002 * i))
                    model.observe(target, robot, close_requested=True)
                angle = .05 if fault == 'rotation' else 0.
                quat = (0., 0., np.sin(angle / 2), np.cos(angle / 2))
                target, robot = frame(5, timestamp=.30 if fault == '100ms_gap' else .25,
                                      quaternion=quat, tcp_delta=(0., 0., .012), body_delta=(0., 0., .012))
                result = model.observe(target, robot, close_requested=True)
                self.assert_unknown(result)
                self.assertTrue(result['probe_failed'])

    def test_fixed_probe_identity_deadline_and_geometry_do_not_chase_later_frames(self):
        model, spec = admitted()
        original = deepcopy(spec)
        spec['target_position'][:] = 100.
        for i in range(1, 4):
            target, robot = frame(i, tcp_delta=(0., 0., i * .002), body_delta=(0., 0., i * .002))
            model.observe(target, robot, close_requested=True)
            again = model.begin_probe(target, robot, actuator(robot))
            self.assertEqual(again['identity'], original['identity'])
            self.assertEqual(again['deadline'], .8)
            np.testing.assert_array_equal(again['target_position'], original['target_position'])

    def test_deadline_expiry_does_not_re_admit_an_already_failed_probe(self):
        model, _ = admitted()
        for i in range(1, 18):
            target, robot = frame(i)
            result = model.observe(target, robot, close_requested=True)
        self.assert_unknown(result)
        self.assertTrue(result['probe_failed'])
        self.assertIsNone(model.begin_probe(target, robot, actuator(robot)))

    def test_open_interrupt_requires_real_release_before_old_probe_can_confirm_again(self):
        model, _ = admitted()
        target, robot = frame(1, tcp_delta=(0., 0., .003), body_delta=(0., 0., .003))
        model.observe(target, robot, close_requested=True)
        target, robot = frame(2, tcp_delta=(0., 0., .006), body_delta=(0., 0., .006))
        self.assert_unknown(model.observe(target, robot, close_requested=False))
        for i in range(3, 10):
            target, robot = frame(i, tcp_delta=(0., 0., .012), body_delta=(0., 0., .012))
            self.assert_unknown(model.observe(target, robot, close_requested=True))
        self.assertIsNone(model.begin_probe(target, robot, actuator(robot)))

    def test_done_from_open_stale_or_nonfinite_executor_state_is_not_admission(self):
        cases = (dict(gripper_target=1.), dict(gripper_done=False),
                 dict(reference_clock=-.05), dict(reference_clock=float('nan')),
                 dict(reference_valid='true'))
        for override in cases:
            with self.subTest(override=override):
                model = ActiveVisualAttachment();target, robot = frame(0)
                model.observe(target, robot, close_requested=True)
                try:
                    result = model.begin_probe(target, robot, actuator(robot, **override))
                except (TypeError, ValueError):
                    continue
                self.assertIsNone(result)

    def test_confirmed_lift_is_not_capped_at_probe_radius_but_visual_loss_revokes_evidence(self):
        model, _ = admitted()
        for i in range(1, 6):
            target, robot = frame(i, tcp_delta=(0., 0., .012 * i / 5), body_delta=(0., 0., .012 * i / 5))
            result = model.observe(target, robot, close_requested=True)
        self.assertEqual(result['mode'], 'ATTACHED_PUBLIC')
        target, robot = frame(6, tcp_delta=(0., 0., .050), body_delta=(0., 0., .050))
        self.assertEqual(model.observe(target, robot, close_requested=True)['mode'], 'ATTACHED_PUBLIC')
        _, robot = frame(7, tcp_delta=(0., 0., .050), body_delta=(0., 0., .050))
        self.assert_unknown(model.observe(None, robot, close_requested=True))
        target, robot = frame(8, tcp_delta=(0., 0., .050), body_delta=(0., 0., .050))
        self.assert_unknown(model.observe(target, robot, close_requested=True))

    def test_only_observed_open_and_separation_allow_a_new_attempt_after_failure(self):
        model, old = admitted()
        _, robot = frame(1)
        model.observe(None, robot, close_requested=True)
        for i in range(2, 6):
            target, robot = frame(i, tcp_delta=(.09, 0., 0.), fingers=(1., 1.))
            result = model.observe(target, robot, close_requested=False)
            if i < 5:
                self.assert_unknown(result)
                self.assertNotEqual(result['mode'], 'FREE')
        self.assertEqual(result['mode'], 'FREE')
        self.assertEqual(result['evidence'].event, 'release_confirmed')
        target, robot = frame(6, tcp_delta=(.09, 0., 0.))
        model.observe(target, robot, close_requested=True)
        new = model.begin_probe(target, robot, actuator(robot))
        self.assertIsNotNone(new)
        self.assertNotEqual(new['identity'], old['identity'])
        self.assertEqual(new['attempt_id'], old['attempt_id'] + 1)
        self.assertAlmostEqual(new['deadline'], .30 + .8)

    def test_wrapper_keeps_legacy_positive_diagnostic_from_authorizing_bridge(self):
        from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
        from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionEvidence
        target, robot = frame(0)
        legacy = dict(mode='ATTACHED_PUBLIC', timestamp=0., evidence=PublicMotionEvidence(
            'attachment_confirmed', 0., ('rgbd_relative_pose_tracking', 'public_robot_pose'),
            Pose([0., 0., -.015])))
        called = []
        def bridge_observe(observation, robot, *, close_requested, evidence):
            called.append(evidence)
            return dict(timestamp=robot.timestamp, mode='UNRESOLVED', object_observation=target)
        # Isolate the evidence-routing method. These local doubles do not pass
        # the real constructor's approved-source checks or enter a Native run.
        wrapper = ActiveVisualPublicBridge.__new__(ActiveVisualPublicBridge)
        wrapper.tracker = SimpleNamespace(observe=lambda observation: (target, {}))
        wrapper.cache = SimpleNamespace(stage_public_target=lambda *args, **kwargs: None)
        wrapper.estimator = SimpleNamespace(observe=lambda *args, **kwargs: legacy)
        wrapper.active = ActiveVisualAttachment()
        wrapper.bridge = SimpleNamespace(observe=bridge_observe)
        wrapper.binding = SimpleNamespace(receipt=lambda: {}, bind_current=lambda *args: None)
        wrapper._last_open_visual = None;wrapper._last_close_requested = False
        wrapper.close_visual_baseline = None
        result = wrapper.observe({}, robot, close_requested=True)
        self.assertIs(result['legacy_attachment_diagnostic'], legacy)
        self.assertEqual(result['legacy_attachment_diagnostic']['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(result['attachment_estimate']['mode'], 'UNRESOLVED')
        self.assertEqual(called[0].event, 'manipulation_unresolved')
        self.assertFalse(result['legacy_attachment_controls_bridge'])

    def test_final_hold_uses_visible_body_height_and_real_half_second_not_controlled_pose(self):
        from our_mani.local_mpc_control_20261103.public_grasp import PublicLiftHoldMetric
        model, _ = admitted();metric = PublicLiftHoldMetric();baseline, _ = frame(0)
        for i in range(1, 6):
            target, robot = frame(i, tcp_delta=(0., 0., .012 * i / 5), body_delta=(0., 0., .012 * i / 5))
            attachment = model.observe(target, robot, close_requested=True)
        def measured_result(target):
            return dict(mode='ATTACHED_PUBLIC', controlled_mode_available=True,
                        object_observation=target,
                        controlled={'estimated_world_pose': {'position': [0., 0., 100.]}})
        for i in range(6, 18):
            target, robot = frame(i, tcp_delta=(0., 0., .012), body_delta=(0., 0., .012))
            attachment = model.observe(target, robot, close_requested=True)
            metric.observe(measured_result(target), robot, attachment, baseline)
            self.assertFalse(metric.passed)
        for i in range(18, 29):
            target, robot = frame(i, tcp_delta=(0., 0., .050), body_delta=(0., 0., .050))
            attachment = model.observe(target, robot, close_requested=True)
            metric.observe(measured_result(target), robot, attachment, baseline)
            if i < 28:
                self.assertFalse(metric.passed)
        self.assertTrue(metric.passed)
        self.assertAlmostEqual(metric.latest['public_hold_seconds'], .5)
        self.assertAlmostEqual(metric.latest['visual_lift_m'], .05)


if __name__ == '__main__':
    unittest.main()
