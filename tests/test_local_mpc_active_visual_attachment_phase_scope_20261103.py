"""CPU phase-scope contracts; synthetic public sensors, no Native/contact labels."""
from dataclasses import asdict
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState, Pose
from our_mani.local_mpc_control_20261103.active_visual_attachment import (
    ActiveVisualAttachment, ActiveVisualConfig,
)
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors


TCP0 = np.array([.2, -.15, .790])
RELATIVE_BODY = np.array([.003, -.002, -.015])


def sensor_frame(index, *, height=0., angle=0., relative_shift=(0., 0., 0.),
                 confidence=1., timestamp=None, identity='target'):
    now = index * .05 if timestamp is None else timestamp
    rotation = Rotation.from_rotvec([0., angle, 0.])
    tcp = TCP0 + [0., 0., height]
    # Object center is independently observed; this synthetic case explicitly
    # supplies a fixed TCP-relative center through a non-axis-aligned rotation.
    body = tcp + rotation.apply(RELATIVE_BODY + relative_shift)
    robot = PublicRobotSensors(now, tcp, rotation.as_quat(), np.zeros(7), np.zeros(7), [.2, .65])
    target = ObjectState(identity, now, Pose(body),
                         body + np.array([[0., 0., 0.], [.005, 0., 0.], [0., .005, 0.]]),
                         np.zeros(3), np.zeros(3), confidence)
    return target, robot


def actual_h(robot):
    return dict(reference_valid=True, reference_positions=robot.q.copy(),
                reference_velocities=np.zeros(7), reference_accelerations=np.zeros(7),
                target_joints=robot.q.copy(), reference_clock=robot.timestamp,
                gripper_target=0., gripper_done=True, reference_is_measured_state=False)


def probe(config):
    estimator = ActiveVisualAttachment(config)
    target, robot = sensor_frame(0)
    estimator.observe(target, robot, close_requested=True)
    spec = estimator.begin_probe(target, robot, actual_h(robot))
    assert spec is not None
    return estimator, spec


def confirmed(config):
    estimator, spec = probe(config)
    for index in range(1, 6):
        target, robot = sensor_frame(index, height=.012 * index / 5)
        result = estimator.observe(target, robot, close_requested=True)
        if index < 5:
            assert result['mode'] == 'UNRESOLVED'
    assert result['mode'] == 'ATTACHED_PUBLIC'
    return estimator, spec


class ActiveVisualPhaseScopeTests(unittest.TestCase):
    def test_configuration_default_preserves_original_guard_and_requires_explicit_bool(self):
        self.assertIs(ActiveVisualConfig().enforce_probe_anchor_orientation_after_confirmation, True)
        for invalid in [0, 1, None, 'false', np.bool_(False)]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'bool'):
                ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=invalid)
        original = asdict(ActiveVisualConfig())
        scoped = asdict(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
        self.assertEqual({k: v for k, v in original.items() if k != 'enforce_probe_anchor_orientation_after_confirmation'},
                         {k: v for k, v in scoped.items() if k != 'enforce_probe_anchor_orientation_after_confirmation'})

    def test_default_and_explicit_true_still_revoke_after_confirmed_anchor_rotation(self):
        for config in [ActiveVisualConfig(), ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=True)]:
            estimator, _ = confirmed(config)
            target, robot = sensor_frame(6, height=.05, angle=.08)
            result = estimator.observe(target, robot, close_requested=True)
            self.assertEqual(result['mode'], 'UNRESOLVED')
            self.assertTrue(result['probe_failed'])
            self.assertEqual(result['reason'], 'current_attached_relative_or_orientation_consistency_lost')

    def test_false_only_allows_current_visible_center_consistency_after_real_confirmation(self):
        estimator, spec = confirmed(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
        self.assertEqual(estimator.confirmation['real_probe_frames'], 6)
        for index in range(6, 9):
            target, robot = sensor_frame(index, height=.05, angle=.08 + .03 * (index - 6))
            result = estimator.observe(target, robot, close_requested=True)
            self.assertEqual(result['mode'], 'ATTACHED_PUBLIC')
            np.testing.assert_allclose(result['evidence'].tcp_relative_pose.position,
                                       spec['relative_r0'], atol=1e-12, rtol=0)
            self.assertFalse(result['evidence'].orientation_observable)
            self.assertFalse(result['object_orientation_observable'])
            self.assertFalse(result['full_SE3_attachment_qualified'])
            self.assertFalse(result['force_or_frictional_grasp_proof'])
            self.assertEqual(result['probe_anchor_orientation_guard_scope'], 'probe_only')
            self.assertIn('translation_only', result['post_confirmation_evidence_semantics'])

    def test_false_keeps_probe_angle_guard_including_the_potential_confirmation_frame(self):
        for bad_index in [1, 5]:
            with self.subTest(bad_index=bad_index):
                estimator, _ = probe(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
                for index in range(1, bad_index + 1):
                    target, robot = sensor_frame(index, height=.012 * index / 5,
                                                 angle=.08 if index == bad_index else 0.)
                    result = estimator.observe(target, robot, close_requested=True)
                self.assertEqual(result['mode'], 'UNRESOLVED')
                self.assertTrue(result['probe_failed'])
                self.assertIsNone(result['diagnostic']['confirmation'])
                self.assertEqual(result['reason'], 'actual_probe_15mm_or_orientation_bound_exceeded')

    def test_false_keeps_probe_displacement_and_real_signed_motion_gates(self):
        estimator, _ = probe(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
        target, robot = sensor_frame(1, height=.016)
        result = estimator.observe(target, robot, close_requested=True)
        self.assertTrue(result['probe_failed'])
        self.assertEqual(result['reason'], 'actual_probe_15mm_or_orientation_bound_exceeded')
        estimator, _ = probe(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
        for index in range(1, 17):
            target, robot = sensor_frame(index, height=.007)
            result = estimator.observe(target, robot, close_requested=True)
            self.assertEqual(result['mode'], 'UNRESOLVED')
        self.assertIsNone(result['diagnostic']['confirmation'])
        self.assertEqual(estimator.spec['deadline'], .8)

    def test_false_still_revokes_on_current_visual_identity_gap_intent_or_relative_failure(self):
        for fault in ['missing', 'low_confidence', 'wrong_identity', 'gap', 'open', 'relative_drift']:
            with self.subTest(fault=fault):
                estimator, _ = confirmed(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
                target, robot = sensor_frame(6, height=.05, angle=.08,
                    confidence=.49 if fault == 'low_confidence' else 1.,
                    identity='another_red_body' if fault == 'wrong_identity' else 'target',
                    timestamp=.35 if fault == 'gap' else .30,
                    relative_shift=(.0061, 0., 0.) if fault == 'relative_drift' else (0., 0., 0.))
                result = estimator.observe(None if fault == 'missing' else target, robot,
                                           close_requested=fault != 'open')
                self.assertEqual(result['mode'], 'UNRESOLVED')
                self.assertTrue(result['probe_failed'])
                self.assertNotEqual(result['evidence'].event, 'attachment_confirmed')
                # Return of a good frame cannot silently retry a failed attempt.
                target, robot = sensor_frame(0, timestamp=robot.timestamp + .05,
                                             height=.05, angle=.08)
                self.assertEqual(estimator.observe(target, robot, close_requested=True)['mode'], 'UNRESOLVED')


if __name__ == '__main__':
    unittest.main()
