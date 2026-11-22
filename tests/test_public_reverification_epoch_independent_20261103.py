"""Independent CPU contract checks; no Native, model forward or TEST inputs."""
import unittest
import numpy as np
from our_mani.contracts import ObjectState, Pose
from tests.test_public_reverification_epoch_prototype_20261103 import fixture, frame, begin
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import actual_h


class IndependentReverificationTests(unittest.TestCase):
    def test_stale_body_latches_witness_and_fresh_reappearance_cannot_restart(self):
        owner, old, _ = fixture()
        spec, _, _ = begin(owner)
        r, robot = frame(9, .015)
        body = r['object_observation']
        r['object_observation'] = ObjectState('target', robot.timestamp-.05, body.pose,
            body.points_world, body.linear_velocity, body.angular_velocity, body.confidence)
        failed = owner.observe(r, robot, close_requested=True)
        self.assertTrue(failed['witness_failed'])
        r, robot = frame(10, .018)
        after = owner.observe(r, robot, close_requested=True)
        self.assertEqual(after['reason'], 'failed_witness_latched_no_silent_restart')
        self.assertIsNone(owner.begin_witness(r, robot, actual_h(robot)))
        self.assertEqual(owner.spec['deadline'], spec['deadline'])
        self.assertTrue(old.failed)
        self.assertEqual(owner.verification_epoch, 0)

    def test_original_height_guard_survives_fresh_local_witness_anchor(self):
        owner, old, baseline = fixture()
        original_r0, original_anchor = old.r0.copy(), old.object_anchor.copy()
        begin(owner)
        r, robot = frame(9, .012, shift=(0., 0., -.006))
        out = owner.observe(r, robot, close_requested=True)
        self.assertEqual(out['reason'], 'original_probe_height_baseline_not_preserved')
        self.assertTrue(out['witness_failed'])
        np.testing.assert_array_equal(old.r0, original_r0)
        np.testing.assert_array_equal(old.object_anchor, original_anchor)
        np.testing.assert_array_equal(owner.original_preclose_baseline.pose.position, baseline.pose.position)
        self.assertTrue(old.failed)

    def test_mutating_current_body_after_good_witness_cannot_commit(self):
        owner, old, _ = fixture()
        begin(owner)
        for i in range(9, 14):
            r, robot = frame(i, .012+.012*(i-8)/5)
            current = owner.observe(r, robot, close_requested=True)
        self.assertTrue(current['current_fresh_witness_passed'])
        body = r['object_observation']
        r['object_observation'] = ObjectState('target', body.timestamp,
            Pose(body.pose.position+[.001, 0., 0.]), body.points_world,
            body.linear_velocity, body.angular_velocity, body.confidence)
        with self.assertRaises(ValueError):
            owner.commit_verified_epoch(r, robot)
        self.assertEqual(owner.verification_epoch, 0)
        self.assertTrue(old.failed)

    def test_fixed_deadline_expires_without_static_attachment_or_rollforward(self):
        owner, old, _ = fixture()
        spec, _, _ = begin(owner)
        for i in range(9, 26):
            r, robot = frame(i)
            current = owner.observe(r, robot, close_requested=True)
        self.assertTrue(current['witness_failed'])
        self.assertEqual(owner.first_failure['reason'], 'fixed_witness_deadline_expired_no_rollforward')
        self.assertEqual(owner.spec['deadline'], spec['deadline'])
        self.assertIsNone(owner.active_test_proposal())
        self.assertFalse(current['current_fresh_witness_passed'])
        self.assertEqual(owner.verification_epoch, 0)
        self.assertTrue(old.failed)


if __name__ == '__main__':
    unittest.main()
