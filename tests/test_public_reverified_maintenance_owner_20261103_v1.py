"""Unwired owner CPU tests; no renderer, Native, model forward, or TEST data."""
from copy import deepcopy
import unittest
import numpy as np
from tests.test_public_reverification_epoch_prototype_20261103 import fixture, begin, frame
from our_mani.local_mpc_control_20261103.public_reverified_maintenance_owner_20261103_v1 import PublicReverifiedMaintenanceOwner
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionEvidence


def committed():
    prototype, old, baseline = fixture()
    begin(prototype)
    for i in range(9, 14):
        result, robot = frame(i, .012+.012*(i-8)/5)
        current = prototype.observe(result, robot, close_requested=True)
    assert current['current_fresh_witness_passed']
    prototype.commit_verified_epoch(result, robot)
    return prototype, old, baseline


class ReverifiedMaintenanceTests(unittest.TestCase):
    def test_requires_explicit_current_commit_and_has_fresh_typed_evidence(self):
        prototype, old, baseline = fixture()
        begin(prototype)
        for i in range(9, 14):
            result, robot = frame(i)
            prototype.observe(result, robot, close_requested=True)
        with self.assertRaises(ValueError):
            PublicReverifiedMaintenanceOwner(prototype)
        prototype, old, baseline = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        result, robot = frame(14, .026)
        current = owner.observe(result, robot, close_requested=True)
        self.assertTrue(current['current_reverified_position_maintenance_passed'])
        self.assertIs(type(current['evidence']), PublicMotionEvidence)
        self.assertEqual(current['evidence'].timestamp, robot.timestamp)
        self.assertFalse(current['evidence'].orientation_observable)
        self.assertTrue(old.failed)
        self.assertFalse(current['old_failed_owner_reset'])
        np.testing.assert_array_equal(current['original_preclose_baseline'].pose.position, baseline.pose.position)

    def test_missing_current_body_never_restores_on_reappearance(self):
        prototype, old, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        result, robot = frame(14, .026)
        result['object_observation'] = None
        first = owner.observe(result, robot, close_requested=True)
        self.assertTrue(first['failed'])
        self.assertIsNone(first['evidence'])
        result, robot = frame(15, .028)
        latest = owner.observe(result, robot, close_requested=True)
        self.assertTrue(latest['failed'])
        self.assertFalse(latest['current_reverified_position_maintenance_passed'])
        self.assertIsNone(latest['evidence'])
        self.assertEqual(latest['failure_time'], first['failure_time'])
        self.assertTrue(old.failed)

    def test_gap_attempt_open_and_stale_each_permanently_fail(self):
        for fault in ('gap', 'attempt', 'open', 'stale'):
            with self.subTest(fault=fault):
                prototype, old, _ = committed()
                owner = PublicReverifiedMaintenanceOwner(prototype)
                result, robot = frame(15 if fault == 'gap' else 14, .026)
                if fault == 'attempt':
                    result['original_probe_estimator_diagnostic']['attempt_id'] = 2
                if fault == 'stale':
                    result['timestamp'] -= .05
                out = owner.observe(result, robot, close_requested=fault != 'open')
                self.assertTrue(out['failed'])
                self.assertIsNone(out['evidence'])
                self.assertTrue(old.failed)

    def test_original_anchor_tamper_latches_even_after_restoration(self):
        prototype, old, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        saved = old.r0.copy()
        old.r0 = old.r0+[.001, 0., 0.]
        result, robot = frame(14, .026)
        out = owner.observe(result, robot, close_requested=True)
        self.assertTrue(out['failed'])
        self.assertFalse(out['original_reference_integrity_passed'])
        old.r0 = saved
        result, robot = frame(15, .028)
        out = owner.observe(result, robot, close_requested=True)
        self.assertTrue(out['failed'])
        self.assertIsNone(out['evidence'])
        np.testing.assert_array_equal(owner.latest['original_fixed_r0'], saved)

    def test_same_value_commit_replacement_is_not_original_identity(self):
        prototype, old, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        prototype.committed = deepcopy(prototype.committed)
        result, robot = frame(14, .026)
        out = owner.observe(result, robot, close_requested=True)
        self.assertTrue(out['failed'])
        self.assertIsNone(out['evidence'])
        self.assertTrue(old.failed)

    def test_fixed_original_height_and_relative_guards_are_not_new_anchors(self):
        for shift, reason in (((.011, 0., 0.), 'fixed_original_r0_over10mm_no_anchor_laundering'),
                ((0., 0., -.019), 'fixed_original_r0_over10mm_no_anchor_laundering')):
            with self.subTest(shift=shift):
                prototype, old, _ = committed()
                owner = PublicReverifiedMaintenanceOwner(prototype)
                result, robot = frame(14, .026, shift=shift)
                out = owner.observe(result, robot, close_requested=True)
                self.assertTrue(out['failed'])
                self.assertEqual(out['first_failure_reason'], reason)
                self.assertTrue(old.failed)
        prototype, old, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        result, robot = frame(14, .007)
        out = owner.observe(result, robot, close_requested=True)
        self.assertEqual(out['first_failure_reason'], 'original_probe_height_baseline_not_preserved')
        self.assertTrue(out['failed'])

    def test_duplicate_commit_frame_and_replaced_source_method_fail(self):
        prototype, _, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        out = owner.observe(prototype.last_result, prototype.last_robot, close_requested=True)
        self.assertTrue(out['failed'])
        self.assertEqual(out['first_failure_reason'], 'duplicate_or_nonincreasing_real_frame')
        prototype, _, _ = committed()
        owner = PublicReverifiedMaintenanceOwner(prototype)
        prototype._current = lambda *args: (prototype.last_target, None)
        result, robot = frame(14, .026)
        out = owner.observe(result, robot, close_requested=True)
        self.assertTrue(out['failed'])
        self.assertFalse(out['original_reference_integrity_passed'])


if __name__ == '__main__':
    unittest.main()
