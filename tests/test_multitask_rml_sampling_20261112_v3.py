import unittest
import numpy as np
from our_mani.multitask_executor_cycle_20261112_v3.sequencing import sample_velocity_checked_rml


class ScalarRMLFixture:
    def __init__(self):
        self.active = {}
        self.created = 0

    def simRMLPos(self, n, dt, flags, initial, limits, selected, target):
        self.created += 1
        self.active[self.created] = dict(position=initial[0], velocity=limits[0], target=target[0])
        return self.created

    def simRMLStep(self, handle, dt, n):
        row = self.active[handle]
        row['position'] = min(row['position'] + row['velocity']*dt, row['target'])
        done = row['position'] >= row['target']-1e-12
        return int(done), [row['position'], row['velocity'], 0.]

    def simRMLRemove(self, handle):
        del self.active[handle]


class NativeRMLPreviewRegression(unittest.TestCase):
    def test_second_velocity_preview_starts_at_path_start_and_converges(self):
        path = np.zeros((2, 7));path[-1, 0] = 1.
        sim = ScalarRMLFixture()
        samples, audit = sample_velocity_checked_rml(path, np.ones(7), (2., 4., 1000.), sim)
        self.assertEqual(len(audit['velocity_trials']), 2)
        self.assertAlmostEqual(audit['velocity_trials'][0]['max_native_velocity_ratio'], 2.)
        self.assertLessEqual(audit['velocity_trials'][1]['max_native_velocity_ratio'], 1.001)
        self.assertLessEqual(samples[0][0], .05+1e-12)
        self.assertTrue(np.allclose(samples[-1], path[-1]))
        self.assertFalse(sim.active)

    def test_duplicate_path_vertices_do_not_divide_by_zero(self):
        path = np.zeros((4, 7));path[-1, 0] = .1
        sim = ScalarRMLFixture()
        samples, _ = sample_velocity_checked_rml(path, np.ones(7), (1., 4., 1000.), sim)
        self.assertTrue(np.isfinite(samples).all())
        self.assertFalse(sim.active)

    def test_zero_length_path_no_RML_or_physics_is_needed(self):
        sim = ScalarRMLFixture()
        samples, audit = sample_velocity_checked_rml(np.zeros((2, 7)), np.ones(7), (1., 4., 1000.), sim)
        self.assertTrue(audit['zero_length_path'])
        self.assertEqual(len(samples), 1)
        self.assertEqual(sim.created, 0)


if __name__ == '__main__':
    unittest.main()
