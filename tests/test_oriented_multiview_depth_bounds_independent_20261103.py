"""Independent synthetic checks; no recorded images, labels or simulator."""
import importlib
import unittest

import numpy as np


class ObservedDepthLowerBoundIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(
            'our_mani.local_mpc_control_20261103.oriented_cube_multiview_free_space_prototype_20261103')

    def call(self, depth, renderer='CLASSIC'):
        result = self.module.observed_depth_lower_bound(depth, renderer)
        self.assertIsInstance(result, dict)
        lower, valid = np.asarray(result['lower_depth']), np.asarray(result['valid'])
        self.assertEqual(lower.shape, depth.shape)
        self.assertEqual(valid.shape, depth.shape)
        self.assertEqual(valid.dtype, np.dtype(bool))
        return result, lower, valid

    @staticmethod
    def plane(a, b, q=1.25):
        v, u = np.indices((5, 5), dtype=float)
        # Classic fixed max-slope polygon offset decreases inverse depth.
        return 1 / (q + a * (u - 2) + b * (v - 2) - .5 * max(abs(a), abs(b)))

    def test_affine_inverse_depth_corners_and_both_signed_slopes(self):
        for a, b in ((.012, .004), (-.012, .004), (.004, -.012), (-.004, -.012), (0., 0.)):
            with self.subTest(a=a, b=b):
                depth = self.plane(a, b)
                _, lower, valid = self.call(depth)
                self.assertTrue(valid[2, 2])
                expected = 1 / (1.25 + .5 * abs(a) + .5 * abs(b))
                self.assertAlmostEqual(lower[2, 2], expected, delta=1e-12)
                self.assertLessEqual(lower[2, 2], 1 / 1.25 + 1e-12)
                self.assertFalse(valid[0].any())
                self.assertFalse(valid[-1].any())
                self.assertFalse(valid[:, 0].any())
                self.assertFalse(valid[:, -1].any())

    def test_raw_center_and_actual_metric_residual_follow_frozen_formula(self):
        depth = self.plane(.008, -.003)
        depth[2, 2] += .0001
        _, lower, valid = self.call(depth)
        self.assertTrue(valid[2, 2])
        v, u = np.indices((3, 3), dtype=float)
        design = np.c_[(u - 1).ravel(), (v - 1).ravel(), np.ones(9)]
        z = depth[1:4, 1:4].ravel()
        a, b, c = np.linalg.lstsq(design, 1 / z, rcond=None)[0]
        residual = np.max(np.abs(z - 1 / (design @ [a, b, c])))
        self.assertGreater(residual, 0)
        self.assertLess(residual, .00025)
        q_center = 1 / depth[2, 2] + .5 * max(abs(a), abs(b))
        q_max_corner = q_center + .5 * abs(a) + .5 * abs(b)
        expected = 1 / q_max_corner - residual
        self.assertAlmostEqual(lower[2, 2], expected, delta=1e-12)
        ls_plane_min = 1 / (c + .5 * max(abs(a), abs(b)) + .5 * abs(a) + .5 * abs(b))
        self.assertLessEqual(lower[2, 2], ls_plane_min + 1e-12)

    def test_missing_depth_or_nonplanar_stencil_creates_no_negative_evidence(self):
        for value in (np.nan, np.inf, 0., -.1):
            depth = self.plane(.008, .003)
            depth[1, 1] = value
            _, _, valid = self.call(depth)
            self.assertFalse(valid[2, 2])
        depth = self.plane(.008, .003)
        depth[2, 2] += .002
        _, _, valid = self.call(depth)
        self.assertFalse(valid[2, 2])

    def test_gl3_keeps_current_raw_depth_and_does_not_require_neighborhood(self):
        depth = np.array([[np.nan, .7, 0.], [.9, 1.1, -1.], [np.inf, .4, .2]], dtype=np.float32)
        _, lower, valid = self.call(depth, 'GL3')
        expected = np.isfinite(depth) & (depth > 0)
        np.testing.assert_array_equal(valid, expected)
        np.testing.assert_array_equal(lower[valid], depth[valid])
        self.assertTrue(valid[0, 1])
        self.assertTrue(valid[-1, -1])

    def test_renderer_identity_is_explicit_and_unknown_is_rejected(self):
        for value in ('front', 'classic', '8', '', None, True):
            with self.subTest(renderer=value), self.assertRaises((ValueError, TypeError)):
                self.module.observed_depth_lower_bound(self.plane(.008, .003), value)


if __name__ == '__main__':
    unittest.main()
