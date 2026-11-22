"""Synthetic public-camera algebra; no real images, poses, calibration or fit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PURE = ROOT / 'artifacts/framework/near_goal_segmentation_20261103/visible_cube_faces_measurement_D_function_v1/measurement.py'
WRAPPER = ROOT / 'artifacts/framework/near_goal_segmentation_20261103/full32_FIT_selection_visible_cube_D_diagnostic_v1/analyze.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_at(origin, point, roll=.37):
    z = np.asarray(point) - np.asarray(origin)
    z = z / np.linalg.norm(z)
    up = np.array([0., 0., 1.])
    if abs(z @ up) > .9:
        up = np.array([0., 1., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    c, s = np.cos(roll), np.sin(roll)
    R = R @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, origin
    return T


def synthetic_face(axis=0, sign=1, classic=False, normal=None, origin=None):
    centre = np.array([.13, -.08, .93])
    n = np.eye(3)[axis] if normal is None else np.asarray(normal, float)
    face = centre + sign * .025 * np.eye(3)[axis]
    if origin is None:
        origin = face + sign * .31 * np.eye(3)[axis] + np.array([.09, .12, .08])
    T = camera_at(origin, face)
    K = np.array([[-87., 0., 8.5], [0., -103., 8.5], [0., 0., 1.]])
    # Plane n·(P-face)=0, P=o+R*z*K^-1*x.  q=h·x.
    m = (T[:3, :3].T @ n) / (n @ (face - T[:3, 3]))
    h = np.linalg.solve(K.T, m)
    v, u = np.indices((17, 17))
    q = h[0] * (u + .5) + h[1] * (v + .5) + h[2]
    if classic:
        q = q - .5 * max(abs(h[0]), abs(h[1]))
    if not (q > 0).all():
        raise AssertionError('Synthetic camera must see a positive-depth plane.')
    return 1 / q, np.ones((17, 17), bool), K, T, centre, face


class VisibleCubeFacesDIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = load('independent_visible_cube_D', PURE)

    def proposals(self, z, mask, K, T, classic=False, pixels=None):
        if pixels is None:
            pixels = np.array([[8, 8], [7, 8], [8, 7]], np.int64)
        return self.D.point_face_proposals(z, mask, pixels, K, T, classic=classic)

    def test_signed_intrinsics_nonaxis_camera_and_both_visible_face_signs(self):
        for classic in (False, True):
            for axis in range(3):
                for sign in (-1, 1):
                    with self.subTest(classic=classic, axis=axis, sign=sign):
                        z, raw, K, T, centre, _ = synthetic_face(axis, sign, classic)
                        r = self.proposals(z, raw, K, T, classic)
                        self.assertTrue(r['accepted'].all())
                        np.testing.assert_array_equal(r['axis'], np.full(3, axis))
                        np.testing.assert_array_equal(r['sign'], np.full(3, sign))
                        np.testing.assert_allclose(np.abs(r['normals_world']), np.tile(np.eye(3)[axis], (3, 1)), atol=2e-11, rtol=0)
                        np.testing.assert_allclose(r['coordinate_proposal'], centre[axis], atol=2e-12, rtol=0)
                        np.testing.assert_allclose(r['point_world'][0], centre + sign * .025 * np.eye(3)[axis], atol=2e-12, rtol=0)

    def test_front_never_receives_classic_offset_and_normal_uses_global_pixel_coefficients(self):
        z, raw, K, T, centre, face = synthetic_face(1, 1, False)
        r = self.proposals(z, raw, K, T, False, np.array([[8, 8]]))
        self.assertTrue(r['accepted'][0])
        self.assertEqual(r['depth_corrected'][0], z[8, 8])
        np.testing.assert_allclose(r['point_world'][0], face, atol=2e-12, rtol=0)
        self.assertAlmostEqual(r['coordinate_proposal'][0], centre[1], places=12)
        a, b, c = r['coefficients'][0]
        h = np.array([a, b, c - a * 8.5 - b * 8.5])
        nw = T[:3, :3] @ (K.T @ h)
        nw /= np.linalg.norm(nw)
        np.testing.assert_allclose(np.abs(nw), [0., 1., 0.], atol=2e-11, rtol=0)

    def test_original_centre_depth_not_replaced_by_plane_intercept(self):
        z, raw, K, T, _, _ = synthetic_face(2, 1, True)
        z[8, 8] += .0001
        r = self.proposals(z, raw, K, T, True, np.array([[8, 8]]))
        self.assertTrue(r['accepted'][0])
        a, b, c = r['coefficients'][0]
        offset = .5 * max(abs(a), abs(b))
        expected = 1 / (1 / z[8, 8] + offset)
        wrong = 1 / (c + offset)
        self.assertAlmostEqual(r['depth_corrected'][0], expected, places=13)
        self.assertGreater(abs(r['depth_corrected'][0] - wrong), .00005)

    def test_front_also_requires_current_raw_nine_pixel_plane_support(self):
        z, raw, K, T, _, _ = synthetic_face()
        raw[7, 7] = False
        r = self.proposals(z, raw, K, T, False, np.array([[8, 8], [0, 0]]))
        self.assertFalse(r['accepted'].any())
        self.assertEqual(len(r['reasons']), 2)
        raw[:] = True
        z[7, 7] = np.nan
        self.assertFalse(self.proposals(z, raw, K, T, False, np.array([[8, 8]]))['accepted'][0])

    def test_offaxis_faces_and_camera_on_axis_plane_do_not_manufacture_coordinate(self):
        for angle, expected in ((4., True), (6., False)):
            n = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.]
            z, raw, K, T, _, _ = synthetic_face(0, 1, False, normal=n)
            self.assertEqual(bool(self.proposals(z, raw, K, T, False, np.array([[8, 8]]))['accepted'][0]), expected)
        # A near-X plane viewed at a grazing angle.  The centre ray has
        # exactly x=0, equal to camera x, so no camera-side sign exists.
        K = np.array([[-1000., 0., 8.5], [0., -1000., 8.5], [0., 0., 1.]])
        v, u = np.indices((17, 17))
        z = 1 / (1 - (u - 8) / (1000 * np.tan(np.deg2rad(4.))))
        r = self.proposals(z, np.ones_like(z, bool), K, np.eye(4), False, np.array([[8, 8]]))
        self.assertFalse(r['accepted'][0])

    def test_crease_rejections_and_empty_input_keep_unknown_ledger(self):
        z, raw, K, T, _, _ = synthetic_face()
        v, u = np.indices(z.shape)
        z = 1 / (1 / z + .03 * np.abs(u - 8))
        r = self.proposals(z, raw, K, T, False, np.array([[8, 8]]))
        self.assertFalse(r['accepted'][0])
        self.assertIn('nonplanar', r['reasons'][0])
        empty = self.proposals(z, raw, K, T, False, np.empty((0, 2), np.int64))
        self.assertEqual(empty['accepted'].shape, (0,))
        self.assertEqual(empty['normals_world'].shape, (0, 3))


class VisibleCubeFacesDWrapperIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = load('independent_visible_cube_D_wrapper', WRAPPER)

    def test_missing_axis_never_borrows_baseline_or_temporal_centre(self):
        base = dict(reasons=[], centre=[.01, .02, 1.], diagnostic_centre_only=[.01, .02, 1.], refiner_available=True)
        inp = dict(public_sphere_center=np.array([.01, .02, 1.]))
        points = np.zeros((60, 3)); views = np.repeat([0, 1, 2], 20)
        axes = np.repeat([0, 1], 30); proposals = np.repeat([.01, .02], 30)
        r = self.D.measurement_d(inp, base, points, views, axes, proposals)
        self.assertFalse(r['refiner_available'])
        self.assertIsNone(r['centre']); self.assertIsNone(r['diagnostic_centre_only'])
        self.assertEqual(r['axis_coordinate_samples'], [30, 30, 0])
        self.assertIn('insufficient_current_visible_face_coordinate_Z', r['reasons'])
        empty = self.D.measurement_d(inp, base, np.empty((0, 3)), np.empty(0, int), np.empty(0, int), np.empty(0))
        self.assertIsNone(empty['diagnostic_centre_only'])

    def test_current_axis_medians_and_original_support_crop_scope_guards(self):
        centre = np.array([.01, -.02, 1.])
        inp = dict(public_sphere_center=centre)
        base = dict(reasons=[], centre=[9., 9., 9.], diagnostic_centre_only=[9., 9., 9.], refiner_available=True)
        points = np.zeros((30, 3)); views = np.repeat([0, 1, 2], 10)
        axes = np.repeat([0, 1, 2], 10); proposals = np.repeat(centre, 10)
        proposals[[0, 10, 20]] += .001
        r = self.D.measurement_d(inp, base, points, views, axes, proposals)
        self.assertTrue(r['refiner_available'])
        np.testing.assert_array_equal(r['centre'], centre)
        self.assertEqual(r['geometry_confidence'], .5)
        self.assertFalse(r['old_quantile_centre_used'])
        self.assertFalse(r['orientation_observed'])
        one_view = self.D.measurement_d(inp, base, points, np.zeros(30, int), axes, proposals)
        self.assertFalse(one_view['refiner_available'])
        few = self.D.measurement_d(inp, base, points[:9], views[:9], np.repeat([0, 1, 2], 3), np.repeat(centre, 3))
        self.assertFalse(few['refiner_available'])
        guard = 'raw_Cube_mask_touches_crop_boundary_before_erosion'
        cropped = self.D.measurement_d(inp, dict(base, reasons=[guard]), points, views, axes, proposals)
        self.assertFalse(cropped['refiner_available']); self.assertIn(guard, cropped['reasons'])
        outside = self.D.measurement_d(dict(public_sphere_center=centre+[.081, 0., 0.]), base, points, views, axes, proposals)
        self.assertFalse(outside['refiner_available'])

    def test_fixed140_denominator_and_all_unknown_remain_explicit(self):
        rows = [dict(scene_id=str(i), condition='near_goal' if i < 140 else 'sphere_only',
            A=dict(measurement=dict(refiner_available=i < 140)),
            D=dict(measurement=dict(refiner_available=i < 126), error_xyz_m=[.001, 0., 0.] if i < 126 else None))
            for i in range(224)]
        r = self.D.previous.stats(rows, 'D')
        self.assertEqual(r['fixed_baseline_eligible'], 140)
        self.assertEqual(r['coverage'], .9); self.assertEqual(r['negatives'], 84)
        self.assertEqual(len(r['missing_from_baseline']), 14)
        for row in rows:
            row['D']['measurement']['refiner_available'] = False
            row['D']['error_xyz_m'] = None
        r = self.D.previous.stats(rows, 'D')
        self.assertEqual(r['coverage'], 0.)
        self.assertIsNone(r['available_measurement_error']['p95_m'])


if __name__ == '__main__':
    unittest.main()
