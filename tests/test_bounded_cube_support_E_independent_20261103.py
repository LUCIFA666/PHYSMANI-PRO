"""Synthetic E support intervals only; no real RGBD/GT/calibration or fit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PURE = ROOT / 'artifacts/framework/near_goal_segmentation_20261103/bounded_cube_support_measurement_E_function_v1/measurement.py'
WRAPPER = ROOT / 'artifacts/framework/near_goal_segmentation_20261103/full32_FIT_selection_bounded_cube_E_diagnostic_v2/analyze.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def two_faces(centre=(.12, -.08, .94), span_y=.024, side=1):
    centre = np.array(centre)
    r = np.linspace(-.024, .024, 30)
    y = np.linspace(-span_y, span_y, 30)
    xface = centre + np.c_[np.full(30, side * .025), y, r]
    zface = centre + np.c_[r, y, np.full(30, side * .025)]
    return np.r_[xface, zface], np.repeat([0, 2], 30), np.repeat(centre[[0, 2]], 30), centre


class BoundedCubeSupportEIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = load('independent_bounded_cube_E', PURE)

    def aggregate(self, p, a, c):
        return self.E.aggregate_known_cube_support(p, a, c)

    def test_missing_axis_tight_interval_and_both_face_sides_preserve_medians(self):
        for side in (-1, 1):
            p, a, c, centre = two_faces(side=side)
            r = self.aggregate(p, a, c)
            self.assertTrue(r['support_available'])
            np.testing.assert_allclose(r['centre'], centre, rtol=0, atol=1e-15)
            self.assertEqual(r['observed_axis'], [True, False, True])
            self.assertEqual(r['axis_counts'], [30, 0, 30])
            np.testing.assert_allclose(np.array(r['intervals'])[1], [centre[1] - .00125, centre[1] + .00125], rtol=0, atol=1e-15)
            np.testing.assert_allclose(r['radius_each'], [.00025, .00125, .00025], rtol=0, atol=1e-15)
            self.assertAlmostEqual(r['joint_radius'], np.linalg.norm([.00025, .00125, .00025]), places=14)
            self.assertIn('visible_face', r['coordinate_sources'][0])
            self.assertNotIn('visible_face', r['coordinate_sources'][1])

    def test_fewer_than_three_face_proposals_do_not_override_support_interval(self):
        p, a, c, centre = two_faces()
        a[-2:] = 1; c[-2:] = 9.  # two incompatible proposals must not become an observed face
        r = self.aggregate(p, a, c)
        self.assertTrue(r['support_available'])
        self.assertEqual(r['axis_counts'], [30, 2, 28])
        self.assertFalse(r['observed_axis'][1])
        self.assertAlmostEqual(r['centre'][1], centre[1], places=14)
        # Exactly three proposals invoke the unchanged face-median branch.
        p, a, c, centre = two_faces()
        a[-3:] = 1; c[-3:] = centre[1] + np.array([.001, 0., -.001])
        r = self.aggregate(p, a, c)
        self.assertTrue(r['observed_axis'][1])
        self.assertAlmostEqual(r['radius_each'][1], .00025, places=14)
        self.assertAlmostEqual(r['centre'][1], centre[1], places=14)

    def test_wide_empty_and_joint_radius_not_per_axis_radius_are_unknown(self):
        p, a, c, _ = two_faces(span_y=.001)
        r = self.aggregate(p, a, c)
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])
        p, a, c, _ = two_faces(span_y=.026)
        r = self.aggregate(p, a, c)
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])
        # Two missing axes each have a 4mm radius, but their joint radius exceeds 5mm.
        centre = np.array([.12, -.08, .94])
        p = centre + np.c_[np.full(60, .025), np.linspace(-.02125, .02125, 60), np.linspace(-.02125, .02125, 60)]
        r = self.aggregate(p, np.zeros(60, int), np.full(60, centre[0]))
        np.testing.assert_allclose(r['radius_each'], [.00025, .004, .004], atol=1e-15, rtol=0)
        self.assertGreater(r['joint_radius'], .005)
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])

    def test_all_observed_axes_keep_original_medians_not_point_bbox_centres(self):
        p, a, c, centre = two_faces()
        a = np.repeat([0, 1, 2], 20)
        c = np.repeat(centre, 20)
        c[[0, 20, 40]] += .001
        r = self.aggregate(p, a, c)
        self.assertTrue(r['support_available'])
        np.testing.assert_allclose(r['centre'], centre, atol=1e-15, rtol=0)
        np.testing.assert_allclose(r['radius_each'], [.00025] * 3, atol=1e-15, rtol=0)
        self.assertEqual(r['observed_axis'], [True] * 3)

    def test_no_points_nonfinite_or_derived_overflow_cannot_fabricate_centre(self):
        r = self.aggregate(np.empty((0, 3)), np.empty(0, int), np.empty(0))
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])
        p, a, c, _ = two_faces()
        p[0, 1] = np.nan
        r = self.aggregate(p, a, c)
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])
        p, a, c, _ = two_faces()
        # Finite supported proposals can overflow numpy's even-N median.
        c[a == 0] = 1e308
        with np.errstate(over='ignore', invalid='ignore'):
            r = self.aggregate(p, a, c)
        self.assertFalse(r['support_available']); self.assertIsNone(r['centre'])


class BoundedCubeSupportEWrapperIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = load('independent_bounded_cube_E_wrapper', WRAPPER)
        cls.pure = load('independent_bounded_cube_E_wrapper_function', PURE)

    def measure(self, inp, base, p, views, axes, proposals):
        return self.E.measurement_e(inp, base, p, views, axes, proposals, self.pure.aggregate_known_cube_support)

    @staticmethod
    def empty_row():
        return dict(eroded_points=0, D=dict(camera_point_ledger=[dict(camera=i,
            baseline_candidate_points=0, accepted=[], accepted_points=0,
            point_world=[], axis=[], coordinate_proposal=[]) for i in range(4)]))

    def test_empty_camera_and_empty_frame_serialized_ledger_remain_unknown(self):
        row = self.empty_row()
        p, views, axes, proposals = self.E.frozen_d_points(row)
        self.assertEqual(p.shape, (0, 3)); self.assertEqual(views.shape, (0,))
        self.assertEqual(axes.shape, (0,)); self.assertEqual(proposals.shape, (0,))
        base = dict(reasons=[], centre=[0., 0., 1.], diagnostic_centre_only=[0., 0., 1.], refiner_available=True)
        r = self.measure(dict(public_sphere_center=np.array([0., 0., 1.])), base, p, views, axes, proposals)
        self.assertFalse(r['refiner_available']); self.assertIsNone(r['centre'])
        row['eroded_points'] = 3
        row['D']['camera_point_ledger'][1].update(baseline_candidate_points=3,
            accepted=[True, False, True], accepted_points=2,
            point_world=[[.1, .2, 1.], [None, None, None], [.1, .2, 1.025]],
            axis=[0, -1, 2], coordinate_proposal=[.075, None, 1.])
        p, views, axes, proposals = self.E.frozen_d_points(row)
        self.assertEqual(p.shape, (2, 3))
        self.assertEqual(views.tolist(), [1, 1]); self.assertEqual(axes.tolist(), [0, 2])
        np.testing.assert_array_equal(proposals, [.075, 1.])

    def test_nonboolean_serialized_accepted_flag_cannot_be_coerced_to_truth(self):
        for poison in (0, 1, 'True', None, 1.):
            row = self.empty_row(); row['eroded_points'] = 1
            row['D']['camera_point_ledger'][0].update(baseline_candidate_points=1,
                accepted=[poison], accepted_points=1, point_world=[[.1, .2, 1.]],
                axis=[0], coordinate_proposal=[.075])
            with self.subTest(poison=poison), self.assertRaises((ValueError, TypeError)):
                self.E.frozen_d_points(row)

    def test_valid_missing_axis_support_cannot_override_original_frame_gates(self):
        p, a, c, centre = two_faces()
        inp = dict(public_sphere_center=centre)
        base = dict(reasons=[], centre=[9., 9., 9.], diagnostic_centre_only=[9., 9., 9.], refiner_available=True)
        views = np.repeat([0, 1, 2], 20)
        r = self.measure(inp, base, p, views, a, c)
        self.assertTrue(r['bounded_support']['support_available'])
        self.assertTrue(r['refiner_available'])
        np.testing.assert_allclose(r['centre'], centre, atol=1e-15, rtol=0)
        guard = 'raw_Cube_mask_touches_crop_boundary_before_erosion'
        for changed_base, changed_views, changed_input, reason in (
            (dict(base, reasons=[guard]), views, inp, guard),
            (base, np.zeros(60, int), inp, 'insufficient_current_camera_views'),
            (base, views, dict(public_sphere_center=centre+[.081, 0., 0.]), 'outside_predeclared_near_scope_0p08m')):
            bad = self.measure(changed_input, changed_base, p, changed_views, a, c)
            self.assertTrue(bad['bounded_support']['support_available'])
            self.assertFalse(bad['refiner_available']); self.assertIsNone(bad['centre'])
            self.assertIn(reason, bad['reasons'])
        # All axes observed means the aggregate succeeds, while total9/current
        # confidence still fails the unchanged outer geometry requirements.
        low = self.measure(inp, base, p[:9], np.repeat([0, 1, 2], 3), np.repeat([0, 1, 2], 3), np.repeat(centre, 3))
        self.assertTrue(low['bounded_support']['support_available'])
        self.assertFalse(low['refiner_available'])
        self.assertIn('insufficient_current_depth_points', low['reasons'])
        self.assertIn('original_geometry_confidence_below_half', low['reasons'])

    def test_unknown_interval_and_no_points_never_reuse_old_centre(self):
        p, a, c, centre = two_faces(span_y=.001)
        inp = dict(public_sphere_center=centre)
        base = dict(reasons=[], centre=centre.tolist(), diagnostic_centre_only=centre.tolist(), refiner_available=True)
        bad = self.measure(inp, base, p, np.repeat([0, 1, 2], 20), a, c)
        self.assertFalse(bad['refiner_available']); self.assertIsNone(bad['centre'])
        self.assertFalse(bad['bounded_support']['support_available'])
        none = self.measure(inp, base, np.empty((0, 3)), np.empty(0, int), np.empty(0, int), np.empty(0))
        self.assertFalse(none['refiner_available']); self.assertIsNone(none['centre'])
        self.assertIsNone(none['diagnostic_centre_only'])

    def test_fixed140_denominator_and_unknown_never_imputed_as_zero_error(self):
        rows = [dict(scene_id=str(i), condition='near_goal' if i < 140 else 'sphere_only',
            A=dict(measurement=dict(refiner_available=i < 140)),
            E=dict(measurement=dict(refiner_available=i < 126), error_xyz_m=[.001, 0., 0.] if i < 126 else None))
            for i in range(224)]
        r = self.E.stats(rows, 'E')
        self.assertEqual(r['fixed_baseline_eligible'], 140)
        self.assertEqual(r['coverage'], .9); self.assertEqual(r['negatives'], 84)
        self.assertEqual(len(r['missing_from_baseline']), 14)
        self.assertEqual(r['available_measurement_error']['n'], 126)
        for row in rows:
            row['E']['measurement']['refiner_available'] = False
            row['E']['error_xyz_m'] = None
        r = self.E.stats(rows, 'E')
        self.assertEqual(r['fixed_baseline_eligible'], 140)
        self.assertEqual(r['coverage'], 0.)
        self.assertIsNone(r['available_measurement_error']['p95_m'])


if __name__ == '__main__':
    unittest.main()
