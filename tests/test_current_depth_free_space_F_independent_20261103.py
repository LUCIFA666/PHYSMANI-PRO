"""Synthetic whole-pixel free-space geometry; no real data, GT or fitting."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/near_goal_segmentation_20261103'
PURE = BASE / 'current_depth_free_space_measurement_F_function_v1/measurement.py'
E_SOURCE = BASE / 'bounded_cube_support_measurement_E_function_v1/measurement.py'
WRAPPER = BASE / 'full32_FIT_selection_free_space_cube_F_diagnostic_v1/analyze.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_scene(E, span_x=.001):
    centre = np.array([.12, -.08, .94])
    r = np.linspace(-.024, .024, 30)
    x = np.linspace(-span_x, span_x, 30)
    points = centre + np.r_[np.c_[x, np.full(30, .025), r], np.c_[x, r, np.full(30, .025)]]
    support = E.aggregate_known_cube_support(points, np.repeat([1, 2], 30), np.repeat(centre[[1, 2]], 30))
    K = np.array([[-100., 0., 32.], [0., -100., 32.], [0., 0., 1.]])
    T = np.eye(4); T[:3, :3] = np.diag([1., -1., -1.]); T[:3, 3] = centre + [0., 0., .2]
    v, u = np.indices((64, 64))
    rays = np.c_[u.ravel() + .5, v.ravel() + .5, np.ones(u.size)] @ np.linalg.inv(K).T
    top_depth = .175
    world = (rays * top_depth) @ T[:3, :3].T + T[:3, 3]
    hit = np.max(np.abs(world[:, :2] - centre[:2]), axis=1) <= .025
    depth = np.where(hit.reshape(64, 64), top_depth, .5)
    return depth, K, T, support, centre


class CurrentDepthFreeSpaceFIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.F = load('independent_current_free_space_F', PURE)
        cls.E = load('independent_current_free_space_E_source', E_SOURCE)

    def test_strict_open_subtraction_keeps_endpoints_and_all_components(self):
        f = self.F.subtract_open_forbidden
        np.testing.assert_array_equal(f([[0., 1.]], [[.25, .75]]), [[0., .25], [.75, 1.]])
        np.testing.assert_array_equal(f([[0., 1.]], [[0., 1.]]), [[0., 0.], [1., 1.]])
        np.testing.assert_array_equal(f([[0., 1.]], [[1., 2.]]), [[0., 1.]])
        np.testing.assert_array_equal(f([[0., 1.]], [[.5, .5]]), [[0., 1.]])
        self.assertEqual(len(f([[0., 1.]], [[-1., 2.]])), 0)
        np.testing.assert_array_equal(f([[0., .3], [.7, 1.]], [[.2, .8]]), [[0., .2], [.8, 1.]])
        np.testing.assert_array_equal(f([[0., 1.]], [[.2, .5], [.5, .8]]), [[0., .2], [.5, .5], [.8, 1.]])

    def test_existing_E_pass_is_exact_and_does_not_query_front(self):
        _, _, _, support, _ = synthetic_scene(self.E, span_x=.024)
        self.assertTrue(support['support_available'])
        before = deepcopy(support)
        result = self.F.refine_with_front_free_space(object(), object(), object(), support)
        self.assertEqual({key: result[key] for key in before}, before)
        self.assertTrue(result['free_space_trace']['E_passthrough'])
        self.assertEqual(support, before)

    def test_all_eight_corners_use_cameraZ_not_normalized_range_and_strict_margin(self):
        K = np.array([[-4., 0., 1.], [0., -4., 1.], [0., 0., 1.]])
        T = np.eye(4); T[:3, :3] = np.diag([1., -1., -1.]); T[2, 3] = 1.2
        fn = self.F.whole_pixel_top_footprints
        r = fn(np.full((2, 2), .1754), K, T, 1.)
        self.assertTrue(bool(r['camera_above_planes']))
        self.assertTrue(np.asarray(r['valid_geometry']).all())
        np.testing.assert_allclose(r['max_expected_depth'], .17525, atol=1e-15, rtol=0)
        np.testing.assert_allclose(r['footprint_min'][0], [0., -.0438125, 1.02475], atol=1e-15, rtol=0)
        np.testing.assert_allclose(r['footprint_max'][0], [.0438125, 0., 1.02525], atol=1e-15, rtol=0)
        self.assertFalse(np.asarray(r['free']).any())
        exact_margin = np.asarray(r['max_expected_depth']).reshape(2, 2) + .00025
        boundary = fn(exact_margin, K, T, 1.)
        self.assertFalse(np.asarray(boundary['free']).any())
        farther = fn(exact_margin + .000001, K, T, 1.)
        self.assertTrue(np.asarray(farther['free']).all())
        # Non-axis camera: explicitly intersect each of the eight rays in
        # world coordinates, retaining the local camera-Z ray parameter.
        a, b = .2, np.pi - .35
        rz = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        rx = np.array([[1., 0., 0.], [0., np.cos(b), -np.sin(b)], [0., np.sin(b), np.cos(b)]])
        T[:3, :3] = rz @ rx; T[:3, 3] = [.1, -.2, 1.2]
        r = fn(np.full((2, 2), .5), K, T, 1.)
        points, expected_z = [], []
        for u, v in ((0., 0.), (1., 0.), (0., 1.), (1., 1.)):
            ray = T[:3, :3] @ np.linalg.solve(K, [u, v, 1.])
            for height in (1.02475, 1.02525):
                t = (height - T[2, 3]) / ray[2]
                points.append(T[:3, 3] + t * ray); expected_z.append(t)
        np.testing.assert_allclose(r['footprint_min'][0], np.min(points, axis=0), atol=1e-15, rtol=0)
        np.testing.assert_allclose(r['footprint_max'][0], np.max(points, axis=0), atol=1e-15, rtol=0)
        self.assertAlmostEqual(r['max_expected_depth'][0], max(expected_z), places=14)

    def test_analytic_current_top_depth_free_background_narrows_only_missing_axis(self):
        depth, K, T, support, centre = synthetic_scene(self.E)
        self.assertFalse(support['support_available'])
        before = deepcopy(support)
        result = self.F.refine_with_front_free_space(depth, K, T, support)
        self.assertTrue(result['support_available'])
        self.assertLessEqual(result['joint_radius'], .005)
        for j in range(3):
            self.assertEqual(len(result['feasible_components'][j]), 1)
            lo, hi = result['feasible_components'][j][0]
            self.assertLessEqual(lo, centre[j]); self.assertGreaterEqual(hi, centre[j])
        np.testing.assert_allclose(result['centre'], centre, atol=.002, rtol=0)
        self.assertEqual(result['intervals'][1:], before['intervals'][1:])
        self.assertEqual(support, before)

    def test_invalid_near_occluder_or_uncertainty_band_supply_no_free_evidence(self):
        _, K, T, support, _ = synthetic_scene(self.E)
        # Max expected top-plane camera Z is .17525; the additional free
        # margin is .00025. .1754 must remain non-free despite centre-only
        # expected depth .175, and nearer occluders likewise are not free.
        for value in (np.nan, 0., -.1, .1, .1754):
            result = self.F.refine_with_front_free_space(np.full((64, 64), value), K, T, support)
            self.assertFalse(result['support_available'])
            self.assertIsNone(result['centre'])
            for j, initial in enumerate(support['intervals']):
                np.testing.assert_allclose(result['feasible_components'][j], [initial], rtol=0, atol=0)

    def test_absent_visible_top_face_cannot_be_supplied_by_support_interval(self):
        depth, K, T, support, _ = synthetic_scene(self.E)
        support['axis_counts'][2] = 0; support['observed_axis'][2] = False
        result = self.F.refine_with_front_free_space(depth, K, T, support)
        self.assertFalse(result['support_available']); self.assertIsNone(result['centre'])

    def test_two_surviving_components_are_not_replaced_by_their_hull_midpoint(self):
        _, K, T, support, _ = synthetic_scene(self.E)
        K[0, 2] = 31.5
        depth = np.full((64, 64), .175)
        depth[31, 31] = .5
        result = self.F.refine_with_front_free_space(depth, K, T, support)
        self.assertEqual(len(result['feasible_components'][0]), 2)
        self.assertFalse(result['support_available']); self.assertIsNone(result['centre'])

    def test_whole_pixel_other_axis_containment_cannot_use_just_pixel_centre(self):
        _, K, T, support, _ = synthetic_scene(self.E)
        K[1, 2] = 31.5
        depth = np.full((64, 64), .175)
        # Centre y=.0245 is inside the common y interval (+/-.02475),
        # but its far corner y>.0253 is outside. This cannot cut X support.
        depth[45, 31] = .5
        result = self.F.refine_with_front_free_space(depth, K, T, support)
        self.assertFalse(result['support_available'])
        np.testing.assert_allclose(result['feasible_components'][0], [support['intervals'][0]], rtol=0, atol=0)


class CurrentDepthFreeSpaceFWrapperIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.F = load('independent_free_space_F_wrapper', WRAPPER)
        cls.E = load('independent_free_space_E_wrapper_source', E_SOURCE)

    @staticmethod
    def forbidden_query(*args):
        raise AssertionError('No free-space query may cross this gate.')

    def row(self, support, available=False, reasons=None):
        return dict(A=dict(measurement=dict(reasons=[] if reasons is None else reasons)),
            E=dict(measurement=dict(refiner_available=available, bounded_support=support,
                reasons=[], centre=support['centre'], diagnostic_centre_only=support['centre'])))

    def test_E_pass_preserved_and_original_gate_failure_never_queries_front(self):
        _, _, _, passed, centre = synthetic_scene(self.E, span_x=.024)
        row = self.row(passed, True)
        measurement, receipt = self.F.measurement_f(object(), row, object(), object(), object(), object(), self.forbidden_query)
        self.assertEqual(measurement, row['E']['measurement'])
        self.assertFalse(receipt['free_space_called'])
        _, _, _, wide, _ = synthetic_scene(self.E)
        p = np.zeros((60, 3)); views = np.repeat([0, 1, 2], 20)
        for reasons, use_points, use_views in (
            (['raw_Cube_mask_touches_crop_boundary_before_erosion'], p, views),
            ([], p, np.zeros(60, int)),
            ([], p[:9], np.repeat([0, 1, 2], 3))):
            row = self.row(wide, False, reasons)
            measurement, receipt = self.F.measurement_f(dict(public_sphere_center=centre), row,
                use_points, use_views, np.zeros(len(use_points), int), np.zeros(len(use_points)), self.forbidden_query)
            self.assertFalse(measurement['refiner_available'])
            self.assertFalse(receipt['free_space_called'])

    def test_final_scope_and_original_support_guard_remain_after_refinement(self):
        depth, K, T, wide, centre = synthetic_scene(self.E)
        _, _, _, passed, _ = synthetic_scene(self.E, span_x=.024)
        # Keep the actual D-axis count provenance while narrowing only X.
        refined = deepcopy(passed)
        refined['feasible_components'] = [[bounds] for bounds in refined['intervals']]
        refined['free_space_trace'] = dict(free_space_applied=True)
        inputs = dict(public_sphere_center=centre+[.081, 0., 0.],
            depth_m=depth[None], K=K[None], T=T[None])
        points = np.zeros((60, 3)); views = np.repeat([0, 1, 2], 20)
        axes = np.repeat([1, 2], 30); proposals = np.repeat(centre[[1, 2]], 30)
        measurement, receipt = self.F.measurement_f(inputs, self.row(wide), points, views, axes, proposals,
            lambda *args: deepcopy(refined))
        self.assertTrue(receipt['free_space_called'])
        self.assertFalse(measurement['refiner_available']); self.assertIsNone(measurement['centre'])
        self.assertIn('outside_predeclared_near_scope_0p08m', measurement['reasons'])

    def test_fixed140_denominator_unknown_and_negatives_are_preserved(self):
        rows = [dict(scene_id=str(i), condition='near_goal' if i < 140 else 'sphere_only',
            A=dict(measurement=dict(refiner_available=i < 140)),
            F=dict(measurement=dict(refiner_available=i < 126), error_xyz_m=[.001, 0., 0.] if i < 126 else None))
            for i in range(224)]
        result = self.F.stats(rows, 'F')
        self.assertEqual(result['fixed_baseline_eligible'], 140)
        self.assertEqual(result['coverage'], .9); self.assertEqual(result['negatives'], 84)
        self.assertEqual(len(result['missing_from_baseline']), 14)
        for row in rows:
            row['F']['measurement']['refiner_available'] = False
            row['F']['error_xyz_m'] = None
        result = self.F.stats(rows, 'F')
        self.assertEqual(result['coverage'], 0.)
        self.assertIsNone(result['available_measurement_error']['p95_m'])


if __name__ == '__main__':
    unittest.main()
