"""Analytic geometry and ambiguity checks; no real observations or labels."""
from copy import deepcopy
import math
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103 import known_normal_single_pixel_plane_prototype_20261103 as m


def rot(axis, angle):
    v = np.zeros(3); v[axis] = 1.
    x, y, z = v
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)


def packets_for_point(point, camera=None):
    point = np.asarray(point, float)
    origin = np.array([0., 0., 1.]) if camera is None else np.asarray(camera, float)
    z = point-origin; depth = float(np.linalg.norm(z)); z /= depth
    x = np.cross([0., 1., 0.], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.c_[x, y, z]; T[:3, 3] = origin
    K = np.array([[-100., 0., .5], [0., -100., .5], [0., 0., 1.]])
    packets = [dict(view_id=view, renderer=renderer, timestamp=1., depth=np.ones((1, 1)),
                    pixels_uv=np.empty((0, 2), dtype=int), K=K.copy(), T=T.copy())
               for view, renderer in zip(m.VIEW_IDS, m.RENDERERS)]
    packets[0]['depth'][0, 0] = depth
    packets[0]['pixels_uv'] = np.array([[0, 0]], dtype=int)
    mode = dict(nuisance_rotation=np.eye(3).tolist(), reference_point=[0., 0., 0.],
                support={'intervals':[[-.00025, .00025], [-.00025, .00025], [.793, .807]],
                         'axis_counts':[3, 3, 0], 'source':'synthetic_current_support'})
    return packets, mode


class KnownNormalSinglePixelIndependentTests(unittest.TestCase):
    def test_signed_K_nonunit_ray_exact_GL3_and_classic_source_formula(self):
        K = np.array([[-200., 1.7, 128.], [0., -190., 128.], [0., 0., 1.]])
        T = np.eye(4); T[:3, :3] = rot(2, .3)@rot(1, -.4)@rot(0, .2); T[:3, 3] = [.2, -.3, 1.1]
        uv = np.array([37, 164]); ray = np.linalg.inv(K)@np.r_[uv+.5, 1.]
        self.assertGreater(abs(np.linalg.norm(ray)-1.), .01)
        for normal in ([0., 0., 1.], [.25, -.35, .9]):
            n = np.asarray(normal, float); n /= np.linalg.norm(n)
            B = (T[:3, :3].T@n)@np.linalg.inv(K)
            b = float(B@np.r_[uv+.5, 1.]); D = math.copysign(.7, b)
            true_z = D/b
            for renderer in ('GL3', 'CLASSIC'):
                q_raw = b/D - (0. if renderer == 'GL3' else .5*max(abs(B[0]), abs(B[1]))/abs(D))
                raw_z = 1/q_raw
                result = m.plane_from_known_normal(raw_z, uv, K, T, n, renderer)
                self.assertTrue(result['valid'])
                self.assertAlmostEqual(result['corrected_depth_m'], true_z, delta=1e-12)
                self.assertAlmostEqual(result['camera_plane_offset'], D, delta=1e-12)
                self.assertAlmostEqual(result['world_plane_offset'], float(n@T[:3, 3]+D), delta=1e-12)
                np.testing.assert_allclose(result['point_world'], T[:3, 3]+T[:3, :3]@ray*true_z, rtol=0, atol=1e-12)
                self.assertFalse(result['qualified'])
                opposite = m.plane_from_known_normal(raw_z, uv, K, T, -n, renderer)
                np.testing.assert_allclose(opposite['point_world'], result['point_world'], rtol=0, atol=1e-12)
                self.assertAlmostEqual(opposite['world_plane_offset'], -result['world_plane_offset'], delta=1e-12)

    def test_common_world_SE3_preserves_camera_depth_and_transforms_plane(self):
        K = np.array([[-90., 0., .5], [0., -100., .5], [0., 0., 1.]])
        T = np.eye(4); T[:3, :3] = rot(1, .4); T[:3, 3] = [.2, -.1, .7]
        n = np.array([0., 0., 1.]); uv = np.array([3, 2])
        initial = m.plane_from_known_normal(.6, uv, K, T, n, 'CLASSIC')
        R, shift = rot(2, .7)@rot(0, -.2), np.array([.4, -.3, .8])
        moved_T = T.copy(); moved_T[:3, :3] = R@T[:3, :3]; moved_T[:3, 3] = R@T[:3, 3]+shift
        moved = m.plane_from_known_normal(.6, uv, K, moved_T, R@n, 'CLASSIC')
        self.assertTrue(moved['valid'])
        self.assertAlmostEqual(moved['corrected_depth_m'], initial['corrected_depth_m'], delta=1e-12)
        self.assertAlmostEqual(moved['camera_plane_offset'], initial['camera_plane_offset'], delta=1e-12)
        np.testing.assert_allclose(moved['point_world'], R@initial['point_world']+shift, rtol=0, atol=1e-12)

    def test_grazing_invalid_depth_or_negative_classic_factor_is_unknown(self):
        K = np.array([[-100., 0., .5], [0., -100., .5], [0., 0., 1.]])
        T = np.eye(4); uv = np.array([0, 0])
        for normal in ([1., 0., 0.], [1., 0., .001]):
            result = m.plane_from_known_normal(1., uv, K, T, normal, 'CLASSIC')
            self.assertFalse(result['valid'])
        for depth in (0., -1., np.nan, np.inf):
            self.assertFalse(m.plane_from_known_normal(depth, uv, K, T, [0., 0., 1.], 'GL3')['valid'])
        with self.assertRaises(ValueError):m.plane_from_known_normal(1., [0.5, 0], K, T, [0., 0., 1.], 'GL3')
        with self.assertRaises(ValueError):m.plane_from_known_normal(1., uv, K, T, [0., 0., 1.], 'unknown')

    def test_one_pure_top_pixel_can_only_make_conditional_unqualified_candidate(self):
        packets, mode = packets_for_point([0., 0., .825])
        before = deepcopy(mode)
        result = m.refine_current_single_pixels(packets, mode)
        self.assertEqual(result['status'], 'CONDITIONAL_CANDIDATE')
        np.testing.assert_allclose(result['centre'], [0., 0., .8], rtol=0, atol=1e-12)
        self.assertEqual(len(result['pixels'][0]['assignments']), 6)
        self.assertFalse(result['pure_pixel_ownership_verified'])
        self.assertFalse(result['position_measurement_qualified'])
        self.assertFalse(result['qualified'])
        self.assertEqual(mode, before)

    def test_side_pixel_keeps_face_ambiguity_instead_of_selecting_a_false_Z_plane(self):
        # The point is on the true +X side, near its top. Calling it +Z would
        # invent centreZ=.794, inside the initial interval but6mm below .800.
        packets, mode = packets_for_point([.025, 0., .819], [.4, 0., 1.2])
        result = m.refine_current_single_pixels(packets, mode)
        assignments = result['pixels'][0]['assignments']
        faces = {(row['axis'], row['face_sign']) for row in assignments if row['camera_facing']}
        self.assertIn((0, 1), faces)
        self.assertIn((2, 1), faces)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['centre'])
        self.assertIn('joint_radius_exceeds_unchanged5mm', result['reasons'])

    def test_depth_influence_semantics_does_not_certify_a_pure_surface_depth(self):
        K = np.array([[-100., 0., .5], [0., -100., .5], [0., 0., 1.]])
        # A semantic counterexample, NOT a claim that this renderer literally
        # linearly averages depth: base1.02 changes when Cube is hidden to1.10,
        # so it is a positive depth-influence pixel although true face is1.00.
        result = m.plane_from_known_normal(1.02, [0, 0], K, np.eye(4), [0., 0., 1.], 'GL3')
        self.assertTrue(result['valid'])
        self.assertGreater(abs(result['world_plane_offset']-1.), .005)
        self.assertFalse(result['pure_surface_or_pixel_ownership_verified'])
        self.assertFalse(result['qualified'])

    def test_all_views_pixels_contradictions_and_invalid_samples_are_retained(self):
        packets, mode = packets_for_point([0., 0., .825])
        packets[1]['pixels_uv'] = np.array([[0, 0]], dtype=int)
        packets[1]['depth'][0, 0] = .165  # top .835 ->centre .810 contradicts initialZ
        result = m.refine_current_single_pixels(packets, mode)
        self.assertEqual(len(result['pixels']), 2)
        self.assertEqual(len(result['intersection_trace']), 2)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('all_current_pixel_face_disjunctions_inconsistent', result['reasons'])
        self.assertTrue(all(len(pixel['assignments']) == 6 for pixel in result['pixels']))
        packets, mode = packets_for_point([0., 0., .825])
        packets[1]['pixels_uv'] = np.array([[0, 0]], dtype=int)
        packets[1]['depth'][0, 0] = np.nan
        invalid = m.refine_current_single_pixels(packets, mode)
        self.assertEqual(invalid['status'], 'UNKNOWN')
        self.assertEqual(len(invalid['pixels']), 2)
        packets[2]['timestamp'] += .05
        with self.assertRaises(ValueError):m.refine_current_single_pixels(packets, mode)


if __name__ == '__main__':
    unittest.main()
