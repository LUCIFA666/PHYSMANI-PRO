"""Synthetic multiview fusion only; no stored observations or truth poses."""
from copy import deepcopy
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103 import oriented_cube_multiview_free_space_prototype_20261103 as m


def rotation(axis, angle):
    unit = np.zeros(3); unit[axis] = 1
    x, y, z = unit
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def fixture(x_bounds=(-.002, .002)):
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1., -1., -1.])
    camera[:3, 3] = [0., 0., 1.]
    K = np.array([[-100., 0., 2.5], [0., -100., 2.5], [0., 0., 1.]])
    packets = [dict(view_id=ident, timestamp=1.25, renderer=renderer,
                    depth=np.full((5, 5), np.nan), K=K.copy(), T=camera.copy())
               for ident, renderer in zip(m.VIEW_IDS, m.RENDERERS)]
    support = dict(axis_counts=[0, 3, 3], intervals=[list(x_bounds), [-.00025, .00025], [-.00025, .00025]],
                   coordinate_sources=['current_support', 'current_face_Y', 'current_face_Z'],
                   centre=None, diagnostic_centre_only=[0., 0., 0.], support_available=False,
                   radius_each=[.002, .00025, .00025], joint_radius=.00204,
                   measured_points=17, visible_views=3, point_count_by_camera=[4, 5, 4, 4])
    estimate = dict(modes=[dict(support=support, nuisance_rotation=np.eye(3).tolist(),
                                reference_point=[0., 0., 0.])], original_positive_point_count=17)
    return packets, estimate


def result_support(result):
    return result['modes'][0]['ray_support']


class OrientedMultiviewFreeSpaceIndependentTests(unittest.TestCase):
    def test_view_identity_renderer_order_and_same_actual_time_are_strict(self):
        packets, estimate = fixture()
        variants = []
        bad = deepcopy(packets); bad.pop(); variants.append(bad)
        bad = deepcopy(packets); bad[0], bad[1] = bad[1], bad[0]; variants.append(bad)
        bad = deepcopy(packets); bad[2]['view_id'] = '8'; variants.append(bad)
        bad = deepcopy(packets); bad[1]['renderer'] = 'GL3'; variants.append(bad)
        bad = deepcopy(packets); bad[3]['timestamp'] += .05; variants.append(bad)
        bad = deepcopy(packets); bad[3]['timestamp'] = True; variants.append(bad)
        bad = deepcopy(packets); bad[3]['timestamp'] = np.nan; variants.append(bad)
        bad = deepcopy(packets); bad[0]['future_position'] = [0., 0., 0.]; variants.append(bad)
        for bad in variants:
            with self.subTest(packet=bad[0]['view_id']), self.assertRaises(ValueError):
                m.refine_multiview_oriented_support(bad, estimate)

    def test_common_SE3_transform_preserves_local_support_and_world_result(self):
        packets, estimate = fixture((-.018, .025))
        packets[0]['depth'][2, 2] = 1.2
        # Build an actually constrained case in a non-axis world orientation.
        # Thus a wrong world/local camera transform changes the surviving cut,
        # rather than passing merely because no depth constraint was active.
        initial_R = rotation(2, .37) @ rotation(0, -.23)
        initial_translation = np.array([.1, -.2, .05])
        for packet in packets:
            packet['T'][:3, :3] = initial_R @ packet['T'][:3, :3]
            packet['T'][:3, 3] = initial_R @ packet['T'][:3, 3] + initial_translation
        estimate['modes'][0]['nuisance_rotation'] = initial_R.tolist()
        estimate['modes'][0]['reference_point'] = initial_translation.tolist()
        baseline = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(baseline['status'], 'AVAILABLE')
        self.assertGreater(result_support(baseline)['multiview_trace']['view_contributions'][0]['constraints'], 0)
        self.assertGreater(result_support(baseline)['intervals'][0][0], .02)
        global_R = rotation(0, .4) @ rotation(1, -.31) @ rotation(2, .22)
        translation = np.array([.7, -.3, .2])
        transformed_packets, transformed_estimate = deepcopy(packets), deepcopy(estimate)
        for packet in transformed_packets:
            packet['T'][:3, :3] = global_R @ packet['T'][:3, :3]
            packet['T'][:3, 3] = global_R @ packet['T'][:3, 3] + translation
        mode = transformed_estimate['modes'][0]
        mode['nuisance_rotation'] = (global_R @ np.asarray(mode['nuisance_rotation'])).tolist()
        mode['reference_point'] = (global_R @ np.asarray(mode['reference_point']) + translation).tolist()
        transformed = m.refine_multiview_oriented_support(transformed_packets, transformed_estimate)
        self.assertEqual(transformed['status'], baseline['status'])
        np.testing.assert_allclose(transformed['centre'], global_R @ baseline['centre'] + translation, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result_support(transformed)['intervals'], result_support(baseline)['intervals'], rtol=0, atol=1e-12)
        self.assertAlmostEqual(transformed['joint_radius'], baseline['joint_radius'], delta=1e-12)
        self.assertEqual(len(result_support(transformed)['multiview_trace']['view_contributions']), 4)

    def test_later_classic_view_can_contradict_front_success_without_view_selection(self):
        packets, estimate = fixture((-.018, .025))
        packets[0]['depth'][2, 2] = 1.2
        front_only = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(front_only['status'], 'AVAILABLE')
        # A valid CLASSIC 3x3 plane admits its centre ray and excludes the
        # remaining component. It cannot be dropped to retain front success.
        packets[1]['depth'][1:4, 0:3] = 1.2
        fused = m.refine_multiview_oriented_support(packets, estimate)
        support = result_support(fused)
        self.assertEqual(fused['status'], 'UNKNOWN')
        self.assertIsNone(fused['centre'])
        self.assertEqual(support['feasible_components'][0], [])
        contributions = support['multiview_trace']['view_contributions']
        self.assertGreater(contributions[0]['constraints'], 0)
        self.assertGreater(contributions[1]['constraints'], 0)
        self.assertIn('8', [item['view_id'] for item in support['multiview_trace']['contradictions']])
        self.assertEqual([item['view_id'] for item in contributions], list(m.VIEW_IDS))
        self.assertTrue(any(step.get('skipped') == 'empty_current_component_no_reset'
                            for step in support['multiview_trace']['rounds']))

    def test_initially_narrow_solution_still_considers_last_actual_view(self):
        packets, estimate = fixture()
        untouched = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(untouched['status'], 'AVAILABLE')
        packets[3]['depth'][1:4, 1:4] = 1.2
        fused = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(fused['status'], 'UNKNOWN')
        trace = result_support(fused)['multiview_trace']
        self.assertGreater(trace['view_contributions'][3]['constraints'], 0)
        self.assertIn('36', [item['view_id'] for item in trace['contradictions']])

    def test_no_reliable_classic_plane_supplies_no_negative_constraint(self):
        packets, estimate = fixture()
        packets[1]['depth'][2, 2] = 1.2  # raw far sample but no 3x3 support
        before_packets, before_estimate = deepcopy(packets), deepcopy(estimate)
        result = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(result['status'], 'AVAILABLE')
        support = result_support(result)
        np.testing.assert_array_equal(support['intervals'], estimate['modes'][0]['support']['intervals'])
        self.assertEqual(support['multiview_trace']['view_contributions'][1]['constraints'], 0)
        self.assertEqual(support['multiview_trace']['view_contributions'][1]['observed_depth_profile']['reliable_lower_depth_pixels'], 0)
        self.assertEqual(estimate, before_estimate)
        for got, expected in zip(packets, before_packets):
            for key in ('depth', 'K', 'T'):
                np.testing.assert_array_equal(got[key], expected[key])

    def test_open_interval_endpoints_and_multiple_components_are_not_repaired(self):
        self.assertEqual(m.subtract_open_forbidden([[0., 2.]], [[0., 1.], [1., 2.]]),
                         [[0., 0.], [1., 1.], [2., 2.]])
        packets, estimate = fixture((-.03, .03))
        packets[0]['depth'][2, 2] = 1.2
        result = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(len(result_support(result)['feasible_components'][0]), 2)
        self.assertIsNone(result['centre'])

    def test_known_face_padding_counts_and_multiple_nuisance_modes_remain_explicit(self):
        packets, estimate = fixture()
        # A narrow fitted known-axis interval cannot reduce its fixed allowance.
        estimate['modes'][0]['support']['intervals'][1] = [-.0001, .0001]
        original = deepcopy(estimate)
        result = m.refine_multiview_oriented_support(packets, estimate)
        support = result_support(result)
        self.assertEqual(support['radius_each'][1], .00025)
        self.assertEqual(support['radius_each'][2], .00025)
        self.assertEqual(support['axis_counts'], [0, 3, 3])
        self.assertEqual(support['measured_points'], 17)
        self.assertEqual(result['original_positive_point_count'], 17)
        self.assertEqual(estimate, original)
        estimate['modes'].append(deepcopy(estimate['modes'][0]))
        ambiguous = m.refine_multiview_oriented_support(packets, estimate)
        self.assertEqual(ambiguous['status'], 'UNKNOWN')
        self.assertIsNone(ambiguous['centre'])
        self.assertIn('no_unique_current_nuisance_frame', ambiguous['reasons'])
        self.assertTrue(all(mode['ray_support']['support_available'] for mode in ambiguous['modes']))


if __name__ == '__main__':
    unittest.main()
