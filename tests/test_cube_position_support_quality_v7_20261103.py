"""Synthetic private-label scoring contracts only; no model/calibration data."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.cube_position_measurement_20261103_v7.support_quality import (
    score_current_support, extract_current_support_masks, GEOMETRY_VERSION,
)


def fixture():
    crops = dict(valid_crop=np.ones((4, 64, 64), bool),
        valid_depth=np.ones((4, 64, 64), bool), origins_xy=np.zeros((4, 2), np.int64))
    ledger = [dict(camera=c, baseline_candidate_points=0, accepted_points=0,
                   pixels_uv=[], accepted=[]) for c in range(4)]
    ledger[0].update(baseline_candidate_points=2, accepted_points=2,
                     pixels_uv=[[10, 10], [11, 10]], accepted=[True, True])
    components = [[[-.01, .01]] for _ in range(3)]
    measurement = dict(geometry_profile=GEOMETRY_VERSION, camera_point_ledger=ledger,
        measured_points=2, visible_views=1, refiner_available=False, centre=None,
        diagnostic_centre_only=[0., 0., 0.], bounded_support={'intervals': [[-.01, .01]]*3},
        free_space_receipt=dict(free_space_applied=False, feasible_components=components, free_space_trace=[]))
    semantic = np.zeros((4, 64, 64), np.uint8)
    semantic[0, 10, 10] = 1; semantic[0, 10, 11] = 255
    sample = dict(semantic=semantic, crops=crops, cube_center_world=np.zeros(3))
    return sample, measurement


class SupportQualityV7(unittest.TestCase):
    def test_unique_stencil_union_and_UNKNOWN_worst_case_without_relabel(self):
        sample, measurement = fixture()
        scored = score_current_support(sample, measurement)
        self.assertEqual(scored['point_pixels']['TP'], 1)
        self.assertEqual(scored['point_pixels']['U'], 1)
        self.assertEqual(scored['point_pixels']['FP'], 0)
        self.assertEqual(scored['point_pixels']['conservative_precision_lowerbound'], .5)
        self.assertEqual(scored['counts']['used_stencil_union'], 12)
        self.assertEqual(scored['counts']['total_stencil_occurrences'], 18)
        self.assertFalse(scored['geometry_available'])
        self.assertTrue(scored['position_qualification_not_granted_by_helper'])

    def test_outside_domain_invalid_depth_duplicate_support_failclosed(self):
        for mode in ('domain', 'depth', 'duplicate', 'edge'):
            sample, measurement = fixture()
            if mode == 'domain': sample['crops']['valid_crop'][0, 9, 9] = False
            if mode == 'depth': sample['crops']['valid_depth'][0, 9, 9] = False
            if mode == 'duplicate': measurement['camera_point_ledger'][0]['pixels_uv'] = [[10, 10], [10, 10]]
            if mode == 'edge': measurement['camera_point_ledger'][0]['pixels_uv'][0] = [0, 10]
            with self.assertRaises(ValueError): extract_current_support_masks(sample['crops'], measurement)

    def test_free_space_is_separate_negative_evidence_and_reports_GT_exclusion(self):
        sample, measurement = fixture()
        receipt = measurement['free_space_receipt']; receipt['free_space_applied'] = True
        receipt['feasible_components'][0] = [[.001, .01]]
        receipt['free_space_trace'] = dict(initial_components=[[[-.01, .01]] for _ in range(3)],
            rounds=[dict(axis=0, other_axis=1, mandatory_other_axis_span=[-.01, .01],
                constraints=[dict(forbidden_open=[-.001, .001], observed_depth=2., max_expected_depth=1.,
                                  footprint_min=[0., 0., 0.], footprint_max=[0., 0., 0.])])])
        result = score_current_support(sample, measurement)
        self.assertEqual(result['point_pixels']['predicted_positive_count'], 2)
        self.assertTrue(result['free_space_posthoc']['GT_excluded_by_free_space_any'])
        self.assertEqual(result['free_space_posthoc']['GT_strictly_in_forbidden_records'], 1)
        self.assertEqual(result['free_space_posthoc']['current_pixel_proof_invalid_records'], 0)

    def test_missing_or_nonfinite_GT_is_not_dropped(self):
        sample, measurement = fixture(); sample['cube_center_world'][0] = np.nan
        with self.assertRaises(ValueError): score_current_support(sample, measurement)


if __name__ == '__main__': unittest.main()
