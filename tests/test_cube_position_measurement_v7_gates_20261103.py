"""Acceptance boundaries for the independent current-position certificate."""
import copy
import unittest
from our_mani.cube_position_measurement_20261103_v7.evaluation import position_gate


class PositionGateTests(unittest.TestCase):
    def setUp(self):
        error = dict(count=18, p95_m=.005, max_m=.010)
        self.positions = dict(samples=32, oracle_geometry_eligible=20,
            oracle_eligible_coverage=.90, oracle_eligible_refiner_available=18,
            qualified_visible_centre_error=error,
            all_available_centre_error=copy.deepcopy(error),
            negatives={g: dict(samples=n, falseavailable=0)
                       for g, n in [('sphere_only', 8), ('cube_far', 4)]},
            owner_unobserved_falseavailable=0)
        self.support = dict(TP=995, FP=3, U=2)

    def gate(self, **kwargs):
        return position_gate(kwargs.get('positions', self.positions),
            kwargs.get('points', self.support), kwargs.get('stencils', self.support))

    def test_boundaries_pass_only_position_qualification(self):
        result = self.gate()
        self.assertTrue(result['position_measurement_qualified'])
        self.assertFalse(result['raw_mask_model_qualified'])
        self.assertFalse(result['grasp_qualified'])
        self.assertFalse(result['Native_control_qualified'])

    def test_unknowns_remain_in_used_support_denominator(self):
        for key in ('points', 'stencils'):
            with self.subTest(key=key):
                self.assertFalse(self.gate(**{key: dict(TP=995, FP=0, U=6)})['position_measurement_qualified'])

    def test_position_error_and_coverage_failures(self):
        for field, value in [('p95_m', .005001), ('max_m', .010001)]:
            for key in ('qualified_visible_centre_error', 'all_available_centre_error'):
                p = copy.deepcopy(self.positions); p[key][field] = value
                self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])
        p = copy.deepcopy(self.positions); p['oracle_eligible_coverage'] = .89999
        self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])

    def test_falseavailable_and_filtered_cohort_fail(self):
        for group in ('sphere_only', 'cube_far'):
            p = copy.deepcopy(self.positions); p['negatives'][group]['falseavailable'] = 1
            self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])
        p = copy.deepcopy(self.positions); p['owner_unobserved_falseavailable'] = 1
        self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])
        p = copy.deepcopy(self.positions); p['samples'] = 31
        self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])

    def test_empty_or_nonfinite_cannot_qualify(self):
        self.assertFalse(self.gate(points=dict(TP=0, FP=0, U=0))['position_measurement_qualified'])
        p = copy.deepcopy(self.positions); p['all_available_centre_error']['max_m'] = float('nan')
        self.assertFalse(self.gate(positions=p)['position_measurement_qualified'])


if __name__ == '__main__':
    unittest.main()
