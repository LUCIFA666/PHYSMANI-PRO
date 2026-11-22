import copy
import unittest
from our_mani.cube_position_measurement_20261103_v8.evaluation import (
    position_gate, validate_prediction_route,
)
from our_mani.cube_position_measurement_20261103_v8.prediction import VERSION


class AdaptiveEvaluationTests(unittest.TestCase):
    def prediction(self, route):
        primary = route == 'first_0p995'
        available = route != 'UNKNOWN'
        tried = [.995] if primary else [.995, .9]
        return dict(adaptive_public_proposal_version=VERSION, proposal_thresholds=[.995, .9],
            thresholds_tried=tried, proposal_route=route, route_uses_GT_or_future=False,
            measurement=dict(refiner_available=available), position_measurement_available=available,
            threshold=tried[-1], selected_threshold=tried[-1] if available else None,
            centre=[0., 0., 0.] if available else None,
            proposal_trace=[dict(threshold=t, geometry_available=available if i == len(tried)-1 else False)
                            for i, t in enumerate(tried)])

    def test_three_public_routes(self):
        for route in ['first_0p995', 'fallback_0p9', 'UNKNOWN']:
            validate_prediction_route(self.prediction(route))

    def test_no_oracle_route_or_unknown_centre(self):
        p = self.prediction('fallback_0p9'); p['proposal_trace'][0]['geometry_available'] = True
        with self.assertRaises(ValueError): validate_prediction_route(p)
        p = self.prediction('UNKNOWN'); p['centre'] = [1., 2., 3.]
        with self.assertRaises(ValueError): validate_prediction_route(p)
        p = self.prediction('first_0p995'); p['route_uses_GT_or_future'] = True
        with self.assertRaises(ValueError): validate_prediction_route(p)

    def test_unmodified_gates_and_new64_count(self):
        error = dict(count=36, p95_m=.005, max_m=.01)
        pos = dict(samples=64, oracle_geometry_eligible=40, oracle_eligible_coverage=.9,
            qualified_visible_centre_error=error, all_available_centre_error=copy.deepcopy(error),
            negatives={g:dict(samples=n,falseavailable=0) for g,n in [('sphere_only',16),('cube_far',8)]},
            owner_unobserved_falseavailable=0)
        support = dict(TP=995, FP=3, U=2)
        decision = position_gate(pos, support, support)
        self.assertTrue(decision['position_measurement_qualified'])
        self.assertFalse(decision['raw_mask_model_qualified'])
        self.assertEqual(decision['proposal_thresholds'], [.995, .9])
        self.assertNotIn('fixed_threshold', decision)
        pos['samples'] = 32
        self.assertFalse(position_gate(pos,support,support)['position_measurement_qualified'])


if __name__ == '__main__': unittest.main()
