"""Independent current-public V8 routing fixtures; no NN, labels or fitting."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.cube_position_measurement_20261103_v8 import prediction as p


def public_inputs():
    return dict(rgb=np.zeros((4, 256, 256, 3), np.uint8),
                depth_m=np.ones((4, 256, 256), np.float32),
                K=np.repeat(np.array([[-100., 0, 128], [0, -100., 128], [0, 0, 1.]])[None], 4, 0),
                T=np.repeat(np.eye(4)[None], 4, 0), public_sphere_center=np.array([0., 0., 1.]),
                camera_names=np.array(['front', '8', '16', '36']), timestamp=np.array(0.))


def probabilities():
    result = np.empty((4, 2, 64, 64), np.float32)
    result[:, 1] = .95; result[:, 0] = 1 - result[:, 1]
    result[:, 1, 20:30, 20:30] = .999; result[:, 0, 20:30, 20:30] = 1 - result[:, 1, 20:30, 20:30]
    return result


def measurement(available, centre=None):
    return dict(refiner_available=available, geometry_profile=p.GEOMETRY_VERSION,
                centre=([.01, .02, 1.] if available and centre is None else centre),
                reasons=[] if available else ['current_support_unknown'], measured_points=24 if available else 0,
                visible_views=3 if available else 0, geometry_confidence=1. if available else 0.)


class AdaptivePublicV8Tests(unittest.TestCase):
    def setUp(self):
        self.prob = probabilities()
        self.crops = {'valid_crop': np.ones((4, 64, 64), bool)}
        self.crops['valid_crop'][:, :2] = False

    def test_primary_available_shortcircuits_and_preserves_exact_F_measurement(self):
        inputs = public_inputs(); before = deepcopy(inputs); primary = measurement(True)
        with patch.object(p, 'measure_current_cube', return_value=primary) as geometry:
            result = p.predict_current_position(inputs, self.crops, self.prob)
        geometry.assert_called_once()
        np.testing.assert_array_equal(geometry.call_args.args[2], (self.prob[:, 1] >= .995) & self.crops['valid_crop'])
        self.assertIs(result['measurement'], primary)
        self.assertEqual(result['thresholds_tried'], [.995])
        self.assertEqual((result['selected_threshold'], result['proposal_route']), (.995, 'first_0p995'))
        self.assertFalse(result['raw_mask_qualified']); self.assertFalse(result['completion'])
        for key in inputs: np.testing.assert_array_equal(inputs[key], before[key])

    def test_fallback_only_after_unavailable_primary_and_fixed_domain(self):
        primary, fallback = measurement(False), measurement(True, [.02, -.01, 1.])
        with patch.object(p, 'measure_current_cube', side_effect=[primary, fallback]) as geometry:
            result = p.predict_current_position(public_inputs(), self.crops, self.prob)
        self.assertEqual(geometry.call_count, 2)
        for call, cutoff in zip(geometry.call_args_list, (.995, .9)):
            np.testing.assert_array_equal(call.args[2], (self.prob[:, 1] >= cutoff) & self.crops['valid_crop'])
        self.assertIs(result['measurement'], fallback)
        self.assertEqual(result['thresholds_tried'], [.995, .9])
        self.assertEqual(result['proposal_route'], 'fallback_0p9')
        self.assertEqual(result['selected_threshold'], .9)
        np.testing.assert_array_equal(result['raw_Cube_mask'], (self.prob[:, 1] >= .9) & self.crops['valid_crop'])
        self.assertFalse(result['route_uses_GT_or_future'])

    def test_both_rejected_stay_unknown_no_old_centre_fallback(self):
        first, second = measurement(False, [99., 99., 99.]), measurement(False, [88., 88., 88.])
        result = p.route_current_public_proposals(self.crops, self.prob, [(.995, first), (.9, second)])
        self.assertIsNone(result['centre']); self.assertIsNone(result['selected_threshold'])
        self.assertEqual(result['threshold'], .9); self.assertEqual(result['proposal_route'], 'UNKNOWN')
        self.assertFalse(result['position_measurement_available'])
        with self.assertRaises(ValueError):
            p.route_current_public_proposals(self.crops, self.prob, [(.995, first)])

    def test_fixed_order_types_and_finite_available_centres(self):
        for proposals in (
            [], [(.9, measurement(True))], [(.995, measurement(False)), (.999, measurement(True))],
            [(np.float32(.995), measurement(True))],
            [(.995, dict(measurement(True), refiner_available=1))],
            [(.995, dict(measurement(True), geometry_profile='legacy'))],
            [(.995, measurement(True, [0., float('nan'), 1.]))],
        ):
            with self.assertRaises(ValueError):
                p.route_current_public_proposals(self.crops, self.prob, proposals)

    def test_invalid_probabilities_or_crop_domain_never_reach_geometry(self):
        bad = []
        bad.append(self.prob.astype(np.float64)); bad.append(self.prob[:, :, :-1])
        value = self.prob.copy(); value[0, 0, 0, 0] = float('nan'); bad.append(value)
        value = self.prob.copy(); value[:, 0] = 0.; bad.append(value)
        value = self.prob.copy(); value[0, :, 0, 0] = [-.1, 1.1]; bad.append(value)
        with patch.object(p, 'measure_current_cube', side_effect=AssertionError('no geometry')) as geometry:
            for value in bad:
                with self.assertRaises(ValueError): p.predict_current_position(public_inputs(), self.crops, value)
            geometry.assert_not_called()
        for domain in (np.ones((4, 64, 64), np.uint8), np.ones((4, 63, 64), bool)):
            with self.assertRaises(ValueError):
                p.route_current_public_proposals({'valid_crop': domain}, self.prob, [(.995, measurement(True))])

    def test_GT_robot_and_future_inputs_rejected_before_geometry(self):
        self.assertEqual(list(inspect.signature(p.predict_current_position).parameters), ['inputs', 'crops', 'probabilities'])
        with patch.object(p, 'measure_current_cube', side_effect=AssertionError('no privileged geometry')) as geometry:
            for key in ('cube_center_world', 'semantic_role_ids', 'Native_state', 'q', 'future_observation'):
                inputs = public_inputs(); inputs[key] = np.zeros(3)
                with self.assertRaises(ValueError): p.predict_current_position(inputs, self.crops, self.prob)
            geometry.assert_not_called()


if __name__ == '__main__': unittest.main()
