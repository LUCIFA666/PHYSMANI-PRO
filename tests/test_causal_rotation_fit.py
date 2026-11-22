import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.multitask.causal_rotation_fit import fit_rotation_history, forecast_rotation


class CausalRotationFitTests(unittest.TestCase):
    def history(self, omega):
        times = np.arange(20) * .05
        base = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
        matrices = Rotation.from_rotvec(times[:, None] * omega).as_matrix() @ base
        return dict(times=times, rotations=matrices, valid=np.ones(20, bool),
            orientation_observable=np.ones(20, bool), modes=np.zeros(20, int),
            confidence=np.ones(20), cutoff=times[-1])

    def test_exact_arbitrary_axis_motion_and_forecast(self):
        w = np.array([.31, -.5, .7])
        data = self.history(w)
        fit = fit_rotation_history(**data)
        np.testing.assert_allclose(fit.angular_velocity_world, w, atol=1e-12)
        expected = Rotation.from_rotvec(w * .7).as_matrix() @ data['rotations'][-1]
        np.testing.assert_allclose(forecast_rotation(fit, .7), expected, atol=1e-12)

    def test_public_noise_reduction_in_stationary_history(self):
        data = self.history(np.zeros(3))
        noise = np.random.default_rng(1018).normal(0, .01, (20, 3))
        data['rotations'] = Rotation.from_rotvec(noise).as_matrix() @ data['rotations']
        fit = fit_rotation_history(**data)
        raw = Rotation.from_matrix(data['rotations'][-1] @ self.history(np.zeros(3))['rotations'][-1].T).magnitude()
        smoothed = Rotation.from_matrix(fit.smoothed_current_rotation @ self.history(np.zeros(3))['rotations'][-1].T).magnitude()
        self.assertLess(smoothed, raw)

    def test_future_timestamp_cannot_be_read_as_causal(self):
        data = self.history(np.array([0, 0, .3]))
        data['cutoff'] -= .05
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            fit_rotation_history(**data)

    def test_mode_boundary_rejects_insufficient_new_segment(self):
        data = self.history(np.array([0, 0, .3]))
        data['modes'][-4:] = 2
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            fit_rotation_history(**data)

    def test_gap_and_missing_rotation_are_not_bridged(self):
        for name in ('gap', 'orientation'):
            data = self.history(np.array([0, 0, .3]))
            if name == 'gap':
                data['times'][-3:] += .1
                data['cutoff'] += .1
            else:
                data['orientation_observable'][-4] = False
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                fit_rotation_history(**data)

    def test_native_labels_are_not_an_accepted_API_argument(self):
        data = self.history(np.array([0, 0, .3]))
        with self.assertRaises(TypeError):
            fit_rotation_history(**data, native_target_rotations=data['rotations'])

    def test_unknown_mode_is_rejected(self):
        data = self.history(np.array([0, 0, .3]))
        data['modes'][-1] = 99
        with self.assertRaises(ValueError):
            fit_rotation_history(**data)


if __name__ == '__main__':
    unittest.main()
