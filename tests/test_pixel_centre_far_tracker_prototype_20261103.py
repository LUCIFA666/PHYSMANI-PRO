"""Synthetic geometry/schema tests; no model, Native or qualification."""
from copy import deepcopy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.local_mpc_control_20261103 import pixel_centre_far_tracker_prototype_20261103 as p
from our_mani.local_mpc_control_20261103.goal_aware_perception import VisualGoalSphere
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors, CHECKPOINT_SHA256
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row


def fixture():
    sphere = VisualGoalSphere([1., 1., 1.], .03, 0., .001, .004, [.0, .0, .015])
    K = np.array([[100., 0., 4.], [0., 100., 4.], [0., 0., 1.]])
    c = CameraCalibration(K, Pose([0., 0., 0.]))
    obs = {'timestamp': 0., 'observation/images': {}, 'observation/depths': {}, 'observation/calibrations': {}}
    for name in ('front', '8', '16', '36'):
        image = np.zeros((8, 8, 3), np.uint8); image[..., 0] = 220
        obs['observation/images'][name] = image
        obs['observation/depths'][name] = np.full((8, 8), .8)
        obs['observation/calibrations'][name] = c
    return sphere, obs


class PixelCentreFarPrototype(unittest.TestCase):
    def test_only_ray_offset_signed_K_proper_world_transform_and_invalid_pixels(self):
        K = np.array([[-101., 0., 3.], [0., 92., 2.], [0., 0., 1.]])
        pose = Pose([.2, -.1, .4], Rotation.from_rotvec([.3, -.2, .1]).as_matrix())
        old = CameraCalibration(K, pose, 2.)
        new = p.PixelCentreFarCalibration(K, pose, 2.)
        z = np.array([[1., np.nan, 0.], [2., 3., -1.]])
        mask = np.ones(z.shape, bool); y, x = np.nonzero(np.isfinite(z)&(z > 0))
        expected = pose.apply((np.c_[x+.5, y+.5, np.ones(len(x))]@np.linalg.inv(K).T)*(2*z[y, x, None]))
        np.testing.assert_array_equal(new.backproject(z, mask), expected)
        old_points = old.backproject(z, mask)
        expected_delta = np.tile(np.array([.5, .5, 0.])@np.linalg.inv(K).T, (len(x), 1))*(2*z[y, x, None])
        np.testing.assert_allclose(new.backproject(z, mask)-old_points, expected_delta@pose.rotation.T, atol=1e-15, rtol=0)

    def test_original_R100_function_reused_and_inputs_immutable(self):
        sphere, obs = fixture(); before = deepcopy(obs)
        tracker = p.PixelCentreFarTrackerPrototype(sphere)
        target, summary = tracker.observe(obs)
        self.assertIsNotNone(target)
        self.assertIs(tracker._raw_tracker.observe.__func__, RedBoxTracker.observe)
        self.assertIs(tracker.sphere, sphere)
        for name in obs['observation/images']:
            np.testing.assert_array_equal(obs['observation/images'][name], before['observation/images'][name])
            np.testing.assert_array_equal(obs['observation/depths'][name], before['observation/depths'][name])
            self.assertIs(type(obs['observation/calibrations'][name]), CameraCalibration)
            private = tracker.last_filtered_observation['observation/calibrations'][name]
            self.assertIs(type(private), p.PixelCentreFarCalibration)
        self.assertEqual(len(tracker.last_measured_cloud), 256)
        self.assertEqual(len(target.points_world), 256)
        self.assertFalse(summary['template_is_measured_cloud'])

    def test_shell_filter_and_tracker_backprojection_share_exact_half_calibration(self):
        sphere, obs = fixture(); tracker = p.PixelCentreFarTrackerPrototype(sphere)
        target, _ = tracker.observe(obs)
        filtered = tracker.last_filtered_observation
        # A direct original R100 using the very same private observation must
        # reproduce every point and centre, not only counts or extrema.
        raw = RedBoxTracker(prior=tracker.prior, complete_geometry=False)
        state, _ = raw.observe(filtered)
        np.testing.assert_array_equal(state.points_world, tracker.last_measured_cloud)
        np.testing.assert_array_equal(state.pose.position, target.pose.position)

    def test_all_unfavorable_points_retained_and_unknown_clears_only_own_history(self):
        sphere, obs = fixture(); tracker = p.PixelCentreFarTrackerPrototype(sphere)
        first, _ = tracker.observe(obs)
        self.assertIsNotNone(first)
        obs['timestamp'] = .05
        for image in obs['observation/images'].values(): image[:] = 0
        missing, summary = tracker.observe(obs)
        self.assertIsNone(missing); self.assertFalse(summary['current_measurement_available'])
        self.assertEqual(len(tracker.history), 0)
        obs['timestamp'] = .1
        for image in obs['observation/images'].values(): image[..., 0] = 220
        current, _ = tracker.observe(obs)
        self.assertIsNotNone(current); self.assertEqual([t for t, _ in tracker.history], [.1])
        np.testing.assert_array_equal(current.linear_velocity, np.zeros(3))

    def test_public_whitelist_pack_is_schema_only_and_all_certification_false(self):
        sphere, obs = fixture(); tracker = p.PixelCentreFarTrackerPrototype(sphere)
        target, summary = tracker.observe(obs)
        for key in ('qualified', 'position_measurement_qualified', 'raw_mask_model_qualified',
                    'semantic_ownership_verified', 'physical_or_statistical_certificate', 'live_consumer_admission'):
            self.assertIs(summary[key], False)
        bad = dict(summary, private_component=tracker.last_private)
        with self.assertRaises(ValueError): p.project_public_far_evidence(bad)
        robot = PublicRobotSensors(0., [.1, .1, 1.], [0., 0., 0., 1.], np.zeros(7), np.zeros(7), [.5, .5])
        result = dict(timestamp=0., mode='UNRESOLVED', available=False, free_forecast_available=False,
            controlled_mode_available=False, frozen_checkpoint_sha256=CHECKPOINT_SHA256,
            Native_inputs_used=False, future_observations_used=False, observations_fabricated=False,
            forecaster_is_injected_test_double=False, object_observation=target, forecast=None,
            far_pixel_centre_research=summary)
        packed = pack_public_row(result, robot, False, 0, 'rgbd/0000.npz')
        self.assertFalse(packed['result']['available'])
        self.assertEqual(packed['result']['mode'], 'UNRESOLVED')
        self.assertNotIn('far_pixel_centre_research', packed['result'])
        np.testing.assert_array_equal(packed['result']['object_observation']['pose']['position'], target.pose.position)

    def test_duplicate_and_gap_do_not_fabricate_intermediate_history(self):
        sphere, obs = fixture(); tracker = p.PixelCentreFarTrackerPrototype(sphere)
        tracker.observe(obs)
        with self.assertRaises(ValueError): tracker.observe(obs)
        obs['timestamp'] = .2
        target, summary = tracker.observe(obs)
        self.assertIsNone(target); self.assertEqual(len(tracker.history), 0)
        self.assertEqual(summary['reason'], 'non50ms_or_stale_actual_frame_no_interpolation')


if __name__ == '__main__': unittest.main()
