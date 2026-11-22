import inspect
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.depth_pose_consistency import FreeSpaceConfig, current_depth_pose_cost


class CurrentDepthPoseCostTests(unittest.TestCase):
    def setUp(self):
        self.pose = Pose(np.zeros(3))
        self.calibration = CameraCalibration(np.array([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]]), self.pose)
        self.point = {'handle': np.array([[0., 0., 1.]])}

    def evaluate(self, depth, points=None, pose=None, **kwargs):
        return current_depth_pose_cost(self.point if points is None else points,
            self.pose if pose is None else pose, {'front': (self.calibration, depth)}, **kwargs)

    def test_front_surface_contradicts_observed_free_space_and_is_bounded(self):
        result = self.evaluate(np.full((11, 11), 2.))
        self.assertAlmostEqual(result['cost_m2'], 2. * .03 ** 2)
        self.assertEqual(result['counts']['contradictory_pixels'], 1)
        self.assertEqual(result['counts']['clipped_contradictory_pixels'], 1)

    def test_true_foreground_occlusion_is_never_a_penalty(self):
        result = self.evaluate(np.full((11, 11), .6))
        self.assertEqual(result['cost_m2'], 0.)
        self.assertEqual(result['counts']['occluded_pixels'], 1)
        self.assertTrue(result['evidence_available'])

    def test_tolerance_and_metric_depth_scale(self):
        self.calibration = CameraCalibration(self.calibration.intrinsics, self.pose, depth_scale=.001)
        self.assertEqual(self.evaluate(np.full((11, 11), 1003.))['cost_m2'], 0.)
        self.assertAlmostEqual(self.evaluate(np.full((11, 11), 1010.))['cost_m2'], 2. * .006 ** 2)

    def test_invalid_depth_and_outside_frustum_are_not_evidence(self):
        for value in (0., -1., np.nan, np.inf):
            result = self.evaluate(np.full((11, 11), value))
            self.assertEqual(result['cost_m2'], 0.)
            self.assertEqual(result['counts']['invalid_depth_pixels'], 1)
            self.assertFalse(result['evidence_available'])
        result = self.evaluate(np.ones((11, 11)), {'outside': np.array([[100., 0., 1.], [0., 0., -1.]])})
        self.assertEqual(result['cost_m2'], 0.)
        self.assertEqual(result['counts']['outside_frustum_samples'], 1)
        self.assertEqual(result['counts']['behind_camera_samples'], 1)

    def test_one_frontmost_surface_per_pixel_not_density_or_piece_count(self):
        points = {'body': np.tile([0., 0., 1.], (100, 1)), 'back': np.array([[0., 0., 1.5]])}
        result = self.evaluate(np.full((11, 11), 1.), points)
        self.assertEqual(result['counts']['projected_samples'], 101)
        self.assertEqual(result['counts']['deduplicated_model_pixels'], 1)
        self.assertEqual(result['cost_m2'], 0.)

    def test_current_asymmetric_handle_can_reject_wrong_orientation(self):
        # Current depth only: one cylindrical-body ray and one handle ray.
        # The wrong pi rotation moves the handle into observed free space.
        points = {'cup': np.array([[0., 0., 1.]]), 'handle': np.array([[.2, 0., 1.]])}
        depth = np.full((11, 11), 2.)
        depth[5, 5] = depth[5, 7] = 1.
        correct = self.evaluate(depth, points)
        wrong = self.evaluate(depth, points, Pose(np.zeros(3), Rotation.from_euler('z', np.pi).as_matrix()))
        self.assertEqual(correct['cost_m2'], 0.)
        self.assertGreater(wrong['cost_m2'], 0.)
        # If an actual foreground occluder hides the wrong handle ray, this
        # one-sided evidence cannot resolve it and must not claim otherwise.
        depth[5, 3] = .8
        hidden = self.evaluate(depth, points, Pose(np.zeros(3), Rotation.from_euler('z', np.pi).as_matrix()))
        self.assertEqual(hidden['cost_m2'], 0.)
        self.assertFalse(hidden['full_pose_observability_proven'])

    def test_future_depth_explicitly_rejected_and_no_future_or_GT_argument(self):
        with self.assertRaisesRegex(ValueError, 'future or stale'):
            self.evaluate(np.ones((11, 11)), reference_time=1., view_timestamps={'front': 1.05})
        with self.assertRaisesRegex(ValueError, 'future or stale'):
            self.evaluate(np.ones((11, 11)), reference_time=1., view_timestamps={'front': .9})
        result = self.evaluate(np.ones((11, 11)), reference_time=1., view_timestamps={'front': 1.})
        self.assertEqual(result['cost_m2'], 0.)
        self.assertNotIn('native_pose', inspect.signature(current_depth_pose_cost).parameters)
        with self.assertRaises(TypeError):
            self.evaluate(np.ones((11, 11)), future_poses=np.ones((20, 7)))

    def test_multiview_cost_mean_and_signed_calibration(self):
        negative = CameraCalibration(np.diag([-10., -10., 1.]) +
            np.array([[0., 0., 5.], [0., 0., 5.], [0., 0., 0.]]), self.pose)
        result = current_depth_pose_cost(self.point, self.pose,
            {'front': (self.calibration, np.full((11, 11), 1.)),
             'second': (negative, np.full((11, 11), 1.014), np.zeros((11, 11)))})
        self.assertAlmostEqual(result['cost_m2'], 2. * .01 ** 2 / 2.)
        self.assertEqual(result['counts']['evaluated_pixels'], 2)


if __name__ == '__main__':
    unittest.main()
