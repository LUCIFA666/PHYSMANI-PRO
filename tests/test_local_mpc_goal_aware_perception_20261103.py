import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.benchmark_perception import BoxPrior,RedBoxTracker
from our_mani.local_mpc_control_20261103.goal_aware_perception import (
    VisualGoalSphere,GoalAwareRedBoxTracker,sphere_from_public_goal_fit,sphere_surface_mask,
    filter_observed_sphere_surface)


def observation(t=0.):
    cameras=('front','8','16','36')
    return {'timestamp':float(t),'observation/images':{c:np.full((4,4,3),[255,0,0],dtype=np.uint8) for c in cameras},
        'observation/depths':{c:np.ones((4,4),dtype=np.float32) for c in cameras},
        'observation/calibrations':{c:CameraCalibration(np.diag([-100.,-100.,1.]),Pose([0,0,0])) for c in cameras}}


def metadata(residual=.0001):
    return dict(backend='rgbd_red_goal_sphere_train_calibrated_frame',sphere_radius_m=.04,
        median_fit_residual_m=residual,marker_to_goal_m=[0,0,.015],simulator_pose_used=False)


class PublicMarker:
    def __init__(self):self.calls=0
    def observe(self,obs):self.calls+=1;return Pose([0,0,1.315]),metadata()


class GoalAwarePerceptionTests(unittest.TestCase):
    def test_visual_sphere_center_subtracts_offset_goal_frame(self):
        sphere=sphere_from_public_goal_fit(Pose([.2,.1,1.015]),metadata(),0.)
        np.testing.assert_allclose(sphere.center,[.2,.1,1.],atol=1e-12)
        self.assertEqual(sphere.surface_tolerance_m,.004)
        self.assertNotAlmostEqual(sphere.center[2],1.015)

    def test_surface_shell_does_not_cut_ball_interior_or_distant_points(self):
        sphere=VisualGoalSphere([0,0,1.],.04,0.,.0001,.004,[0,0,.015])
        points=np.array([[0,0,1.04],[0,0,1.],[0,0,1.01],[0,0,1.10]])
        self.assertEqual(sphere_surface_mask(points,sphere).tolist(),[True,False,False,False])

    def test_filter_private_RGB_copy_preserves_depth_signed_calibration_and_all_sources(self):
        obs=observation();before={c:im.copy() for c,im in obs['observation/images'].items()}
        sphere=VisualGoalSphere([0,0,.96],.04,0.,.0001,.004,[0,0,.015])
        filtered,diagnostic=filter_observed_sphere_surface(obs,sphere,BoxPrior())
        for camera in before:
            np.testing.assert_array_equal(obs['observation/images'][camera],before[camera])
            np.testing.assert_array_equal(filtered['observation/images'][camera][0,0],[0,0,0])
            np.testing.assert_array_equal(filtered['observation/images'][camera][3,3],[255,0,0])
            self.assertIs(filtered['observation/depths'][camera],obs['observation/depths'][camera])
            self.assertIs(filtered['observation/calibrations'][camera],obs['observation/calibrations'][camera])
            self.assertLess(filtered['observation/calibrations'][camera].intrinsics[0,0],0.)
        self.assertGreater(diagnostic['removed_red_pixels_total'],0)
        self.assertFalse(diagnostic['whole_sphere_volume_cut'])
        self.assertTrue(diagnostic['target_surface_overlap_with_sphere_shell_may_remove_target_pixels'])

    def test_uncertain_or_Native_goal_fit_cannot_initialize_filter(self):
        with self.assertRaisesRegex(ValueError,'uncertainty'):
            sphere_from_public_goal_fit(Pose([0,0,1.015]),metadata(.003),0.)
        bad=metadata();bad['simulator_pose_used']=True
        with self.assertRaises(ValueError):sphere_from_public_goal_fit(Pose([0,0,1.015]),bad,0.)
        bad_obs=observation();bad_obs['Native_goal']=[0,0,1.]
        with self.assertRaises(ValueError):GoalAwareRedBoxTracker().initialize_from_public(bad_obs)

    def test_initial_public_fit_cached_once_across_motion_reset_and_refit_only_new_episode(self):
        marker=PublicMarker();tracker=GoalAwareRedBoxTracker(goal_marker=marker)
        # Red point plane at1m is deliberately outside unchanged table prior.
        for t in (0.,.05):
            with self.assertRaisesRegex(ValueError,'association failed'):tracker.observe(observation(t))
        self.assertEqual(marker.calls,1);sphere=tracker.sphere
        tracker.reset()
        with self.assertRaisesRegex(ValueError,'association failed'):tracker.observe(observation(.1))
        self.assertIs(sphere,tracker.sphere);self.assertEqual(marker.calls,1)
        tracker.reset_episode()
        with self.assertRaises(ValueError):tracker.observe(observation(0.))
        self.assertEqual(marker.calls,2)

    def test_old_box_prior_height_gate_and_confidence_are_not_relaxed(self):
        original=RedBoxTracker();aware=GoalAwareRedBoxTracker(goal_marker=PublicMarker())
        self.assertEqual(aware.prior,original.prior)
        with self.assertRaisesRegex(ValueError,'association failed'):original.observe(observation())
        with self.assertRaisesRegex(ValueError,'association failed'):aware.observe(observation())
        self.assertTrue(aware.last_filter_metadata['filter_available'])

    def test_existing_public_frame_cache_reset_hook_accepts_compatible_public_tracker(self):
        from our_mani.local_mpc_control_20261103.public_attachment import PublicTargetFrameCache
        marker=PublicMarker();aware=GoalAwareRedBoxTracker(goal_marker=marker)
        aware.initialize_from_public(observation());sphere=aware.sphere
        aware.history.append((0.,np.zeros(3)));aware.last_velocity=np.ones(3)
        cache=PublicTargetFrameCache(public_tracker=aware);cache.reset()
        self.assertEqual(len(aware.history),0);np.testing.assert_array_equal(aware.last_velocity,np.zeros(3))
        self.assertIs(aware.sphere,sphere);self.assertEqual(marker.calls,1)

    def test_invalid_depth_pixels_never_become_removed_world_points(self):
        obs=observation();obs['observation/depths']['front'][0,0]=np.nan
        sphere=VisualGoalSphere([0,0,.96],.04,0.,.0001,.004,[0,0,.015])
        filtered,diagnostic=filter_observed_sphere_surface(obs,sphere,BoxPrior())
        np.testing.assert_array_equal(filtered['observation/images']['front'][0,0],[255,0,0])
        self.assertEqual(diagnostic['per_camera']['front']['valid_red_depth_pixels'],15)


if __name__=='__main__':unittest.main()
