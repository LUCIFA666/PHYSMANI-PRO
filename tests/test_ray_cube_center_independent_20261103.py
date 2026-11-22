"""Independent analytic depth/ray counterexamples, no recorded labels or fit."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.local_mpc_control_20261103.ray_cube_center_prototype_20261103 import (
    CurrentRayBundle, RayCubeCenterEstimator, cube_sdf, ray_box_first_hit)


class IndependentRayCubeTests(unittest.TestCase):
    def test_non_rotation_matrices_cannot_change_cube_geometry_silently(self):
        for rotation in (np.zeros((3,3)),np.diag([1.,1.,-1.]),2*np.eye(3),
                         np.array([[1.,.1,0.],[0.,1.,0.],[0.,0.,1.]])):
            with self.subTest(rotation=rotation.tolist()):
                with self.assertRaises(ValueError):
                    cube_sdf([[0.,0.,1.]],[0.,0.,1.025],rotation)
                with self.assertRaises(ValueError):
                    ray_box_first_hit([[0.,0.,0.]],[[0.,0.,1.]],[0.,0.,1.025],rotation)

    def test_numpy_boolean_is_not_a_real_capture_timestamp(self):
        for timestamp in (True,False,np.bool_(True),np.bool_(False)):
            with self.subTest(timestamp=repr(timestamp)),self.assertRaises(ValueError):
                CurrentRayBundle(timestamp,[[0.,0.,0.]],[[0.,0.,1.]],[1.],[0],np.ones(1,bool))

    def test_point_mirror_has_opposite_solution_when_camera_is_behind(self):
        observed = np.array([[0.,0.,1.]])
        centres = [[0.,0.,.975],[0.,0.,1.025]]
        for c in centres:
            np.testing.assert_allclose(cube_sdf(observed,c,np.eye(3)),0.,atol=1e-14)
        for origin,direction,correct,wrong in [([0.,0.,0.],[0.,0.,1.],1,0),
                                               ([0.,0.,2.],[0.,0.,-1.],0,1)]:
            entry = [ray_box_first_hit([origin],[direction],c,np.eye(3))[0][0] for c in centres]
            self.assertAlmostEqual(entry[correct],1.)
            self.assertAlmostEqual(entry[wrong],.95)

    def test_signed_K_camera_z_is_preserved_under_world_rigid_transform(self):
        K = np.array([[-100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        local_ray = np.linalg.solve(K,[100.,80.,1.])
        np.testing.assert_allclose(local_ray,[-.5,.3,1.])
        camera_R = Rotation.from_euler('xyz',[.3,-.2,.6]).as_matrix()
        origin = np.array([.7,-.2,.1]); world_ray = camera_R@local_ray
        centre = origin+camera_R@np.array([-.5,.3,1.025])
        observed = origin+world_ray # exactly1m camera-Z, not1m ray range
        entry,exit,hit = ray_box_first_hit(origin[None],world_ray[None],centre,camera_R)
        self.assertTrue(hit[0]); self.assertAlmostEqual(entry[0],1.)
        np.testing.assert_allclose(cube_sdf(observed[None],centre,camera_R),0.,atol=1e-14)
        normalized_entry,_,_ = ray_box_first_hit(origin[None],world_ray[None]/np.linalg.norm(world_ray),centre,camera_R)
        self.assertAlmostEqual(normalized_entry[0],np.linalg.norm(local_ray))
        self.assertGreater(abs(normalized_entry[0]-entry[0]),.1)

    def test_nearer_measured_occluder_never_claims_background_free_space(self):
        estimator = RayCubeCenterEstimator(); origins=np.zeros((2,3)); rays=np.tile([0.,0.,1.],(2,1))
        def residual(context_depth):
            b=CurrentRayBundle(.5,origins,rays,[1.,context_depth],[0,1],np.array([True,False]))
            return estimator._residual(np.array([0.,0.,1.025,0.,0.,0.]),b)
        occluded = residual(.5); background = residual(1.5)
        # Last2 entries are free-prefix penalties. A real foreground depth.5
        # cannot assert anything about the hypothesized object behind it.
        np.testing.assert_allclose(occluded[-2:],0.,atol=1e-12)
        self.assertGreater(background[-1],100.)
        np.testing.assert_allclose(occluded[:-2],background[:-2],atol=1e-12)

    def test_neighbour_background_ray_rules_out_lateral_box_without_target_guess(self):
        xy=np.array([[x,y,1.] for x in [-.002,0.,.002] for y in [-.002,0.,.002]])
        rays=np.r_[xy,[[.04,0.,1.]]];depth=np.r_[np.ones(9),2.]
        bundle=CurrentRayBundle(.5,np.zeros_like(rays),rays,depth,np.zeros(10),np.r_[np.ones(9,bool),False])
        estimator=RayCubeCenterEstimator()
        correct=estimator._residual(np.array([0.,0.,1.025,0.,0.,0.]),bundle)
        shifted=estimator._residual(np.array([.02,0.,1.025,0.,0.,0.]),bundle)
        # All positive surface points fit both boxes, but the shifted box
        # incorrectly fills the measured background ray's free prefix.
        np.testing.assert_allclose(correct[:18],shifted[:18],atol=1e-10)
        self.assertLess(np.max(np.abs(correct[-10:])),1e-10)
        self.assertGreater(shifted[-1],100.)

    def test_small_collinear_multi_view_patch_does_not_certify_initializer(self):
        xy=np.array([[x,y,1.] for x in np.linspace(-.002,.002,5) for y in np.linspace(-.002,.002,5)])
        origins=[];rays=[];depth=[];views=[]
        for i,z in enumerate([0.,.1]):
            o=np.tile([0.,0.,z],(len(xy),1));d=np.full(len(xy),1.-z)
            origins.append(o);rays.append((xy-o)/d[:,None]);depth.append(d);views.append(np.full(len(xy),i))
        bundle=CurrentRayBundle(.5,np.concatenate(origins),np.concatenate(rays),np.concatenate(depth),
            np.concatenate(views),np.ones(50,bool))
        estimator=RayCubeCenterEstimator()
        for c in [0.,.02,-.02]:
            residual=estimator._residual(np.array([c,0.,1.025,0.,0.,0.]),bundle)
            self.assertLess(np.max(np.abs(residual)),1e-10)
        result=estimator.fit_bundle(bundle,[.015,0.,1.025])
        self.assertFalse(result['available']);self.assertIsNone(result['centre'])
        self.assertFalse(result['statistical_uncertainty_calibrated'])
        self.assertTrue({'near_optimal_mirror_or_centre_ambiguity','centre_unobservable_with_rotation_nuisance'}&set(result['reasons']))

    def test_miss_and_behind_camera_are_explicit_not_finite_clipped_hits(self):
        for origin,ray,centre in [([.1,0.,0.],[0.,0.,1.],[0.,0.,1.]),
                                  ([0.,0.,0.],[0.,0.,1.],[0.,0.,-1.])]:
            entry,exit,hit=ray_box_first_hit([origin],[ray],centre,np.eye(3))
            self.assertFalse(hit[0]);self.assertTrue(np.isposinf(entry[0]));self.assertTrue(np.isneginf(exit[0]))
        bundle=CurrentRayBundle(.5,[[0.,0.,0.]],[[0.,0.,1.]],[1.],[0],np.ones(1,bool))
        residual=RayCubeCenterEstimator()._residual(np.array([.4,0.,1.,0.,0.,0.]),bundle)
        self.assertTrue(np.isfinite(residual).all());self.assertGreater(np.linalg.norm(residual),1.)

    def test_current_bundle_copies_measurements_and_rejects_missing_depth_or_model_channels(self):
        o=np.zeros((2,3));r=np.tile([0.,0.,1.],(2,1));d=np.ones(2)
        b=CurrentRayBundle(.5,o,r,d,[0,1],np.ones(2,bool));o[0]=9.;d[0]=9.
        np.testing.assert_array_equal(b.points,[[0.,0.,1.],[0.,0.,1.]])
        with self.assertRaises(ValueError):b.depth[0]=2.
        with self.assertRaises(ValueError):CurrentRayBundle(.5,o,r,[0.,1.],[0,1],np.ones(2,bool))
        with self.assertRaises(TypeError):RayCubeCenterEstimator().fit_bundle(b,[0.,0.,1.],GRU_center=[0.,0.,1.])
        with self.assertRaises(TypeError):RayCubeCenterEstimator().fit_bundle(b,[0.,0.,1.],TCP=[0.,0.,1.])


if __name__ == '__main__': unittest.main()
