"""Synthetic current-depth geometry contracts, not real-world accuracy claims."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.local_mpc_control_20261103.ray_cube_center_prototype_20261103 import (
    ray_box_first_hit,cube_sdf,CurrentRayBundle,RayCubeCenterEstimator)


def measured_bundle(centre,rotation,*,views=3,patch=False):
    origins=[];directions=[];depths=[];ids=[];targets=[]
    for index,origin in enumerate(([.0,-.2,.0],[.5,.1,.1],[-.3,.3,.15])[:views]):
        origin=np.array(origin);z=centre-origin;z/=np.linalg.norm(z)
        x=np.cross([0.,1.,0.],z);x/=np.linalg.norm(x);y=np.cross(z,x);camera=np.stack([x,y,z],axis=1)
        axis=np.linspace(-.06,.06,13) if not patch else np.linspace(-.003,.003,7)
        u,v=np.meshgrid(axis,axis);rays=np.c_[u.ravel(),v.ravel(),np.ones(u.size)]@camera.T
        o=np.broadcast_to(origin,rays.shape).copy();entry,_,hit=ray_box_first_hit(o,rays,centre,rotation)
        measured=np.where(hit,entry,2.)
        origins.append(o);directions.append(rays);depths.append(measured);targets.append(hit);ids.append(np.full(len(rays),index))
    return CurrentRayBundle(1.,np.concatenate(origins),np.concatenate(directions),np.concatenate(depths),
        np.concatenate(ids),np.concatenate(targets))


class CubeRayGeometryTests(unittest.TestCase):
    def test_point_surface_mirror_is_removed_by_first_surface_depth(self):
        p=np.array([[0.,0.,1.]])
        front=np.array([0.,0.,.975]);back=np.array([0.,0.,1.025])
        np.testing.assert_allclose(cube_sdf(p,front,np.eye(3)),0.,atol=1e-14)
        np.testing.assert_allclose(cube_sdf(p,back,np.eye(3)),0.,atol=1e-14)
        o=np.zeros((1,3));d=np.array([[0.,0.,1.]])
        a,_,_=ray_box_first_hit(o,d,front,np.eye(3));b,_,_=ray_box_first_hit(o,d,back,np.eye(3))
        self.assertAlmostEqual(a[0],.95);self.assertAlmostEqual(b[0],1.)

    def test_depth_is_camera_z_not_unit_ray_distance_and_parallel_outside_misses(self):
        centre=np.array([.2,0.,1.]);o=np.zeros((1,3));d=np.array([[.2,0.,1.]])
        a,_,hit=ray_box_first_hit(o,d,centre,np.eye(3))
        self.assertTrue(hit[0]);self.assertAlmostEqual(a[0],.975)
        b,_,_=ray_box_first_hit(o,d/np.linalg.norm(d),centre,np.eye(3))
        self.assertAlmostEqual(b[0],a[0]*np.linalg.norm(d))
        _,_,hit=ray_box_first_hit(np.array([[.1,0.,0.]]),np.array([[0.,0.,1.]]),[0.,0.,1.],np.eye(3))
        self.assertFalse(hit[0])

    def test_nearer_occluder_is_not_false_free_space(self):
        o=np.zeros((1,3));d=np.array([[0.,0.,1.]])
        entry,_,hit=ray_box_first_hit(o,d,[0.,0.,1.],np.eye(3))
        self.assertTrue(hit[0])
        self.assertEqual(float(np.maximum(0.,.5-entry[0]-.002)),0.)
        self.assertGreater(float(np.maximum(0.,1.5-entry[0]-.002)),.5)

    def test_one_view_or_missing_depth_cannot_certify_prior(self):
        centre=np.array([.02,-.01,.8]);bundle=measured_bundle(centre,np.eye(3),views=1)
        result=RayCubeCenterEstimator().fit_bundle(bundle,centre)
        self.assertFalse(result['available']);self.assertIsNone(result['centre'])
        self.assertIn('insufficient_current_multi_view_target_depth',result['reasons'])
        with self.assertRaises(ValueError):
            CurrentRayBundle(1.,np.zeros((4,3)),np.ones((4,3)),np.zeros(4),np.zeros(4),np.ones(4,bool))

    def test_full_multiview_current_surfaces_recover_rotated_centre(self):
        centre=np.array([.02,-.01,.8]);rotation=Rotation.from_rotvec([.15,-.1,.2]).as_matrix()
        bundle=measured_bundle(centre,rotation)
        result=RayCubeCenterEstimator().fit_bundle(bundle,centre+[.012,-.008,.014])
        self.assertTrue(result['available'],result)
        self.assertLess(np.linalg.norm(np.array(result['centre'])-centre),1e-4)
        self.assertFalse(result['rotation_observed']);self.assertFalse(result['statistical_uncertainty_calibrated'])

    def test_no_robot_or_predicted_state_argument_exists(self):
        centre=np.array([.02,-.01,.8]);bundle=measured_bundle(centre,np.eye(3))
        with self.assertRaises(TypeError):RayCubeCenterEstimator().fit_bundle(bundle,centre,TCP=centre)

    def test_two_front_patch_views_without_silhouette_still_return_unknown(self):
        axis=np.linspace(-.003,.003,5);x,y=np.meshgrid(axis,axis)
        points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        origins=[];directions=[];depths=[];views=[]
        for index,z in enumerate((0.,.1)):
            o=np.broadcast_to([0.,0.,z],points.shape).copy()
            depth=points[:,2]-z;ray=(points-o)/depth[:,None]
            origins.append(o);directions.append(ray);depths.append(depth);views.append(np.full(len(ray),index))
        bundle=CurrentRayBundle(1.,np.concatenate(origins),np.concatenate(directions),np.concatenate(depths),
            np.concatenate(views),np.ones(2*len(points),bool))
        result=RayCubeCenterEstimator().fit_bundle(bundle,[0.,0.,1.025])
        self.assertFalse(result['available']);self.assertIsNone(result['centre'])
        self.assertTrue({'near_optimal_mirror_or_centre_ambiguity','centre_unobservable_with_rotation_nuisance'}&set(result['reasons']))


if __name__=='__main__':unittest.main()
