"""Meaningful synthetic primitive recovery/ambiguity checks, no native/GT seed."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from scripts.experimental_cup_features_r10 import (CupCADFeatures,FeatureConfig,
    estimate_cup_feature_pose,fit_unsigned_cylinder,handle_radial_candidates)


def synthetic_cup():
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    heights=np.linspace(-.0415,.0415,12)
    side=np.concatenate([np.c_[radius*np.cos(angles),radius*np.sin(angles),np.full(len(angles),z)]
        for radius in (.0328,.0293) for z in heights])
    bottom=np.concatenate([np.c_[radius*np.cos(angles),radius*np.sin(angles),np.full(len(angles),-.0415)]
        for radius in np.linspace(0,.0328,8)])
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    handle=np.concatenate([np.c_[np.full(len(theta),x),.047+.017*np.cos(theta),.029*np.sin(theta)]
        for x in (-.003,0,.003)])
    full=np.r_[side,bottom,handle]
    outside=handle[np.linalg.norm(handle[:,:2],axis=1)>.0368]
    model=CupCADFeatures(full,outside,np.zeros(3),.0328,.0293,.0415,Pose(np.zeros(3)),
        {'synthetic_fixture':True,'GT_or_world_pose_used':False})
    return model,full,side


class CupFeaturePrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,cls.full,cls.side=synthetic_cup()

    def test_rigidly_rotated_cylinder_axis_and_centre_are_recovered_without_pose_seed(self):
        R=Rotation.from_euler('xyz',[32,-47,81],degrees=True).as_matrix();centre=np.array([.2,-.1,.8])
        rng=np.random.default_rng(61);observed=self.full@R.T+centre+rng.normal(0,.0004,self.full.shape)
        fit=fit_unsigned_cylinder(observed,self.model)
        error=np.rad2deg(np.arccos(np.clip(abs(fit['axis']@R[:,2]),-1,1)))
        self.assertLess(error,5.)
        self.assertLess(np.linalg.norm(fit['centre']-centre),.005)

    def test_current_handle_produces_finite_pose_candidates_without_policy_authorization(self):
        R=Rotation.from_euler('xyz',[-19,28,112],degrees=True).as_matrix();p=np.array([.13,.21,.9])
        result=estimate_cup_feature_pose(self.full@R.T+p,self.model,{})
        self.assertIsNotNone(result['pose_xyzw'])
        self.assertGreaterEqual(result['candidate_count'],2)
        self.assertFalse(result['policy_authorized']);self.assertFalse(result['full_orientation_promoted'])
        errors=[Rotation.from_matrix(Pose.from_xyzw(row['pose_xyzw']).rotation@R.T).magnitude()
            for row in result['candidates']]
        self.assertLess(np.rad2deg(min(errors)),10.)

    def test_no_handle_support_keeps_unknown_and_small_cloud_is_rejected(self):
        directions,outside=handle_radial_candidates(self.side,np.zeros(3),[0,0,1],self.model)
        self.assertEqual(directions,[]);self.assertEqual(len(outside),0)
        result=estimate_cup_feature_pose(np.zeros((3,3)),self.model,{})
        self.assertIsNone(result['pose_xyzw']);self.assertTrue(result['ambiguous'])
        self.assertEqual(result['status'],'insufficient_current_points')


if __name__=='__main__':unittest.main()
