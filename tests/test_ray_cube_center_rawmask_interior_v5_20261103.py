"""Support order only: no geometry fit/UNKNOWN gate or raw RGBD changes."""
from unittest.mock import patch
import unittest
import numpy as np
from tests.test_ray_cube_center_interior_prototype_20261103 import observation
from our_mani.local_mpc_control_20261103.ray_cube_center_prototype_20261103 import RayCubeCenterEstimator,CubeRayConfig
from our_mani.local_mpc_control_20261103.ray_cube_center_darkred_prototype_20261103 import DarkRedInteriorRayCubeCenterEstimator as V4
from our_mani.local_mpc_control_20261103.ray_cube_center_rawmask_interior_v5_20261103 import RawMaskInteriorDarkRedCubeEstimator as V5
from our_mani.local_mpc_control_20261103.goal_aware_perception import VisualGoalSphere


class RawInteriorV5Tests(unittest.TestCase):
    def test_without_sphere_every_ray_is_V4_exact(self):
        obs=observation(7)
        for image in obs['observation/images'].values():image[image[...,0]>0,0]=128
        a,_=V4().current_bundle(obs,[0.,0.,1.]);b,_=V5().current_bundle(obs,[0.,0.,1.])
        for key in ('origins','directions','depth','view_ids','is_target'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))

    def test_artificial_shell_cut_is_not_a_second_erosion_boundary(self):
        obs=observation(7);before={k:v.copy() for k,v in obs['observation/images'].items()}
        sphere=VisualGoalSphere([0.,0.,1.],.04,0.,0.,.004,[0.,0.,0.])
        def filter_cut(value,*args):
            private=dict(value);private['observation/images']={k:v.copy() for k,v in value['observation/images'].items()}
            for image in private['observation/images'].values():image[:,8]=0
            return private,dict(test_cut=True)
        with patch('our_mani.local_mpc_control_20261103.ray_cube_center_darkred_prototype_20261103.filter_observed_sphere_surface',side_effect=filter_cut):
            a,_=V4().current_bundle(obs,[0.,0.,1.],sphere)
        with patch('our_mani.local_mpc_control_20261103.ray_cube_center_rawmask_interior_v5_20261103.filter_observed_sphere_surface',side_effect=filter_cut):
            b,meta=V5().current_bundle(obs,[0.,0.,1.],sphere)
        self.assertEqual(a.is_target.sum(),40);self.assertEqual(b.is_target.sum(),80)
        for key in ('origins','directions','depth','view_ids'):
            np.testing.assert_array_equal(getattr(a,key)[~a.is_target],getattr(b,key)[~b.is_target])
        self.assertTrue(meta['positive_target_support']['erosion_before_sphere_removal'])
        self.assertFalse(meta['positive_target_support']['artificial_sphere_cut_edge_eroded_again'])
        for k,v in before.items():np.testing.assert_array_equal(v,obs['observation/images'][k])

    def test_fit_cost_config_multistarts_and_unknown_stay_inherited(self):
        for name in ('fit','fit_bundle','_residual','_weights'):
            self.assertIs(getattr(V5,name),getattr(RayCubeCenterEstimator,name))
        self.assertEqual(V5().config,CubeRayConfig())
        result=V5().fit(observation(3),[0.,0.,1.])
        self.assertFalse(result['available']);self.assertIsNone(result['centre'])
        self.assertIn('insufficient_current_multi_view_target_depth',result['reasons'])
        obs=observation();obs['TCP_position']=[0.,0.,1.]
        with self.assertRaises(ValueError):V5().current_bundle(obs,[0.,0.,1.])
        sphere=VisualGoalSphere([0.,0.,1.],.04,2.,0.,.004,[0.,0.,0.])
        with self.assertRaisesRegex(ValueError,'future'):V5().current_bundle(observation(),[0.,0.,1.],sphere)

if __name__=='__main__':unittest.main()
