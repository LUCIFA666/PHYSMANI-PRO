import unittest
import numpy as np
from tests.test_ray_cube_center_interior_prototype_20261103 import observation
from our_mani.local_mpc_control_20261103.ray_cube_center_prototype_20261103 import RayCubeCenterEstimator,CubeRayConfig
from our_mani.local_mpc_control_20261103.ray_cube_center_interior_prototype_20261103 import InteriorRayCubeCenterEstimator
from our_mani.local_mpc_control_20261103.ray_cube_center_darkred_prototype_20261103 import DarkRedInteriorRayCubeCenterEstimator


class DarkRedHypothesisTests(unittest.TestCase):
    def test_bright_input_same_as_v3_and_generic_depth_exact(self):
        obs=observation();a,_=InteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        b,_=DarkRedInteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        for key in ('origins','directions','depth','view_ids','is_target'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))

    def test_fixed_R100_recovers_current_dark_red_without_rgb_fabrication(self):
        obs=observation()
        for image in obs['observation/images'].values():image[image[...,0]>0,0]=128
        before={key:value.copy() for key,value in obs['observation/images'].items()}
        a,_=InteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        b,meta=DarkRedInteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        self.assertEqual(a.is_target.sum(),0);self.assertEqual(b.is_target.sum(),36)
        for key in ('origins','directions','depth','view_ids'):
            np.testing.assert_array_equal(getattr(a,key)[~a.is_target],getattr(b,key)[~b.is_target])
        for key in before:np.testing.assert_array_equal(before[key],obs['observation/images'][key])
        self.assertEqual(meta['positive_target_support']['minimum_R'],100)
        self.assertEqual(meta['positive_target_support']['red_ratio'],3.5)

    def test_ratio_and_all_geometry_gates_stay_unchanged(self):
        obs=observation()
        for image in obs['observation/images'].values():
            mask=image[...,0]>0;image[mask]=[128,50,0]
        bundle,_=DarkRedInteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        self.assertEqual(bundle.is_target.sum(),0)
        self.assertIs(DarkRedInteriorRayCubeCenterEstimator.fit_bundle,RayCubeCenterEstimator.fit_bundle)
        self.assertIs(DarkRedInteriorRayCubeCenterEstimator._residual,RayCubeCenterEstimator._residual)
        self.assertEqual(DarkRedInteriorRayCubeCenterEstimator().config,CubeRayConfig())


if __name__=='__main__':unittest.main()
