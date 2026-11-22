"""Only positive support changes; no inherited acceptance gate is weakened."""
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.ray_cube_center_prototype_20261103 import RayCubeCenterEstimator,CubeRayConfig
from our_mani.local_mpc_control_20261103.ray_cube_center_interior_prototype_20261103 import InteriorRayCubeCenterEstimator


def observation(width=5):
    images={};depths={};calibrations={}
    for camera in ('front','8','16','36'):
        image=np.zeros((16,16,3),np.uint8);start=8-width//2;image[start:start+width,start:start+width,0]=255
        images[camera]=image;depths[camera]=np.ones((16,16))
        calibrations[camera]=CameraCalibration([[-100.,0.,8.],[0.,-100.,8.],[0.,0.,1.]],Pose([0.,0.,0.]))
    return {'timestamp':1.,'observation/images':images,'observation/depths':depths,'observation/calibrations':calibrations}


class InteriorSupportTests(unittest.TestCase):
    def test_exact_one_pixel_positive_boundary_removed_and_context_preserved(self):
        obs=observation();before={k:v.copy() for k,v in obs['observation/images'].items()}
        old,_=RayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        new,metadata=InteriorRayCubeCenterEstimator().current_bundle(obs,[0.,0.,1.])
        self.assertEqual(old.is_target.sum(),100);self.assertEqual(new.is_target.sum(),36)
        for key in ('origins','directions','depth','view_ids'):
            np.testing.assert_array_equal(getattr(old,key)[~old.is_target],getattr(new,key)[~new.is_target])
        for name in before:np.testing.assert_array_equal(before[name],obs['observation/images'][name])
        self.assertEqual(metadata['positive_target_support']['erosion_iterations'],1)
        self.assertEqual(metadata['positive_target_support']['erosion_kernel'],[3,3])

    def test_small_support_remains_unknown_without_size_prior_certification(self):
        result=InteriorRayCubeCenterEstimator().fit(observation(3),[0.,0.,1.])
        self.assertFalse(result['available']);self.assertIsNone(result['centre'])
        self.assertIn('insufficient_current_multi_view_target_depth',result['reasons'])

    def test_original_cost_solver_config_gates_and_clock_contract_inherited_exactly(self):
        self.assertIs(InteriorRayCubeCenterEstimator.fit_bundle,RayCubeCenterEstimator.fit_bundle)
        self.assertIs(InteriorRayCubeCenterEstimator._residual,RayCubeCenterEstimator._residual)
        self.assertEqual(InteriorRayCubeCenterEstimator().config,CubeRayConfig())
        obs=observation();obs['Native_held']=True
        with self.assertRaises(ValueError):InteriorRayCubeCenterEstimator().fit(obs,[0.,0.,1.])


if __name__=='__main__':unittest.main()
