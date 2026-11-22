"""Pure feature/SE3 diagnostics, zero real model/checkpoint/GT calls."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.context_candidate_compatibility_20261103_v1 import (
    predicted_features,reconstruction_difference,world_candidates,validate_prepared)


class CandidatePreparationContracts(unittest.TestCase):
    def test_world_relative_translation_column_rot6_and_unchanged_public_condition(self):
        position=np.array([1.,2.,3.])
        ref=dict(future_tokens=np.zeros((17,16),np.float32),robot=np.zeros(25,np.float32),
            geometry=np.array([.05,.05,.05,1.],np.float32),context_c=np.full(16,.125,np.float32),task_e=np.full(16,.125,np.float32))
        ref['future_tokens'][:,0]=np.arange(17)/16
        ref['robot'][-2:]=1.
        R=Rotation.from_euler('z',90,degrees=True).as_matrix()
        p=np.tile(position,(17,1));p[:,0]+=np.arange(17)*.01
        pred=dict(positions=p[None,None],rotations=np.tile(R,(1,1,17,1,1)),
            velocities=np.tile([.2,0,0],(1,1,17,1)),angular_velocities=np.zeros((1,1,17,3)),contexts=np.full((1,1,16),.125,np.float32))
        before={k:v.copy()for k,v in ref.items()}
        out=predicted_features(ref,position,pred,0,0)
        np.testing.assert_allclose(out['future_tokens'][:,1],np.arange(17)*.01,atol=1e-7)
        np.testing.assert_allclose(out['future_tokens'][:,2:4],0.,atol=1e-7)
        np.testing.assert_allclose(out['future_tokens'][0,4:10],np.r_[R[:,0],R[:,1]],atol=1e-7)
        for key in ('robot','geometry','task_e'):
            np.testing.assert_array_equal(out[key],ref[key])
        for key in ref:np.testing.assert_array_equal(ref[key],before[key])

    def test_own_tau_original_forecast_slerp_and_object_local_composition(self):
        times=np.arange(17)*.05
        p=np.c_[times,times*0,times*0]
        R=Rotation.from_euler('z',times/.8*90,degrees=True).as_matrix()
        pred=dict(positions=p[None,None],rotations=R[None,None],velocities=np.zeros((1,1,17,3)),angular_velocities=np.zeros((1,1,17,3)))
        local_R=Rotation.from_euler('x',30,degrees=True).as_matrix()
        wp,wr=world_candidates(pred,0,0,7.9,np.array([[1.,0,0]]),local_R[None],np.array([.4]))
        object_R=Rotation.from_euler('z',45,degrees=True).as_matrix()
        np.testing.assert_allclose(wp[0],np.array([.4,0,0])+object_R@np.array([1.,0,0]),atol=1e-12)
        np.testing.assert_allclose(wr[0],object_R@local_R,atol=1e-12)

    def test_reconstruction_difference_is_reported_without_selecting_features(self):
        a=dict(context_c=np.zeros(16,np.float32))
        b=dict(context_c=np.zeros(16,np.float32));b['context_c'][1]=.01
        report=reconstruction_difference(a,b)
        self.assertFalse(report['context_c']['exact'])
        self.assertAlmostEqual(report['context_c']['max_abs'],.01,places=7)

    def test_invalid_prepared_identity_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);(p/'feature_freeze_before_TF.json').write_text(json.dumps(dict(complete=False)))
            with patch('torch.load',side_effect=AssertionError('must not load model'))as load:
                with self.assertRaises(ValueError):validate_prepared(p)
            self.assertEqual(load.call_count,0)


if __name__=='__main__':unittest.main()
