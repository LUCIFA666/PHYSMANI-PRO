"""Synthetic features only; no checkpoint, real public rows or NN forward.

The exact Torch class is randomly initialized solely to exercise frozen CPU
guards. The successful orchestration's private forward boundary is mocked;
its output tensors are a declared analytic fixture, never a runtime model.
"""
from copy import deepcopy
from types import MethodType
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.context_gap_dropout_20261103_v1 import compatibility as c


def features():
    f={k:np.zeros(shape,np.float32)for k,shape in c.FEATURE_SHAPES.items()}
    f['future_tokens'][:,0]=np.arange(17,dtype=np.float32)/16
    f['geometry'][:]=[.05,.05,.05,1.]
    f['robot'][-2:]=1.
    f['context_c'][:]=f['task_e'][:]=.125
    return f


def output(position=(0.,0.,0.),angle=0.,tau=.25):
    R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]],np.float32)
    matrices=np.broadcast_to(R,(1,8,3,3)).copy()
    rot6=np.concatenate((matrices[..., :,0],matrices[..., :,1]),axis=-1)
    return dict(positions=np.broadcast_to(np.array(position,np.float32),(1,8,3)).copy(),
        rotation_matrices=matrices,rotation_6d=rot6,rotation_6d_raw=rot6.copy(),
        times=np.full((1,8),tau,np.float32),proposal_logits=np.zeros((1,8),np.float32),
        proposal_score_semantics=c.SCORE_SEMANTICS)


class CandidateCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=c.CandidateTransformerPilot().eval().requires_grad_(False)

    def test_missing_real_bundles_are_not_evaluated(self):
        with patch.object(c,'_forward_pair',side_effect=AssertionError('No forward')):
            for model,a,b in ((None,None,None),(self.fixture,features(),None),(None,features(),features())):
                r=c.compare_candidate_features(model,a,b)
                self.assertEqual(r['status'],'not_evaluated');self.assertEqual(r['model_forward_calls'],0)
                self.assertFalse(r['production_admission']);self.assertFalse(r['real_selection_compatibility_claimed'])

    def test_only_forecast_and_context_may_change_exact_feature_schema(self):
        a=features();b=features();b['future_tokens'][1:,1]=.01
        b['context_c'][0]=.1;b['context_c'][1]=.15
        aa,bb,n,batched=c._checked_pair(a,b)
        self.assertEqual(n,1);self.assertFalse(batched)
        self.assertTrue(all(np.array_equal(aa[k],bb[k])for k in c.IMMUTABLE_FEATURES))
        with patch.object(c,'_forward_pair',side_effect=AssertionError('No forward')):
            for key in c.IMMUTABLE_FEATURES:
                bad=deepcopy(b);bad[key].flat[0]+=.001
                with self.subTest(key=key),self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,a,bad)
            bad=deepcopy(b);bad['labels']=np.zeros(1,np.float32)
            with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,a,bad)
            bad=deepcopy(b);bad['future_tokens']=np.zeros((16,16),np.float32)
            with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,a,bad)
            bad=deepcopy(b);bad['future_tokens'][1,0]=np.nan
            with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,a,bad)

    def test_pure_candidate_delta_has_physical_units_and_no_success_probability(self):
        r=c.summarize_candidate_outputs(output(),output((.003,.004,0.),.2,.29))
        self.assertAlmostEqual(r['max_position_distance_m'],.005,places=8)
        self.assertAlmostEqual(r['max_rotation_geodesic_rad'],.2,places=6)
        self.assertAlmostEqual(r['max_abs_tau_delta_seconds'],.04,places=7)
        self.assertTrue(r['same_query_index_comparison']);self.assertTrue(r['proposal_logits_constant_zero'])
        self.assertFalse(r['calibrated_success_probability']);self.assertFalse(r['world_pose_or_action_equivalence_claimed'])
        bad=output();bad['proposal_logits'][0,0]=.1
        with self.assertRaises(ValueError):c.summarize_candidate_outputs(output(),bad)

    def test_exact_eval_frozen_CPU_model_and_original_forward_are_required(self):
        with patch.object(c,'_forward_pair',side_effect=AssertionError('No forward')):
            with self.assertRaises(TypeError):c.compare_candidate_features(object(),features(),features())
            self.fixture.train()
            try:
                with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,features(),features())
            finally:self.fixture.eval()
            p=next(self.fixture.parameters());p.requires_grad_(True)
            try:
                with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,features(),features())
            finally:p.requires_grad_(False)
            with patch.object(self.fixture,'forward',MethodType(lambda self,**kw:output(),self.fixture)):
                with self.assertRaises(ValueError):c.compare_candidate_features(self.fixture,features(),features())

    def test_diagnostic_orchestration_is_immutable_and_never_a_passing_certificate(self):
        a=features();b=features();b['future_tokens'][1:,2]=.02
        a0=deepcopy(a);b0=deepcopy(b)
        with patch.object(c,'_forward_pair',return_value=(output(),output((0.,.002,0.))))as fake:
            r=c.compare_candidate_features(self.fixture,a,b)
        self.assertEqual(fake.call_count,1);self.assertEqual(r['status'],'evaluated_offline')
        self.assertEqual(r['unchanged_features'],list(c.IMMUTABLE_FEATURES))
        self.assertAlmostEqual(r['comparison']['max_position_distance_m'],.002,places=8)
        self.assertFalse(r['compatibility_qualified']);self.assertFalse(r['production_admission'])
        self.assertFalse(r['checkpoint_and_input_provenance_certified'])
        self.assertTrue(all(np.array_equal(a[k],a0[k])and np.array_equal(b[k],b0[k])for k in a))


if __name__=='__main__':unittest.main()
