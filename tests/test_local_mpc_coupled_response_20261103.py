from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

# Hash-checked archived pure functions; no historical imports or real-data CLI.
import importlib.util
_support_path=Path(__file__).with_name('local_mpc_coupled_research_support.py')
_support_spec=importlib.util.spec_from_file_location('_local_mpc_coupled_research_support',_support_path)
_support=importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
predict=_support.RESEARCH.predict
from our_mani.local_mpc_control_20261103.coupled_response import FrozenCoupledResponse,CoupledWithFingerResponse

ARCHIVE=_support.ARCHIVE
PARAMETERS=ARCHIVE/'B_parameters_frozen_20261103.json'
SHA='d7c7fac5f8de7a1bf3d37c10b94a20585728a7c03c7d1a05920799e69a501f92'


class FingerFixture:
    evidence=dict(finger_fixture_only=True,globally_qualified=False)
    def __init__(self,*,bad=False,mutate=False):self.bad,self.mutate=bad,mutate
    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        n=len(x['q'])
        if self.mutate:x['q'][:]=999.;references['q_ref'][:]=999.
        return dict(q=np.full((n,7),999.),dq=np.full((n,7),999.),
                    finger_open_amounts=np.full((n,2),1.1 if self.bad else .3),
                    held=np.ones(n,bool),feasible=np.ones(n,bool),latent=deepcopy(latent))


class FrozenPredictorTests(unittest.TestCase):
    def setUp(self):
        self.model=FrozenCoupledResponse.from_json(PARAMETERS,expected_sha256=SHA)
        rng=np.random.default_rng(20261103)
        self.q=rng.uniform(-1.,1.,(8,7));self.dq=rng.uniform(-.5,.5,(8,7))
        self.ref=np.c_[self.q+rng.uniform(-.1,.1,(8,7)),rng.uniform(-1.,1.,(8,7)),rng.uniform(-8.,8.,(8,7))]

    def test_matches_research_predictor_exactly_with_frozen_coefficients(self):
        report=json.loads((ARCHIVE/'evaluation_B.json').read_text())
        m=report['parameters']
        for key in ('mean','scale','coefficient'):m[key]=np.asarray(m[key])
        expected_q,expected_v=predict(m,self.q,self.dq,self.ref)
        result=self.model.predict_joint(self.q,self.dq,self.ref[:,:7],self.ref[:,7:14],self.ref[:,14:])
        np.testing.assert_array_equal(result['q'],expected_q)
        np.testing.assert_array_equal(result['dq'],expected_v)
        np.testing.assert_allclose(result['dq'],(result['q']-self.q)/.05,atol=2e-15,rtol=0)

    def test_sha_tampering_and_wrong_feature_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tampered.json';path.write_bytes(PARAMETERS.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                FrozenCoupledResponse.from_json(path,expected_sha256=SHA)
        p=self.model.parameters();p['feature_order'][0]='wrong'
        with self.assertRaises(ValueError):FrozenCoupledResponse(p)

    def test_parameter_archive_contains_exact_train_source_weights_and_counts(self):
        p=self.model.parameters();fit=p['fit_source'];sources=fit['train_episode_sources_and_weights']
        self.assertEqual(fit['parameter_fit_split'],'train');self.assertEqual(fit['train_transition_count'],1658)
        self.assertEqual(len(sources),16);self.assertEqual(fit['train_parent_count'],8)
        self.assertAlmostEqual(sum(x['episode_total_weight'] for x in sources),1.)
        for parent in {x['parent_id'] for x in sources}:
            self.assertAlmostEqual(sum(x['episode_total_weight'] for x in sources if x['parent_id']==parent),1/8)
        self.assertFalse(p['formal_TEST_used']);self.assertEqual(p['normalization']['fit_split'],'train')
        manifest_path=ARCHIVE/'source_snapshot/episode_manifest.json'
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(),fit['split_manifest_sha256'])
        records=json.loads(manifest_path.read_text())['episode_records']
        train={r['episode_id']:r for r in records if r['split']=='train' and r['excluded_reason'] is None}
        self.assertEqual(set(train),{s['episode_id'] for s in sources})
        self.assertEqual(sum(s['ticks'] for s in sources),1658)
        for source in sources:
            original=train[source['episode_id']]
            self.assertEqual(source['parent_id'],original['parent_id'])
            self.assertEqual(source['ticks'],original['transitions'])
            self.assertEqual(source['files_sha256'],original['files'])
            self.assertAlmostEqual(source['per_tick_weight']*source['ticks'],source['episode_total_weight'])

    def test_basic_generic_callback_preserves_latent_and_exposes_only_joints(self):
        latent={'marker':np.array([1.])};before=deepcopy(latent)
        result=self.model(self.q,self.dq,self.ref[:,:7],self.ref[:,7:14],self.ref[:,14:],latent,.05)
        self.assertNotIn('finger_open_amounts',result);self.assertNotIn('held',result)
        self.assertTrue(result['feasible'].all());np.testing.assert_array_equal(result['latent']['marker'],before['marker'])
        self.assertFalse(hasattr(self.model,'predict_execution'))

    def test_rich_composition_discards_nn_joint_and_held_outputs(self):
        combined=CoupledWithFingerResponse(self.model,FingerFixture(mutate=True))
        x=dict(q=self.q.copy(),dq=self.dq.copy(),finger_open_amounts=np.full((8,2),.8))
        refs=dict(q_ref=self.ref[:,:7].copy(),dq_ref=self.ref[:,7:14].copy(),ddq_ref=self.ref[:,14:].copy(),gripper_target=np.zeros(8))
        before=deepcopy(x);ref_before=deepcopy(refs)
        result=combined.predict_execution(x,refs,{},.05,held_anchor=np.zeros(8,bool))
        expected=self.model.predict_joint(self.q,self.dq,self.ref[:,:7],self.ref[:,7:14],self.ref[:,14:])
        np.testing.assert_array_equal(result['q'],expected['q']);np.testing.assert_array_equal(result['dq'],expected['dq'])
        np.testing.assert_array_equal(result['finger_open_amounts'],np.full((8,2),.3))
        self.assertNotIn('held',result);self.assertFalse(result['globally_qualified'])
        for key in x:np.testing.assert_array_equal(x[key],before[key])
        for key in refs:np.testing.assert_array_equal(refs[key],ref_before[key])

    def test_rich_requires_two_real_fingers_and_rejects_invalid_finger_prediction(self):
        x=dict(q=self.q,dq=self.dq,finger_open_amounts=np.full((8,2),.8))
        refs=dict(q_ref=self.ref[:,:7],dq_ref=self.ref[:,7:14],ddq_ref=self.ref[:,14:],gripper_target=np.zeros(8))
        combined=CoupledWithFingerResponse(self.model,FingerFixture(bad=True))
        with self.assertRaisesRegex(ValueError,'outside'):
            combined.predict_execution(x,refs,None,.05,held_anchor=np.zeros(8,bool))
        x['finger_open_amounts']=np.full(8,.8)
        with self.assertRaises(ValueError):combined.predict_execution(x,refs,None,.05,held_anchor=np.zeros(8,bool))

    def test_input_validation_no_clipping_and_no_global_qualification(self):
        q=np.zeros((1,7));extreme=np.full((1,7),1e6)
        result=self.model.predict_joint(q,q,extreme,q,q)
        self.assertGreater(np.max(np.abs(result['q'])),100.)
        self.assertIn('outer joint bounds',result['feasibility_scope'])
        self.assertFalse(self.model.evidence['globally_qualified'])
        self.assertFalse(self.model.evidence['contact_model_trained'])
        self.assertFalse(self.model.evidence['gripper_model_included'])
        with self.assertRaises(ValueError):self.model.predict_joint(q,q,q,q,q,dt=.1)
        with self.assertRaises(ValueError):self.model.predict_joint(q,q,np.full((1,7),np.nan),q,q)

    def test_arrays_and_returned_parameter_metadata_cannot_change_predictor(self):
        self.assertFalse(self.model.mean.flags.writeable);self.assertFalse(self.model.coefficient.flags.writeable)
        expected=self.model.predict_joint(self.q,self.dq,self.ref[:,:7],self.ref[:,7:14],self.ref[:,14:])['q']
        p=self.model.parameters();p['coefficient_scaled'][0][0]=999.
        actual=self.model.predict_joint(self.q,self.dq,self.ref[:,:7],self.ref[:,7:14],self.ref[:,14:])['q']
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),SHA)


if __name__=='__main__':unittest.main()
