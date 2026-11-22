from copy import deepcopy
import unittest
from pathlib import Path

import numpy as np

# Hash-checked archived pure functions; no historical imports or real-data CLI.
import importlib.util
_support_path=Path(__file__).with_name('local_mpc_coupled_research_support.py')
_support_spec=importlib.util.spec_from_file_location('_local_mpc_coupled_research_support',_support_path)
_support=importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
features=_support.RESEARCH.features
fit=_support.RESEARCH.fit
predict=_support.RESEARCH.predict
rollouts=_support.RESEARCH.rollouts
episode,ToyFK=_support.episode,_support.ToyFK


def simple_model(gain=.4):
    c=np.zeros((43,7))
    for j in range(7):c[j,j]=gain;c[7+j,j]=.3
    return dict(mean=np.zeros(42),scale=np.ones(42),coefficient=c)


class CoupledResearchTests(unittest.TestCase):
    def test_feature_order_is_42_continuous_plus_bias_and_cross_joint_response(self):
        q=np.zeros((1,7));dq=q.copy();ref=np.zeros((1,21));ref[0,1]=.2
        f=features(q,dq,ref)
        self.assertEqual(f.shape,(1,42));self.assertEqual(f[0,1],.2)
        np.testing.assert_array_equal(f[0,28:35],np.zeros(7))
        np.testing.assert_array_equal(f[0,35:42],np.ones(7))
        m=simple_model(0.);m['coefficient'][1,0]=.5
        nq,nv=predict(m,q,dq,ref)
        self.assertAlmostEqual(nq[0,0],.1);self.assertAlmostEqual(nv[0,0],2.)
        self.assertEqual(nq[0,1],0.)

    def test_sin_cos_use_current_actual_q_and_velocity_is_finite_difference(self):
        q=np.full((2,7),.2);dq=np.zeros_like(q);ref=np.zeros((2,21));m=simple_model(0.)
        m['coefficient'][28+2,0]=.1;m['coefficient'][35+4,1]=.2
        nq,nv=predict(m,q,dq,ref)
        self.assertAlmostEqual(nq[0,0]-.2,.1*np.sin(.2))
        self.assertAlmostEqual(nq[0,1]-.2,.2*np.cos(.2))
        np.testing.assert_allclose(nv,(nq-q)/.05,atol=2e-15,rtol=0)

    def test_validation_poison_does_not_change_fit_or_scaler(self):
        train,val=episode(split='train'),episode(split='validation')
        expected=fit([train,val]);poisoned=deepcopy(val)
        for key in ('q','dq','refs'):poisoned[key][:]=np.nan
        actual=fit([train,poisoned])
        for key in ('mean','scale','coefficient','physical_coefficient','physical_intercept'):
            np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(actual['train_transition_count'],len(train['refs']))
        self.assertEqual(actual['input_dimension'],43);self.assertEqual(actual['output_dimension'],7)

    def test_parent_balance_is_invariant_to_duplicate_episode_within_parent(self):
        first,second=episode(),episode()
        first['parent_id']='A';second['parent_id']='B';second['q']*=.8;second['refs'][:,:7]*=.9
        expected=fit([first,second]);actual=fit([first,deepcopy(first),second])
        for key in ('mean','scale','coefficient'):
            np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)

    def test_free_rollout_never_uses_future_actual_labels(self):
        e=episode();poisoned=deepcopy(e)
        poisoned['q'][1:]+=9.;poisoned['dq'][1:]-=7.;poisoned['tcp'][1:]+=10.
        for mode in ('recorded_reference','regenerated_reference'):
            with self.subTest(mode=mode):
                a=rollouts(simple_model(),e,mode,ToyFK(),return_predictions=True)
                b=rollouts(simple_model(),poisoned,mode,ToyFK(),return_predictions=True)
                np.testing.assert_array_equal(a['available'][0],b['available'][0])
                np.testing.assert_allclose(a['predicted_q'][0],b['predicted_q'][0],atol=0,rtol=0,equal_nan=True)
                np.testing.assert_allclose(a['predicted_dq'][0],b['predicted_dq'][0],atol=0,rtol=0,equal_nan=True)

    def test_infeasible_prediction_is_reported_not_replaced_with_hold(self):
        r=rollouts(simple_model(100.),episode(),'regenerated_reference',ToyFK(),return_predictions=True)
        failures=[x for x in r['failures'] if x['anchor_before_tick']==0]
        self.assertTrue(failures);step=failures[0]['horizon_tick']-1
        self.assertFalse(r['available'][0,step:].any())
        self.assertTrue(np.isnan(r['predicted_q'][0,step:]).all())
        self.assertTrue(r['labelled'][0,step])


if __name__=='__main__':unittest.main()
