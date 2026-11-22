"""Frozen-weight probe binding only, no fit or Native execution."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_control_20261103 import probe_onpolicy_response_v55 as helper
from our_mani.local_mpc_control_20261103.coupled_response import FrozenCoupledResponse,CoupledWithFingerResponse
from our_mani.local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel
from tests.test_local_mpc_probe_nominal_cache_v53_20261103 import fixture


class FingerSpy:
    def __init__(self):self.calls=[]
    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        self.calls.append(dict(latent=deepcopy(latent),held=held_anchor.copy()))
        return dict(q=x['q']+999.,dq=x['dq']+999.,finger_open_amounts=x['finger_open_amounts'].copy(),
            feasible=np.ones(len(x['q']),bool),latent=deepcopy(latent),held=np.ones(len(x['q']),bool))


def base_fixture():
    base,state,candidates=fixture();finger=FingerSpy();initial=[]
    old=FrozenCoupledResponse.from_json(helper.ROOT/'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json',
        expected_sha256='d7c7fac5f8de7a1bf3d37c10b94a20585728a7c03c7d1a05920799e69a501f92')
    def initialize(actual,n):
        initial.append((actual['time'],actual['q'].copy(),n));return dict(anchor_time=actual['time'])
    model=JointResidualFKExecutionModel(base.execution_model.reference_step,base.execution_model.limits,
        CoupledWithFingerResponse(old,finger),base.execution_model.fk_batch,config=base.execution_model.config,
        latent_initializer=initialize,model_id='original_global_B_plus_same_finger')
    return model,state,finger,initial


class ProbeOnPolicyBindingTests(unittest.TestCase):
    def test_strict_source_identity_and_small_cohort_limits_are_explicit(self):
        sources=helper.probe_response_sources()
        self.assertGreaterEqual(len(sources),25)
        self.assertTrue(all(Path(p).is_absolute() for p in sources))
        base,state,finger,initial=base_fixture();new,receipt=helper.build_onpolicy_probe_execution(base)
        self.assertEqual(receipt['parameter_sha256'],helper.PARAMETER_SHA256)
        self.assertEqual(receipt['internal_TRAIN_parents'],2);self.assertEqual(receipt['internal_probe_ticks'],14)
        self.assertEqual(receipt['internal_pure_probe_400ms']['predictable'],7)
        self.assertEqual(receipt['internal_pure_probe_800ms']['labelled'],0)
        self.assertFalse(receipt['pure_probe_800ms_qualified']);self.assertFalse(receipt['global_800ms_qualified'])
        self.assertTrue(receipt['deployment_scope_selected_using_internal_TRAIN_report'])
        self.assertFalse(receipt['parameter_fit_hyperparameter_selection_used_internal'])
        wrong=dict(helper.EXPECTED);wrong[helper.FIT+'/parameters.json']='0'*64
        with patch.object(helper,'EXPECTED',wrong),self.assertRaisesRegex(ValueError,'SHA mismatch'):
            helper.build_onpolicy_probe_execution(base)

    def test_all_original_callbacks_limits_config_latent_initializer_and_finger_object_preserved(self):
        base,state,finger,initial=base_fixture();old_transition=base.transition_batch
        new,receipt=helper.build_onpolicy_probe_execution(base)
        for name in ('reference_step','limits','fk_batch','config','latent_initializer'):
            self.assertIs(getattr(new,name),getattr(base,name))
        self.assertIs(new.transition_batch.finger_response,finger)
        self.assertIs(base.transition_batch,old_transition)
        self.assertIsNot(new.transition_batch.joint_response,old_transition.joint_response)
        self.assertEqual(base.model_id,'original_global_B_plus_same_finger')
        self.assertFalse(receipt['actual_state_or_h_cached'])
        self.assertEqual(finger.calls,[]);self.assertEqual(initial,[])

    def test_each_plan_uses_new_actual_h_fingers_remain_exact_and_no_fake_held(self):
        base,state,finger,initial=base_fixture();new,_=helper.build_onpolicy_probe_execution(base)
        x,h=new.initialize_batch(state,2)
        goal=np.tile(state['q'],(2,1));u=dict(joint_goal=goal,opening=np.zeros(2))
        nxt,hn,info=new.step_batch(x,u,h)
        np.testing.assert_array_equal(nxt['finger_open_amounts'],x['finger_open_amounts'])
        self.assertFalse(nxt['held'].any());self.assertEqual(len(finger.calls),2)
        self.assertTrue(all(not call['held'].any() for call in finger.calls))
        self.assertTrue(all(call['latent']=={'anchor_time':0.} for call in finger.calls))
        later=deepcopy(state);later['time']=.1;later['q'][0]+=1e-6;later['actuator_state']['reference_clock']=.1
        y,hy=new.initialize_batch(later,1)
        np.testing.assert_array_equal(y['q'][0],later['q'])
        self.assertEqual(hy['latent'],{'anchor_time':.1});self.assertEqual(len(initial),2)
        self.assertEqual(initial[-1][0],.1)

    def test_stale_virtual_open_or_held_initialization_and_other_phase_steps_rejected(self):
        base,state,finger,initial=base_fixture();new,_=helper.build_onpolicy_probe_execution(base)
        for fault in ('held','open','hopen','stale','nan','boolclock','virtual','predicted'):
            with self.subTest(fault=fault):
                bad=deepcopy(state)
                if fault=='held':bad['held']=True
                elif fault=='open':bad['requested_opening']=1.
                elif fault=='hopen':bad['actuator_state']['gripper_target']=1.
                elif fault=='stale':bad['actuator_state']['reference_clock']=.05
                elif fault=='nan':bad['actuator_state']['reference_clock']=float('nan')
                elif fault=='boolclock':bad['actuator_state']['reference_clock']=False
                elif fault=='virtual':bad['actuator_state']['source']='virtual_rollout'
                else:bad['actuator_state']['predicted']=True
                with self.assertRaises(ValueError):new.initialize_batch(bad,1)
        x,h=new.initialize_batch(state,1)
        with self.assertRaises(ValueError):new.step_batch(x,dict(joint_goal=x['q'],opening=np.ones(1)),h)
        x['held'][:]=True
        with self.assertRaises(ValueError):new.step_batch(x,dict(joint_goal=x['q'],opening=np.zeros(1)),h)

    def test_arbitrary_wrappers_are_not_silently_flattened(self):
        base,state,finger,initial=base_fixture()
        with self.assertRaises(TypeError):helper.build_onpolicy_probe_execution(object())
        base.transition_batch=FingerSpy()
        with self.assertRaisesRegex(TypeError,'do not flatten'):helper.build_onpolicy_probe_execution(base)


if __name__=='__main__':unittest.main()
