"""Independent v55 production-hook CPU contracts, no fit or Native."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103 import probe_onpolicy_response_v55 as v55
from our_mani.local_mpc_control_20261103.probe_deadline_v54 import FixedDeadlineVerificationProbePlanner
from our_mani.local_mpc_20261102.records import freeze_sources,sha
from tests import test_local_mpc_probe_deadline_v54_integration_20261103 as v54test

ROOT=Path(__file__).resolve().parents[1]


class ProbeResponseHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v54test.FixedDeadlineHookTests.setUpClass()
        cls.fixture=v54test.FixedDeadlineHookTests(methodName='test_default_false_is_v53_and_invalid_combination_rejected')
        cls.fixture.harness=v54test.FixedDeadlineHookTests.harness
        cls.fixture.first=v54test.FixedDeadlineHookTests.first

    def coordinator(self,**kwargs):
        return self.fixture.harness.coordinator(cache_fixed_probe_nominal_ik=True,probe_terminal_at_fixed_deadline=True,**kwargs)

    def components(self,control):
        spec=v54test.numpy_spec(self.fixture.first['decision']['probe']);spec['deadline']=spec['trigger_time']+.8
        return self.fixture.components(control,self.fixture.first,spec)

    def test_default_false_keeps_original_response_and_illegal_protocols_reject(self):
        for options in ({},{'probe_onpolicy_joint_response':False}):
            control=self.coordinator(**options);state,rows,receipt,planner=self.components(control)
            self.assertFalse(control.probe_onpolicy_joint_response)
            self.assertIsNone(control.probe_execution_model)
            self.assertIs(planner.execution_model,control.core.execution_model)
            self.assertIs(type(planner),FixedDeadlineVerificationProbePlanner)
            self.assertEqual(len(control.probe_cache_sources),9)
        for cache,deadline in ((False,False),(True,False),(False,True)):
            with self.assertRaises(ValueError):
                self.fixture.harness.coordinator(cache_fixed_probe_nominal_ik=cache,probe_terminal_at_fixed_deadline=deadline,probe_onpolicy_joint_response=True)
        for invalid in (1,None,'true'):
            with self.assertRaises(ValueError):self.coordinator(probe_onpolicy_joint_response=invalid)

    def test_true_swaps_joint_only_preserves_base_objects_and_fresh_h(self):
        control=self.coordinator(probe_onpolicy_joint_response=True);base=control.core.execution_model
        old_joint=base.transition_batch.joint_response;old_finger=base.transition_batch.finger_response
        state,rows,_,planner=self.components(control);model=planner.execution_model
        self.assertIs(type(model),v55.ProbeOnlyOnPolicyExecution)
        self.assertIs(control.core.execution_model,base);self.assertIs(control.probe_execution_base,base)
        self.assertIs(base.transition_batch.joint_response,old_joint)
        self.assertIs(model.transition_batch.finger_response,old_finger)
        self.assertIsNot(model.transition_batch.joint_response,old_joint)
        self.assertEqual(model.transition_batch.joint_response.parameter_sha256,v55.PARAMETER_SHA256)
        for key in ('reference_step','limits','fk_batch','config','latent_initializer'):
            self.assertIs(getattr(model,key),getattr(base,key))
        actual=deepcopy(state);x,h=model.initialize_batch(actual,1)
        np.testing.assert_array_equal(x['q'][0],actual['q'])
        np.testing.assert_array_equal(x['finger_open_amounts'][0],actual['finger_open_amounts'])
        self.assertFalse(h['predicted']);self.assertFalse(x['held'].any())
        bad=deepcopy(actual);bad['actuator_state']['reference_clock']-=.05
        with self.assertRaisesRegex(ValueError,'fresh actual'):model.initialize_batch(bad,1)
        bad=deepcopy(actual);bad['held']=True
        with self.assertRaises(ValueError):model.initialize_batch(bad,1)
        with self.assertRaisesRegex(ValueError,'held/reopen'):
            model.step_batch(x,{'opening':np.ones(1)},h)
        result=planner.plan(actual,rows)
        self.assertEqual(result['budget']['sequences'],256);self.assertEqual(result['budget']['horizon'],8)
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80])
        self.assertIs(control.core.execution_model,base)
        self.assertFalse(control.probe_response_receipt['new_weights_used_in_other_phases'])
        self.assertFalse(control.probe_response_receipt['actual_state_or_h_cached'])

    def test_all_model_evidence_sources_are_real_and_conditionally_archived(self):
        sources=v55.probe_response_sources();self.assertEqual(len(sources),25)
        for name,expected in sources.items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),expected)
        control=self.coordinator(probe_onpolicy_joint_response=True)
        for name,digest in sources.items():self.assertEqual(control.probe_cache_sources[name],digest)
        config=json.loads((ROOT/'config/framework/P4_student_closedloop_devVAL_v54_v1/00_val_static.json').read_text())
        with TemporaryDirectory(prefix='v55_CPU_inventory_',dir=ROOT/'config/framework') as directory:
            path=Path(directory)/'config.json';config['probe_onpolicy_joint_response']=False;path.write_text(json.dumps(config));old=freeze_sources(path,'p1')
            config['probe_onpolicy_joint_response']=True;path.write_text(json.dumps(config));new=freeze_sources(path,'p1')
        expected={str(Path(name).relative_to(ROOT))for name in sources}-set(old)
        self.assertEqual(set(new)-set(old),expected)
        parameter=str((ROOT/v55.FIT/'parameters.json').relative_to(ROOT))
        self.assertEqual(new[parameter],v55.PARAMETER_SHA256)
        self.assertIn('our_mani/local_mpc_control_20261103/probe_onpolicy_response_v55.py',new)


if __name__=='__main__':unittest.main()
