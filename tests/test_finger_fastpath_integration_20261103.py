"""Real frozen CPU factory integration; no Native, fitting or TEST inputs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_control_20261103.coupled_response import CoupledWithFingerResponse
from our_mani.local_mpc_control_20261103.finger_response_fastpath_20261103 import FrozenFingerOnlyResponse
from our_mani.local_mpc_control_20261103.finger_fastpath_integration_20261103 import enable_finger_fastpath
from our_mani.local_mpc_control_20261103.probe_onpolicy_response_v55 import build_onpolicy_probe_execution

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT/'runtime/framework/local_mpc_20261102/P2_onpolicy_probe_TRAIN8_20261103_v1/00_train_static'


def assert_equal_tree(test, a, b):
    if isinstance(a, np.ndarray):
        test.assertIsInstance(b, np.ndarray); np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a: assert_equal_tree(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(type(a), type(b)); test.assertEqual(len(a), len(b))
        for x,y in zip(a,b): assert_equal_tree(test,x,y)
    else:
        test.assertEqual(a,b)


class FingerFastpathIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        snapshot = json.loads((TRAIN/'initial_snapshot.json').read_text())
        native = snapshot['joint_limits']
        cls.limits = JointLimits(native['lower'],native['upper'],native['velocity'],acceleration=12.,jerk=1000.)
        # This input is an already-recorded TRAIN public scene/robot/reference
        # state. Neither held/success GT nor any TEST result is used to choose it.
        with (TRAIN/'public_grasp_candidates_v5_decisions.jsonl').open() as stream:
            cls.state = next(json.loads(line)['decision']['state'] for line in stream
                if json.loads(line)['decision'].get('state') is not None)

    def core(self):
        core,helper,provenance = build_planner(ROOT,self.limits,controller='coupled',action_space='joint',cpu_threads=2)
        return core,helper,provenance

    def test_same_model_weights_callbacks_and_original_evidence_are_unchanged(self):
        core,helper,provenance = self.core(); model = core.execution_model
        old = model.transition_batch; original = old.finger_response
        previous_provenance = deepcopy(provenance); previous_evidence = deepcopy(original.evidence)
        previous_metadata = deepcopy(original.metadata)
        weights = {key:value.clone() for key,value in original.model.state_dict().items()}
        callbacks = {name:getattr(model,name) for name in ('reference_step','limits','fk_batch','config','latent_initializer','model_id')}
        updated = enable_finger_fastpath(model,provenance)
        self.assertIs(type(model.transition_batch),CoupledWithFingerResponse)
        self.assertIs(model.transition_batch.joint_response,old.joint_response)
        view = model.transition_batch.finger_response
        self.assertIs(type(view),FrozenFingerOnlyResponse)
        self.assertIs(view.original,original); self.assertIs(view.model,original.model)
        self.assertIs(core.planner.execution_model,model)
        for name,value in callbacks.items(): self.assertIs(getattr(model,name),value)
        for key,value in original.model.state_dict().items(): self.assertTrue(torch.equal(value,weights[key]))
        self.assertEqual(original.evidence,previous_evidence)
        self.assertEqual(original.metadata,previous_metadata)
        self.assertEqual(provenance,previous_provenance)
        self.assertIsNot(updated,provenance)
        self.assertTrue(view.evidence['finger_computation_view']['production_integrated'])
        self.assertTrue(updated['finger_response_fastpath']['runtime_enabled'])
        self.assertFalse(updated['finger_response_fastpath']['finger_response_requalified'])
        self.assertTrue(model.transition_batch.evidence['finger_response_evidence']['finger_computation_view']['production_integrated'])

    def test_real_model_rich_response_and_two_tick_outputs_are_exact(self):
        core,_,provenance = self.core(); model = core.execution_model
        state = deepcopy(self.state); before = deepcopy(state)
        x,h = model.initialize_batch(state,7)
        x['finger_open_amounts'][:] = np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.],[.2,.65],[.4,.9],[.7,.1]])
        refs = dict(q_ref=x['q']+.001,dq_ref=x['dq'].copy(),ddq_ref=np.zeros((7,7)),gripper_target=np.arange(7)%2)
        latent = {'explicit_CPU_fixture':np.arange(7),'nested':[{'time':state['time']}]}
        original = model.transition_batch
        expected_rich = original.predict_execution(x,refs,latent,.05,held_anchor=x['held'])
        action = dict(joint_goal=x['q']+.001,opening=(np.arange(7)%2).astype(float))
        expected_step = model.step_batch(deepcopy(x),deepcopy(action),deepcopy(h))
        enable_finger_fastpath(model,provenance)
        actual_rich = model.transition_batch.predict_execution(x,refs,latent,.05,held_anchor=x['held'])
        actual_step = model.step_batch(deepcopy(x),deepcopy(action),deepcopy(h))
        assert_equal_tree(self,actual_rich,expected_rich)
        assert_equal_tree(self,actual_step,expected_step)
        assert_equal_tree(self,state,before)
        np.testing.assert_array_equal(latent['explicit_CPU_fixture'],np.arange(7))
        self.assertFalse(actual_step[0]['held'].any())

    def test_unknown_wrappers_duplicate_and_bad_identity_fail_before_mutation(self):
        core,_,provenance = self.core(); model = core.execution_model; old = model.transition_batch
        with self.assertRaises(TypeError): enable_finger_fastpath(SimpleNamespace(transition_batch=old),provenance)
        model.transition_batch = SimpleNamespace(finger_response=old.finger_response)
        with self.assertRaisesRegex(TypeError,'not flattened'): enable_finger_fastpath(model,provenance)
        model.transition_batch = old
        bad = deepcopy(provenance); bad['checkpoint']['sha256'] = '0'*64
        with self.assertRaises(ValueError): enable_finger_fastpath(model,bad)
        self.assertIs(model.transition_batch,old)
        flagged = deepcopy(provenance); flagged['finger_response_fastpath']={'runtime_enabled':True}
        with self.assertRaises(ValueError): enable_finger_fastpath(model,flagged)
        self.assertIs(model.transition_batch,old)
        enabled = enable_finger_fastpath(model,provenance); replacement=model.transition_batch
        for receipt in (provenance,enabled):
            with self.assertRaises(ValueError): enable_finger_fastpath(model,receipt)
            self.assertIs(model.transition_batch,replacement)

    def test_fresh_cores_do_not_share_models_or_h_and_probe_derivation_keeps_view(self):
        first,_,p1 = self.core(); second,_,p2 = self.core()
        model1,model2 = first.execution_model,second.execution_model
        enable_finger_fastpath(model1,p1); enable_finger_fastpath(model2,p2)
        view1,view2 = model1.transition_batch.finger_response,model2.transition_batch.finger_response
        self.assertIsNot(view1,view2); self.assertIsNot(view1.model,view2.model)
        probe,_ = build_onpolicy_probe_execution(model1)
        self.assertIs(probe.transition_batch.finger_response,view1)
        self.assertIsNot(probe.transition_batch.joint_response,model1.transition_batch.joint_response)
        x,h = model1.initialize_batch(self.state,1)
        later = deepcopy(self.state); later['time'] += .1; later['q'][0] += 1e-6
        later['actuator_state']['reference_clock'] = later['time']
        y,hy = model1.initialize_batch(later,1)
        self.assertFalse(h['predicted']);self.assertFalse(hy['predicted'])
        self.assertIsNot(x['q'],y['q']);self.assertIsNot(h,hy)
        np.testing.assert_array_equal(y['q'][0],later['q'])
        np.testing.assert_array_equal(y['finger_open_amounts'][0],later['finger_open_amounts'])
        self.assertEqual(hy['feedback_time'],later['time'])
        self.assertFalse(any(name in vars(view1) for name in ('x','h','actual_state','last_actual')))


if __name__ == '__main__': unittest.main()
