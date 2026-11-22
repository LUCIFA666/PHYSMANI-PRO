"""Small v54 production-hook CPU audit, frozen real response weights only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.probe_deadline_v54 import FixedDeadlineVerificationProbePlanner
from our_mani.local_mpc_control_20261103.probe_nominal_cache_v53 import CachedNominalVerificationProbePlanner
from our_mani.local_mpc_control_20261103.joint_goal_cem import norm,qlog,qmul,qinv
from our_mani.local_mpc_20261102.records import freeze_sources,sha
from scripts.replay_probe_nominal_counterfactual_20261103 import candidate_ledger,numpy_spec
from tests import test_local_mpc_probe_nominal_cache_v53_integration_20261103 as prior

ROOT=Path(__file__).resolve().parents[1]


class FixedDeadlineHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior.ProductionHookTests.setUpClass()
        cls.harness=prior.ProductionHookTests(methodName='test_default_and_false_delegate_to_original_components')
        cls.harness.core=prior.ProductionHookTests.core;cls.harness.helper=prior.ProductionHookTests.helper
        cls.harness.inference=prior.ProductionHookTests.inference
        rows=[json.loads(line) for line in (prior.CASE/'public_grasp_v41_decisions.jsonl').read_text().splitlines()]
        cls.first=next(row for row in rows if row['decision']['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE')
        cls.later=next(row for row in rows if abs(row['time']-cls.first['time']-.5)<1e-8)

    def components(self,coordinator,row,spec):
        actual=deepcopy(row['decision']['state']);planning=candidate_ledger(row)
        for c in planning:c['parallel_gripper_symmetry_choice']=0
        oracle=prior.SyntheticCollisionOnly()
        candidates,receipt,planner=coordinator._verification_components(oracle,actual,planning,coordinator.core.execution_model,spec)
        return actual,candidates,receipt,planner

    def test_default_false_is_v53_and_invalid_combination_rejected(self):
        for options in ({},{'probe_terminal_at_fixed_deadline':False}):
            coordinator=self.harness.coordinator(cache_fixed_probe_nominal_ik=True,**options)
            spec=numpy_spec(self.first['decision']['probe']);spec['deadline']=spec['trigger_time']+.8
            _,_,_,planner=self.components(coordinator,self.first,spec)
            self.assertIs(type(planner),CachedNominalVerificationProbePlanner)
            self.assertFalse(coordinator.probe_terminal_at_fixed_deadline)
            self.assertEqual(len(coordinator.probe_cache_sources),8)
        with self.assertRaisesRegex(ValueError,'requires the explicit cached'):
            self.harness.coordinator(cache_fixed_probe_nominal_ik=False,probe_terminal_at_fixed_deadline=True)
        for invalid in (1,None,'true'):
            with self.assertRaisesRegex(ValueError,'explicit boolean fixed probe terminal'):
                self.harness.coordinator(cache_fixed_probe_nominal_ik=True,probe_terminal_at_fixed_deadline=invalid)

    def test_true_uses_actual_ninth_source_and_conditional_source_inventory(self):
        coordinator=self.harness.coordinator(cache_fixed_probe_nominal_ik=True,probe_terminal_at_fixed_deadline=True)
        self.assertEqual(len(coordinator.probe_cache_sources),9)
        for name,expected in coordinator.probe_cache_sources.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),expected)
        config=json.loads((ROOT/'config/framework/P4_student_closedloop_devVAL_v52_v1/00_val_static.json').read_text())
        with TemporaryDirectory(prefix='v54_CPU_inventory_',dir=ROOT/'config/framework') as directory:
            path=Path(directory)/'config.json';config.update(cache_fixed_probe_nominal_ik=True,probe_terminal_at_fixed_deadline=False)
            path.write_text(json.dumps(config));old=freeze_sources(path,'p1')
            config['probe_terminal_at_fixed_deadline']=True;path.write_text(json.dumps(config));new=freeze_sources(path,'p1')
            name='our_mani/local_mpc_control_20261103/probe_deadline_v54.py'
            self.assertEqual(set(new)-set(old),{name});self.assertEqual(new[name],sha(ROOT/name))
            config['cache_fixed_probe_nominal_ik']=False;path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'requires explicit cached'):
                freeze_sources(path,'p1')

    def test_real_state_response_first_H8_exact_then_phi_at_fixed_remaining_300ms(self):
        coordinator=self.harness.coordinator(cache_fixed_probe_nominal_ik=True,probe_terminal_at_fixed_deadline=True)
        # Recorded feedback is unchanged. The test explicitly supplies the
        # current .8s protocol, rather than claiming the old1.2s was executed.
        spec=numpy_spec(self.first['decision']['probe']);spec['deadline']=spec['trigger_time']+.8
        state,candidates,_,new=self.components(coordinator,self.first,spec)
        self.assertIs(type(new),FixedDeadlineVerificationProbePlanner)
        old=CachedNominalVerificationProbePlanner(new.config,execution_model=new.execution_model,goal_helper=new.goal_helper,
            constraint_cost=new.constraint_cost,terminal_config=new.terminal_config,probe_spec=new.probe_spec,
            nominal_cache=new.nominal_cache,source_sha256=new.source_sha256,nominal_mask=new.nominal_mask,qualification=new.qualification)
        first_new=new.plan(state,candidates);first_old=old.plan(state,candidates)
        self.assertEqual(first_new['cost'],first_old['cost']);self.assertEqual(first_new['budget'],first_old['budget'])
        np.testing.assert_array_equal(first_new['action']['joint_goal'],first_old['action']['joint_goal'])
        self.assertEqual(first_new['diagnostics']['probe_terminal_index'],8)
        state,candidates,_,later=self.components(coordinator,self.later,spec)
        result=later.plan(state,candidates)
        self.assertEqual(result['diagnostics']['probe_terminal_index'],3)
        self.assertAlmostEqual(result['diagnostics']['probe_terminal_time'],spec['deadline'])
        endpoint=result['predicted_states'][3]
        self.assertAlmostEqual(endpoint['time'],spec['deadline'])
        error=(endpoint['tcp_position']-spec['target_position'])/.003
        angle=norm(qlog(qmul(endpoint['tcp_quaternion'],qinv(spec['target_quaternion']))))
        expected=5.*(np.sum(error**2)+.2*(angle/.04)**2)
        self.assertAlmostEqual(result['cost_terms']['probe_terminal_phi'],expected,places=11)
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80]);self.assertEqual(result['budget']['horizon'],8)
        self.assertEqual(result['action']['opening'],0.)
        self.assertEqual(coordinator.probe_nominal_cache.solve_count,1)
        self.assertTrue(result['diagnostics']['eight_running_costs_unchanged'])
        self.assertTrue(result['diagnostics']['fixed_probe_deadline_unchanged'])


if __name__=='__main__':unittest.main()
