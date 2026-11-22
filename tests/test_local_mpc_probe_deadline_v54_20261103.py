"""Fixed-deadline terminal-only contracts; CPU synthetic data, no fit/Native."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_control_20261103.probe_deadline_v54 import FixedDeadlineVerificationProbePlanner
from our_mani.local_mpc_control_20261103.probe_nominal_cache_v53 import CachedNominalVerificationProbePlanner,ProbeNominalQCache
from tests.test_local_mpc_probe_nominal_cache_v53_20261103 import fixture,qualified,SyntheticNoStepCollision


def planners(time=0.):
    base,state,candidates=fixture();state['time']=time
    state['actuator_state']['reference_clock']=time
    spec=deepcopy(base.probe_spec);spec.update(attempt_id=1,deadline=spec['trigger_time']+.8)
    old,rows,receipt,oracle=qualified(base,state,candidates,ProbeNominalQCache(),spec=spec)
    new=FixedDeadlineVerificationProbePlanner(old.config,execution_model=old.execution_model,
        constraint_cost=old.constraint_cost,goal_helper=old.goal_helper,terminal_config=old.terminal_config,
        probe_spec=spec,nominal_cache=old.nominal_cache,source_sha256=old.source_sha256,
        nominal_mask=old.nominal_mask,qualification=receipt)
    return old,new,state,rows,oracle


class ProbeDeadlineTests(unittest.TestCase):
    def test_first_plan_at_original_H8_is_exactly_equivalent(self):
        old,new,state,rows,_=planners()
        before=deepcopy(new.probe_spec)
        a=old.plan(state,rows);b=new.plan(state,rows)
        self.assertEqual(a['cost'],b['cost']);self.assertEqual(a['budget'],b['budget'])
        self.assertEqual(b['budget']['sequences'],256);self.assertEqual(b['budget']['horizon'],8)
        self.assertEqual(b['diagnostics']['execution_model_calls'],24)
        np.testing.assert_array_equal(a['action']['joint_goal'],b['action']['joint_goal'])
        self.assertEqual(a['action']['opening'],b['action']['opening'])
        for left,right in zip(a['sequence'],b['sequence']):
            np.testing.assert_array_equal(left['joint_goal'],right['joint_goal'])
        for key,value in a['cost_terms'].items():self.assertEqual(value,b['cost_terms'][key])
        self.assertEqual(b['cost_terms']['probe_terminal_phi_delta'],0.)
        self.assertEqual(b['diagnostics']['probe_terminal_index'],8)
        self.assertEqual(new.probe_spec['deadline'],before['deadline'])

    def test_earlier_deadline_terminal_flips_late_arrival_without_changing_running_or_actions(self):
        _,planner,state,rows,_=planners(.5)
        anchor=np.asarray(planner.probe_spec['anchor_position']);target=np.asarray(planner.probe_spec['target_position'])
        states=dict(tcp_position=np.tile(target,(2,9,1)),
                    tcp_quaternion=np.tile(planner.probe_spec['target_quaternion'],(2,9,1)))
        states['tcp_position'][0,3]=anchor+[0.,0.,.004]
        states['tcp_position'][1,8]=target+[0.,0.,.003]
        parts=dict(probe_position=np.array([1.,1.]),probe_orientation=np.zeros(2),
                   smoothness=np.zeros(2),effort=np.zeros(2),constraint=np.zeros(2),
                   probe_terminal_phi=np.array([0.,5.]))
        original=np.array([1.,6.]);actions={'joint_goal':np.zeros((2,8,7))}
        with patch.object(CachedNominalVerificationProbePlanner,'_rollout_batch',
                          return_value=(original,parts,states,actions)) as parent:
            cost,out,returned_states,returned_actions=planner._rollout_batch(
                state,rows,np.array([0,1]),np.zeros((8,2,7)),state['phase_state'],None,None)
        self.assertEqual(parent.call_count,1)
        self.assertEqual(int(np.argmin(original)),0);self.assertEqual(int(np.argmin(cost)),1)
        self.assertIs(returned_states,states);self.assertIs(returned_actions,actions)
        for key in ('probe_position','probe_orientation','smoothness','effort','constraint'):
            np.testing.assert_array_equal(out[key],parts[key])
        np.testing.assert_array_equal(out['probe_terminal_index'],[3,3])
        np.testing.assert_allclose(out['probe_terminal_time'],.8)
        np.testing.assert_allclose(out['probe_terminal_original_time'],1.3)

    def test_expired_future_or_ambiguous_time_rejected_before_model_or_qualification(self):
        _,planner,state,rows,_=planners()
        for time in (.8,.9,-.1,.05,.55,float('nan'),True):
            with self.subTest(time=time):
                bad=deepcopy(state);bad['time']=time
                with patch.object(CachedNominalVerificationProbePlanner,'plan',side_effect=AssertionError('expired model call')):
                    with self.assertRaises(ValueError):planner.plan(bad,rows)
        for changes in ({'deadline':.85},{'deadline':1.2},{'trigger_time':.05}, {'deadline':None}):
            with self.subTest(changes=changes):
                previous=deepcopy(planner.probe_spec);planner.probe_spec.update(changes)
                with self.assertRaises(ValueError):planner.plan(state,rows)
                planner.probe_spec=previous

    def test_actual_state_h_stays_fresh_and_does_not_import_future_truth(self):
        _,planner,state,rows,_=planners(.1)
        bad=deepcopy(state);bad['actuator_state']['reference_clock']=0.
        with self.assertRaisesRegex(ValueError,'fresh same-time'):planner.plan(bad,rows)
        result=planner.plan(state,rows)
        np.testing.assert_array_equal(result['predicted_states'][0]['q'],state['q'])
        self.assertEqual(result['predicted_states'][0]['time'],.1)
        self.assertFalse(result['diagnostics']['virtual_h_used_as_next_real_feedback'])
        self.assertEqual(result['diagnostics']['probe_terminal_index'],7)

    def test_selected_action_is_model_first_action_exact_FK_and_original_collision_path(self):
        _,planner,state,rows,_=planners(.5)
        result=planner.plan(state,rows);action=result['action']
        self.assertIsNotNone(action)
        np.testing.assert_array_equal(action['joint_goal'],result['sequence'][0]['joint_goal'])
        np.testing.assert_array_equal(action['tcp_position'],planner.goal_helper.kinematics(action['joint_goal'])[:3,3])
        self.assertEqual(action['opening'],0.)
        oracle=SyntheticNoStepCollision()
        q,reason,receipt=planner.goal_helper.qualify_action(oracle,state,action,require_predicted_joint_goal=True)
        self.assertIsNone(reason);np.testing.assert_array_equal(q,action['joint_goal'])
        self.assertEqual(receipt['DLS_solves'],0);self.assertEqual(len(oracle.calls),8)
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80])
        self.assertFalse(result['diagnostics']['selected_action_postprocessed'])
        self.assertTrue(result['diagnostics']['eight_running_costs_unchanged'])
        self.assertTrue(result['diagnostics']['fixed_probe_deadline_unchanged'])


if __name__=='__main__':unittest.main()
