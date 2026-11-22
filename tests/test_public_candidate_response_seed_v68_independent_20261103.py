"""Independent V68 contracts. Synthetic arithmetic/spies only, no learned forward/Native."""
import ast
from copy import deepcopy
import inspect
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_control_20261103 import response_seed_generator_v68 as generator
from our_mani.local_mpc_control_20261103 import response_seed_hold_planner_v68 as planner_module
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v68 as producer
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_v67 as original_producer
from our_mani.local_mpc_control_20261103 import public_candidate_local_hold_v66 as original_hold
from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner
from our_mani.local_mpc_control_20261103.hold_orientation_guard_prototype_20261103 import HoldMicroOrientationGuard, _identity
from our_mani.local_mpc_20261102.records import plain
from tests.test_public_candidate_local_hold_v66_independent_20261103 import fixture as hold_fixture
from tests.test_hold_response_feedforward_prototype_20261103 import actual as seed_actual


class LinearResponse:
    """Analytic test response; records the entire optimizer trial domain."""
    def __init__(self, execution=None, helper=None, actual=None, fail_step=None):
        self.execution = execution or SimpleNamespace(limits=SimpleNamespace(lower=-np.ones(7), upper=np.ones(7)))
        self.calls=[]; self.model_batch_calls=0; self.modeled_macro_rows=0; self.modeled_micro_rows=0
        self.fail_step=fail_step

    def initialize(self, actual):
        return dict(time=np.array([actual['time']]),q=np.asarray(actual['q'])[None].copy(),
            dq=np.zeros((1,7)),tcp_position=np.zeros((1,3)),object_position=np.zeros((1,3)),
            tcp_quaternion=np.array([[0.,0.,0.,1.]])), dict(counter=np.array([0]),predicted=False)

    def evaluate(self, x, h, goals):
        goals=np.asarray(goals); n=len(goals); step=int(np.asarray(h.get('counter',[0]))[0])
        self.calls.append(dict(goals=goals.copy(),input_time=float(np.asarray(x.get('time',[0.]))[0]),counter=step))
        self.model_batch_calls+=1; self.modeled_macro_rows+=n; self.modeled_micro_rows+=2*n
        now=float(np.asarray(x.get('time',[0.]))[0])+.1
        return dict(x=dict(time=np.full(n,now),q=goals.copy(),dq=np.zeros((n,7)),
            tcp_position=.25*goals[:,:3],object_position=.25*goals[:,:3],
            tcp_quaternion=np.tile([0.,0.,0.,1.],(n,1))),
            h=dict(counter=np.full(n,step+1),predicted=True),
            feasible=np.full(n,step!=self.fail_step,bool),micro_angles_rad=np.zeros((n,2)),
            q_bounds=np.ones(n,bool),actual_q_bounds=np.ones(n,bool),actual_micro_velocity_bounds=np.ones(n,bool))


def fixture():
    _,state,candidates,oracle,base=hold_fixture()
    state.update(Native_object_state_used=False,Native_held_used=False,Native_success_used=False)
    planner=planner_module.ResponseSeedHoldPlannerV68(base.config,execution_model=base.execution_model,
        goal_helper=base.goal_helper,terminal_config=base.terminal_config,backend=oracle)
    return planner,state,candidates,oracle


def prepare_nominals(planner, state, candidates, old):
    basis=np.asarray(state['q'])
    planner._nominals=dict(joint_goal=np.broadcast_to(basis,(3,8,7)).copy(),feasible=np.ones((3,8),bool),
        pose=np.broadcast_to(np.r_[state['tcp_position'],state['tcp_quaternion']],(3,8,7)).copy(),
        time=np.full((3,8),state['time']+.1))
    planner.nominal_receipt={'synthetic':True}; planner.fresh_nominal_qualification={'synthetic':True}
    for row in candidates: row['native_pose_qualified']=True


def proposal(state, candidate_index=0):
    sequence=[dict(joint_goal=np.asarray(state['q'])+np.r_[.01*(k+1),np.zeros(6)])for k in range(8)]
    return dict(candidate_index=candidate_index,feedback_time=state['time'],feedback_sha256=_identity(state),
        complete_feedback_sha256=planner_module._complete_feedback_sha(state),history_warm_used=False,
        sequence_shift_steps=0,sequence=sequence)


def rollout_spy(calls, *, fresh_only=False):
    def run(self, state, candidates, modes, controls, old, committed, fresh):
        n=len(modes)
        calls.append(dict(modes=modes.copy(),controls=controls.copy(),committed=committed,fresh=deepcopy(fresh)))
        q=(np.asarray(state['q'])[None,None,:]+controls.transpose(1,0,2)).copy()
        scores=np.sum(controls*controls,axis=(0,2))
        if fresh_only: scores[:]=np.inf
        if fresh is not None:
            q[fresh['row_index']]=np.array([step['joint_goal']for step in fresh['sequence']])
            scores[fresh['row_index']]=.125
        actions=dict(joint_goal=q,opening=np.zeros((n,8)),phase_index=np.full((n,8),3,np.int64),
            nominal_kind=np.full((n,8),2,np.int64))
        states=dict(q=np.broadcast_to(state['q'],(n,9,7)).copy(),held=np.ones((n,9),bool),
            time=np.broadcast_to(state['time']+.1*np.arange(9),(n,9)).copy())
        return scores,{'synthetic':scores.copy()},states,actions
    return run


class GeneratorContracts(unittest.TestCase):
    def test_inverse_domain_is_static_not_progressive_guess_and_selected_is_evaluated(self):
        model=LinearResponse();model.execution.limits.upper=np.r_[.85,np.full(6,.2)]
        static=np.r_[.8,np.zeros(6)];initial=np.full(7,-5.)
        result=generator.inverse_in_static_domain(model,{}, {},initial,static,np.array([.18,.01,-.01]),np.array([0.,0.,0.,1.]))
        lower=np.maximum(-np.ones(7),static-.30);upper=np.minimum(model.execution.limits.upper,static+.30)
        np.testing.assert_array_equal(result['lower'],lower);np.testing.assert_array_equal(result['upper'],upper)
        for call in model.calls:
            self.assertTrue(np.all(call['goals']>=lower));self.assertTrue(np.all(call['goals']<=upper))
        np.testing.assert_array_equal(model.calls[0]['goals'][0],np.maximum(initial,lower))
        q=result['selected']['joint_goal']
        self.assertTrue(any(np.array_equal(q,row)for call in model.calls for row in call['goals']))
        self.assertFalse(result['selected_action_clipped']);self.assertFalse(result['Native_collision_qualified'])
        self.assertEqual(result['numerical_constants']['iterations'],12)
        self.assertEqual(result['numerical_constants']['damping'],.02)

    def test_all8_fresh_h_pairing_C0_no_shift_and_failure_never_becomes_seed(self):
        actual=seed_actual();before=plain(actual);execution=SimpleNamespace(limits=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7)))
        helper=SimpleNamespace(adapter=SimpleNamespace(solve=lambda q,p,r:dict(joint_goal=np.full((8,7),1.2))))
        made=[]
        def factory(execution,helper,actual):
            value=LinearResponse(execution,helper,actual);made.append(value);return value
        with patch.object(generator,'ModelOnlyHoldResponse',side_effect=factory):
            result=generator.generate_response_seed(execution,helper,actual,np.full(7,.05))
        self.assertTrue(result['model_feasible_horizon']);self.assertEqual(result['sequence_shift_steps'],0)
        self.assertEqual(len(result['proposed_q_sequence']),8);self.assertEqual(plain(actual),before)
        for call in made[0].calls:
            self.assertAlmostEqual(call['input_time'],actual['time']+.1*call['counter'])
            self.assertTrue(np.all(call['goals']>=-.25));self.assertTrue(np.all(call['goals']<=.35))
        np.testing.assert_array_equal(result['proposed_q_sequence'][0],result['steps'][0]['proposed_q'])
        self.assertFalse(np.array_equal(result['proposed_q_sequence'][0],result['proposed_q_sequence'][1]))
        self.assertEqual(result['model_calls']['batch_calls'],len(made[0].calls))
        def failed(execution,helper,actual):return LinearResponse(execution,helper,actual,fail_step=2)
        with patch.object(generator,'ModelOnlyHoldResponse',side_effect=failed):
            invalid=generator.generate_response_seed(execution,helper,actual,np.full(7,.05))
        self.assertFalse(invalid['model_feasible_horizon']);self.assertIsNone(invalid['proposed_q_sequence'])
        stale=deepcopy(actual);stale['actuator_state']['reference_clock']-=.05
        with self.assertRaises(ValueError):generator.generate_response_seed(execution,helper,stale,np.zeros(7))


class FreshCEMContracts(unittest.TestCase):
    def test_invalid_seed_identical_cold_rng_controls_budget_and_action(self):
        planner,state,candidates,_=fixture();old_calls=[];new_calls=[]
        def invalid(self,state,candidates):
            np.random.normal(size=37)  # Must not consume the CEM's independent generator.
            self.response_seed_receipt={'injected':False,'rejection_reason':'synthetic_invalid'};return None
        with patch.object(planner,'_prepare_nominals',side_effect=lambda s,c,o:prepare_nominals(planner,s,c,o)), \
             patch.object(planner,'_compute_response_seed',new=invalid.__get__(planner)), \
             patch.object(planner,'_rollout_batch',new=rollout_spy(new_calls).__get__(planner)):
            new=planner._plan_cold_with_fresh_seed(state,candidates)
        with patch.object(planner,'_prepare_nominals',side_effect=lambda s,c,o:prepare_nominals(planner,s,c,o)), \
             patch.object(planner,'_rollout_batch',new=rollout_spy(old_calls).__get__(planner)):
            old=JointGoalCEMPlanner.plan(planner,state,candidates,warm_start=None)
        self.assertEqual([len(c['modes'])for c in new_calls],[88,88,80])
        self.assertEqual(sum(len(c['modes'])for c in new_calls),256)
        for a,b in zip(new_calls,old_calls):
            np.testing.assert_array_equal(a['controls'],b['controls']);np.testing.assert_array_equal(a['modes'],b['modes'])
            self.assertIsNone(a['fresh']);self.assertIsNone(a['committed'])
        np.testing.assert_array_equal(new['action']['joint_goal'],old['action']['joint_goal'])
        self.assertEqual(new['cost'],old['cost'])

    def test_one_random_row_only_C0_exact_unique_feasible_best_and_no_fake_elite(self):
        planner,state,candidates,_=fixture();calls=[];fixed=proposal(state)
        def compute(self,state,candidates):self.response_seed_receipt={'injected':True};return deepcopy(fixed)
        with patch.object(planner,'_prepare_nominals',side_effect=lambda s,c,o:prepare_nominals(planner,s,c,o)), \
             patch.object(planner,'_compute_response_seed',new=compute.__get__(planner)), \
             patch.object(planner,'_rollout_batch',new=rollout_spy(calls,fresh_only=True).__get__(planner)):
            result=planner._plan_cold_with_fresh_seed(state,candidates)
        self.assertEqual([len(c['modes'])for c in calls],[88,88,80])
        self.assertEqual(calls[0]['fresh']['row_index'],1)
        self.assertTrue(any(calls[0]['controls'][:,1].ravel()!=0))
        self.assertIsNone(calls[1]['fresh']);self.assertIsNone(calls[2]['fresh'])
        for call in calls:np.testing.assert_array_equal(call['controls'][:,0],np.zeros((8,7)))
        self.assertEqual(result['diagnostics']['fresh_response_seed_evaluations_counted'],1)
        self.assertEqual(result['budget']['sequences'],256)
        np.testing.assert_array_equal(result['action']['joint_goal'],fixed['sequence'][0]['joint_goal'])
        for k in range(8):np.testing.assert_array_equal(result['sequence'][k]['joint_goal'],fixed['sequence'][k]['joint_goal'])

    def test_current_binding_masks_and_complete_sequence_cannot_be_bypassed(self):
        planner,state,candidates,_=fixture();prepare_nominals(planner,state,candidates,state['phase_state'])
        value=proposal(state);value['row_index']=1;modes=np.repeat(np.arange(8),11)
        planner._validate_fresh_rollout_binding(state,candidates,modes,value)
        for change in ({'sequence_shift_steps':1},{'history_warm_used':True},{'feedback_time':-.05},{'row_index':12}):
            bad=dict(value,**change)
            with self.assertRaises(ValueError):planner._validate_fresh_rollout_binding(state,candidates,modes,bad)
        bad=deepcopy(value);bad['sequence'][7]['joint_goal']=state['q']+.31
        with self.assertRaises(ValueError):planner._validate_fresh_rollout_binding(state,candidates,modes,bad)
        candidates[0]['native_pose_qualified']=False
        with self.assertRaises(ValueError):planner._validate_fresh_rollout_binding(state,candidates,modes,value)
        candidates[0]['native_pose_qualified']=True;planner._nominals['feasible'][2,0]=False
        with self.assertRaises(ValueError):planner._validate_fresh_rollout_binding(state,candidates,modes,value)

    def test_history_warm_ignored_fresh_guard_action_identity_and_exception_restore(self):
        planner,state,candidates,_=fixture();execution=planner.execution_model;q=state['q'].copy();action={'joint_goal':q,'opening':0.};seen=[]
        def cold(actual,candidates):
            seen.append(actual['time']);self.assertIs(type(planner.execution_model),HoldMicroOrientationGuard)
            planner.response_seed_receipt={'injected':False}
            return dict(action=action,diagnostics={},warm_start={'sequence':[]})
        with patch.object(planner,'_plan_cold_with_fresh_seed',side_effect=cold):
            result=planner.plan(state,candidates,warm_start={'nominal_basis_version':'future_or_old','time':999.,'sequence':['bad']})
        self.assertEqual(seen,[state['time']]);self.assertIs(result['action'],action);self.assertIs(result['action']['joint_goal'],q)
        self.assertTrue(result['diagnostics']['V68_hold_control']['incoming_warm_discarded'])
        self.assertIs(planner.execution_model,execution)
        with patch.object(planner,'_plan_cold_with_fresh_seed',side_effect=RuntimeError('isolated synthetic failure')):
            with self.assertRaises(RuntimeError):planner.plan(state,candidates)
        self.assertIs(planner.execution_model,execution)

    def test_rollout_cost_feasibility_and_runner_dispatch_math_unchanged(self):
        def node(function):return ast.parse(textwrap.dedent(inspect.getsource(function)))
        class RemoveInjection(ast.NodeTransformer):
            def visit_Assign(self,n):
                if any(isinstance(t,ast.Name)and t.id in ('warm_mask','fresh_mask')for t in n.targets):return None
                return self.generic_visit(n)
            def visit_If(self,n):
                text=ast.unparse(n.test)
                if text in ('fixed_sequence is not None','np.any(warm_mask)','np.any(fresh_mask)'):return None
                return self.generic_visit(n)
        self.assertEqual(ast.dump(RemoveInjection().visit(node(planner_module.ResponseSeedHoldPlannerV68._rollout_batch)),include_attributes=False),
            ast.dump(RemoveInjection().visit(node(JointGoalCEMPlanner._rollout_batch)),include_attributes=False))
        text=textwrap.dedent(inspect.getsource(producer._V68V4Dispatch.decide)).replace('ResponseSeedHoldPlannerV68','LocalHoldPlannerV66')
        self.assertEqual(ast.dump(ast.parse(text),include_attributes=False),ast.dump(node(original_hold._V66V4Dispatch.decide),include_attributes=False))
        self.assertIs(producer.ObservedPublicCandidateResponseSeedV68Backend.sample,original_producer.ObservedPublicCandidateScope12V67Backend.sample)
        before=next(n for n in ast.walk(node(original_producer.run_public_candidate_scope12_v67))if isinstance(n,ast.While))
        after=next(n for n in ast.walk(node(producer.run_public_candidate_response_seed_v68))if isinstance(n,ast.While))
        class Versions(ast.NodeTransformer):
            def visit_Constant(self,n):
                if isinstance(n.value,str):n.value=n.value.replace('v68','v67')
                return n
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(Versions().visit(after),include_attributes=False))


if __name__=='__main__':unittest.main()
