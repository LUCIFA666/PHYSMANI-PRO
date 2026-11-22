"""Independent pure V71 selection contracts; no learned model, data or Native."""
from copy import deepcopy
import ast
import inspect
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_local_mpc_forecast_control_20261103 import bound_fixture
from our_mani.local_mpc_control_20261103.forecast_control import (
    BoundForecastExecutionModel,ForecastJointGoalCEMPlanner)
from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner
from our_mani.local_mpc_control_20261103 import lazy_free_joint_goal_cem_v71 as lazy
from our_mani.local_mpc_control_20261103 import public_candidate_lazy_free_v71 as producer
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v69 as control
from our_mani.local_mpc_control_20261103.response_seed_hold_planner_v69 import ResponseSeedHoldPlannerV69
from our_mani.local_mpc_20261102 import mpc


class QueryOnlyBackend:
    def __init__(self,now=0.,reject=0):self.now=now;self.reject=reject;self.queries=[]
    def clock(self):return self.now
    def probe_joint_collision(self,q):
        self.queries.append(np.asarray(q).copy())
        return self.reject=='all' or self.reject==1 and len(self.queries)==1


def public_fixture(reject=0):
    base,state,candidates,forecast,_=bound_fixture()
    state.update(grasp_motion_mode='FREE_GRU',Native_object_state_used=False,Native_held_used=False,Native_success_used=False)
    bound=BoundForecastExecutionModel.from_public_bridge(base.execution_model,dict(available=True,free_forecast_available=True,
        mode='FREE',timestamp=state['time'],forecast=forecast,Native_inputs_used=False,future_observations_used=False,
        observations_fabricated=False,rotation_observable=False))
    backend=QueryOnlyBackend(state['time'],reject)
    planner=lazy.LazyQualifiedForecastJointGoalCEMPlannerV71(execution_model=bound,goal_helper=base.goal_helper,
        terminal_config=base.terminal_config,backend=backend)
    return planner,state,candidates,backend


def equal(test,left,right):
    if isinstance(left,np.ndarray)or isinstance(right,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left:equal(test,left[key],right[key])
    elif isinstance(left,(tuple,list)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):equal(test,a,b)
    else:test.assertEqual(left,right)


class LazyFreeIndependentContracts(unittest.TestCase):
    def test_rank0_matches_original_all_controls_selection_phase_and_warm(self):
        original,state,candidates,forecast,_=bound_fixture()
        state.update(grasp_motion_mode='FREE_GRU',Native_object_state_used=False,Native_held_used=False,Native_success_used=False)
        result=dict(available=True,free_forecast_available=True,mode='FREE',timestamp=state['time'],forecast=forecast,
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,rotation_observable=False)
        old=ForecastJointGoalCEMPlanner(execution_model=BoundForecastExecutionModel.from_public_bridge(original.execution_model,result),
            goal_helper=original.goal_helper,terminal_config=original.terminal_config)
        new=lazy.LazyQualifiedForecastJointGoalCEMPlannerV71(execution_model=BoundForecastExecutionModel.from_public_bridge(original.execution_model,result),
            goal_helper=original.goal_helper,terminal_config=original.terminal_config,backend=QueryOnlyBackend())
        old_calls=[];new_calls=[]
        def observe(method,ledger):
            def call(*args):
                ledger.append((args[2].copy(),args[3].copy()))
                return method(*args)
            return call
        with patch.object(old,'_rollout_batch',side_effect=observe(old._rollout_batch,old_calls)):
            a=old.plan(deepcopy(state),deepcopy(candidates))
        with patch.object(new,'_rollout_batch',side_effect=observe(new._rollout_batch,new_calls)):
            b=new.plan(deepcopy(state),deepcopy(candidates))
        equal(self,old_calls,new_calls)
        self.assertEqual([len(c[0])for c in new_calls],[88,88,80])
        for key in ('action','sequence','selected_candidate_id','phase_state','warm_start','cost','cost_terms','predicted_states','budget'):
            equal(self,a[key],b[key])
        diag=b['diagnostics']['V71_lazy_free_qualification']
        self.assertEqual(diag['selected_rank'],0)
        self.assertEqual(diag['Native_collision_queries'],8)
        self.assertIsNone(new._caller_state)

    def test_rank1_replaces_complete_bundle_before_phase_and_warm(self):
        planner,state,candidates,backend=public_fixture(reject=1)
        captured={};real=lazy.select_first_qualified
        def select(pool,*args,**kwargs):
            captured['pool']=pool.ranked()
            result=real(pool,*args,**kwargs);captured['selection']=deepcopy(result['selection_bundle'])
            return result
        before=deepcopy(state)
        with patch.object(lazy,'select_first_qualified',side_effect=select):result=planner.plan(state,candidates)
        selected=captured['selection']
        self.assertEqual(selected['selected_rank'],1)
        candidate=next(c for c in candidates if c['id']==selected['candidate_id'])
        self.assertEqual(result['selected_candidate_id'],selected['candidate_id'])
        equal(self,result['sequence'],selected['sequence'])
        equal(self,result['action'],selected['sequence'][0])
        equal(self,result['predicted_states'],selected['predicted_states'])
        equal(self,result['cost_terms'],selected['cost_terms'])
        self.assertEqual(result['cost'],selected['cost'])
        expected=mpc.update_phase(state,candidate,before['phase_state'],planner.config)
        equal(self,result['phase_state'],expected)
        equal(self,result['warm_start']['phase_state'],expected)
        equal(self,result['warm_start']['sequence'],selected['sequence'])
        self.assertEqual(result['warm_start']['selected_candidate_id'],selected['candidate_id'])
        self.assertEqual(len(backend.queries),9)
        equal(self,state,before)

    def test_all_rejected_preserves_real_phase_without_warm_or_action(self):
        planner,state,candidates,backend=public_fixture(reject='all');before=deepcopy(state)
        result=planner.plan(state,candidates)
        self.assertIsNone(result['action']);self.assertIsNone(result['selected_candidate_id']);self.assertIsNone(result['warm_start'])
        self.assertEqual(result['sequence'],[])
        equal(self,result['phase_state'],before['phase_state'])
        self.assertEqual(result['diagnostics']['phase_real_updates'],0)
        self.assertEqual(result['diagnostics']['reason'],'all_retained_first_actions_rejected_by_original_qualification')
        self.assertEqual(len(backend.queries),8)
        self.assertEqual(result['budget']['sequences'],256)
        equal(self,state,before)
        self.assertIsNone(planner._caller_state)

    def test_original_loop_rng_and_elite_AST_exact_after_pool_instrumentation(self):
        def node(function):return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        old=node(JointGoalCEMPlanner.plan);new=node(lazy._LazyJointGoalCEMPlannerV71.plan)
        def loop(function):return next(x for x in function.body if isinstance(x,ast.For)and isinstance(x.target,ast.Tuple)and any(isinstance(v,ast.Name)and v.id=='iteration'for v in x.target.elts))
        original=loop(old);changed=deepcopy(loop(new))
        changed.body=[x for x in changed.body if not (isinstance(x,ast.Expr)and isinstance(x.value,ast.Call)
            and isinstance(x.value.func,ast.Attribute)and (x.value.func.attr=='_assert_caller_current'
                or isinstance(x.value.func.value,ast.Name)and x.value.func.value.id=='pool'and x.value.func.attr=='offer_batch'))]
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(changed,include_attributes=False))
        def rng(function):return next(x for x in function.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='rng'for t in x.targets))
        self.assertEqual(ast.dump(rng(old),include_attributes=False),ast.dump(rng(new),include_attributes=False))
        self.assertIs(lazy.LazyQualifiedForecastJointGoalCEMPlannerV71._rollout_batch,JointGoalCEMPlanner._rollout_batch)
        self.assertIs(lazy.LazyQualifiedForecastJointGoalCEMPlannerV71._prepare_nominals,ForecastJointGoalCEMPlanner._prepare_nominals)

    def test_dispatch_only_declared_FREE_planner_and_backend_delta(self):
        def function(fn):return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        old=function(control._V69V4Dispatch.decide);new=function(producer._V71V4Dispatch.decide)
        class Normalize(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='LazyQualifiedForecastJointGoalCEMPlannerV71':node.id='ForecastJointGoalCEMPlanner'
                return node
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name)and node.func.id=='planner_class':
                    extras=[k for k in node.keywords if k.arg is None]
                    self_outer.assertEqual(len(extras),1)
                    expected=ast.parse("({'backend':backend} if mode=='FREE_GRU' else {})",mode='eval').body
                    self_outer.assertEqual(ast.dump(extras[0].value,include_attributes=False),ast.dump(expected,include_attributes=False))
                    node.keywords=[k for k in node.keywords if k.arg is not None]
                return node
        self_outer=self;new=Normalize().visit(new)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        self.assertIs(producer.ResponseSeedHoldPlannerV69,ResponseSeedHoldPlannerV69)
        self.assertIs(producer.JointGoalCEMPlanner,JointGoalCEMPlanner)
        mro=producer.CandidateLazyFreeV71Coordinator.mro()
        self.assertLess(mro.index(producer._V71V4Dispatch),mro.index(control._V69V4Dispatch))

    def test_nonpublic_mode_and_clock_failure_reject_without_stale_context(self):
        planner,state,candidates,backend=public_fixture()
        for key,value in (('held',True),('grasp_motion_mode','ATTACHED_PUBLIC'),('Native_held_used',True)):
            bad=deepcopy(state);bad[key]=value
            with self.assertRaises(ValueError):planner.plan(bad,candidates)
        backend.now=True
        with self.assertRaises(ValueError):planner.plan(state,candidates)
        self.assertIsNone(planner._caller_state)
        self.assertEqual(backend.queries,[])


if __name__=='__main__':unittest.main()
