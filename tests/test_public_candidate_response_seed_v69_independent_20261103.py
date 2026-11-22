"""V69 raw-public/numeric boundary negatives and actual-sanitizer composition.

The generator is real. Only its execution response is an analytic synthetic
test model, and CEM rollouts are explicit spies. No learned forward/Native.
"""
import ast
from copy import deepcopy
import inspect
import textwrap
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_control_20261103 import response_seed_hold_planner_v69 as subject
from our_mani.local_mpc_control_20261103 import response_seed_hold_planner_v68 as previous
from our_mani.local_mpc_control_20261103 import response_seed_generator_v68 as generator
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v69 as producer
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v68 as old_producer
from our_mani.local_mpc_p2_20261103.execution_models import validate_feedback_state
from our_mani.local_mpc_20261102.records import plain
from tests.test_public_candidate_response_seed_v68_independent_20261103 import (
    fixture as old_fixture, prepare_nominals, rollout_spy, LinearResponse)


def fixture():
    old,state,candidates,oracle=old_fixture()
    planner=subject.ResponseSeedHoldPlannerV69(old.config,execution_model=old._base_execution,
        goal_helper=old.goal_helper,terminal_config=old.terminal_config,backend=oracle)
    return planner,state,candidates,oracle


def empty_result():
    return dict(diagnostics={},action=None,warm_start=None)


def clear_context(test,planner):
    test.assertIsNone(planner._original_public_feedback)
    test.assertIsNone(planner._canonical_numeric_feedback)
    test.assertIsNone(planner._original_feedback_sha)


class V69Boundary(unittest.TestCase):
    def test_actual_sanitizer_drops_scope_but_real_generator_receives_original(self):
        planner,state,candidates,_=fixture();before=plain(state)
        numeric=validate_feedback_state(state)
        for key in ('grasp_motion_mode',)+subject.PROVENANCE_KEYS:self.assertNotIn(key,numeric)
        calls=[];snapshots=[];compute=generator.generate_response_seed
        def synthetic_rollout(initial,cs,modes,controls,old,committed,fresh):
            for key in ('grasp_motion_mode',)+subject.PROVENANCE_KEYS:self.assertNotIn(key,initial)
            snapshots.append(deepcopy(initial))
            if fresh is not None:planner._validate_fresh_rollout_binding(initial,cs,modes,fresh)
            return rollout_spy(calls)(planner,initial,cs,modes,controls,old,committed,fresh)
        with patch.object(planner,'_prepare_nominals',side_effect=lambda s,c,o:prepare_nominals(planner,s,c,o)), \
             patch.object(planner.goal_helper.adapter,'solve',return_value={'joint_goal':np.broadcast_to(state['q'],(8,7)).copy()}), \
             patch.object(generator,'ModelOnlyHoldResponse',side_effect=lambda e,h,a:LinearResponse(e,h,a)), \
             patch.object(previous,'generate_response_seed',wraps=compute) as real_generator, \
             patch.object(planner,'_rollout_batch',side_effect=synthetic_rollout):
            result=planner.plan(state,candidates,warm_start={'time':999.,'sequence':['historical']})
        self.assertEqual(real_generator.call_count,1)
        raw=real_generator.call_args.args[2]
        self.assertIsNot(raw,state);self.assertEqual(raw['grasp_motion_mode'],'ATTACHED_PUBLIC')
        for key in subject.PROVENANCE_KEYS:self.assertIs(raw[key],False)
        self.assertEqual(plain(raw),before);self.assertEqual(plain(state),before)
        self.assertEqual([len(c['modes'])for c in calls],[88,88,80])
        self.assertTrue(any(c['fresh']is not None for c in calls))
        boundary=result['diagnostics']['V69_original_public_feedback_boundary']
        self.assertFalse(boundary['provenance_fields_added_to_numeric_state'])
        self.assertTrue(boundary['generator_received_original_public_feedback'])
        self.assertTrue(boundary['CEM_received_original_numeric_validator_output'])
        clear_context(self,planner)

    def test_missing_or_nonfalse_original_provenance_never_fabricated(self):
        for key in subject.PROVENANCE_KEYS:
            for value in ('missing',True,None,0,np.bool_(False)):
                planner,state,candidates,_=fixture()
                if value=='missing':state.pop(key)
                else:state[key]=value
                with patch.object(previous.ResponseSeedHoldPlannerV68,'plan',side_effect=AssertionError('must reject before original planner')):
                    with self.subTest(key=key,value=str(value)),self.assertRaises(ValueError):planner.plan(state,candidates)
                clear_context(self,planner)
        planner,state,candidates,_=fixture();state=validate_feedback_state(state)
        with self.assertRaises(ValueError):planner.plan(state,candidates)
        clear_context(self,planner)

    def test_all_shared_numeric_values_checked_before_crossing_boundary(self):
        changes=[('q',lambda n:n['q'].__setitem__(0,n['q'][0]+.001)),
            ('dq',lambda n:n['dq'].__setitem__(0,.001)),
            ('h',lambda n:n['actuator_state'].__setitem__('gripper_target',1.)),
            ('phase',lambda n:n['phase_state']['lift_goal_position'].__setitem__(0,999.)),
            ('tcp',lambda n:n['tcp_position'].__setitem__(0,999.)),
            ('body',lambda n:n['object_position'].__setitem__(0,999.)),
            ('goal',lambda n:n['lift_goal_position'].__setitem__(0,999.)),
            ('clock',lambda n:n.__setitem__('time',.05))]
        for name,change in changes:
            planner,state,candidates,_=fixture()
            def old_plan(self,actual,cs,*,warm_start):
                numeric=validate_feedback_state(actual);change(numeric)
                self._verified_original_feedback(numeric)
                raise AssertionError('inconsistent numeric state must not cross')
            with patch.object(previous.ResponseSeedHoldPlannerV68,'plan',new=old_plan):
                with self.subTest(field=name),self.assertRaisesRegex(ValueError,'Numeric CEM state differs'):
                    planner.plan(state,candidates)
            clear_context(self,planner)

    def test_snapshot_mutation_nested_context_and_sanitizer_failure_cleanup(self):
        planner,state,candidates,_=fixture();before=plain(state)
        def mutate(self,actual,cs,*,warm_start):
            self._original_public_feedback['object_position'][0]+=1.
            self._verified_original_feedback(validate_feedback_state(actual))
        with patch.object(previous.ResponseSeedHoldPlannerV68,'plan',new=mutate):
            with self.assertRaisesRegex(ValueError,'authenticated feedback changed'):planner.plan(state,candidates)
        self.assertEqual(plain(state),before);clear_context(self,planner)
        def nested(self,actual,cs,*,warm_start):return self.plan(actual,cs)
        with patch.object(previous.ResponseSeedHoldPlannerV68,'plan',new=nested):
            with self.assertRaisesRegex(ValueError,'nested or stale'):planner.plan(state,candidates)
        clear_context(self,planner)
        bad=deepcopy(state);bad['tcp_position'][0]=np.nan
        with self.assertRaises(ValueError):planner.plan(bad,candidates)
        clear_context(self,planner)
        def successful(self,actual,cs,*,warm_start):
            raw=self._verified_original_feedback(validate_feedback_state(actual))
            self.assertion_not_a_model_call = raw is not actual
            return empty_result()
        with patch.object(previous.ResponseSeedHoldPlannerV68,'plan',new=successful):planner.plan(state,candidates)
        self.assertTrue(planner.assertion_not_a_model_call);clear_context(self,planner)

    def test_raw_identity_validation_only_and_no_numeric_provenance_mutation(self):
        planner,state,candidates,_=fixture();numeric=validate_feedback_state(state);before=plain(numeric)
        seen=[]
        def old_compute(self,raw,cs):seen.append(raw);return {'same':raw['grasp_motion_mode']}
        def old_plan(self,actual,cs,*,warm_start):
            n=validate_feedback_state(actual)
            answer=self._compute_response_seed(n,cs)
            if answer['same']!='ATTACHED_PUBLIC':raise AssertionError('raw scope missing')
            return empty_result()
        with patch.object(previous.ResponseSeedHoldPlannerV68,'_compute_response_seed',new=old_compute), \
             patch.object(previous.ResponseSeedHoldPlannerV68,'plan',new=old_plan):planner.plan(state,candidates)
        self.assertEqual(len(seen),1);self.assertEqual(plain(numeric),before)
        for key in subject.PROVENANCE_KEYS:self.assertNotIn(key,numeric)

    def test_inherits_original_numeric_math_and_only_hold_boundary_changes(self):
        cls=subject.ResponseSeedHoldPlannerV69;old=previous.ResponseSeedHoldPlannerV68
        self.assertIs(cls._plan_cold_with_fresh_seed,old._plan_cold_with_fresh_seed)
        self.assertIs(cls._rollout_batch,old._rollout_batch)
        self.assertIs(cls._prepare_nominals,old._prepare_nominals)
        def node(fn):return ast.parse(textwrap.dedent(inspect.getsource(fn)))
        text=textwrap.dedent(inspect.getsource(producer._V69V4Dispatch.decide)).replace('ResponseSeedHoldPlannerV69','ResponseSeedHoldPlannerV68')
        self.assertEqual(ast.dump(ast.parse(text),include_attributes=False),ast.dump(node(old_producer._V68V4Dispatch.decide),include_attributes=False))
        self.assertIs(producer.ObservedPublicCandidateResponseSeedV69Backend.sample,old_producer.ObservedPublicCandidateResponseSeedV68Backend.sample)
        before=next(n for n in ast.walk(node(old_producer.run_public_candidate_response_seed_v68))if isinstance(n,ast.While))
        after=next(n for n in ast.walk(node(producer.run_public_candidate_response_seed_v69))if isinstance(n,ast.While))
        class Versions(ast.NodeTransformer):
            def visit_Constant(self,n):
                if isinstance(n.value,str):n.value=n.value.replace('v69','v68')
                return n
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(Versions().visit(after),include_attributes=False))


if __name__=='__main__':unittest.main()
