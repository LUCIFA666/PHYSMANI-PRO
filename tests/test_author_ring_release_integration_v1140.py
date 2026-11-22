"""CPU contracts for author release plus unchanged semantic continuation."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.contracts import Pose
from our_mani.multitask import author_ring_release_integration_v1140 as subject
from our_mani.multitask.physmani_ring_compensation_v580 import CorrectedRingIntents
from our_mani.multitask.public_roles import PublicRoleResolutionError
from tests.test_committed_public_role_integration_v1137 import fixture, state

ROOT=Path(__file__).resolve().parents[1]


def config(task):
    path=ROOT/'artifacts/framework/multitask_r10/post_training_qualification_v680_setup'/(task+'_ours_configuration.json')
    return json.loads(path.read_text())


class Dispatch:
    def __init__(self,source):
        self.source_policy=source
        self.qualify_opening=Mock(return_value=(1.,{'original1121':True},None))
        self.confirm_opening=Mock(return_value='original_confirmation')
        self.receipt_calls=0
    def receipt(self):
        self.receipt_calls+=1
        return dict(parent1121=True,source=self.source_policy.receipt())


def owned_fixture():
    f=fixture();f.source.receipt=lambda:{'original_source':True}
    f.source._done=Mock(return_value='original_done')
    f.source.qualify_opening=Mock(return_value=(1.,{'original_source':True},None))
    f.source.candidates=Mock(return_value=['original_candidate'])
    f.frontend=NS(ready=Mock(return_value=True),collision_port=object())
    f.dispatched=Dispatch(f.source);f.executor=NS(skill_policy=f.dispatched)
    f.order=[]
    @contextmanager
    def parent(*args):
        f.order.append('1121_enter')
        try:yield f.frontend,f.dispatched,f.executor
        finally:f.order.append('1121_exit')
    f.parent=parent
    return f


class BindingContracts(unittest.TestCase):
    def test_real_binding_is_direct1121_class_only1137_and_no_seating_or_drop_steps(self):
        value=subject.binding(ROOT,config(subject.RING_TASKS[0])['baseline_repo'])
        self.assertEqual(value['schema'],subject.SCHEMA)
        self.assertEqual(value['parent']['schema'],subject.parent.SCHEMA)
        self.assertEqual(value['comparison_source_overlay_lock']['sha256'],subject.COMPARISON_LOCK_SHA)
        self.assertEqual(set(value['sources_sha256']),set(subject.SOURCES))
        self.assertFalse(value['semantic_class_reuse']['original_integrated_components_called'])
        self.assertFalse(value['semantic_class_reuse']['original1131_parent_scope_entered'])
        self.assertFalse(value['ring_seating_v1129_installed'])
        self.assertFalse(value['new_numeric_thresholds']);self.assertEqual(value['author_drop_steps_added'],0)
        self.assertFalse(value['release_success_or_physical_drop_hit_guaranteed'])
        json.dumps(value,allow_nan=False)

    def test_two_real_ring_rules_pin_author_3cm_native_sources_and_original_config(self):
        for task in subject.RING_TASKS:
            current=config(task);rule=subject.ring_rule_binding(ROOT,current,task)
            self.assertEqual(rule['task'],task)
            self.assertEqual(rule['source_sha256'],hashlib.sha256((ROOT/subject.SOURCE).read_bytes()).hexdigest())
            self.assertEqual(rule['inherited_author_stage_binding']['phase_positions']['insert'],.03)
            self.assertEqual(rule['inherited_author_release_binding']['current_public_release_distance_m'],.03)
            self.assertEqual(rule['inherited_author_release_binding']['lookahead_seconds'],.5)
            self.assertFalse(rule['ring_seating_v1129_installed']);self.assertEqual(rule['extra_drop_steps'],0)
            self.assertEqual(rule['inherited_author_stage_binding']['native_source_sha256'],
                rule['inherited_author_release_binding']['native_source_sha256'])
            json.dumps(rule,allow_nan=False)
            current['public_ring_compensation_v580'][task]['phase_positions']['insert']=.02
            with self.assertRaises(ValueError):subject.ring_rule_binding(ROOT,current,task)

    def test_changed_comparison_lock_is_rejected_before_parent_binding(self):
        with patch.object(subject,'_sha',return_value='changed'),patch.object(subject.parent,'binding') as parent:
            with self.assertRaises(ValueError):subject.binding(ROOT,'/native')
            parent.assert_not_called()


class ScopeContracts(unittest.TestCase):
    def enter(self,f,**kwargs):
        return subject.integrated_components(ROOT,{'baseline_repo':'/native'},subject.RING_TASKS[0],f.actor,**kwargs)

    def test_ring_keeps_done_qualifiers_ready_port_and_final_executor_without1131(self):
        f=owned_fixture();before=(f.source._done,f.source.qualify_opening,f.dispatched.qualify_opening,
            f.dispatched.confirm_opening,f.frontend.ready,f.frontend.collision_port)
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={'author_wp4':True}), \
             patch.object(subject.semantic,'integrated_components',side_effect=AssertionError('no1137 parent scope')), \
             patch.object(subject.semantic.parent,'integrated_components',side_effect=AssertionError('no1131 scope')):
            with self.enter(f) as triple:
                self.assertEqual(triple,(f.frontend,f.dispatched,f.executor))
                self.assertIs(f.executor.skill_policy,f.dispatched)
                self.assertEqual(before,(f.source._done,f.source.qualify_opening,f.dispatched.qualify_opening,
                    f.dispatched.confirm_opening,f.frontend.ready,f.frontend.collision_port))
                self.assertEqual(f.dispatched.qualify_opening(1.,0.)[0],1.)
                self.assertEqual(f.dispatched.confirm_opening(1.,{},None),'original_confirmation')
                self.assertEqual(f.source._done(None),'original_done')
        f.dispatched.qualify_opening.assert_called_once_with(1.,0.)
        f.dispatched.confirm_opening.assert_called_once();f.source._done.assert_called_once_with(None)
        self.assertEqual(f.order,['1121_enter','1121_exit'])

    def test_real_author_insert_3cm_guard_accepts_hover_without_installing_seating(self):
        f=owned_fixture();f.source.comp_rule={'phase_positions':{'insert':.03}}
        f.source.frontend=NS(state=NS(robot=NS(pose=Pose([0.,0.,.02]))))
        f.source._completion_target=Mock(return_value=Pose([0.,0.,0.]))
        f.source._done=MethodType(CorrectedRingIntents._done,f.source);original=f.source._done
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={}):
            with self.enter(f):
                self.assertEqual(f.source._done,original)
                self.assertTrue(f.source._done({'phase':'insert'}))
                f.source.frontend.state.robot.pose=Pose([0.,0.,.031])
                self.assertFalse(f.source._done({'phase':'insert'}))
        self.assertEqual(f.source._completion_target.call_count,2)

    def test_real1137_resolver_runs_once_per_call_and_semantics_survive_phase_change(self):
        f=owned_fixture();original=f.resolver.resolve;events={};old_receipt=f.dispatched.receipt
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={}):
            with self.enter(f,events=events):
                continuation=vars(f.resolver)['_committed_public_role_continuation_v1137']
                self.assertIs(type(continuation),subject.semantic.PublicCommitmentContinuation)
                first=f.resolver.resolve(('red peg',),state(1.),{'timestamp':1.},f.templates)
                self.assertEqual(first[0]['selected_peg'],('peg0',))
                f.source.phase_index=5;f.source.selected=None
                second=f.resolver.resolve(('red peg',),state(2.,direct=False),{'timestamp':2.},f.templates)
                self.assertEqual(second[0]['selected_peg'],('peg0',))
                self.assertFalse(second[1]['current_colour_revalidated'])
                self.assertEqual(continuation.calls,2);self.assertEqual(continuation.uses,1)
                self.assertEqual(f.resolver._role_colour.call_count,3)
                receipt=f.dispatched.receipt();self.assertEqual(f.dispatched.receipt_calls,1)
                self.assertFalse(receipt['committed_public_role_continuation_v1137']['ring_seating_v1129_installed'])
                self.assertNotIn('later_ready_collision_forecast_and1129_guard_unchanged',receipt['committed_public_role_continuation_v1137'])
                self.assertFalse(receipt['author_ring_release_integration_v1140']['ring_seating_v1129_installed'])
        self.assertEqual(f.resolver.resolve,original);self.assertEqual(f.dispatched.receipt,old_receipt)
        self.assertTrue(events['restored']);self.assertTrue(events['witness_cleared'])
        self.assertIsNone(continuation.witness)

    def test_real1137_exact_exception_guard_and_rewind_latch_remain(self):
        f=owned_fixture()
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={}):
            with self.enter(f):
                f.resolver.resolve(('red peg',),state(5.),{'timestamp':5.},f.templates)
                f.resolver.resolve(('red peg',),state(5.1,direct=False),{'timestamp':5.1},f.templates)
                f.resolver.resolve(('red peg',),state(4.),{'timestamp':4.},f.templates)
                with self.assertRaises(PublicRoleResolutionError):
                    f.resolver.resolve(('red peg',),state(4.1,direct=False),{'timestamp':4.1},f.templates)
                continuation=vars(f.resolver)['_committed_public_role_continuation_v1137']
                self.assertEqual(continuation.uses,1);self.assertTrue(continuation.scope_invalidated)

    def test_held_false_and_open_do_not_relax_semantic_eligibility(self):
        for bad in ('held_false','open'):
            f=owned_fixture()
            with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
                 patch.object(subject,'ring_rule_binding',return_value={}):
                with self.enter(f):
                    f.resolver.resolve(('red peg',),state(1.),{'timestamp':1.},f.templates)
                    if bad=='open':f.actor.requested_opening=1.
                    with self.assertRaises(PublicRoleResolutionError):
                        f.resolver.resolve(('red peg',),state(2.,direct=False,contact=bad!='held_false'),{'timestamp':2.},f.templates)

    def test_original_execution_exception_identity_and_resolver_restoration(self):
        f=owned_fixture();original=f.resolver.resolve;error=RuntimeError('original actuator');events={}
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={}):
            with self.assertRaises(RuntimeError) as caught:
                with self.enter(f,events=events):raise error
        self.assertIs(caught.exception,error);self.assertEqual(f.resolver.resolve,original)
        self.assertTrue(events['restored']);self.assertNotIn('_committed_public_role_continuation_v1137',vars(f.resolver))

    def test_other_routes_are_exact1121_triples_and_receipts(self):
        for task in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed',
                     'push_moving_button','push_moving_button_high_speed','put_cube_in_drawer'):
            f=owned_fixture();before=f.dispatched.receipt
            with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
                 patch.object(subject,'ring_rule_binding',side_effect=AssertionError('ring only')), \
                 patch.object(subject.buzz,'derive_buzz_approach_rule',side_effect=AssertionError('Buzz only')):
                with subject.integrated_components(None,{},task,f.actor) as actual:
                    self.assertEqual(actual,(f.frontend,f.dispatched,f.executor));self.assertEqual(f.dispatched.receipt,before)

    def test_buzz_uses_same1130_source_scope_and_exact1121_wrapper_receipt(self):
        f=owned_fixture();before=f.dispatched.receipt;rule={'same1130':True}
        @contextmanager
        def scope(source,actual):
            self.assertIs(source,f.source);self.assertIs(actual,rule);f.order.append('1130_enter')
            try:yield source
            finally:f.order.append('1130_exit')
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject.buzz,'derive_buzz_approach_rule',return_value=rule) as derive, \
             patch.object(subject.buzz,'owned_buzz_approach_scope',side_effect=scope), \
             patch.object(subject.semantic,'PublicCommitmentContinuation',side_effect=AssertionError('ring only')):
            with subject.integrated_components(ROOT,{},subject.BUZZ_TASKS[0],f.actor) as actual:
                self.assertEqual(actual,(f.frontend,f.dispatched,f.executor));self.assertEqual(f.dispatched.receipt,before)
        derive.assert_called_once_with(ROOT,{},subject.BUZZ_TASKS[0],f.source)
        self.assertEqual(f.order,['1121_enter','1130_enter','1130_exit','1121_exit'])

    def test_buzz_real1130_trace_remains_in_original1121_receipt(self):
        from tests.test_public_buzz_approach_curve_v1130 import ApproachCurveTests
        from our_mani.multitask.retained_preclose_policy_integration_v1121 import RetainedPrecloseWitness
        cases=ApproachCurveTests();cases.setUpClass();source,_,_=cases.fixture()
        dispatched=RetainedPrecloseWitness(source);executor=NS(skill_policy=dispatched)
        original_receipt=dispatched.receipt;before=(source.candidates,source._done,source.receipt)
        @contextmanager
        def parent(*args):yield source.frontend,dispatched,executor
        with patch.object(subject.parent,'integrated_components',side_effect=parent):
            with subject.integrated_components(ROOT,cases.configs[source.task],source.task,source.frontend.actor):
                self.assertEqual(dispatched.receipt,original_receipt)
                self.assertLess(source.candidates('approach')[0]['curve_index'],20)
                receipt=dispatched.receipt()
                self.assertIn('public_buzz_approach_curve_v1130',receipt)
                self.assertIn('retained_preclose_dispatch_v1121',receipt)
        self.assertEqual(before,(source.candidates,source._done,source.receipt))

    def test_nested_resolver_scope_is_rejected_without_overwriting_existing_marker(self):
        f=owned_fixture();marker=object();f.resolver._committed_public_role_continuation_v1137=marker
        original=f.resolver.resolve
        with patch.object(subject.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject,'ring_rule_binding',return_value={}):
            with self.assertRaises(RuntimeError):
                with self.enter(f):pass
        self.assertIs(f.resolver._committed_public_role_continuation_v1137,marker)
        self.assertEqual(f.resolver.resolve,original)


if __name__=='__main__':unittest.main()
