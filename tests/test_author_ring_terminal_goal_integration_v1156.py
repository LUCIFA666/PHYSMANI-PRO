"""CPU-only local author terminal adapter contracts; no Native launch."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PhaseTemplate, COMMAND_GOAL
from our_mani.multitask.grasp_frame_compensation_native_compat import compose, transform
from our_mani.multitask import author_ring_terminal_goal_integration_v1156 as subject
from tests.test_author_ring_release_integration_v1140 import config, owned_fixture

ROOT=Path(__file__).resolve().parents[1]


def assert_pose(case,actual,expected):
    np.testing.assert_allclose(actual.position,expected.position,atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual.rotation,expected.rotation,atol=1e-12,rtol=0)


def fixture(task=subject.RING_TASKS[0], *, corrected=False):
    cfg=config(task);rule=subject.ring_rule_binding(ROOT,cfg,task)
    data=json.loads((ROOT/cfg['learned_artifacts'][task]['interaction']['path']).read_text())
    base=PhaseTemplate(**next(t for t in data['templates'] if t['phase']=='insert'))
    delta=Pose([.004,-.003,.002],Rotation.from_euler('xyz',[.07,-.04,.03]).as_matrix())
    template=transform(base,delta) if corrected else base
    future=Pose([.3,-.1,.8],Rotation.from_euler('xyz',[.2,.1,-.4]).as_matrix())
    current=Pose([.31,-.08,.801],Rotation.from_euler('xyz',[.21,.12,-.38]).as_matrix())
    curve=template.at(1.)[0]
    row=dict(phase='insert',template=template,reference_id='peg0',manipulated_ids=('ring',),
        pose=compose(future,curve),present_pose=compose(current,curve),opening=0.,
        selected_at=3.,predicted_reference_time=3.5,lookahead_seconds=.5,score=-1.)
    robot=NS(pose=Pose([0.,0.,0.]),timestamp=3.)
    generator=NS(observed_reference=Mock(side_effect=AssertionError('no repeated public helper')))
    controller=NS(library=NS(templates=(template,)),generator=generator)
    frontend=NS(state=NS(robot=robot),actor=NS(controller=controller),
        forecast=NS(at=Mock(side_effect=AssertionError('no repeated prediction'))))
    source=NS(task=task,frontend=frontend,original_templates=(base,),
        candidates=Mock(return_value=[row]),receipt=Mock(return_value={'parent_source':True}),
        _done=Mock(return_value='original_done'),qualify_opening=Mock(return_value='original_qualify'))
    return NS(source=source,rule=rule,row=row,base=base,template=template,delta=delta,
        future=future,current=current,config=cfg)


class BindingContracts(unittest.TestCase):
    def test_real_source_only_binding_pins_both_comparisons_and_parent(self):
        value=subject.binding(ROOT,config(subject.RING_TASKS[0])['baseline_repo'])
        self.assertEqual(value['schema'],subject.SCHEMA)
        self.assertEqual(value['changed_factor'],subject.CHANGED_FACTOR)
        self.assertEqual(value['parent']['schema'],subject.parent.SCHEMA)
        self.assertEqual(set(value['sources_sha256']),set(subject.SOURCES))
        self.assertEqual({p['path']:p['sha256'] for p in value['comparison_source_overlay_locks']},subject.COMPARISON_LOCKS)
        self.assertFalse(value['universal_COMMAND_GOAL_contract_bug_claimed'])
        self.assertFalse(value['source_TRAIN_curve_or_weights_modified'])
        self.assertEqual(value['additional_observed_reference_predict_observe_Native_getter_or_success_calls'],0)
        self.assertFalse(value['Native_success_or_physical_seating_guaranteed'])
        json.dumps(value,allow_nan=False)

    def test_real_two_ring_models_and_original_author_rules_remain(self):
        for task in subject.RING_TASKS:
            cfg=config(task);r=subject.ring_rule_binding(ROOT,cfg,task)
            self.assertEqual(r['task'],task);self.assertEqual(r['changed_factor'],subject.CHANGED_FACTOR)
            self.assertEqual(r['interaction'],cfg['learned_artifacts'][task]['interaction'])
            self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/subject.SOURCE).read_bytes()).hexdigest())
            self.assertEqual(r['inherited_author_stage_binding']['phase_positions']['insert'],.03)
            self.assertEqual(r['inherited_author_release_binding']['lookahead_seconds'],.5)
            self.assertTrue(r['original_delta_retained_once']);self.assertEqual(r['extra_drop_steps'],0)
            self.assertTrue(r['original_insert_template_sha256']);json.dumps(r,allow_nan=False)

    def test_comparison_and_model_pointer_changes_rejected(self):
        with patch.object(subject,'_sha',return_value='bad'):
            with self.assertRaises(ValueError):subject.binding(ROOT,'/native')
        cfg=config(subject.RING_TASKS[0]);cfg['learned_artifacts'][subject.RING_TASKS[0]]['interaction']['sha256']='bad'
        with self.assertRaises(ValueError):subject.ring_rule_binding(ROOT,cfg,subject.RING_TASKS[0])


class ConversionContracts(unittest.TestCase):
    def test_both_models_submit_canonical_in_same_future_and_current_frames(self):
        for task in subject.RING_TASKS:
            for phase in subject.PHASES:
                f=fixture(task);f.row['phase']=phase;f.row['opening']=float(phase=='release')
                row,e=subject.canonical_terminal_row(f.source,f.row,f.rule)
                canonical=Pose.from_xyzw(f.template.completion_geometry_xyzw)
                assert_pose(self,row['pose'],compose(f.future,canonical))
                assert_pose(self,row['present_pose'],compose(f.current,canonical))
                self.assertIs(row['template'],f.template)
                for k in set(f.row)-{'pose','present_pose','score'}:self.assertIs(row[k],f.row[k])
                self.assertEqual(row['score'],-float(np.linalg.norm(row['pose'].position)))
                self.assertEqual(e['additional_reference_or_prediction_queries'],0)
                f.source.frontend.actor.controller.generator.observed_reference.assert_not_called()
                f.source.frontend.forecast.at.assert_not_called()

    def test_actual_corrected_template_keeps_measured_delta_once_not_twice(self):
        f=fixture(subject.RING_TASKS[1],corrected=True)
        row,e=subject.canonical_terminal_row(f.source,f.row,f.rule)
        original_canonical=Pose.from_xyzw(f.base.completion_geometry_xyzw)
        expected=compose(compose(f.future,original_canonical),f.delta)
        assert_pose(self,row['pose'],expected)
        self.assertTrue(e['measured_compensation_already_in_template'])
        self.assertGreater(np.linalg.norm(row['pose'].position-compose(expected,f.delta).position),.001)
        np.testing.assert_array_equal(f.row['pose'].position,compose(f.future,f.template.at(1.)[0]).position)

    def test_at_override_semantics_used_instead_of_only_array_endpoint(self):
        f=fixture(corrected=True)
        original=subject.CorrectedTemplate.at
        with patch.object(subject.CorrectedTemplate,'at',autospec=True,
                side_effect=lambda t,u:original(t,u)) as at:
            row,_=subject.canonical_terminal_row(f.source,f.row,f.rule)
            at.assert_called_once_with(f.template,1.)
        expected=compose(compose(f.future,Pose.from_xyzw(f.base.completion_geometry_xyzw)),f.delta)
        assert_pose(self,row['pose'],expected)

    def test_template_source_mutation_unknown_reference_kind_and_unknown_subclass_rejected(self):
        f=fixture()
        class Unsupported(PhaseTemplate):pass
        variants=[replace(f.base,reference_kind='translation_only'),
                  replace(f.base,positions=f.base.positions+np.array([0.,0.,.001])),
                  Unsupported(**{x.name:getattr(f.base,x.name) for x in fields(PhaseTemplate)})]
        for bad in variants:
            f.row['template']=bad;f.source.frontend.actor.controller.library.templates=(bad,)
            with self.assertRaises(ValueError):subject.canonical_terminal_row(f.source,f.row,f.rule)


    def test_real_590_MRO_and_529_593_candidates_keep_exact_helper_call_counts(self):
        from our_mani.contracts import RobotState,ObjectState,Forecast
        from our_mani.multitask.interaction import InteractionLibrary
        from our_mani.multitask.physmani_ring_release_contact_v657 import SourceContactRingIntents
        from our_mani.multitask import physmani_geometry_goals_v529 as goals529
        from our_mani.multitask import physmani_geometry_goals_v593 as goals593
        from our_mani.multitask import physmani_skill_policy_v548 as skill
        for task in subject.RING_TASKS:
            for corrected in (False,True):
                cfg=config(task);rule=subject.ring_rule_binding(ROOT,cfg,task)
                if task.endswith('_high_speed'):
                    from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary as Library
                else:Library=InteractionLibrary
                library=Library.load(ROOT/cfg['learned_artifacts'][task]['interaction']['path'])
                original_templates=library.templates
                t=library.for_phase('insert')[0];now=3.
                reference=ObjectState('peg0',now,Pose([.3,-.1,.8]),np.zeros((3,3)),np.zeros(3),np.zeros(3))
                generator=NS(observed_reference=Mock(return_value=reference))
                controller=NS(library=library,generator=generator,committed_roles={})
                robot=RobotState(now,Pose([.1,.2,.7]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
                forecast=Forecast(('peg0',),np.array([now,now+1.]),
                    np.broadcast_to(reference.pose.position,(2,1,3)).copy(),
                    np.broadcast_to(reference.pose.rotation,(2,1,3,3)).copy(),
                    np.zeros((2,1,3)),np.zeros((2,1,3)),metadata={'observed_until':now})
                frontend=NS(actor=NS(controller=controller),forecast=forecast,
                    state=NS(robot=robot,role_bindings={t.reference_role:('peg0',),t.manipulated_role:('ring',)},
                        perception_metadata={'bodies':{'peg0':{'motion_mode':t.reference_mode}}}))
                comp=cfg['public_ring_compensation_v580'][task]
                source=SourceContactRingIntents(frontend,comp,json.loads((ROOT/comp['prior']['path']).read_text()),
                    cfg['author_ring_terminal_release_v590'][task],contact_rule=cfg['ring_release_contact_v657'][task])
                delta=Pose([.003,-.002,.001],Rotation.from_euler('z',.03).as_matrix())
                # Existing parent compensation is already admitted; no extra measure.
                source.compensation=delta if corrected else Pose([0.,0.,0.])
                if corrected:library.templates=tuple(transform(x,delta) if x.phase in ('align','insert') else x for x in original_templates)
                before=(source._done,source.qualify_opening,source.confirm_opening)
                original_at=Forecast.at
                for helper in (goals529.pose_goals,goals593.pose_goals):
                    for phase in subject.PHASES:
                        generator.observed_reference.reset_mock()
                        with patch.object(skill,'pose_goals',side_effect=helper) as goals, \
                             patch.object(Forecast,'at',autospec=True,side_effect=lambda f,*a,**k:original_at(f,*a,**k)) as at, \
                             subject.owned_terminal_goal_scope(source,rule):
                            rows=source.candidates(phase)
                            self.assertEqual(len(rows),1);row=rows[0]
                            self.assertEqual(row['phase'],phase);self.assertEqual(row['opening'],float(phase=='release'))
                            self.assertEqual(row['predicted_reference_time'],3.5)
                            actual=library.for_phase('insert')[0]
                            assert_pose(self,row['pose'],compose(reference.pose,Pose.from_xyzw(actual.completion_geometry_xyzw)))
                            goals.assert_called_once_with(frontend,'insert',origin=None,committed=None)
                            generator.observed_reference.assert_called_once();at.assert_called_once()
                            self.assertEqual((source._done,source.qualify_opening,source.confirm_opening),before)
                self.assertEqual(source.original_templates,original_templates)

    def test_wrong_ownership_wrong_lead_and_missing_pose_are_unknown_not_guessed(self):
        for key,value in [('lookahead_seconds',1.),('predicted_reference_time',3.4),
                          ('manipulated_ids',('other',)),('present_pose',None)]:
            f=fixture();f.row[key]=value
            with self.assertRaises(ValueError):subject.canonical_terminal_row(f.source,f.row,f.rule)
        f=fixture();f.source.original_templates=()
        with self.assertRaises(ValueError):subject.canonical_terminal_row(f.source,f.row,f.rule)


class ScopeContracts(unittest.TestCase):
    def test_candidates_once_original_curve_immutable_and_only_two_methods_patched(self):
        f=fixture(corrected=True);source=f.source
        before=(source.candidates,source.receipt,source._done,source.qualify_opening)
        original_position=f.template.positions.copy();events={}
        with subject.owned_terminal_goal_scope(source,f.rule,events=events):
            result=source.candidates('insert')
            self.assertEqual(len(result),1);self.assertIsNot(result[0],f.row)
            before[0].assert_called_once_with('insert')
            self.assertEqual((source._done,source.qualify_opening),before[2:])
            receipt=source.receipt();before[1].assert_called_once_with()
            self.assertTrue(receipt['parent_source'])
            self.assertEqual(receipt['author_ring_terminal_goal_integration_v1156']['converted_candidates'],1)
        self.assertEqual((source.candidates,source.receipt,source._done,source.qualify_opening),before)
        self.assertTrue(events['restored']);np.testing.assert_array_equal(f.template.positions,original_position)
        source._done.assert_not_called();source.qualify_opening.assert_not_called()

    def test_other_phase_exact_original_list_and_unknown_terminal_not_dispatched(self):
        f=fixture();original=f.source.candidates.return_value
        with subject.owned_terminal_goal_scope(f.source,f.rule) as source:
            self.assertIs(source.candidates('align'),original)
            f.row['present_pose']=None
            self.assertEqual(source.candidates('insert'),[])
            r=source.receipt()['author_ring_terminal_goal_integration_v1156']
            self.assertEqual(r['unknown_candidates'],1);self.assertFalse(r['events'][0]['target_invented'])

    def test_same_public_distance_ranking_is_recomputed_for_multiple_candidates(self):
        f=fixture();far=dict(f.row,reference_id='peg1',pose=Pose([2.,0.,0.]))
        f.source.candidates.return_value=[far,f.row]
        with subject.owned_terminal_goal_scope(f.source,f.rule):
            result=f.source.candidates('insert')
            self.assertEqual(result[0]['reference_id'],'peg0')
            self.assertGreater(result[0]['score'],result[1]['score'])

    def test_original_candidate_and_execution_exception_identity_survive_and_restore(self):
        for where in ('candidate','execution'):
            f=fixture();before=(f.source.candidates,f.source.receipt);error=RuntimeError('original '+where);events={}
            if where=='candidate':before[0].side_effect=error
            with self.assertRaises(RuntimeError) as caught:
                with subject.owned_terminal_goal_scope(f.source,f.rule,events=events):
                    if where=='candidate':f.source.candidates('insert')
                    else:raise error
            self.assertIs(caught.exception,error);self.assertTrue(events['restored'])
            self.assertEqual((f.source.candidates,f.source.receipt),before)

    def test_bounded_receipt_and_nested_scope_rejection(self):
        f=fixture();events={}
        with subject.owned_terminal_goal_scope(f.source,f.rule,events=events,max_events=1):
            f.source.candidates('insert');f.source.candidates('insert')
            with self.assertRaises(RuntimeError):
                with subject.owned_terminal_goal_scope(f.source,f.rule):pass
        self.assertEqual(len(events['events']),1);self.assertEqual(events['dropped_events'],1)
        self.assertTrue(events['restored'])

    def test_integration_complete_parent_lifetime_same_triple_and_restoration(self):
        f=fixture();dispatch=NS(source_policy=f.source);executor=NS(skill_policy=dispatch);order=[];events={}
        @contextmanager
        def parent(*args,events=None,**kwargs):
            order.append('parent_enter')
            try:yield f.source.frontend,dispatch,executor
            finally:order.append('parent_exit');events.update(restored=True)
        before=f.source.candidates
        with patch.object(subject.parent,'integrated_components',side_effect=parent), \
             patch.object(subject,'ring_rule_binding',return_value=f.rule):
            with subject.integrated_components(ROOT,f.config,f.rule['task'],object(),events=events) as triple:
                self.assertEqual(triple,(f.source.frontend,dispatch,executor));self.assertIs(executor.skill_policy,dispatch)
                self.assertIsNot(f.source.candidates,before);f.source.candidates('insert')
                order.append('body')
        self.assertEqual(order,['parent_enter','body','parent_exit'])
        self.assertEqual(f.source.candidates,before);self.assertTrue(events['restored'])

    def test_real_1140_semantic_scope_receipt_and_guard_restore_survive_new_wrapper(self):
        from tests.test_committed_public_role_integration_v1137 import state
        f=owned_fixture();events={};before=(f.source.candidates,f.source.receipt,f.source._done,
            f.source.qualify_opening,f.resolver.resolve,f.dispatched.receipt)
        with patch.object(subject.parent.parent,'integrated_components',side_effect=f.parent), \
             patch.object(subject.parent,'ring_rule_binding',return_value={}), \
             patch.object(subject,'ring_rule_binding',return_value={'task':subject.RING_TASKS[0]}):
            with subject.integrated_components(ROOT,{},subject.RING_TASKS[0],f.actor,events=events) as triple:
                self.assertEqual(triple,(f.frontend,f.dispatched,f.executor))
                f.resolver.resolve(('red peg',),state(1.),{'timestamp':1.},f.templates)
                f.source.phase_index=5;f.source.selected=None
                f.resolver.resolve(('red peg',),state(2.,direct=False),{'timestamp':2.},f.templates)
                c=vars(f.resolver)['_committed_public_role_continuation_v1137']
                self.assertEqual(c.calls,2);self.assertEqual(c.uses,1)
                receipt=f.dispatched.receipt()
                self.assertIn('author_ring_terminal_goal_integration_v1156',receipt['source'])
                self.assertIn('committed_public_role_continuation_v1137',receipt)
                self.assertEqual((f.source._done,f.source.qualify_opening),before[2:4])
        self.assertTrue(events['restored']);self.assertTrue(events['parent_scope']['restored'])
        self.assertEqual((f.source.candidates,f.source.receipt,f.source._done,f.source.qualify_opening,
            f.resolver.resolve,f.dispatched.receipt),before)

    def test_nonring_routes_exact_parent_objects_receipts_and_no_new_rule(self):
        f=owned_fixture();triple=(f.frontend,f.dispatched,f.executor);before=f.dispatched.receipt
        @contextmanager
        def parent(*args,**kwargs):yield triple
        for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed',
                     'remove_cups_from_rotating_frame','push_moving_button','stack_blocks_in_order'):
            with patch.object(subject.parent,'integrated_components',side_effect=parent), \
                 patch.object(subject,'ring_rule_binding',side_effect=AssertionError('ring only')):
                with subject.integrated_components(None,{},task,f.actor) as actual:
                    self.assertIs(actual,triple);self.assertEqual(f.dispatched.receipt,before)


if __name__=='__main__':unittest.main()
