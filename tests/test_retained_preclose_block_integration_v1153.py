"""CPU-only first-edge comparison through the complete original v7 MRO."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import Pose,ObjectState
from our_mani.multitask import retained_preclose_block_integration_v1153 as new
from tests import test_author_aligned_block_integration_v1120 as fixture

ACTION,CONFIG,Env=fixture.ACTION,fixture.CONFIG,fixture.Env


def choose(policy,env,*,opening=0.,phase='acquire'):
    action=ACTION.copy();action[7]=opening;policy.phase=phase
    policy.selected=dict(pose=Pose.from_xyzw(action[:7]),opening=float(opening),selected_at=float(env.time),phase=phase,action=action.copy())
    return action


def fresh_object(frontend,policy,*,distance=.01,stale=False):
    robot=frontend.robot
    grasp=Pose(robot.pose.position+np.array([distance,0.,0.]),robot.pose.rotation)
    rotation=grasp.rotation@policy.grasp.rotation.T
    pose=Pose(grasp.position-rotation@policy.grasp.position,rotation)
    frontend.object=ObjectState('target',robot.timestamp-(.05 if stale else 0.),pose,
        np.array([pose.position,pose.position+[.01,0,0],pose.position+[0,.01,0]]),np.zeros(3),np.zeros(3))


class Tests(unittest.TestCase):
    def setUp(self):
        self.base=fixture.original_fixture.Tests();self.base.setUp();self.addCleanup(self.base.doCleanups)

    def objects(self,env,*,distance=lambda time:.01):
        frontend,source=fixture.policy_for(env)
        def observe(observation,observed):
            frontend.robot=fixture.robot(observed.timestamp)
            fresh_object(frontend,source,distance=distance(observed.timestamp))
        frontend.observe=observe;observe({},NS(timestamp=env.time))
        _,parent,executor=new.parent.adapt(CONFIG,frontend,source)
        active=new.RetainedBlockPrecloseWitness(parent);executor.skill_policy=active
        return frontend,source,parent,active,executor

    def test_defer_far_first_edge_then_one_real_later_close_uses_original_proposal(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env,distance=lambda time:.046 if time<.5 else .01)
        source.admit_pose=Mock(side_effect=AssertionError('no legacy pose veto'))
        source.admit_joint=Mock(side_effect=AssertionError('no collision predicate'))
        original_qualify=source.qualify_opening;original_confirm=source.confirm_opening
        with patch.object(source,'qualify_opening',wraps=original_qualify) as raw, \
             patch.object(source,'confirm_opening',wraps=original_confirm) as confirm, \
             patch.object(parent,'qualify_opening',wraps=parent.qualify_opening) as forwarded:
            self.base.run_action(env,executor)
            self.assertEqual(env.ledger.grasp_requests,0);self.assertIsNone(source.close_witness);self.assertFalse(frontend.close_confirmed)
            decision=active.qualification_events[-1]
            self.assertTrue(decision['retained_original_first_preclose']);self.assertEqual(decision['qualified_opening'],1.)
            self.assertIsNone(decision['proposed']);self.assertAlmostEqual(decision['proof']['distance_m'],.046)
            self.assertFalse(decision['proof']['legacy_SE3_defer_is_diagnostic_only'])
            choose(source,env);self.base.run_action(env,executor)
            self.assertEqual(env.ledger.grasp_requests,1);self.assertTrue(frontend.close_confirmed)
            self.assertAlmostEqual(source.close_witness['issued_at'],.95);self.assertAlmostEqual(source.close_witness['confirmed_after'],1.)
            self.assertNotIn('witness_kind',source.close_witness)  #538 proposal, not1120 synthetic proposal.
            self.assertAlmostEqual(source.close_witness['distance_m'],.01)
            self.assertEqual((raw.call_count,forwarded.call_count,confirm.call_count),(20,20,20))
        source.admit_pose.assert_not_called();source.admit_joint.assert_not_called()
        self.assertIsNone(executor.admit_joint_target);self.assertEqual(executor.collision_diagnostic_limit,0)
        self.assertFalse(frontend.robot.contact);self.assertIsNone(frontend.collision.held_local)

    def test_raw_missing_or_stale_current_grasp_keeps_open_without_fabricated_close(self):
        for state in ('missing','stale'):
            env=Env();frontend,source,parent,active,executor=self.objects(env)
            if state=='missing':frontend.object=None
            else:fresh_object(frontend,source,stale=True)
            with self.subTest(state=state),patch.object(source,'qualify_opening',wraps=source.qualify_opening) as raw:
                command,proof,proposed=active.qualify_opening(0.,1.)
                self.assertEqual(command,1.);self.assertIsNone(proposed);self.assertIsNone(proof['distance_m'])
                self.assertEqual(proof['reason'],'defer_close_until_current_public_block_physical_SE3')
                raw.assert_called_once()
            self.assertIsNone(source.close_witness);self.assertEqual(parent.events,[])

    def test_original_angle_and_public_opening_requirements_are_not_relaxed(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env,distance=lambda time:0.)
        current=source.current_grasp();angle=.2
        desired=Pose.from_xyzw([*current.position,0.,0.,np.sin(angle/2),np.cos(angle/2)])
        rotation=desired.rotation@source.grasp.rotation.T
        pose=Pose(desired.position-rotation@source.grasp.position,rotation)
        frontend.object=replace(frontend.object,pose=pose)
        command,proof,proposal=active.qualify_opening(0.,1.)
        self.assertEqual(command,1.);self.assertIsNone(proposal);self.assertAlmostEqual(proof['angle_rad'],.2)
        fresh_object(frontend,source,distance=.01)
        frontend.robot=replace(frontend.robot,gripper_open=False,contact=False)
        command,proof,proposal=active.qualify_opening(0.,1.)
        self.assertEqual(command,1.);self.assertIsNone(proposal)

    def test_already_closed_without_witness_never_reopens_when_raw_qualifier_defers(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env,distance=lambda time:.1)
        command,proof,proposed=active.qualify_opening(0.,0.)
        self.assertEqual(command,0.);self.assertIsNone(proposed)
        self.assertEqual(proof['legacy_qualification']['qualified_opening'],1.)
        self.assertTrue(proof['legacy_SE3_defer_is_diagnostic_only'])

    def test_repeated_close_and_actual_release_delegate_exactly_to_parent(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env)
        self.base.run_action(env,executor);old=deepcopy(source.close_witness)
        choose(source,env);self.base.run_action(env,executor)
        self.assertEqual(env.ledger.grasp_requests,1);self.assertEqual(source.close_witness,old)
        opening=choose(source,env,opening=1.,phase='transport')
        self.base.run_action(env,executor,action=opening)
        self.assertEqual(env.ledger.release_requests,1);self.assertEqual(env._requested_opening,1.)
        self.assertEqual(parent.events[-1]['opening'],1.)
        self.assertTrue(parent.events[-1]['proof']['legacy_SE3_defer_is_diagnostic_only'])
        self.assertFalse(active.qualification_events[-1]['retained_original_first_preclose'])

    def test_other_phase_and_release_match_parent_even_when_raw_rejects(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env)
        for phase,requested,current in (('approach',0.,1.),('transport',1.,0.),('acquire',1.,0.)):
            source.phase=phase
            with patch.object(source,'qualify_opening',return_value=(1.-requested,
                    dict(time=0.,phase=phase,desired_opening=requested,qualified_opening=1.-requested),None)) as raw:
                expected=parent.qualify_opening(requested,current);raw.reset_mock()
                actual=active.qualify_opening(requested,current)
                self.assertEqual(actual,expected);raw.assert_called_once_with(requested,current)

    def test_partial_budget_cannot_publish_new_close_edge_or_support_confirmation(self):
        env=Env(ticks=195);frontend,source,parent,active,executor=self.objects(env)
        report=self.base.run_action(env,executor)
        self.assertEqual(report['ticks'],5);self.assertEqual(env.ledger.grasp_requests,0)
        self.assertIsNone(source.close_witness);self.assertFalse(frontend.close_confirmed)
        self.assertEqual(len(active.qualification_events),5)
        self.assertTrue(all(row['proposed'] is None for row in active.qualification_events))

    def test_when_original_qualifier_accepts_full_native_effects_and_getter_counts_match_parent(self):
        baseline,env=Env(),Env()
        _,_,parent,_,parent_executor=self.objects(baseline);parent_executor.skill_policy=parent
        _,_,_,active,executor=self.objects(env)
        self.base.run_action(baseline,parent_executor);self.base.run_action(env,executor)
        self.base.equal_native_effects(baseline,env)
        self.assertEqual(env.counts,baseline.counts);self.assertEqual(env.arm.calls,baseline.arm.calls)
        self.assertEqual(executor.collision_diagnostic_limit,0)

    def test_qualified_proposal_is_not_a_close_witness_before_original_confirmation(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env)
        command,proof,proposal=active.qualify_opening(0.,1.)
        self.assertEqual(command,0.);self.assertIsNotNone(proposal)
        self.assertIsNone(source.close_witness);self.assertFalse(frontend.close_confirmed);self.assertEqual(parent.events,[])
        active.confirm_opening(command,proof,proposal,confirmed_after=.05)
        self.assertAlmostEqual(source.close_witness['confirmed_after'],.05);self.assertTrue(frontend.close_confirmed)

    def test_failed_native_close_apply_never_confirms_the_proposed_edge(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env)
        original_tick=env._tick
        def tick(phase):
            if env.ledger.ticks==9:raise new.ProtocolInvariantError('original tenth tick failed')
            return original_tick(phase)
        env._tick=tick
        with patch.object(source,'confirm_opening',wraps=source.confirm_opening) as confirm:
            with self.assertRaises(new.ProtocolInvariantError):self.base.run_action(env,executor)
        self.assertEqual(confirm.call_count,9);self.assertEqual(len(parent.events),9)
        self.assertIsNotNone(active.qualification_events[-1]['proposed'])
        self.assertEqual(active.qualification_events[-1]['evidence_kind'],'qualification_only_not_applied_or_confirmed')
        self.assertIsNone(source.close_witness);self.assertFalse(frontend.close_confirmed)

    def test_no_contact_or_supported_held_geometry_does_not_force_lift_after_close(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env)
        self.base.run_action(env,executor)
        frontend.goal_pose=Pose([.5,0.,1.]);frontend.frames=10;frontend.config.update(warmup_ticks=1,preposition_height=1.)
        frontend.forecast=None
        self.assertIsNone(active.select());self.assertEqual(source.phase,'acquire')
        frontend.robot=fixture.robot(env.time,contact=True)
        self.assertIsNone(active.select());self.assertEqual(source.phase,'acquire')
        frontend.collision.held_local=Pose([0.,0.,-.05])
        self.assertIsNotNone(active.select());self.assertEqual(source.phase,'lift')

    def test_receipt_marks_old_parent_dispatch_metadata_historical_and_active_preclose_honest(self):
        env=Env();frontend,source,parent,active,executor=self.objects(env,distance=lambda time:.046)
        self.base.run_action(env,executor);receipt=active.receipt()
        self.assertNotIn('author_aligned_block_integration_v1120',receipt)
        self.assertFalse(receipt['legacy_parent_block_integration_v1120']['active_dispatch_description'])
        active_receipt=receipt['retained_preclose_block_integration_v1153']
        self.assertTrue(active_receipt['active_first_preclose_uses_original_qualifier'])
        self.assertFalse(active_receipt['native_capture_claimed']);self.assertFalse(active_receipt['public_threshold_is_native_necessity_or_sufficiency_claim'])
        self.assertEqual(active_receipt['confirmed_opening_events'][-1]['opening'],1.)
        self.assertFalse(receipt['held_geometry']['held_target_vs_arm_and_static_obstacles_checked'])
        self.assertFalse(frontend.close_confirmed)

    def test_scope_uses_same_parent_objects_and_restores_final_skill_policy_before_parent_exit(self):
        env=Env();frontend,source,parent,_,executor=self.objects(env);executor.skill_policy=parent;events=[]
        @contextmanager
        def scope(*args):
            events.append('parent_enter')
            try:yield frontend,parent,executor
            finally:
                self.assertIs(executor.skill_policy,parent);events.append('parent_exit')
        with patch.object(new.parent,'integrated_block_components',side_effect=scope):
            with self.assertRaisesRegex(RuntimeError,'owned failure'):
                with new.integrated_block_components(None,CONFIG,new.parent.BLOCKS[0],'context_geometry',env) as result:
                    f,active,owned=result
                    self.assertIs(f,frontend);self.assertIs(owned,executor);self.assertIs(active.source_policy,source)
                    self.assertIs(executor.skill_policy,active);raise RuntimeError('owned failure')
        self.assertEqual(events,['parent_enter','parent_exit'])

    def test_binding_is_full_parent_plus_pinned_original_qualifier_and_no_new_threshold(self):
        root=Path(__file__).resolve().parents[1]
        with patch.object(new.parent,'binding',return_value={'complete_parent_binding':True}) as parent:
            result=new.binding(root,'existing_native_source_root')
        parent.assert_called_once_with(root,'existing_native_source_root')
        self.assertEqual(result['parent'],{'complete_parent_binding':True})
        self.assertEqual(result['sources_sha256'][new.QUALIFIER_SOURCE],new.QUALIFIER_SHA)
        self.assertFalse(result['active_first_preclose']['new_numeric_thresholds'])
        self.assertFalse(result['public_threshold_is_native_grasp_necessity_or_sufficiency_claim'])
        self.assertEqual(result['additional_native_getters_observations_predictions_or_success_calls'],0)

    def test_bounded_qualification_and_confirmations_remain_separate_receipts(self):
        env=Env();frontend,source,_,_,_=self.objects(env)
        parent=new.parent.AuthorBlockRequestedOpeningWitness(source,max_events=1)
        active=new.RetainedBlockPrecloseWitness(parent)
        for index in range(3):
            command,proof,proposal=active.qualify_opening(1.,1.)
            active.confirm_opening(command,proof,proposal,confirmed_after=(index+1)*.05)
        receipt=active.receipt()['retained_preclose_block_integration_v1153']
        self.assertEqual(len(receipt['qualification_events']),1);self.assertEqual(receipt['dropped_qualifications'],2)
        self.assertEqual(len(receipt['confirmed_opening_events']),1);self.assertEqual(receipt['dropped_confirmations'],2)


if __name__=='__main__':unittest.main()
