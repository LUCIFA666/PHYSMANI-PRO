"""CPU block protocol and full original-v7-MRO tests; no Native construction."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.multitask import author_aligned_block_integration_v1120 as block
from our_mani.multitask.physmani_block_policy_v538 import BlockGeometryIntents, BlockPublicFrontend
from tests import test_author_aligned_geometry_executor_v1114 as original_fixture

ACTION, CONFIG, Env = original_fixture.ACTION, original_fixture.CONFIG, original_fixture.Env


def robot(timestamp=0., *, contact=False, position=(0.,0.,1.)):
    return RobotState(timestamp, Pose(position), np.zeros(3), np.zeros(3),
                      np.zeros(7), np.zeros(7), gripper_open=not contact, contact=contact)


def policy_for(env):
    frontend = NS(config=dict(CONFIG,local_max_speed=.5,local_max_acceleration=3.),method='cv_geometry',
        robot=robot(env.time),object=None,close_confirmed=False,events=[],dynamics=None,
        collision=NS(held_local=None,receipt=lambda:dict(first_contact_support_events=[],collision_queries=0,
            held_target_vs_arm_and_static_obstacles_checked=True,gripper_only_contact_exemption=True,
            original_held_support_geometry_reused=True)))
    def observe(observation, observed_robot):
        frontend.robot=robot(observed_robot.timestamp)
    frontend.observe=observe
    policy=BlockGeometryIntents(frontend); policy.phase='acquire'
    policy.selected=dict(pose=Pose.from_xyzw(ACTION[:7]),opening=float(ACTION[7]),selected_at=float(env.time),
                         phase='acquire',action=ACTION.copy())
    return frontend,policy


class Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=original_fixture.Tests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_full_v7_mro_native_effects_match_author_while_legacy_block_SE3_defer_is_diagnostic(self):
        baseline,env=Env(),Env()
        expected=self.fixture.run_action(baseline)
        frontend,policy=policy_for(env)
        self.assertFalse(hasattr(policy,'phase_index'))
        self.assertNotIn('manipulated_ids',policy.selected)
        policy.admit_pose=Mock(side_effect=AssertionError('legacy pose callback must not run'))
        policy.admit_joint=Mock(side_effect=AssertionError('legacy joint collision must not run'))
        original_qualify=policy.qualify_opening; original_confirm=policy.confirm_opening
        with patch.object(policy,'qualify_opening',wraps=original_qualify) as qualify,\
             patch.object(policy,'confirm_opening',wraps=original_confirm) as confirm:
            _,dispatched,executor=block.adapt(CONFIG,frontend,policy)
            result=self.fixture.run_action(env,executor)
        self.fixture.equal_native_effects(baseline,env); self.assertEqual(expected,result)
        policy.admit_pose.assert_not_called(); policy.admit_joint.assert_not_called()
        self.assertEqual(qualify.call_count,10); self.assertEqual(confirm.call_count,10)
        self.assertIsNone(executor.admit_joint_target); self.assertEqual(executor.collision_diagnostic_limit,0)
        self.assertTrue(frontend.close_confirmed); self.assertIsNotNone(policy.close_witness)
        event=dispatched.events[-1]
        self.assertEqual(event['opening'],0.)
        self.assertEqual(event['proof']['qualified_opening'],0.)
        legacy=event['proof']['legacy_qualification']
        self.assertEqual(legacy['qualified_opening'],1.)
        self.assertEqual(legacy['proof']['reason'],'defer_close_until_current_public_block_physical_SE3')
        self.assertIsNone(event['proposed']['distance_m'])
        self.assertFalse(event['native_capture_claimed'])

    def test_partial_budget_does_not_fabricate_new_block_close_or_support_confirmation(self):
        env=Env(ticks=195); frontend,policy=policy_for(env)
        _,dispatched,executor=block.adapt(CONFIG,frontend,policy)
        report=self.fixture.run_action(env,executor)
        self.assertEqual(report['ticks'],5); self.assertEqual(env.ledger.grasp_requests,0)
        self.assertIsNone(policy.close_witness); self.assertFalse(frontend.close_confirmed)
        self.assertTrue(all(event['proposed'] is None for event in dispatched.events))

    def test_failed_native_application_never_confirms_close_and_original_helpers_restore(self):
        env=Env(apply_error=block.ProtocolInvariantError('original counted tick failed'))
        frontend,policy=policy_for(env)
        _,dispatched,executor=block.adapt(CONFIG,frontend,policy)
        with patch.object(policy,'confirm_opening',wraps=policy.confirm_opening) as confirm:
            with self.assertRaises(block.ProtocolInvariantError): self.fixture.run_action(env,executor)
        confirm.assert_not_called(); self.assertEqual(dispatched.events,[])
        self.assertFalse(frontend.close_confirmed); self.assertIsNone(policy.close_witness)

    def test_protocol_admission_uses_public_block_DTO_and_exact_record_only(self):
        env=Env(); frontend,policy=policy_for(env)
        accepted,proof=block.protocol_block_admission(policy)(ACTION)
        self.assertTrue(accepted); self.assertFalse(proof['collision_predicate_called'])
        frontend.collision=object()  # No collision API can be reached by the admission.
        policy.selected['action']=ACTION.copy(); policy.selected['action'][0]=.1
        self.assertFalse(block.protocol_block_admission(policy)(ACTION)[0])
        policy.selected['action']=ACTION.copy(); policy.selected['selected_at']=-.05
        self.assertFalse(block.protocol_block_admission(policy)(ACTION)[0])
        native_pose=NS(xyzw=Mock(side_effect=AssertionError('Native getter forbidden')))
        policy.selected['pose']=native_pose
        self.assertFalse(block.protocol_block_admission(policy)(ACTION)[0]); native_pose.xyzw.assert_not_called()

    def test_original_phase_rules_still_require_public_supported_capture(self):
        env=Env(); frontend,policy=policy_for(env)
        frontend.goal_pose=Pose([.5,0.,1.]); frontend.frames=10
        frontend.config.update(warmup_ticks=1,preposition_height=1.)
        frontend.forecast=None; frontend.robot=robot(contact=True)
        policy.close_witness=dict(issued_at=0.,confirmed_after=.05)
        self.assertIsNone(policy.select()); self.assertEqual(policy.phase,'acquire')
        frontend.collision.held_local=Pose([0.,0.,0.])
        action=policy.select()
        self.assertEqual(policy.phase,'lift'); self.assertEqual(action[7],0.)
        self.assertEqual(policy.events[0]['event'],'own_close_and_supported_public_capture_enter_lift')

    def test_original_frontend_still_observes_tracker_and_support_once(self):
        frontend=object.__new__(BlockPublicFrontend)
        raw=ObjectState('target',.05,Pose([0.,0.,.1]),[[0.,0.,.1],[.01,0.,.1],[0.,.01,.1]],
                        np.zeros(3),np.zeros(3))
        prior=NS(dimensions=np.full(3,.05))
        frontend.tracker=NS(prior=prior,observe=Mock(return_value=(raw,{})))
        frontend.marker=NS(observe=Mock(return_value=(Pose([.5,0.,.1]),{})))
        frontend.collision=NS(held_local=Pose([0.,0.,-.05]),observe_contact=Mock())
        frontend.config=dict(warmup_ticks=1,forecast_seconds=.5,dt=.05)
        frontend.dynamics=None; frontend.close_confirmed=True; frontend.frames=0
        frontend.events=[]; frontend.last_time=None; frontend.goal_pose=None
        observed=robot(.05,contact=True)
        frontend.observe({'timestamp':.05},observed)
        frontend.tracker.observe.assert_called_once(); frontend.marker.observe.assert_called_once()
        frontend.collision.observe_contact.assert_called_once_with(observed,raw,prior,True)
        self.assertEqual(frontend.object.object_id,'target'); self.assertTrue(frontend.events[0]['held_transform'])

    def test_active_receipt_does_not_claim_unexecuted_joint_collision_checks(self):
        env=Env(); frontend,policy=policy_for(env)
        dispatched=block.AuthorBlockRequestedOpeningWitness(policy,max_events=1)
        for index in range(2):
            opening,proof,proposal=dispatched.qualify_opening(1.,1.)
            dispatched.confirm_opening(opening,proof,proposal,confirmed_after=.05*(index+1))
        receipt=dispatched.receipt()
        self.assertFalse(receipt['held_geometry']['held_target_vs_arm_and_static_obstacles_checked'])
        self.assertTrue(receipt['legacy_held_geometry_receipt']['held_target_vs_arm_and_static_obstacles_checked'])
        self.assertTrue(receipt['held_geometry']['original_held_support_geometry_reused'])
        self.assertEqual(receipt['author_aligned_block_integration_v1120']['dropped_events'],1)

    def test_explicit_original_block_constructor_order_and_scope_use_no_extra_environment_methods(self):
        env=NS(robot_model=object()); events=[]; frontend=NS(); policy=NS(); sentinel=(frontend,policy,object())
        def collision(current):self.assertIs(current,env); events.append('collision'); return 'original_collision'
        def make_frontend(root,config,task,method,model,owned):
            self.assertIs(model,env.robot_model); self.assertEqual(owned,'original_collision')
            events.append('frontend'); return frontend
        def make_policy(current):self.assertIs(current,frontend); events.append('policy'); return policy
        with patch.object(block,'PublicBlockCollision',side_effect=collision),\
             patch.object(block,'BlockPublicFrontend',side_effect=make_frontend),\
             patch.object(block,'BlockGeometryIntents',side_effect=make_policy),\
             patch.object(block,'adapt',return_value=sentinel) as adapt:
            with block.integrated_block_components(Path('/fixture'),CONFIG,block.BLOCKS[0],'cv_geometry',env) as result:
                self.assertEqual(result,sentinel)
            self.assertEqual(events,['collision','frontend','policy'])
            adapt.assert_called_once_with(CONFIG,frontend,policy)
        with self.assertRaises(ValueError):block.construct(Path('/fixture'),CONFIG,'remove_cups_from_rotating_frame','cv_geometry',env)


if __name__=='__main__':unittest.main()
