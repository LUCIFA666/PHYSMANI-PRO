"""Pure CPU contracts for retaining already evaluated first-release gates."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.contracts import Pose, RobotState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_basketball_release_v558 import AuthorBasketballReleaseIntents
from our_mani.multitask.physmani_terminal_opening_v620 import BuzzCurveIntents
from our_mani.multitask.retained_preclose_policy_integration_v1121 import RetainedPrecloseWitness
from our_mani.multitask.action_conditioned_mode_observer_v1115b import ActionConditionedPublicModeObserver
from our_mani.multitask import author_native_release_core_v1164 as m

ROOT=Path(__file__).resolve().parents[1]


def rule(task):
    basketball=task in m.BASKETBALL_TASKS
    return dict(schema=m.SCHEMA,task=task,changed_factor=m.CHANGED_FACTOR,
        controlled_phase='deposit' if basketball else 'release',
        inherited_author_release_binding=dict(task=task,
            **{'release_distance_m' if basketball else 'author_terminal_release_distance_m':.03},
            source_sha256=m.PINS[m.BALL_SOURCE if basketball else m.BUZZ_SOURCE]))


def fixture(distance=.82,*,task=m.TASKS[0],angle=False):
    basketball=task in m.BASKETBALL_TASKS
    source_type=AuthorBasketballReleaseIntents if basketball else BuzzCurveIntents
    source=source_type.__new__(source_type)
    robot=RobotState(2.7,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),
                     gripper_open=False,contact=True)
    source.frontend=NS(state=PolicyState(robot,{}, {},{}),actor=NS(requested_opening=0.))
    phase='deposit' if basketball else 'release'
    source.task=task;source.phase_order=(('approach','acquire','lift','deposit') if basketball
                                        else ('approach','acquire','lift','follow','release'))
    source.phase_index=len(source.phase_order)-1
    source.phase_epoch=2.25;source.finished=False;source.events=[];source.gripper_events=[]
    identity='ball' if basketball else 'wand'
    source.selected=dict(phase=phase,reference_id='hoop' if basketball else 'target',manipulated_ids=(identity,),opening=1.,selected_at=2.25)
    source.close_witness={'phase_index':1,'confirmed_after':1.7,'manipulated_ids':[identity]}
    source.release_rule=dict(release_distance_m=.03)
    if not basketball:source.curve_rule={};source.cursors={};source.curve_events=[]
    calls=[];hook=[];target=[None if distance is None else Pose(np.array([distance,0.,0.]),
        np.diag([-1.,-1.,1.]) if angle else np.eye(3))]
    def completion(owned,goal):calls.append(goal);return target[0]
    source._completion_target=MethodType(completion,source)
    def confirmed(raw,opening,proof,proposed,*,confirmed_after):
        hook.append(dict(opening=opening,proposed=deepcopy(proposed),
            confirmed_after=confirmed_after,own_close=deepcopy(raw.close_witness)))
    dispatched=RetainedPrecloseWitness(source,on_confirmed_opening=confirmed)
    active=m.RetainedAuthorReleaseWitness(dispatched,rule(task))
    return NS(source=source,dispatched=dispatched,active=active,calls=calls,hook=hook,target=target)


class QualificationContracts(unittest.TestCase):
    def test_actual558_defer_is_kept_once_and_zero_does_not_create_close_or_open_witness(self):
        f=fixture();before=f.source.close_witness
        command,proof,proposed=f.active.qualify_opening(1.,0.)
        self.assertEqual(command,0.);self.assertIsNone(proposed);self.assertEqual(len(f.calls),1)
        self.assertEqual(proof['legacy_qualification']['qualified_opening'],0.)
        self.assertFalse(proof['legacy_geometric_qualification_is_diagnostic_only'])
        self.assertEqual(f.hook,[]);self.assertEqual(f.source.gripper_events,[])
        self.assertIs(f.source.close_witness,before)
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertEqual(f.source.frontend.actor.requested_opening,0.)
        self.assertIs(f.source.close_witness,before);self.assertEqual(len(f.hook),1)
        self.assertEqual(f.hook[0]['opening'],0.);self.assertIsNone(f.hook[0]['proposed'])
        self.assertFalse(f.source.gripper_events[-1]['own_close_edge'])

    def test_actual558_allow_is_retained_and_angle_does_not_add_an_extra_guard(self):
        f=fixture(.02,angle=True)
        command,proof,proposed=f.active.qualify_opening(1.,0.)
        self.assertEqual(command,1.);self.assertIsNone(proposed);self.assertEqual(len(f.calls),1)
        self.assertGreater(proof['current_orientation_error_rad'],3.)
        self.assertEqual(f.hook,[])
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertEqual(f.hook[0]['opening'],1.)
        self.assertEqual(f.source.frontend.actor.requested_opening,1.)

    def test_unknown_target_keeps_original_defer_without_retry_or_added_helper(self):
        f=fixture(None)
        command,proof,proposed=f.active.qualify_opening(1.,0.)
        self.assertEqual(command,0.);self.assertIsNone(proposed);self.assertEqual(len(f.calls),1)
        self.assertIsNone(proof['current_position_error_m'])
        self.assertEqual(f.hook,[])

    def test_both_Buzz_tasks_keep_original602_first_gate_and_terminal_opening_MRO(self):
        for task in m.BUZZ_TASKS:
            with self.subTest(task=task):
                f=fixture(.10,task=task)
                command,proof,proposed=f.active.qualify_opening(1.,0.)
                self.assertEqual(command,0.);self.assertIsNone(proposed);self.assertEqual(len(f.calls),1)
                self.assertEqual(proof['reason'],'defer_Buzz_current_public3cm_release')
                self.assertEqual(proof['legacy_qualification']['qualified_opening'],0.)
                f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
                self.assertEqual(f.hook[-1]['opening'],0.)
                f.target[0]=Pose(np.array([.02,0.,0.]))
                command,proof,proposed=f.active.qualify_opening(1.,0.)
                self.assertEqual(command,1.);self.assertIsNone(proposed);self.assertEqual(len(f.calls),2)
                self.assertEqual(proof['reason'],'original_author_Buzz_current_public3cm_release')
                f.active.confirm_opening(command,proof,proposed,confirmed_after=3.25)
                self.assertEqual(f.hook[-1]['opening'],1.)
                f.target[0]=Pose(np.array([10.,0.,0.]))
                command,proof,proposed=f.active.qualify_opening(1.,1.)
                self.assertEqual(command,1.);self.assertEqual(len(f.calls),2)
                self.assertEqual(proof['legacy_qualification']['proof']['reason'],'preserve_actual_terminal_open_request')

    def test_original_V6_apply_closed_defer_does_not_issue_grasp_or_release(self):
        from our_mani.sensor_alignment.physmani_executor_v6 import _apply_direct_target
        calls=[];target=[np.zeros(7)]
        class ArmCPU:
            def get_joint_positions(self):calls.append('get_q');return np.zeros(7)
            def set_joint_target_positions(self,q):calls.append('set_q');target[0]=np.asarray(q)
            def get_joint_target_positions(self):calls.append('get_target');return target[0]
        class GripperCPU:
            def grasp(self,obj):calls.append('grasp')
            def release(self):calls.append('release')
            def actuate(self,opening,*,velocity):calls.append(('actuate',opening))
        env=NS(ledger=NS(before=lambda:calls.append('before')),arm=ArmCPU(),gripper=GripperCPU(),
            task=NS(get_graspable_objects=lambda:(_ for _ in ()).throw(AssertionError('unexpected regrasp'))),
            robot_model=NS(lower=np.full(7,-10.),upper=np.full(7,10.)),_requested_opening=0.,
            protocol={'gripper_velocity':.2},_tick=lambda phase:calls.append(('tick',phase)))
        # Native API names above are CPU stubs. The exact originalV6 function
        # is invoked;1164 neither adds an actuator call nor synthesizes an edge.
        f=fixture(.82)
        command,proof,proposed=f.active.qualify_opening(1.,0.)
        _apply_direct_target(env,np.zeros(7),opening=command,phase='original_grip_tick')
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertEqual(calls,['before','get_q','set_q',('actuate',0.),('tick','original_grip_tick'),'get_target'])
        self.assertEqual(env._requested_opening,0.);self.assertEqual(f.hook[-1]['opening'],0.)
        self.assertFalse(any(x in ('grasp','release') for x in calls))
        self.assertEqual(len(f.source.gripper_events),1)

    def test_Buzz_earlier_arm_ticks_and_follow_are_exact_parent_not_first_open(self):
        f=fixture(.82,task=m.BUZZ_TASKS[0])
        command,proof,proposed=f.active.qualify_opening(0.,0.)
        self.assertEqual(command,0.);self.assertEqual(f.calls,[])
        self.assertEqual(proof['legacy_qualification']['proof']['reason'],'await_original_RML_new_gripper_request')
        self.assertFalse(f.active.events[-1]['retained_original_release'])
        f.source.selected.update(phase='follow',opening=0.)
        command,proof,proposed=f.active.qualify_opening(0.,0.)
        self.assertEqual(command,0.);self.assertEqual(f.calls,[])
        self.assertFalse(f.active.events[-1]['retained_original_release'])

    def test_already_open_is_exact_parent_and_never_reclosed_at_far_target(self):
        f=fixture()
        command,proof,proposed=f.active.qualify_opening(1.,1.)
        self.assertEqual(command,1.);self.assertIsNone(proposed);self.assertEqual(f.calls,[])
        self.assertNotIn('original_public_release_qualification_retained',proof)
        self.assertFalse(f.active.events[-1]['retained_original_release'])

    def test_original_acquire_preclose_and_proposal_remain_parent1121(self):
        f=fixture(.04)
        f.source.selected.update(phase='acquire',reference_id='ball',opening=0.)
        f.source.phase_index=1;f.source.close_witness=None
        current=f.source.frontend.state.robot
        f.source.frontend.state=PolicyState(RobotState(current.timestamp,current.pose,np.zeros(3),np.zeros(3),
            np.zeros(7),np.zeros(7),gripper_open=True,contact=False),{},{},{})
        command,proof,proposed=f.active.qualify_opening(0.,1.)
        self.assertEqual(command,1.);self.assertIsNone(proposed)
        self.assertTrue(proof['original_public_preclose_qualification_retained'])
        f.target[0]=Pose(np.array([.01,0.,0.]))
        command,proof,proposed=f.active.qualify_opening(0.,1.)
        self.assertEqual(command,0.);self.assertIsNotNone(proposed)
        self.assertIsNone(f.source.close_witness);self.assertEqual(f.hook,[])
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertIsNotNone(f.source.close_witness)
        self.assertEqual(f.hook[-1]['proposed'],proposed)
        self.assertEqual(len(f.calls),2)

    def test_actual_closed_defer_does_not_release_public_held_association(self):
        f=fixture();observer=ActionConditionedPublicModeObserver(('ball','hoop'),('ball',),.15)
        observer.held='ball';observer.held_provenance='own_confirmed_close_intent_and_shared_contact_and_public_candidate'
        def hook(source,opening,proof,proposed,*,confirmed_after):
            return observer.on_confirmed_opening(selected=source.selected,opening=opening,proof=proof,
                proposed=proposed,confirmed_after=confirmed_after,previous_states={},
                confirmed_witness=source.close_witness,phase_index=source.phase_index)
        f.dispatched.on_confirmed_opening=hook
        command,proof,proposed=f.active.qualify_opening(1.,0.)
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertEqual(observer.held,'ball');self.assertFalse(observer.open_command_active)
        self.assertFalse(any(e['event']=='confirmed_public_open_dispatch' for e in observer.events))
        self.assertIsNone(observer.pending_close)

    def test_valid_pending_token_is_not_consumed_by_a_same_closed_confirmation(self):
        # A real pending acquire edge must not leak into deposit. This test
        # exercises the unchanged acquire path, where pending is still valid.
        f=fixture();observer=ActionConditionedPublicModeObserver(('ball','hoop'),('ball',),.15)
        observer.pending_close=dict(identity='ball',phase_index=1,issued_at=1.,confirmed_after=1.05)
        f.source.selected.update(phase='acquire',reference_id='ball',opening=0.)
        f.source.phase_index=1;f.source.close_witness=dict(phase_index=1,manipulated_ids=['ball'])
        def hook(source,opening,proof,proposed,*,confirmed_after):
            observer.on_confirmed_opening(selected=source.selected,opening=opening,proof=proof,proposed=proposed,
                confirmed_after=confirmed_after,previous_states={},confirmed_witness=source.close_witness,phase_index=1)
        f.dispatched.on_confirmed_opening=hook
        token=observer.pending_close
        command,proof,proposed=f.active.qualify_opening(0.,0.)
        self.assertEqual(command,0.);self.assertIsNone(proposed)
        f.active.confirm_opening(command,proof,proposed,confirmed_after=2.75)
        self.assertIs(observer.pending_close,token)
        self.assertFalse(observer.open_command_active)

    def test_parent_exception_identity_and_bad_legacy_proof_are_not_substituted(self):
        f=fixture();error=ValueError('original qualifier')
        with patch.object(f.dispatched,'qualify_opening',side_effect=error):
            try:f.active.qualify_opening(1.,0.)
            except ValueError as actual:self.assertIs(actual,error)
            else:self.fail('original exception lost')
        self.assertEqual(f.hook,[])
        for bad in ({}, {'legacy_qualification':{'qualified_opening':None,'proof':{}}}):
            with patch.object(f.dispatched,'qualify_opening',return_value=(1.,bad,None)) as called:
                with self.assertRaises(m.ProtocolInvariantError):f.active.qualify_opening(1.,0.)
                called.assert_called_once_with(1.,0.)

    def test_receipt_is_bounded_and_qualification_never_claims_confirmation(self):
        f=fixture();f.active=m.RetainedAuthorReleaseWitness(f.dispatched,rule(f.source.task),max_events=1)
        for _ in range(2):f.active.qualify_opening(1.,0.)
        out=f.active.receipt();own=out['author_native_release_core_v1164']
        self.assertEqual(own['dropped_qualifications'],1);self.assertEqual(len(own['qualification_events']),1)
        self.assertEqual(f.hook,[]);self.assertEqual(f.source.gripper_events,[])
        self.assertTrue(own['qualification_does_not_confirm_dispatch_or_release'])
        self.assertIn('original558',out['physical_grasp_gate']['release_and_other_openings'])


class ScopeAndBindingContracts(unittest.TestCase):
    def test_full_parent_lifetime_same_executor_final_dispatch_and_exception_restore(self):
        f=fixture();frontend=object();executor=NS(skill_policy=f.dispatched);order=[];original_error=RuntimeError('body')
        @contextmanager
        def parent(*args,**kwargs):
            order.append(('parent_enter',kwargs));yield frontend,f.dispatched,executor;order.append('parent_exit')
        events={}
        with patch.object(m.parent,'integrated_components',side_effect=parent),\
             patch.object(m,'task_rule_binding',return_value=rule(f.source.task)):
            with m.integrated_components(ROOT,{},f.source.task,object(),events=events,max_events=20) as triple:
                self.assertIs(triple[0],frontend);self.assertIs(triple[2],executor)
                self.assertIs(executor.skill_policy,triple[1]);self.assertIs(triple[1].source_policy,f.source)
            self.assertIs(executor.skill_policy,f.dispatched);self.assertTrue(events['restored'])
            try:
                with m.integrated_components(ROOT,{},f.source.task,object()):raise original_error
            except RuntimeError as actual:self.assertIs(actual,original_error)
            self.assertIs(executor.skill_policy,f.dispatched)
        self.assertEqual(order[0][1],dict(events=events,max_events=20))

    def test_every_other_route_returns_exact_parent_objects_and_receipt(self):
        triple=(object(),object(),object());sink={};calls=[]
        @contextmanager
        def parent(*args,**kwargs):calls.append(kwargs);yield triple
        for task in (*m.parent.RING_TASKS,'push_moving_button','remove_cups_from_rotating_frame','put_rubbish_in_moving_bin'):
            with patch.object(m.parent,'integrated_components',side_effect=parent),\
                 patch.object(m,'task_rule_binding',side_effect=AssertionError('other route altered')):
                with m.integrated_components(ROOT,{},task,object(),events=sink,max_events=7) as result:
                    self.assertIs(result,triple)
        self.assertTrue(all(c==dict(events=sink,max_events=7) for c in calls))
        sentinel=object()
        with patch.object(m.parent,'ring_rule_binding',return_value=sentinel):
            self.assertIs(m.ring_rule_binding(ROOT,{},m.parent.RING_TASKS[0]),sentinel)

    def test_real_four_task_rules_use_existing_source3cm_and_configuration(self):
        for task in m.TASKS:
            p=ROOT/'artifacts/framework/multitask_r10/post_training_qualification_v680_setup'/(task+'_ours_configuration.json')
            config=json.loads(p.read_text());out=m.task_rule_binding(ROOT,config,task)
            basketball=task in m.BASKETBALL_TASKS
            threshold='release_distance_m' if basketball else 'author_terminal_release_distance_m'
            config_key='author_basketball_release_v558' if basketball else 'public_Buzz_curve_v602'
            self.assertEqual(out['inherited_author_release_binding'][threshold],.03)
            self.assertEqual(out['changed_factor'],m.CHANGED_FACTOR);json.dumps(out,allow_nan=False)
            bad=deepcopy(config);bad[config_key][task][threshold]=.04
            with self.assertRaises(m.ProtocolInvariantError):m.task_rule_binding(ROOT,bad,task)

    def test_full_binding_has_complete_parent_and_no_new_physical_rule(self):
        p=ROOT/'artifacts/framework/multitask_r10/post_training_qualification_v680_setup'/(m.TASKS[0]+'_ours_configuration.json')
        config=json.loads(p.read_text());value=m.binding(ROOT,config['baseline_repo'])
        self.assertEqual(value['parent']['schema'],m.parent.SCHEMA)
        self.assertEqual(set(value['sources_sha256']),set(m.SOURCES))
        self.assertTrue(set(m.parent.SOURCES)<=set(m.SOURCES))
        self.assertFalse(value['new_numeric_thresholds']);self.assertEqual(value['original_qualifier_calls_per_counted_command'],1)
        self.assertEqual(value['additional_native_getters_observe_predict_predicate_calls'],0)
        self.assertFalse(value['Native_grasp_identity_verified']);self.assertFalse(value['release_success_guaranteed'])
        json.dumps(value,allow_nan=False)


if __name__=='__main__':unittest.main()
