"""v5 controller regressions, including public acquire/lift handover."""
from pathlib import Path
import sys,types
# The native environment has an unrelated installed 'tests' package.
package=types.ModuleType('tests');package.__path__=[str(Path(__file__).resolve().parent)];sys.modules['tests']=package
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.multitask.causal_safety_brake_reference_v12 import ContinuousReferenceManager,SCHEMA,install_continuous_reference
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.interaction import (CandidateConfig,MPCConfig,GripperTimingConfig,CandidateGenerator,
    SE3InteractionMPC,InteractionController)
from our_mani.reachability import TimedReachability
from our_mani.contracts import Pose,ObjectState
from our_mani.multitask.pipeline import PolicyState
from our_mani.dynamics import ConstantVelocityDynamics
from scipy.spatial.transform import Rotation
from tests.test_feedback_interaction import GateModel,robot
from tests.test_persistent_reference_feedforward import owner
from tests.test_servo_response_actor_backend import prepared


def fixture():
    model,controller,_=owner()
    controller.generator.config.dt=.05;controller.mpc=SimpleNamespace(config=SimpleNamespace(dt=.05));controller.phase_index=0;controller.library=SimpleNamespace(phase_order=('approach','acquire','lift'))
    manager=ContinuousReferenceManager(controller);manager.forecast_observed_until=.5
    return model,controller,manager


def seed_reference(manager,q=.4,v=.7,time=.5):
    """A previously confirmed reference is a unit-test initial condition."""
    manager.last_confirmed=dict(time=time,joints=np.full(7,q).tolist(),nominal_velocity=np.full(7,v).tolist(),command='joint_trajectory')


def execution(q,time):
    tick=round(time/.05)
    return dict(schema='r10_continuous_reference_direct_counted_setpoint_v12',tick_before=tick-1,tick_after=tick,
        time_before=time-.05,time_after=time,applied_joint_reference=np.asarray(q).tolist(),
        actual_window_velocity_within_limits=True,actual_window_acceleration_within_limits=True)


def installed_actor():
    data,library,_=prepared();config=data['config'];model=GateModel()
    collision=lambda **kw:dict(feasible=True,evidence={});collision.bind_candidate=lambda **kw:None
    generator=CandidateGenerator(library,TimedReachability(model),collision,CandidateConfig(**config['candidate_config']))
    base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**config['mpc_config'])),
        predictor='cv',ranking='geometry',continuation_sha256='v5fixture',
        gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config']))
    controller=ObservedFeedbackController.from_controller(base,{'phase_tracking_slack_seconds':2.})
    actor=SimpleNamespace(controller=controller,requested_opening=1.,events=[],last_diagnostics={},
        metrics_report=lambda:{},act=lambda *args:dict(command='hold'))
    install_continuous_reference(actor)
    return actor,library,model


class ContinuousTests(unittest.TestCase):
    def test_lift_uses_TRAIN_closed_curve_even_when_open_proxy_remains_true(self):
        actor,library,_=installed_actor();controller=actor.controller
        template=library.for_phase('lift')[0]
        controller.goal=SimpleNamespace(phase='lift',template=template,selected_at=5.,
            candidate_id='lift',manipulated_ids=('ball',),reference_id='phase_origin:lift')
        measured=replace(robot(t=5.05),gripper_open=True,contact=True)
        opening=controller._gripper_opening(SimpleNamespace(robot=measured),1.,None)
        self.assertEqual(opening,0.)
        self.assertEqual(controller.gripper_timing_events[-1]['learned_phase_opening'],0.)
        self.assertIsNone(controller._close_edge_issued_at)

    def test_deposit_still_obeys_learned_release_curve(self):
        actor,library,_=installed_actor();controller=actor.controller
        template=library.for_phase('deposit')[0]
        controller.goal=SimpleNamespace(phase='deposit',template=template,selected_at=5.,
            candidate_id='deposit',manipulated_ids=('ball',),reference_id='hoop')
        state=SimpleNamespace(robot=replace(robot(t=6.),gripper_open=False,contact=True))
        self.assertEqual(controller._gripper_opening(state,0.,None),1.)

    def completed_acquire_fixture(self):
        from our_mani.multitask.interaction import InteractionCandidate,_world_target
        from our_mani.multitask.candidate_success import FEATURE_NAMES
        actor,library,model=installed_actor();c=actor.controller;m=actor.continuous_reference_manager
        template=library.for_phase('acquire')[0];now=.8
        ball=ObjectState('ball',now,Pose(np.array([.1,.2,.7])),np.array([[.1,.2,.7],[.11,.2,.7],[.1,.21,.7]]),
            np.zeros(3),np.zeros(3))
        target=_world_target(ball.pose,np.zeros(3),np.zeros(3),template,1.)[0]
        q=model.inverse(target,np.zeros(7))
        measured=replace(robot(q,np.zeros(7),now),pose=target,gripper_open=True,contact=True)
        state=PolicyState(measured,{'ball':ball},{'bodies':{'ball':dict(directly_visible=True,registration_valid=True,
            motion_mode='attached',mode_provenance='caller_controller_touch_action_proprioception',
            orientation_observable=False)}},{'ball':('ball',)})
        forecast=ConstantVelocityDynamics().predict([ball],1.1,.05);forecast.metadata['observed_until']=now
        c.phase_index=1;c.origin=target;c._close_edge_issued_at=.75
        reach=m.native_gate.check_joints(measured,q,np.zeros(7),.3)
        c.goal=InteractionCandidate('acquire', 'acquire',1,'ball',('ball',),template,0.,.3,target,
            np.zeros(3),np.zeros(3),ball.pose,np.zeros(3),np.zeros(3),ball,reach,0.,template.duration_seconds,
            2.,.5,True,{})
        c.attempts.begin('acquire','acquire',0.,np.zeros(len(FEATURE_NAMES)))
        # This already granted effective deadline is the old phase's fixed cap.
        c._phase_slack_tokens.add(('acquire',0.,1))
        m.last_confirmed=dict(time=now,joints=q.tolist(),nominal_velocity=np.zeros(7).tolist(),command='joint_trajectory')
        m.forecast_observed_until=now
        return actor,state,forecast

    def test_public_acquire_completes_and_selects_lift_in_same_decision_without_hold_tick(self):
        actor,state,forecast=self.completed_acquire_fixture();c=actor.controller;m=actor.continuous_reference_manager
        command=c.step(state,forecast,9.2)
        self.assertEqual(c.phase_index,2)
        self.assertEqual(command['phase'],'lift')
        self.assertEqual(command['command'],'joint_trajectory',c.last_diagnostics)
        self.assertEqual(command['opening'],0.)
        self.assertEqual(len(m.handover_events),1)
        self.assertEqual(m.handover_events[-1]['retired_phase_deadline'],2.)
        self.assertEqual(m.handover_events[-1]['additional_physics_ticks'],0)
        self.assertEqual(m.last_confirmed['time'],.8)
        self.assertIsNotNone(m.pending)
        self.assertTrue(c.attempts.records[0]['skill_success'])
        self.assertIsNone(c.attempts.records[0]['official_task_success'])
        self.assertEqual(len(c.events),1)

    def test_handover_never_bypasses_collision_or_suffix_budget(self):
        for collision,remaining in [(True,9.2),(False,.1)]:
            actor,state,forecast=self.completed_acquire_fixture();c=actor.controller;m=actor.continuous_reference_manager
            m.native_gate.model.dynamic_block=collision
            command=c.step(state,forecast,remaining)
            self.assertEqual(command['command'],'hold')
            self.assertIsNone(c.goal)
            self.assertIsNone(m.pending)
            self.assertEqual(c.phase_index,2)
            self.assertIsNone(m.handover_events[-1]['next_candidate_id'])
            self.assertTrue(m.handover_events[-1]['controller_diagnostics']['candidate_search']['rejections'])

    def test_safety_brake_can_finish_after_expired_task_phase_without_extending_it(self):
        model,c,m=fixture();seed_reference(m,q=.4,v=.4,time=.5)
        measured=replace(robot(t=.5),joints=np.full(7,.4),joint_velocities=np.full(7,.4))
        m.absolute_phase_deadline=.49
        self.assertTrue(m.make_brake(measured,1.,'expired_phase_requires_safe_stop'))
        self.assertEqual(m.absolute_phase_deadline,.49)
        self.assertLessEqual(m.plan.end,1.5)
        self.assertTrue(m.events[-1]['task_candidate_phase_deadline_unchanged'])
        self.assertFalse(m.events[-1]['total_budget_extended'])

    def test_acquire_stop_endpoint_retains_complete_dual_collision_gates(self):
        model,c,m=fixture();c.phase_index=1;seed_reference(m,q=.4,v=0.,time=.5)
        measured=replace(robot(t=.5),joints=np.full(7,.4),joint_velocities=np.zeros(7))
        result=m.dual_gate.check_joints(measured,np.full(7,.45),np.full(7,.2),.9)
        self.assertTrue(result.feasible,result.reason)
        self.assertTrue(np.array_equal(result.joint_velocities,np.zeros(7)))
        self.assertTrue(result.evidence['actual_state_reachability_retained'])
        self.assertTrue(result.evidence['continuous_reference_proof']['feasible'])
        model.dynamic_block=True
        self.assertFalse(m.dual_gate.check_joints(measured,np.full(7,.45),np.full(7,.2),.9).feasible)

    def test_close_requires_actual_deceleration_and_stationary_confirmed_reference(self):
        actor,library,_=installed_actor();c=actor.controller;m=actor.continuous_reference_manager
        template=library.for_phase('acquire')[0];target=Pose(np.array([.1,.2,.7]),template.at(1.)[0].rotation)
        c.goal=SimpleNamespace(phase='acquire',template=template,selected_at=5.,candidate_id='stop_gate',manipulated_ids=('ball',),reference_id='ball')
        measured=replace(robot(t=5.75),pose=target)
        state=SimpleNamespace(robot=measured,perception_metadata={},role_bindings={})
        seed_reference(m,q=0.,v=.5,time=5.75)
        self.assertEqual(c._gripper_opening(state,1.,target),1.)
        self.assertIsNone(c._close_edge_issued_at)
        seed_reference(m,q=0.,v=0.,time=5.75)
        moving=replace(measured,joint_velocities=np.full(7,.4))
        self.assertEqual(c._gripper_opening(SimpleNamespace(robot=moving,perception_metadata={},role_bindings={}),1.,target),1.)
        self.assertEqual(c._gripper_opening(state,1.,target),0.)
        self.assertEqual(c._close_edge_issued_at,5.75)

    def test_active_reference_learned_close_uses_unchanged_geometry_gate_and_one_edge(self):
        actor,library,_=installed_actor();controller=actor.controller;template=library.for_phase('acquire')[0]
        target=Pose(np.array([.1,.2,.7]),template.at(1.)[0].rotation)
        controller.goal=SimpleNamespace(phase='acquire',template=template,selected_at=5.,candidate_id='stable_acquire',
            manipulated_ids=('ball',),reference_id='ball')
        measured=robot(t=5.75)
        measured=type(measured)(measured.timestamp,Pose(target.position+np.array([.015,0,0]),target.rotation),
            measured.linear_velocity,measured.angular_velocity,measured.joints,measured.joint_velocities)
        state=SimpleNamespace(robot=measured,perception_metadata={},role_bindings={})
        seed_reference(actor.continuous_reference_manager,q=0.,v=0.,time=5.75)
        result=controller._gripper_opening(state,1.,target)
        self.assertEqual(result,0.)
        self.assertEqual(controller._close_edge_issued_at,5.75)
        self.assertEqual(controller.gripper_timing_events[-1]['reason'],'observed_final_grasp_SE3_reached_issue_one_close_edge')
        self.assertEqual(controller._gripper_opening(state,1.,target),0.)
        self.assertEqual(controller._close_edge_issued_at,5.75)
        self.assertEqual(controller.gripper_timing_events[-1]['reason'],'one_close_edge_already_issued_this_attempt')
        # Starting another real attempt with the same2cm gate cannot close at
        # Context's observed2.8cm or before learned phase asks to close.
        controller.goal.candidate_id='new_acquire'
        far=type(measured)(5.75,Pose(target.position+np.array([.028,0,0]),target.rotation),
            measured.linear_velocity,measured.angular_velocity,measured.joints,measured.joint_velocities)
        self.assertEqual(controller._gripper_opening(SimpleNamespace(robot=far),1.,target),1.)
        self.assertIsNone(controller._close_edge_issued_at)
        self.assertEqual(controller.gripper_timing_events[-1]['reason'],'defer_close_until_observed_final_grasp_SE3')
        early=type(measured)(5.05,measured.pose,measured.linear_velocity,measured.angular_velocity,measured.joints,measured.joint_velocities)
        self.assertEqual(controller._gripper_opening(SimpleNamespace(robot=early),1.,target),1.)
        self.assertIsNone(controller._close_edge_issued_at)

    def rolling_fixture(self):
        actor,library,model=installed_actor();c=actor.controller;m=actor.continuous_reference_manager
        template=library.for_phase('acquire')[0];now=.8
        ball=ObjectState('ball',now,Pose(np.array([.1,.2,.7])),np.array([[.1,.2,.7],[.11,.2,.7],[.1,.21,.7]]),
            np.array([.1,0,0]),np.zeros(3))
        local=template.at(1.)[0];target=Pose(ball.pose.position+local.position,local.rotation)
        q=model.inverse(target,np.zeros(7));q[0]-=.02;v=np.zeros(7)
        measured=robot(q,v,now)
        state=PolicyState(measured,{'ball':ball},{'bodies':{'ball':dict(directly_visible=True,registration_valid=True,
            motion_mode='external',orientation_observable=False)}},{'ball':('ball',)})
        forecast=ConstantVelocityDynamics().predict([ball],1.1,.05)
        forecast.metadata['observed_until']=now
        c.goal=SimpleNamespace(phase='acquire',template=template,selected_at=0.,phase_budget=2.,suffix_budget=.5,
            candidate_id='stable_acquire',reference_id='ball',manipulated_ids=('ball',),reachability=SimpleNamespace(evidence={}))
        c.origin=Pose(np.zeros(3))
        m.current_state=state;m.current_forecast=forecast;m.forecast_observed_until=now
        m.last_confirmed=dict(time=now,joints=q.tolist(),nominal_velocity=v.tolist(),command='joint_trajectory')
        m.plan=type(owner()[1].trajectory)(.6,now,q.copy(),v.copy(),q.copy(),v.copy())
        m.plan_kind='braking';m.plan_candidate_id='completed_brake';m.invalidated=False
        return actor,model,measured

    def test_completed_brake_rolls_fresh_current_target_with_full_dual_gate_without_resetting_goal(self):
        actor,model,measured=self.rolling_fixture();m=actor.continuous_reference_manager;c=actor.controller;goal=c.goal
        result=m.check_stream(measured,.05)
        self.assertTrue(result.feasible,result.reason)
        self.assertEqual(m.plan_kind,'current_acquire_rolling')
        self.assertIs(c.goal,goal);self.assertEqual(goal.selected_at,0.);self.assertEqual(goal.phase_budget,2.)
        self.assertEqual(goal.candidate_id,'stable_acquire')
        self.assertTrue(m.rolling_events[-1]['actual_and_reference_full_gate_feasible'])
        self.assertTrue(m.rolling_events[-1]['committed'])
        self.assertTrue(np.any(abs(result.joints-measured.joints)>1e-8))
        self.assertEqual(m.pending['reference_kind'],'current_acquire_rolling')

    def test_rolling_stale_forecast_or_fresh_collision_never_commits_new_plan(self):
        actor,model,measured=self.rolling_fixture();m=actor.continuous_reference_manager;old=m.plan
        m.current_forecast.metadata['observed_until']=.0
        self.assertFalse(m.roll_current_reference(measured));self.assertIs(m.plan,old)
        self.assertFalse(m.rolling_events[-1]['committed'])
        m.current_forecast.metadata['observed_until']=measured.timestamp;model.dynamic_block=True
        self.assertFalse(m.roll_current_reference(measured));self.assertIs(m.plan,old)
        self.assertTrue(m.rolling_events[-1]['trials'])
        self.assertFalse(m.rolling_events[-1]['committed'])

    def test_pending_reference_does_not_advance_confirmed_ledger_and_only_tick_callback_can_commit(self):
        _,controller,m=fixture();seed_reference(m)
        measured=robot(np.full(7,.38),np.full(7,.5),.5)
        dual=m.dual_gate.check_joints(measured,np.full(7,.7),np.zeros(7),.6)
        self.assertTrue(dual.feasible,dual.reason)
        controller.goal.reachability=dual;controller.goal.arrival_time=1.1
        m.adopt_selected_plan(measured);before=dict(m.last_confirmed)
        checked=m.check_stream(measured,.05);self.assertTrue(checked.feasible)
        pending=m.pending;m.pending_dispatch=pending
        self.assertEqual(m.last_confirmed,before);self.assertFalse(pending['issued'])
        cert=dict(pending,dispatch_authorized=True)
        command=dict(command='joint_trajectory',feedforward_execution=cert)
        q=np.asarray(pending['reference_to_joints']);v=np.asarray(pending['reference_to_velocity'])
        with self.assertRaises(ValueError):m.confirm_actual_issued(time=.55,joints=q,nominal_velocity=v,command=command,execution={})
        self.assertEqual(m.last_confirmed,before)
        m.confirm_actual_issued(time=.55,joints=q,nominal_velocity=v,command=command,execution=execution(q,.55))
        self.assertEqual(m.last_confirmed['joints'],q.tolist());self.assertTrue(pending['issued'])
        with self.assertRaises(ValueError):m.confirm_actual_issued(time=.55,joints=q,nominal_velocity=v,command=command,execution=execution(q,.55))

    def test_selected_trial_proof_has_exact_confirmed_reference_start_not_last_trial_or_actual_start(self):
        _,controller,m=fixture();seed_reference(m)
        measured=robot(np.full(7,.38),np.full(7,.5),.5)
        selected=m.dual_gate.check_joints(measured,np.full(7,.7),np.zeros(7),.6)
        other=m.dual_gate.check_joints(measured,np.full(7,.8),np.zeros(7),.9)
        self.assertTrue(selected.feasible);self.assertTrue(other.feasible)
        controller.goal.reachability=selected;controller.goal.arrival_time=1.1
        self.assertTrue(m.adopt_selected_plan(measured))
        self.assertTrue(np.allclose(m.plan.q0,.4));self.assertTrue(np.allclose(m.plan.v0,.7))
        self.assertTrue(np.allclose(m.plan.q1,.7))
        self.assertFalse(np.array_equal(m.plan.q0,measured.joints))

    def test_actual_feasible_retarget_is_rejected_when_reference_continuity_violates_original_caps(self):
        _,controller,m=fixture();seed_reference(m,q=.9,v=0.)
        measured=robot(np.zeros(7),np.zeros(7),.5)
        actual=m.native_gate.check_joints(measured,np.zeros(7),np.zeros(7),.3)
        self.assertTrue(actual.feasible)
        result=m.dual_gate.check_joints(measured,np.zeros(7),np.zeros(7),.3)
        self.assertFalse(result.feasible);self.assertTrue(result.reason.startswith('reference_continuity_'))
        self.assertEqual(m.retarget.successful_reissues_by_phase_index,{})
        self.assertEqual(m.confirmed_reissues_by_phase_index,{})

    def test_invalidation_brakes_from_last_reference_without_resuming_old_wallclock(self):
        _,controller,m=fixture();seed_reference(m)
        measured=robot(np.full(7,.38),np.full(7,.5),.5)
        old=controller.trajectory;m.plan=old;m.plan_kind='tracking';m.plan_candidate_id='old_virtual_plan'
        m.invalidate('observed_acceleration_limit',.5)
        self.assertIsNone(m.plan)
        self.assertTrue(m.make_brake(measured,9.5,'observed_acceleration_limit'))
        self.assertEqual(m.plan_kind,'braking');self.assertIsNot(m.plan,old)
        self.assertTrue(np.allclose(m.plan.q0,.4));self.assertTrue(np.allclose(m.plan.v0,.7))
        checked=m.check_stream(measured,.05);self.assertTrue(checked.feasible,checked.reason)
        self.assertTrue(np.all(checked.joints>.4));self.assertTrue(np.all(checked.joints<.44))
        self.assertEqual(controller.goal.selected_at,0.);self.assertEqual(controller.goal.phase_budget,3.3)

    def test_fresh_collision_blocks_both_reference_tracking_and_braking(self):
        model,_,m=fixture();seed_reference(m);model.dynamic_block=True
        measured=robot(np.full(7,.38),np.full(7,.5),.5)
        self.assertFalse(m.make_brake(measured,9.5,'collision'))
        self.assertEqual(m.events[-1]['event'],'no_feasible_continuous_brake')
        self.assertFalse(m.events[-1]['acceleration_limited_stop_verified'])

    def test_real_installer_final_hold_is_continuous_reference_and_pending_until_dispatch(self):
        data,library,_=prepared();config=data['config'];model=GateModel()
        collision=lambda **kw:dict(feasible=True,evidence={});collision.bind_candidate=lambda **kw:None
        generator=CandidateGenerator(library,TimedReachability(model),collision,CandidateConfig(**config['candidate_config']))
        base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**config['mpc_config'])),
            predictor='cv',ranking='geometry',continuation_sha256='continuous_fixture',
            gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config']))
        controller=ObservedFeedbackController.from_controller(base,{'phase_tracking_slack_seconds':2.})
        actor=SimpleNamespace(controller=controller,requested_opening=1.,events=[],last_diagnostics={},
            metrics_report=lambda:{},last_command={'command':'hold'})
        def original_act(observation,measured,remaining):
            actor.last_diagnostics={'collision_scene_readiness':{'ready':True}}
            actor.continuous_reference_manager.forecast_observed_until=measured.timestamp
            actor.requested_opening=0.;controller._close_edge_issued_at=measured.timestamp
            controller.gripper_timing_events.append(dict(time=measured.timestamp,commanded_opening=0.))
            return dict(command='hold',phase='approach',public_command_kind='hold')
        actor.act=original_act;install_continuous_reference(actor)
        m=actor.continuous_reference_manager;seed_reference(m)
        command=actor.act({},robot(np.full(7,.38),np.full(7,.5),.5),9.5)
        self.assertEqual(command['command'],'joint_trajectory')
        cert=command['feedforward_execution'];self.assertEqual(cert['schema'],SCHEMA)
        self.assertTrue(cert['dispatch_authorized']);self.assertFalse(cert['issued'])
        self.assertAlmostEqual(cert['issued_at'],.5);self.assertAlmostEqual(cert['reference_target_at'],.55)
        self.assertEqual(m.last_confirmed['time'],.5)
        self.assertEqual(command['joints'].tolist(),cert['reference_to_joints'])
        self.assertEqual(m.metrics()['schema'],SCHEMA)
        self.assertEqual(command['opening'],1.)
        self.assertFalse(controller.gripper_timing_events[-1]['actual_opening_request_issued'])
        plan=m.plan;target=np.asarray(cert['reference_to_joints']);velocity=np.asarray(cert['reference_to_velocity'])
        m.confirm_actual_issued(time=.55,joints=target,nominal_velocity=velocity,command=command,execution=execution(target,.55))
        self.assertTrue(controller.gripper_timing_events[-1]['actual_opening_request_issued'])
        self.assertEqual(controller.gripper_timing_events[-1]['actual_opening_request'],1.)
        current=np.full(7,.38)+.65*(target-.38)
        second=actor.act({},robot(current,(current-.38)/.05,.55),9.45)
        self.assertEqual(second['command'],'joint_trajectory')
        self.assertIs(m.plan,plan)
        self.assertEqual(second['feedforward_execution']['candidate_id'],cert['candidate_id'])


if __name__=='__main__':unittest.main()
