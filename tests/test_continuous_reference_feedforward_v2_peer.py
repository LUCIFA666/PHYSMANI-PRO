"""Independent CPU boundaries; native execution is not claimed by fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.intercept_planner import JointTrajectory
from our_mani.reachability import TimedReachability
from our_mani.multitask.continuous_reference_feedforward_v2 import ContinuousReferenceManager,SCHEMA
from tests.test_feedback_interaction import GateModel,robot
from tests.test_direct_counted_setpoint import Environment
from our_mani.multitask.continuous_direct_counted_setpoint_v2 import CountedDirectDispatcher

DISPATCH='r10_continuous_reference_direct_counted_setpoint_v2'


def execution(time,q,*,velocity=True,acceleration=True):
    tick=round(time/.05)
    return dict(schema=DISPATCH,tick_before=tick-1,tick_after=tick,time_after=time,
        applied_joint_reference=np.asarray(q).tolist(),actual_window_velocity_within_limits=velocity,
        actual_window_acceleration_within_limits=acceleration)


def fixture():
    model=GateModel();gate=TimedReachability(model)
    controller=SimpleNamespace(original_reachability=gate,phase_index=0,
        generator=SimpleNamespace(config=SimpleNamespace(dt=.05,maximum_observation_age=.1)),
        mpc=SimpleNamespace(config=SimpleNamespace(dt=.05)),goal=None,trajectory=None,
        feedback_events=[],gripper_timing_events=[])
    manager=ContinuousReferenceManager(controller)
    observed=robot(t=.05)
    manager.confirm_actual_issued(time=.05,joints=np.zeros(7),nominal_velocity=None,
        command=dict(command='hold'),execution=execution(.05,np.zeros(7)))
    checked=manager.dual_gate.check_joints(observed,np.full(7,.3),np.zeros(7),1.)
    assert checked.feasible,checked.reason
    controller.goal=SimpleNamespace(candidate_id='complete_dual_proof',reachability=checked,
        phase='approach',phase_index=0,selected_at=0.,phase_budget=3.3,suffix_budget=2.,arrival_time=1.05)
    controller.trajectory=JointTrajectory(.05,1.05,np.zeros(7),np.zeros(7),np.full(7,.3),np.zeros(7))
    manager.forecast_observed_until=.05
    return model,controller,manager


def propose(manager,observed):
    manager.pending=None
    manager.forecast_observed_until=observed.timestamp
    checked=manager.check_stream(observed,.05)
    assert checked.feasible,checked.reason
    manager.pending_dispatch=manager.pending
    certificate=deepcopy(manager.pending)
    certificate.update(issued=False,dispatch_authorized=True)
    return dict(command='joint_trajectory',joints=checked.joints.copy(),joint_velocities=checked.joint_velocities.copy(),
                feedforward_execution=certificate,opening=1.,phase='approach',public_command_kind='joint_trajectory')


def confirm(manager,command,*,acceleration=True):
    cert=command['feedforward_execution'];time=cert['time']+.05
    manager.confirm_actual_issued(time=time,joints=command['joints'],nominal_velocity=command['joint_velocities'],
        command=command,execution=execution(time,command['joints'],acceleration=acceleration))


class ContinuousReferenceIndependentTests(unittest.TestCase):
    def hold_actor(self):
        from tests.test_servo_response_actor_backend import prepared
        from our_mani.multitask.continuous_reference_feedforward_v2 import install_continuous_reference
        from our_mani.multitask.feedback_interaction import ObservedFeedbackController
        from our_mani.multitask.interaction import (InteractionController,CandidateGenerator,SE3InteractionMPC,
            CandidateConfig,MPCConfig,GripperTimingConfig)
        data,library,_=prepared();config=data['config'];model=GateModel()
        collision=lambda **kwargs:dict(feasible=True,evidence={});collision.bind_candidate=lambda **kwargs:None
        generator=CandidateGenerator(library,TimedReachability(model),collision,CandidateConfig(**config['candidate_config']))
        base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**config['mpc_config'])),
            predictor='cv',ranking='geometry',continuation_sha256='independent_continuous_peer_CPU',
            gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config']))
        controller=ObservedFeedbackController.from_controller(base,{'phase_tracking_slack_seconds':2.})
        actor=SimpleNamespace(controller=controller,requested_opening=1.,events=[],last_diagnostics={},
            metrics_report=lambda:{},last_command={'command':'hold'})
        def original_act(observation,measured,remaining):
            actor.last_diagnostics={'collision_scene_readiness':{'ready':True}}
            actor.continuous_reference_manager.forecast_observed_until=measured.timestamp
            return dict(command='hold',phase='approach',public_command_kind='hold')
        actor.act=original_act;install_continuous_reference(actor)
        manager=actor.continuous_reference_manager
        manager.last_confirmed=dict(time=.5,joints=np.full(7,.4).tolist(),nominal_velocity=np.full(7,.7).tolist(),command='joint_trajectory')
        return actor,manager

    def test_active_brake_is_not_restarted_by_repeated_inner_holds(self):
        actor,manager=self.hold_actor()
        first=actor.act({},robot(q=np.full(7,.38),v=np.full(7,.5),t=.5),9.5)
        self.assertEqual(first['command'],'joint_trajectory')
        self.assertEqual(manager.plan_kind,'braking')
        active=manager.plan;end=active.end
        confirm(manager,first)
        self.assertTrue(manager.braking_active(.55))
        second=actor.act({},robot(q=np.full(7,.39),v=np.full(7,.2),t=.55),9.45)
        self.assertEqual(second['command'],'joint_trajectory')
        self.assertIs(manager.plan,active)
        self.assertEqual(manager.plan.end,end)
        self.assertEqual(second['feedforward_execution']['reference_kind'],'braking')

    def test_brake_still_obeys_retained_deadline_after_controller_abandons_goal(self):
        _,controller,manager=fixture()
        manager.last_confirmed['time']=.5
        manager.last_confirmed['nominal_velocity']=np.full(7,.7).tolist()
        manager.absolute_phase_deadline=.52
        controller.goal=None
        self.assertFalse(manager.make_brake(robot(t=.5),9.5,'CPU_goal_abandoned'))
        self.assertIsNone(manager.plan)
        self.assertTrue(all(row['duration']<=.02+1e-8 for row in manager.events[-1]['trials']))

    def environment(self):
        class SevenJointEnvironment(Environment):
            @property
            def time(self):return self.ledger.ticks*self.ledger.dt
        env=SevenJointEnvironment();env.ledger.ticks=1
        env.arm.q=np.zeros(7);env.arm.target=np.zeros(7);env.arm.velocity=np.zeros(7)
        env.robot_model=SimpleNamespace(lower=np.full(7,-2.),upper=np.full(7,2.),
            velocity_limits=np.ones(7),acceleration_limits=np.full(7,6.))
        env.response=.5
        return env

    def test_real_dispatcher_and_manager_commit_only_after_actual_fixture_tick(self):
        _,_,manager=fixture();env=self.environment();dispatcher=CountedDirectDispatcher();dispatcher.bind_manager(manager)
        command=propose(manager,robot(t=.05));before=deepcopy(manager.last_confirmed)
        self.assertEqual(before['time'],env.time)
        result=dispatcher.dispatch(env,command)
        self.assertEqual(env.ledger.ticks,2);self.assertEqual(manager.last_confirmed['time'],env.time)
        np.testing.assert_array_equal(manager.last_confirmed['joints'],env.arm.target)
        np.testing.assert_array_equal(manager.last_confirmed['nominal_velocity'],command['joint_velocities'])
        self.assertFalse(np.array_equal(manager.last_confirmed['joints'],env.arm.q))
        self.assertEqual(result['persistent_reference_execution']['reference_target_at'],env.time)
        self.assertTrue(manager.reference_events[-1]['dispatch_confirmed'])

    def test_failed_fixture_tick_cannot_commit_manager_reference(self):
        _,_,manager=fixture();env=self.environment();dispatcher=CountedDirectDispatcher();dispatcher.bind_manager(manager)
        command=propose(manager,robot(t=.05));before=deepcopy(manager.last_confirmed)
        def fail(phase):raise RuntimeError('CPU injected physical tick failure')
        env._tick=fail
        with self.assertRaises(RuntimeError):dispatcher.dispatch(env,command)
        self.assertEqual(manager.last_confirmed,before)
        self.assertFalse(manager.reference_events[-1]['issued']);self.assertEqual(dispatcher.events,[])

    def test_proposal_does_not_advance_last_actual_reference_until_callback(self):
        _,_,manager=fixture();before=deepcopy(manager.last_confirmed)
        command=propose(manager,robot(t=.05))
        self.assertEqual(manager.last_confirmed,before)
        self.assertFalse(manager.pending['issued']);self.assertFalse(manager.pending['dispatch_confirmed'])
        self.assertFalse(command['feedforward_execution']['issued'])
        confirm(manager,command)
        self.assertEqual(manager.last_confirmed['time'],.1)
        np.testing.assert_array_equal(manager.last_confirmed['joints'],command['joints'])
        self.assertTrue(manager.reference_events[-1]['issued'])
        self.assertIsNone(manager.pending_dispatch)

    def test_wrong_callback_or_duplicate_does_not_consume_actual_reference(self):
        _,_,manager=fixture();command=propose(manager,robot(t=.05));before=deepcopy(manager.last_confirmed)
        with self.assertRaises(ValueError):
            manager.confirm_actual_issued(time=.1,joints=np.full(7,.9),nominal_velocity=command['joint_velocities'],
                command=command,execution=execution(.1,np.full(7,.9)))
        self.assertEqual(manager.last_confirmed,before);self.assertFalse(manager.pending['issued'])
        confirm(manager,command);after=deepcopy(manager.last_confirmed)
        with self.assertRaises(ValueError):confirm(manager,command)
        self.assertEqual(manager.last_confirmed,after)

    def test_stream_start_is_last_issued_reference_and_nominal_derivative_under_lag(self):
        _,_,manager=fixture();command=propose(manager,robot(t=.05));confirm(manager,command)
        issued=deepcopy(manager.last_confirmed)
        lagged=robot(q=np.zeros(7),v=np.zeros(7),t=.1)
        next_command=propose(manager,lagged);event=manager.pending
        np.testing.assert_array_equal(event['reference_from_joints'],issued['joints'])
        np.testing.assert_array_equal(event['reference_from_velocity'],issued['nominal_velocity'])
        self.assertGreater(np.linalg.norm(np.array(event['reference_from_joints'])-lagged.joints),0.)
        self.assertEqual(manager.last_confirmed,issued)
        self.assertFalse(next_command['feedforward_execution']['actual_tracking_guaranteed'])

    def test_full_actual_gate_cannot_be_replaced_by_reference_only_gate(self):
        _,_,manager=fixture();observed=robot(q=np.full(7,-.5),v=np.zeros(7),t=.05)
        result=manager.dual_gate.check_joints(observed,np.full(7,.01),np.zeros(7),.1)
        self.assertFalse(result.feasible)
        self.assertIn(result.reason,('insufficient_arrival_time','trajectory_joint_speed_limit','trajectory_joint_acceleration_limit'))

    def test_failed_reference_retarget_keeps_old_plan_and_does_not_consume_counter(self):
        _,controller,manager=fixture();manager.adopt_selected_plan(robot(t=.05))
        old_goal,old_plan=controller.goal,manager.plan
        # The measured q could reach this small target, but a previously issued
        # distant reference cannot meet its very short arrival deadline.
        manager.last_confirmed['joints']=np.full(7,.8).tolist()
        result=manager.dual_gate.check_joints(robot(t=.05),np.full(7,.01),np.zeros(7),.1)
        self.assertFalse(result.feasible);self.assertTrue(result.reason.startswith('reference_continuity_'))
        self.assertIs(controller.goal,old_goal);self.assertIs(manager.plan,old_plan)
        self.assertEqual(manager.confirmed_reissues_by_phase_index,{})
        self.assertEqual(manager.retarget.successful_reissues_by_phase_index,{})

    def test_real_retarget_selected_proof_uses_confirmed_reference_then_failure_preserves_it(self):
        from tests.test_causal_retarget_interaction import CausalRetargetTests
        source_fixture=CausalRetargetTests()
        controller,_,model,scene=source_fixture.fixture(slack=2.)
        manager=ContinuousReferenceManager(controller,retarget_config=dict(minimum_issue_interval_seconds=.1,
            scheduled_issue_interval_seconds=.1,reference_shift_threshold_m=.0001,arrival_delays_seconds=(.3,.15,.05)))
        controller.original_reachability=manager.dual_gate;controller.generator.reachability=manager.dual_gate
        observed,forecast=source_fixture.observed(.15)
        reference_q=observed.robot.joints.copy()+.02;reference_v=np.full(7,.03)
        manager.last_confirmed=dict(time=.15,joints=reference_q.tolist(),nominal_velocity=reference_v.tolist(),command='joint_trajectory')
        old_deadline=controller.goal.selected_at+controller.goal.phase_budget
        event=manager.retarget.before_step(observed,forecast,9.85)
        self.assertEqual(event['action'],'new_candidate_issue',event)
        self.assertTrue(manager.adopt_selected_plan(observed.robot))
        np.testing.assert_array_equal(manager.plan.q0,reference_q)
        np.testing.assert_array_equal(manager.plan.v0,reference_v)
        self.assertEqual(controller.goal.selected_at+controller.goal.phase_budget,old_deadline)
        self.assertEqual(manager.confirmed_reissues_by_phase_index,{})
        old_goal,old_plan=controller.goal,manager.plan;count=deepcopy(manager.retarget.successful_reissues_by_phase_index)
        observed,forecast=source_fixture.observed(.3);manager.last_confirmed['time']=.3;scene.blocked=True
        event=manager.retarget.before_step(observed,forecast,9.7)
        self.assertEqual(event['action'],'reissue_rejected_original_candidate_retained')
        self.assertIs(controller.goal,old_goal);self.assertIs(manager.plan,old_plan)
        self.assertEqual(manager.retarget.successful_reissues_by_phase_index,count)
        self.assertEqual(manager.confirmed_reissues_by_phase_index,{})

    def test_observed_violation_invalidates_old_wallclock_plan_before_braking(self):
        _,controller,manager=fixture();command=propose(manager,robot(t=.05));confirm(manager,command,acceleration=False)
        self.assertTrue(manager.invalidated);self.assertIsNone(manager.plan)
        self.assertTrue(manager.qualification_failed)
        old_deadline=controller.goal.selected_at+controller.goal.phase_budget
        self.assertTrue(manager.make_brake(robot(t=.1),9.9,'CPU_observed_violation'))
        self.assertEqual(manager.plan_kind,'braking')
        np.testing.assert_array_equal(manager.plan.q0,manager.last_confirmed['joints'])
        np.testing.assert_array_equal(manager.plan.v0,manager.last_confirmed['nominal_velocity'])
        self.assertLessEqual(manager.plan.end,old_deadline+1e-8)
        self.assertEqual(controller.goal.selected_at,0.)

    def test_hold_invalidates_old_plan_and_cannot_resume_at_old_wallclock(self):
        _,controller,manager=fixture();command=propose(manager,robot(t=.05));confirm(manager,command)
        old_plan=manager.plan
        # A source-bound ordinary hold has actually replaced the target.
        manager.confirm_actual_issued(time=.15,joints=np.zeros(7),nominal_velocity=None,
            command=dict(command='hold'),execution=execution(.15,np.zeros(7)))
        self.assertTrue(manager.invalidated);self.assertIsNone(manager.plan)
        manager.forecast_observed_until=.15
        result=manager.check_stream(robot(t=.15),.05)
        self.assertFalse(result.feasible);self.assertIsNone(manager.plan)
        # Freshly dual-verified replan starts from confirmed hold target/v=0,
        # rather than resuming the old trajectory at its elapsed wall time.
        checked=manager.dual_gate.check_joints(robot(t=.15),np.full(7,.3),np.zeros(7),1.)
        self.assertTrue(checked.feasible);controller.goal=SimpleNamespace(
            **{**vars(controller.goal),'candidate_id':'new_after_actual_hold','reachability':checked,'arrival_time':1.15})
        self.assertTrue(manager.adopt_selected_plan(robot(t=.15)))
        np.testing.assert_array_equal(manager.plan.q0,np.zeros(7));np.testing.assert_array_equal(manager.plan.v0,np.zeros(7))
        self.assertIsNot(manager.plan,old_plan)
        self.assertEqual(controller.goal.selected_at,0.);self.assertEqual(controller.goal.phase_budget,3.3)

    def test_fresh_dynamic_collision_rejects_stream_without_advancing_reference(self):
        model,_,manager=fixture();manager.adopt_selected_plan(robot(t=.05));before=deepcopy(manager.last_confirmed)
        manager.pending=None;model.dynamic_block=True
        result=manager.check_stream(robot(t=.05),.05)
        self.assertFalse(result.feasible);self.assertEqual(result.reason,'trajectory_collision')
        self.assertIsNone(manager.pending);self.assertEqual(manager.last_confirmed,before)


if __name__=='__main__':unittest.main()
