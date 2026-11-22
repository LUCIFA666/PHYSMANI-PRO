"""Continuity proofs and actual-dispatch ledger boundaries for the v2 helper."""
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.multitask.continuous_reference_feedforward_v2 import ContinuousReferenceManager,SCHEMA,install_continuous_reference
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.interaction import (CandidateConfig,MPCConfig,GripperTimingConfig,CandidateGenerator,
    SE3InteractionMPC,InteractionController)
from our_mani.reachability import TimedReachability
from tests.test_feedback_interaction import GateModel,robot
from tests.test_persistent_reference_feedforward import owner
from tests.test_servo_response_actor_backend import prepared


def fixture():
    model,controller,_=owner()
    controller.generator.config.dt=.05;controller.mpc=SimpleNamespace(config=SimpleNamespace(dt=.05));controller.phase_index=0
    manager=ContinuousReferenceManager(controller);manager.forecast_observed_until=.5
    return model,controller,manager


def seed_reference(manager,q=.4,v=.7,time=.5):
    """A previously confirmed reference is a unit-test initial condition."""
    manager.last_confirmed=dict(time=time,joints=np.full(7,q).tolist(),nominal_velocity=np.full(7,v).tolist(),command='joint_trajectory')


def execution(q,time):
    tick=round(time/.05)
    return dict(schema='r10_continuous_reference_direct_counted_setpoint_v2',tick_before=tick-1,tick_after=tick,
        time_before=time-.05,time_after=time,applied_joint_reference=np.asarray(q).tolist(),
        actual_window_velocity_within_limits=True,actual_window_acceleration_within_limits=True)


class ContinuousTests(unittest.TestCase):
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
