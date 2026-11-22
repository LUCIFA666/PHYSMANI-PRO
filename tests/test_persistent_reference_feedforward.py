"""Persistent reference gates, measured-window failures and installation chain."""
from dataclasses import replace
from types import MethodType,SimpleNamespace
import unittest
import numpy as np

from our_mani.contracts import Pose,Forecast
from our_mani.intercept_planner import JointTrajectory
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionController,CandidateGenerator,SE3InteractionMPC,
    CandidateConfig,MPCConfig,GripperTimingConfig)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.persistent_reference_feedforward import (
    SCHEMA,PersistentReferenceManager,install_persistent_reference,persistent_reference_binding)
from tests.test_feedback_interaction import GateModel,robot
from tests.test_servo_response_actor_backend import prepared


def owner(q1=.5,end=1.,v1=0.):
    model=GateModel();gate=TimedReachability(model)
    initial=robot();q=np.full(7,q1);v=np.full(7,v1)
    verified=gate.check_joints(initial,q,v,end)
    assert verified.feasible,verified.reason
    plan=JointTrajectory(0.,end,initial.joints.copy(),initial.joint_velocities.copy(),q,v)
    goal=SimpleNamespace(phase='approach',template=SimpleNamespace(reference_time_basis='command_issue'),
        reachability=verified,candidate_id='fully_verified_original',arrival_time=end,selected_at=0.,phase_budget=3.3)
    controller=SimpleNamespace(goal=goal,trajectory=plan,original_reachability=gate,
        generator=SimpleNamespace(config=SimpleNamespace(maximum_observation_age=.1)))
    manager=PersistentReferenceManager(controller);manager.forecast_observed_until=.5
    return model,controller,manager


class PersistentTests(unittest.TestCase):
    def test_lagged_measured_bridge_fails_but_exact_original_reference_segment_passes(self):
        model,controller,manager=owner()
        measured=robot(np.zeros(7),np.zeros(7),.5)
        q,v=controller.trajectory.at(.55)
        rejected=controller.original_reachability.check_joints(measured,q,v,.05)
        self.assertFalse(rejected.feasible)
        result=manager.check_reference(measured,.05)
        self.assertTrue(result.feasible,result.reason)
        self.assertTrue(np.allclose(result.joints,q))
        event=manager.reference_events[-1]
        self.assertTrue(event['checked_state_is_reference_not_measured'])
        self.assertFalse(event['actual_tracking_guaranteed'])
        self.assertLessEqual(event['reference_gate_evidence']['maximum_acceleration_fraction'],1.+1e-6)

    def test_full_candidate_and_fresh_collision_remain_mandatory(self):
        model,controller,manager=owner();model.dynamic_block=True
        result=manager.check_reference(robot(t=.5),.05)
        self.assertFalse(result.feasible);self.assertEqual(result.reason,'trajectory_collision')
        self.assertIsNone(manager.pending)
        controller.trajectory.q1=controller.trajectory.q1+.1
        with self.assertRaises(ValueError):manager.check_reference(robot(t=.5),.05)

    def test_endpoint_hold_only_for_original_zero_terminal_velocity_without_fake_completion(self):
        _,controller,manager=owner();manager.forecast_observed_until=1.2
        result=manager.check_reference(robot(t=1.2),.05)
        self.assertTrue(result.feasible)
        self.assertEqual(manager.reference_events[-1]['reference_mode'],'verified_zero_velocity_endpoint')
        self.assertTrue(np.array_equal(result.joints,controller.trajectory.q1))
        self.assertTrue(np.all(result.joint_velocities==0))
        self.assertTrue(manager.reference_events[-1]['endpoint_reference_not_physical_phase_completion'])
        _,_,moving=owner(v1=.1);moving.forecast_observed_until=1.2
        self.assertIsNone(moving.check_reference(robot(t=1.2),.05))

    def test_reference_analytic_acceleration_gate_is_not_weakened(self):
        _,controller,manager=owner();manager.forecast_observed_until=.0
        # Retaining a feasible endpoint flag cannot certify a newly fabricated
        # high-acceleration trajectory with a different initial velocity.
        controller.trajectory.v0=np.full(7,-.9)
        result=manager.check_reference(robot(),.05)
        self.assertFalse(result.feasible)
        self.assertIn(result.reason,('trajectory_joint_speed_limit','trajectory_joint_acceleration_limit'))

    def test_observed_windows_distinguish_reference_proof_and_actual_acceleration_failure(self):
        _,_,manager=owner()
        self.assertIsNone(manager.observe(robot())['backward_window_acceleration'])
        self.assertIsNone(manager.observe(robot(np.full(7,.005),np.full(7,.1),.05))['backward_window_acceleration'])
        event=manager.observe(robot(np.full(7,.03),np.full(7,.5),.1))
        self.assertIn('observed_backward_window_acceleration_limit',event['violations'])
        self.assertTrue(np.allclose(event['backward_window_acceleration'],8.))
        self.assertTrue(manager.metrics()['observed_motion_qualification_failed'])
        with self.assertRaises(ValueError):manager.observe(robot(t=.1))

    def test_installer_preserves_prior_retarget_step_chain_and_emits_checked_final_reference(self):
        data,library,_=prepared();configuration=data['config'];model=GateModel()
        collision=lambda **kw:dict(feasible=True,evidence={});collision.bind_candidate=lambda **kw:None
        generator=CandidateGenerator(library,TimedReachability(model),collision,
            CandidateConfig(**configuration['candidate_config']))
        base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**configuration['mpc_config'])),
            predictor='cv',ranking='geometry',continuation_sha256='persistent_fixture',
            gripper_timing_config=GripperTimingConfig(**configuration['gripper_timing_config']))
        controller=ObservedFeedbackController.from_controller(base,{'phase_tracking_slack_seconds':2.})
        called=[]
        def already_installed_retarget(instance,state,forecast,remaining):
            called.append(state.robot.timestamp)
            result=instance.feedback_reachability.check_joints(state.robot,np.ones(7)*99.,np.zeros(7),.05)
            self.assertTrue(result.feasible)
            return dict(command='joint_trajectory',joints=result.joints,opening=1.,phase='approach')
        controller.step=MethodType(already_installed_retarget,controller)
        actor=SimpleNamespace(controller=controller,metrics_report=lambda:{'original':True},
            causal_retarget_manager=SimpleNamespace(controller=controller))
        install_persistent_reference(actor)
        _,fixture,unused=owner();controller.goal=fixture.goal;controller.trajectory=fixture.trajectory
        # Use the actual library template to keep command-issue semantics.
        controller.goal.template=library.for_phase('approach')[0]
        original_goal,original_plan=controller.goal,controller.trajectory
        forecast=SimpleNamespace(metadata={'observed_until':.5})
        command=controller.step(SimpleNamespace(robot=robot(t=.5)),forecast,9.5)
        self.assertEqual(called,[.5])
        self.assertIs(actor.causal_retarget_manager.controller,controller)
        self.assertIs(controller.goal,original_goal);self.assertIs(controller.trajectory,original_plan)
        self.assertEqual(controller.goal.selected_at,0.);self.assertEqual(controller.goal.phase_budget,3.3)
        receipt=command['feedforward_execution']
        self.assertEqual(receipt['schema'],SCHEMA);self.assertTrue(receipt['reference_gate_passed']);self.assertTrue(receipt['issued'])
        self.assertEqual(command['joints'].tolist(),receipt['reference_to_joints'])
        self.assertTrue(actor.metrics_report()['persistent_reference_feedforward']['reference_events'][-1]['issued'])
        with self.assertRaises(ValueError):install_persistent_reference(actor)

    def test_stale_forecast_cannot_issue_a_direct_reference_and_config_is_bound(self):
        _,_,manager=owner();manager.forecast_observed_until=.1
        result=manager.check_reference(robot(t=.5),.05)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason,'persistent_reference_requires_current_causal_forecast')
        self.assertIsNone(manager.pending)
        binding=persistent_reference_binding()
        self.assertIn('our_mani/reachability.py',binding['original_sources_sha256'])
        for config in ({'dt':.1},{'observed_acceleration_tolerance':-1.}):
            with self.assertRaises(ValueError):persistent_reference_binding(config)


if __name__=='__main__':unittest.main()
