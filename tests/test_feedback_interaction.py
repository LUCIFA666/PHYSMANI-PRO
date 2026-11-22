"""Original gates on recorded rejects, lagged motion and returned commands."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.contracts import Pose, RobotState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.intercept_planner import JointTrajectory
from our_mani.reachability import TimedReachability, hermite
from our_mani.multitask.feedback_interaction import (SCHEMA, FeedbackConfig,
    ObservedFeedbackController, _FeedbackReachability, bounded_joint_segment,
    feedback_binding, install_feedback_controller)
from our_mani.multitask.interaction import InteractionController, CandidateGenerator, SE3InteractionMPC
from tests.test_multitask_interaction import (CartesianRobot, synthetic_library, state,
    candidates_config, mpc_config)


class GateModel(CartesianRobot):
    velocity_limits = np.ones(7)
    acceleration_limits = np.full(7, 6.)
    def __init__(self): self.collision_calls=[]; self.dynamic_block=False; self.ik_calls=0
    def inverse(self, pose, seed):
        self.ik_calls+=1; return super().inverse(pose, seed)
    def collision(self, q, future_offset):
        self.collision_calls.append((q.copy(),future_offset))
        return self.dynamic_block and future_offset >= .025


def robot(q=None, v=None, t=0.):
    q = np.zeros(7) if q is None else np.asarray(q,float)
    return RobotState(t, Pose(q[:3]), np.zeros(3), np.zeros(3),q,
                      np.zeros(7) if v is None else np.asarray(v,float))


def owner(model, fraction=.5):
    original=TimedReachability(model)
    controller=SimpleNamespace(goal=SimpleNamespace(candidate_id='fixed',arrival_time=1.),
        library=SimpleNamespace(phase_order=('approach',)), phase_index=0,
        generator=SimpleNamespace(config=SimpleNamespace(dt=.05)), feedback_config=FeedbackConfig(acceleration_fraction=fraction),
        feedback_events=[],_feedback_command=None)
    return controller, original, _FeedbackReachability(original,controller)


class FeedbackTests(unittest.TestCase):
    def test_recorded_actual_rejects_become_bounded_segments_under_original_dynamic_caps(self):
        path=Path(__file__).resolve().parents[1]/'artifacts/framework/multitask_r10/ready_tasks_execution_v1/development_demo_only/joint_rejection_diagnostic/basketball_normal_context_v2/joint_checks.json'
        rows=[row for row in json.loads(path.read_text())['checks']
              if abs(row['duration']-.05)<1e-9 and row['reason'] in
              ('insufficient_arrival_time','trajectory_joint_acceleration_limit','trajectory_joint_speed_limit')]
        self.assertGreaterEqual(len(rows),50)
        for row in rows:
            model=GateModel(); model.velocity_limits=np.asarray(row['velocity_limits']); model.acceleration_limits=np.asarray(row['acceleration_limits'])
            controller, original, feedback=owner(model)
            measured=robot(row['observed_joints'],row['observed_joint_velocities'],row['observed_time'])
            q=np.asarray(row['requested_joints']);v=np.asarray(row['requested_joint_velocities'])
            rejected=original.check_joints(measured,q,v,.05)
            self.assertFalse(rejected.feasible)
            accepted=feedback.check_joints(measured,q,v,.05)
            self.assertTrue(accepted.feasible,(row['observed_time'],accepted.reason))
            self.assertLessEqual(accepted.evidence['maximum_speed_fraction'],1.+1e-7)
            self.assertLessEqual(accepted.evidence['maximum_acceleration_fraction'],.5+1e-7)
            self.assertGreater(len(model.collision_calls),0)
            self.assertFalse(controller.feedback_events[-1]['nominal_trajectory_completed'])

    def test_dynamic_collision_and_joint_position_gates_remain_mandatory(self):
        model=GateModel();controller,original,feedback=owner(model);model.dynamic_block=True
        check=feedback.check_joints(robot(),np.ones(7),np.zeros(7),.05)
        self.assertFalse(check.feasible);self.assertEqual(check.reason,'trajectory_collision')
        self.assertIsNone(controller._feedback_command)
        self.assertTrue(any(offset>=.025 for _,offset in model.collision_calls))
        model.dynamic_block=False;model.upper=np.full(7,.001)
        check=feedback.check_joints(robot(),np.ones(7),np.zeros(7),.05)
        self.assertFalse(check.feasible);self.assertEqual(check.reason,'joint_position_limit')

    def test_candidate_search_is_unchanged_and_non_dynamic_failures_are_not_recovered(self):
        model=GateModel();controller,original,feedback=owner(model)
        q=np.ones(7);v=np.zeros(7)
        a=original.check_joints(robot(),q,v,.4);b=feedback.check_joints(robot(),q,v,.4)
        self.assertEqual((a.feasible,a.reason),(b.feasible,b.reason));self.assertEqual(controller.feedback_events,[])
        controller.goal=None
        a=original.check_joints(robot(),q,v,.05);b=feedback.check_joints(robot(),q,v,.05)
        self.assertEqual(a.reason,b.reason);self.assertEqual(controller.feedback_events,[])

    def test_local_mpc_ik_is_obtained_once_and_returned_bounded_target_is_not_nominal(self):
        model=GateModel();controller,original,feedback=owner(model)
        check=feedback.check(robot(),Pose([.5,0,0]),np.zeros(3),np.zeros(3),.05)
        self.assertTrue(check.feasible);self.assertEqual(model.ik_calls,1)
        self.assertLess(check.joints[0],.01);self.assertGreater(check.joints[0],0.)
        self.assertEqual(controller.feedback_events[-1]['nominal_joints'][0],.5)

    def test_sustained_lag_loop_makes_progress_without_teleport_or_weaker_gate(self):
        model=GateModel();controller,original,feedback=owner(model)
        plan=JointTrajectory(0.,1.,np.zeros(7),np.zeros(7),np.r_[.5,np.zeros(6)],np.zeros(7))
        q=np.zeros(7);v=np.zeros(7);holds=0;accepted=0
        for tick in range(100):
            now=tick*.05;measured=robot(q,v,now)
            nominal_q,nominal_v=plan.at(now+.05)
            original_check=original.check_joints(measured,nominal_q,nominal_v,.05)
            holds+=not original_check.feasible
            check=feedback.check_joints(measured,nominal_q,nominal_v,.05)
            self.assertTrue(check.feasible,check.reason)
            self.assertLessEqual(check.evidence['maximum_speed_fraction'],1.+1e-7)
            self.assertLessEqual(check.evidence['maximum_acceleration_fraction'],1.+1e-7)
            # Explicit synthetic motor: measured motion reaches 30% of target
            # displacement, so actual q/v differ from both nominal and command.
            actual_next=q+.3*(check.joints-q)
            actual_velocity=(actual_next-q)/.05
            self.assertLessEqual(np.max(np.abs(actual_velocity)),1.+1e-7)
            self.assertLessEqual(np.max(np.abs((actual_velocity-v)/.05)),6.+1e-7)
            q,v=actual_next,actual_velocity;accepted+=1
        self.assertGreater(holds,70);self.assertEqual(accepted,100)
        # Compare the same motor and original nominal gate without recovery.
        baseline_q=np.zeros(7);baseline_v=np.zeros(7)
        for tick in range(100):
            nominal_q,nominal_v=plan.at((tick+1)*.05)
            check=original.check_joints(robot(baseline_q,baseline_v,tick*.05),nominal_q,nominal_v,.05)
            target=check.joints if check.feasible else baseline_q
            next_q=baseline_q+.3*(target-baseline_q)
            baseline_v=(next_q-baseline_q)/.05;baseline_q=next_q
        self.assertGreater(q[0]-baseline_q[0],.1)
        self.assertGreater(q[0],5*baseline_q[0])
        self.assertFalse(any(event['total_budget_extended'] for event in controller.feedback_events))

    def controller(self):
        library=synthetic_library();model=GateModel()
        generator=CandidateGenerator(library,TimedReachability(model),lambda **kw:dict(feasible=True,evidence={}),
            candidates_config(),allow_synthetic=True)
        base=InteractionController(library,generator,SE3InteractionMPC(mpc_config()),predictor='cv',ranking='geometry',
            continuation_sha256='new_feedback_continuation_fixture',allow_synthetic=True)
        return base,model

    def test_original_super_step_outputs_actual_bounded_commands_and_no_false_phase_completion(self):
        base,model=self.controller();controller=ObservedFeedbackController.from_controller(base,{})
        original=controller.generator.reachability
        q=np.zeros(7);v=np.zeros(7);bounded_count=0
        for tick in range(40):
            observed=state(tick*.05,robot_z=q[2])
            observed=replace(observed,robot=replace(observed.robot,joints=q,joint_velocities=v,
                linear_velocity=v[:3]))
            forecast=ConstantVelocityDynamics().predict(list(observed.bodies.values()),1.,.05)
            command=controller.step(observed,forecast,10.-tick*.05)
            self.assertIs(controller.generator.reachability,original)
            if command['command']=='joint_trajectory':
                if command.get('measured_feedback_command'):
                    bounded_count+=1
                    checked=original.check_joints(observed.robot,command['joints'],command['joint_velocities'],.05)
                    self.assertTrue(checked.feasible)
                    self.assertFalse(command['nominal_trajectory_completion_claimed'])
                    self.assertEqual(command['joints'].tolist(),controller.feedback_events[-1]['bounded_joints'])
                next_q=q+.25*(command['joints']-q);v=(next_q-q)/.05;q=next_q
            else:v=np.zeros(7)
        self.assertGreater(bounded_count,5)
        self.assertGreater(q[2],.015)
        self.assertFalse(any(event.get('official_task_success_claimed') for event in controller.events))

    def test_installation_receipt_and_configuration_are_explicit_and_early_only(self):
        base,_=self.controller()
        actor=SimpleNamespace(controller=base,metrics_report=lambda:{'original':True})
        install_feedback_controller(actor,{'version':SCHEMA,'acceleration_fraction':.5})
        metrics=actor.metrics_report();self.assertTrue(metrics['original'])
        receipt=metrics['observed_feedback_controller']
        self.assertEqual(receipt['normalized_config']['acceleration_fraction'],.5)
        self.assertEqual(receipt['configuration_sha256'],feedback_binding({})['configuration_sha256'])
        self.assertEqual(receipt['step_calls'],0);self.assertTrue(receipt['installed_before_first_step'])
        with self.assertRaises(ValueError):install_feedback_controller(actor,{})
        base,_=self.controller();base.goal=object()
        with self.assertRaises(ValueError):ObservedFeedbackController.from_controller(base,{})
        for config in ({'acceleration_fraction':0.},{'acceleration_fraction':2.},{'acceleration_fraction':True},{'version':'bad'}):
            with self.assertRaises(ValueError):feedback_binding(config)

    def test_optional_phase_slack_is_once_only_and_capped_by_task_budget_and_suffix(self):
        base,model=self.controller();controller=ObservedFeedbackController.from_controller(base,
            {'phase_tracking_slack_seconds':5.})
        observed=state(0.,robot_z=0.);forecast=ConstantVelocityDynamics().predict(list(observed.bodies.values()),1.,.05)
        controller.step(observed,forecast,.8)
        self.assertIsNotNone(controller.goal)
        event=controller.phase_timing_events[-1]
        self.assertLessEqual(event['actual_phase_deadline']+event['suffix_reserve_seconds'],.8+1e-8)
        budget=controller.goal.phase_budget
        controller._grant_phase_slack(observed,.8)
        self.assertEqual(controller.goal.phase_budget,budget);self.assertEqual(len(controller.phase_timing_events),1)
        self.assertFalse(event['total_budget_extended'])


if __name__=='__main__':unittest.main()
