"""CPU reissue bookkeeping and original-gate tests; no native trajectories."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState,Pose,RobotState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionLibrary,CandidateConfig,CandidateGenerator,
    InteractionController,SE3InteractionMPC,MPCConfig,GripperTimingConfig,PolicyState,_world_target)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.causal_retarget_interaction import (CausalRetargetManager,CausalRetargetConfig,
    install_causal_retarget,retarget_binding,TASK)
from tests.test_multitask_interaction import CartesianRobot

ROOT=Path(__file__).resolve().parents[1]


class Scene:
    def __init__(self):self.blocked=False;self.bound=[]
    def bind_candidate(self,**kwargs):self.bound.append(kwargs)
    def __call__(self,**kwargs):return dict(feasible=not self.blocked,reason='CPU scene',evidence={'CPU_fixture':True})


class Model(CartesianRobot):
    velocity_limits=np.full(7,100.)
    acceleration_limits=np.full(7,10000.)
    def __init__(self):self.inverse_seeds=[];self.blocked=False;self.collisions=0
    def inverse(self,pose,seed):self.inverse_seeds.append(np.asarray(seed).copy());return super().inverse(pose,seed)
    def collision(self,q,t):self.collisions+=1;return self.blocked


class CausalRetargetTests(unittest.TestCase):
    def fixture(self,slack=0.,*,start=True):
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/basketball_Context_geometry_TRAIN_development_demo_v1_setup/configuration.json').read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][TASK]['interaction']['path'])
        model=Model();scene=Scene()
        candidate=dict(config['candidate_config']);candidate.update(maximum_linear_speed=100.,maximum_angular_speed=100.)
        generator=CandidateGenerator(library,TimedReachability(model),scene,CandidateConfig(**candidate))
        base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**config['mpc_config'])),
            predictor='cv',ranking='geometry',continuation_sha256='CPU_retarget_fixture_only',
            gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config']))
        controller=ObservedFeedbackController.from_controller(base,{'phase_tracking_slack_seconds':slack})
        local=library.for_phase('approach')[0].at(0.)[0]
        q=np.r_[local.position,Rotation.from_matrix(local.rotation.copy()).as_rotvec(),0.]
        self.q=q;self.controller=controller
        state,forecast=self.observed(0.)
        if start:
            controller.step(state,forecast,10.)
            self.assertIsNotNone(controller.goal)
        manager=CausalRetargetManager(controller,dict(minimum_issue_interval_seconds=.1,
            scheduled_issue_interval_seconds=.1,reference_shift_threshold_m=.0001,
            arrival_delays_seconds=(.3,.15,.05)))
        return controller,manager,model,scene

    def observed(self,time,*,q=None):
        q=self.q.copy() if q is None else np.asarray(q).copy()
        robot=RobotState(time,Pose(q[:3],Rotation.from_rotvec(q[3:6]).as_matrix()),np.zeros(3),np.zeros(3),q,np.zeros(7))
        position=np.array([.1*time,0.,0.])
        body=ObjectState('ball',time,Pose(position),np.tile(position,(3,1)),np.array([.1,0.,0.]),np.zeros(3))
        state=PolicyState(robot,{'ball':body},{'bodies':{'ball':dict(directly_visible=True,registration_valid=True,
            orientation_observable=False,motion_mode='external')}},{'ball':('ball',)})
        forecast=ConstantVelocityDynamics().predict([body],1.5,.05)
        forecast=replace(forecast,metadata={**forecast.metadata,'observed_until':float(time)})
        return state,forecast

    def test_new_issue_preserves_template_phase_progress_deadline_and_parent_attempt(self):
        controller,manager,model,scene=self.fixture();old=controller.goal;rows=deepcopy(controller.attempts.to_report())
        current=self.q.copy();current[0]+=.02
        state,forecast=self.observed(.15,q=current)
        event=manager.before_step(state,forecast,9.85)
        self.assertEqual(event['action'],'new_candidate_issue',event)
        self.assertNotEqual(controller.goal.candidate_id,old.candidate_id)
        self.assertIs(controller.goal.template,old.template)
        self.assertEqual(controller.goal.selected_at,old.selected_at)
        self.assertEqual(controller.goal.phase_budget,old.phase_budget)
        self.assertEqual(controller.goal.selected_at+controller.goal.phase_budget,event['original_phase_deadline'])
        self.assertEqual(event['issued_at'],.15);self.assertGreater(event['template_progress_now'],0.)
        np.testing.assert_array_equal(controller.trajectory.q0,current)
        self.assertTrue(any(np.array_equal(seed,current) for seed in model.inverse_seeds))
        self.assertEqual(rows,controller.attempts.to_report())
        self.assertFalse(event['phase_completion_claimed']);self.assertFalse(event['total_budget_extended'])
        self.assertAlmostEqual(controller.goal.reference_pose.position[0],.1*controller.goal.arrival_time)
        self.assertGreater(model.collisions,0);self.assertTrue(scene.bound)

    def test_new_candidate_does_not_grant_feedback_slack_again(self):
        controller,manager,_,_=self.fixture(slack=2.)
        deadline=controller.goal.selected_at+controller.goal.phase_budget
        count=len(controller.phase_timing_events)
        state,forecast=self.observed(.15);event=manager.before_step(state,forecast,9.85)
        self.assertEqual(event['action'],'new_candidate_issue')
        controller._grant_phase_slack(state,9.85)
        self.assertEqual(len(controller.phase_timing_events),count)
        self.assertEqual(controller.goal.selected_at+controller.goal.phase_budget,deadline)

    def test_interval_and_issue_cap_prevent_every_tick_restarts(self):
        controller,manager,_,_=self.fixture()
        for time in (.15,.25,.35,.45):
            state,forecast=self.observed(time);event=manager.before_step(state,forecast,10.-time)
            self.assertEqual(event['action'],'new_candidate_issue',event)
            state,forecast=self.observed(time+.05)
            self.assertIsNone(manager.before_step(state,forecast,10.-time-.05))
        state,forecast=self.observed(.65)
        self.assertIsNone(manager.before_step(state,forecast,9.35))
        self.assertEqual(len(manager.events),4)
        self.assertEqual(controller.goal.selected_at,0.)

    def test_original_joint_or_dynamic_collision_failure_never_commits_a_reissue(self):
        for kind in ('joint','dynamic'):
            controller,manager,model,scene=self.fixture();old=controller.goal
            if kind=='joint':model.blocked=True
            else:scene.blocked=True
            state,forecast=self.observed(.15)
            event=manager.before_step(state,forecast,9.85)
            self.assertEqual(event['action'],'reissue_rejected_original_candidate_retained')
            self.assertIs(controller.goal,old)
            self.assertFalse(any(trial.get('feasible') for trial in event['trials']))

    def test_missing_or_future_forecast_and_wrong_mode_cannot_issue(self):
        for kind in ('missing','future','mode'):
            controller,manager,model,_=self.fixture();old=controller.goal
            state,forecast=self.observed(.15)
            if kind=='missing':forecast=replace(forecast,metadata={})
            if kind=='future':forecast=replace(forecast,metadata={'observed_until':.2})
            if kind=='mode':state.perception_metadata['bodies']['ball']['motion_mode']='attached'
            before=len(model.inverse_seeds);event=manager.before_step(state,forecast,9.85)
            self.assertEqual(event['action'],'reissue_rejected_original_candidate_retained')
            self.assertIs(controller.goal,old);self.assertEqual(len(model.inverse_seeds),before)

    def test_deadline_and_suffix_budget_are_never_extended_for_new_arrival(self):
        controller,manager,model,_=self.fixture();old=controller.goal
        state,forecast=self.observed(.15);before=len(model.inverse_seeds)
        self.assertIsNone(manager.before_step(state,forecast,old.suffix_budget+.01))
        self.assertIs(controller.goal,old);self.assertEqual(len(model.inverse_seeds),before)
        state,forecast=self.observed(old.selected_at+old.phase_budget+.1)
        self.assertIsNone(manager.before_step(state,forecast,8.))
        self.assertIs(controller.goal,old)

    def test_install_is_early_geometry_only_and_before_outer_servo(self):
        controller,_,_,_=self.fixture()
        actor=SimpleNamespace(controller=controller,metrics_report=lambda:{})
        with self.assertRaises(ValueError):install_causal_retarget(actor,{})
        controller,_,_,_=self.fixture(start=False)
        actor=SimpleNamespace(controller=controller,metrics_report=lambda:{})
        install_causal_retarget(actor,{})
        self.assertTrue(actor.metrics_report()['causal_retarget']['geometry_only'])
        self.assertFalse(actor.metrics_report()['causal_retarget']['Value_required'])
        with self.assertRaises(ValueError):install_causal_retarget(actor,{})
        controller,_,_,_=self.fixture(start=False)
        actor=SimpleNamespace(controller=controller,metrics_report=lambda:{},servo_response_compensator=object())
        with self.assertRaises(ValueError):install_causal_retarget(actor,{})

    def test_installed_step_wraps_actual_feedback_and_preserves_original_attempt(self):
        controller,_,_,_=self.fixture(start=False)
        actor=SimpleNamespace(controller=controller,metrics_report=lambda:{})
        install_causal_retarget(actor,dict(minimum_issue_interval_seconds=.1,scheduled_issue_interval_seconds=.1,
            reference_shift_threshold_m=.0001,arrival_delays_seconds=(.3,.15,.05)))
        state,forecast=self.observed(0.);controller.step(state,forecast,10.)
        old=controller.goal;count=len(controller.attempts.records)
        state,forecast=self.observed(.15);controller.step(state,forecast,9.85)
        event=actor.metrics_report()['causal_retarget']['events'][-1]
        self.assertEqual(event['action'],'new_candidate_issue')
        self.assertEqual(controller.goal.selected_at,old.selected_at)
        self.assertEqual(len(controller.attempts.records),count)
        self.assertEqual(len(controller.selection_events),count)

    def test_failed_attempts_do_not_consume_successful_reissue_limit(self):
        controller,manager,_,scene=self.fixture(slack=2.);scene.blocked=True
        for time in (.15,.3,.45,.6,.75):
            state,forecast=self.observed(time)
            self.assertEqual(manager.before_step(state,forecast,10.-time)['action'],
                'reissue_rejected_original_candidate_retained')
        scene.blocked=False;state,forecast=self.observed(.9)
        event=manager.before_step(state,forecast,9.1)
        self.assertEqual(event['action'],'new_candidate_issue')
        self.assertEqual(next(iter(manager.epochs.values()))['issues'],1)

    def test_same_phase_new_original_attempt_cannot_reset_successful_issue_cap(self):
        controller,manager,_,_=self.fixture(slack=2.)
        for time in (.15,.3,.45,.6):
            state,forecast=self.observed(time)
            self.assertEqual(manager.before_step(state,forecast,10.-time)['action'],'new_candidate_issue')
        controller.attempts.abandon('approach',.65,'CPU_original_same_phase_reselection')
        controller.goal=controller.trajectory=controller.origin=None
        state,forecast=self.observed(.65);controller.step(state,forecast,9.35)
        self.assertIsNotNone(controller.goal);self.assertEqual(controller.phase_index,0)
        old=controller.goal;state,forecast=self.observed(.8)
        self.assertIsNone(manager.before_step(state,forecast,9.2))
        self.assertIs(controller.goal,old)
        self.assertEqual(manager.successful_reissues_by_phase_index,{0:4})

    def test_declared_long_delays_are_checked_within_same_absolute_deadline(self):
        controller,_,_,_=self.fixture(slack=2.)
        manager=CausalRetargetManager(controller,dict(minimum_issue_interval_seconds=.1,
            scheduled_issue_interval_seconds=.1,reference_shift_threshold_m=.0001))
        state,forecast=self.observed(.15);event=manager.before_step(state,forecast,9.85)
        self.assertEqual(event['action'],'new_candidate_issue')
        self.assertIn(1.1,[trial['delay'] for trial in event['trials']])
        self.assertTrue(all(trial['arrival_time']<=event['original_phase_deadline']+1e-8 for trial in event['trials']))

    def test_configuration_and_same_tick_reentry_are_rejected(self):
        for config in ({'minimum_issue_interval_seconds':.05},{'maximum_reissues_per_phase':0},{'dt':.1}):
            with self.assertRaises(ValueError):CausalRetargetConfig(**config)
        controller,manager,_,_=self.fixture();state,forecast=self.observed(.15)
        manager.before_step(state,forecast,9.85)
        with self.assertRaisesRegex(ValueError,'one monotonically'):manager.before_step(state,forecast,9.85)
        self.assertFalse(retarget_binding({})['instantaneous_endpoint_velocity_claimed'])


if __name__=='__main__':unittest.main()
