"""CPU sensor/decision contracts. No Native, fitting or contact qualification."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,tempfile
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_20261102.mpc import MPCConfig
from our_mani.local_mpc_20261102.terminal_mpc import TerminalCostConfig
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.public_grasp_state import PublicGraspStateAssembler
from our_mani.local_mpc_control_20261103.public_grasp import (
    AttachmentAwarePublicBridge,PublicLiftHoldMetric,PublicGraspConfig,ObservedPublicGraspBackend,
    PublicGraspCoordinator,run_public_grasp)
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from tests.test_local_mpc_public_motion_bridge_20261103 import observation
from tests.test_local_mpc_public_pregrasp_20261103 import NumericModel,GoalHelper,Executor


def sample(t,*,opening=1.,moving=False,visible=True):
    tcp=np.array([.2,-.15,.815])+np.array([0.,0.,.08*t if moving else 0.])
    robot=PublicRobotSensors(t,tcp,[0,0,0,1],np.zeros(7),np.zeros(7),[opening,opening])
    p=tcp+[0.,0.,-.04]
    target=ObjectState('target',t,Pose(p),p+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),
        np.array([0.,0.,.08 if moving else 0.]),np.zeros(3)) if visible else None
    return target,robot


def h(t,opening=0.):
    return dict(reference_valid=True,reference_positions=np.zeros(7),reference_velocities=np.zeros(7),reference_accelerations=np.zeros(7),
        target_joints=np.zeros(7),reference_clock=t,gripper_target=opening,gripper_done=True,reference_is_measured_state=False)


class Planner:
    seen=[];closed_free=True
    def __init__(self,*args,**kwargs):self.model=kwargs['execution_model']
    def plan(self,state,candidates,*,warm_start=None):
        type(self).seen.append(dict(state=deepcopy(state),warm=deepcopy(warm_start),model=self.model,candidates=deepcopy(candidates)))
        phase=(state.get('phase_state') or {}).get('phase','approach')
        opening=0. if phase in ('close','hold') or state['grasp_motion_mode']=='FREE_GRU' and self.closed_free else 1.
        phase_state=deepcopy(state.get('phase_state') or dict(phase='close',dwell=0.,lost_hold=0.,updated_at=state['time']))
        action=dict(joint_goal=np.full(7,.1),tcp_position=state['tcp_position'].copy(),tcp_quaternion=state['tcp_quaternion'].copy(),opening=opening)
        return dict(action=action,selected_candidate_id=candidates[0]['id'],phase_state=phase_state,warm_start=None,
            budget=dict(sequences=256,horizon=8),diagnostics=dict(diagnostic_only=True))


class CollisionOnly:
    def __init__(self):self.probes=[]
    def probe_joint_collision(self,q):self.probes.append(np.asarray(q).copy());return False
    def solve_ik(self,*args):raise AssertionError('Native IK forbidden')


class RawBackend(CollisionOnly):
    def __init__(self,budget=26):super().__init__();self.ticks=0;self.budget=budget;self.q=np.zeros(7);self.requested=1.;self.close_start=None;self.captures=0
    def clock(self):return self.ticks*.05
    def remaining_ticks(self):return self.budget-self.ticks
    def terminal(self):raise AssertionError('Native task success must not terminate the public policy')
    def opening(self):return 1. if self.close_start is None else max(.6,1.-2.*(self.clock()-self.close_start))
    def read_state(self):
        _,robot=sample(self.clock(),opening=self.opening())
        return dict(time=robot.timestamp,tcp_position=robot.tcp_position,tcp_quaternion=robot.tcp_quaternion,q=self.q.copy(),dq=np.zeros(7),
            finger_open_amounts=robot.finger_open_amounts,requested_opening=self.requested,
            object_position=[999,888,777],object_quaternion=[0,0,0,1],object_linear_velocity=[55,55,55],object_angular_velocity=[1,1,1],held=True,native_success=True)
    def observe_rgbd(self):self.captures+=1;return observation(self.clock())
    def tick(self,phase):
        if self.requested==0. and self.close_start is None:self.close_start=self.clock()
        self.ticks+=1;return self.read_state()


class GraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Components are genuinely constructed with the frozen CPU checkpoint.
        # Only perceptor method outputs are mocked as explicit synthetic sensor
        # fixtures below; no such mock is used in a Native canary or artifact.
        cls.wrapper=AttachmentAwarePublicBridge()
    def setUp(self):
        self.wrapper.reset();self.assembler=PublicGraspStateAssembler(self.wrapper.binding)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        Planner.seen=[];Planner.closed_free=True
    def feed(self,t,*,opening=1.,close=False,moving=False,visible=True):
        target,robot=sample(t,opening=opening,moving=moving,visible=visible)
        def track(_):
            if target is None:raise ValueError('explicit synthetic visual occlusion')
            return target,dict(backend='synthetic_CPU_fixture',simulator_pose_used=False,future_observations_used=False)
        with patch.object(self.wrapper.tracker,'observe',side_effect=track),\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
            result=self.wrapper.observe(observation(t),robot,close_requested=close)
        self.assembler.observe_robot(robot)
        return result,robot
    def core(self):
        return SimpleNamespace(config=MPCConfig(),execution_model=NumericModel(),planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig())

    def test_only_real_cache_approved_flags_retained_and_forecaster_goal_injection_rejected(self):
        for i in range(5):result,robot=self.feed(i*.05)
        self.assertTrue(result['tracker_dependency_injected']);self.assertFalse(self.assembler.allow_test_doubles)
        state=self.assembler.build(result,robot,h(.2,1.))
        self.assertIsNotNone(state);self.assertTrue(result['tracker_dependency_injected'])
        self.assertTrue(state['approved_tracker_binding']['tracker_dependency_injected'])
        for key in ('forecaster_is_injected_test_double','goal_marker_dependency_injected'):
            result[key]=True
            with self.assertRaises(ValueError):self.assembler.build(result,robot,h(.2,1.))
            result[key]=False
        class Unapproved(type(self.wrapper.tracker)):pass
        original=self.wrapper.binding.tracker;self.wrapper.binding.tracker=Unapproved()
        try:
            with self.assertRaisesRegex(ValueError,'real RedBoxTracker'):self.wrapper.binding.approve(result)
        finally:self.wrapper.binding.tracker=original

    def test_pending_never_FREE_never_held_and_does_not_observe_GRU(self):
        for i in range(5):result,robot=self.feed(i*.05)
        before=len(self.wrapper.bridge.forecaster.history.get('target',[]))
        with patch.object(self.wrapper.bridge.forecaster,'observe',wraps=self.wrapper.bridge.forecaster.observe) as spy:
            result,robot=self.feed(.25,opening=.9,close=True)
            self.assertEqual(spy.call_count,0)
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertFalse(result['free_forecast_available'])
        self.assertIsNone(self.assembler.build(result,robot,h(.25)))
        state=self.assembler.build_pending(result,robot,h(.25))
        self.assertFalse(state['held']);self.assertFalse(state['free_motion_asserted'])
        self.assertEqual(state['attachment_status'],'unknown')
        self.assertIn('not_free_motion_assertion',state['held_source'])
        self.assertEqual(self.wrapper.close_visual_baseline.timestamp,.2)
        self.assertEqual(self.wrapper.latest_attachment['closure_baseline']['timestamp'],.2)
        np.testing.assert_array_equal(self.wrapper.latest_attachment['closure_baseline']['finger_open_amounts'],[1.,1.])

    def test_pending_requires_current_visual_fresh_h_and_no_Native_phase_channels(self):
        self.feed(0.);result,robot=self.feed(.05,opening=.8,close=True)
        with self.assertRaisesRegex(ValueError,'clock'):self.assembler.build_pending(result,robot,h(0.))
        with self.assertRaises(ValueError):self.assembler.build_pending(result,robot,h(.05),phase_state={'phase':'close','Native_held':True})
        result,robot=self.feed(.1,opening=.7,close=True,visible=False)
        self.assertIsNone(self.assembler.build_pending(result,robot,h(.1)))

    def test_mode_changes_close_pending_timeout_reopen_and_visual_recovery_keep_reopen_intent(self):
        helper=GoalHelper();coordinator=PublicGraspCoordinator(self.core(),helper,self.assembler)
        raw=CollisionOnly()
        with patch('our_mani.local_mpc_control_20261103.public_grasp.ForecastJointGoalCEMPlanner',Planner),\
             patch('our_mani.local_mpc_control_20261103.public_grasp.JointGoalCEMPlanner',Planner):
            for i in range(5):result,robot=self.feed(i*.05)
            free=coordinator.decide(raw,result,robot,h(.2,1.));coordinator.note_executed(free['action'],robot)
            self.assertIsInstance(Planner.seen[-1]['model'],BoundForecastExecutionModel)
            coordinator.warm={'sentinel':'incompatible_GRU_warm'}
            result,robot=self.feed(.25,opening=.9,close=True)
            pending=coordinator.decide(raw,result,robot,h(.25))
            self.assertEqual(pending['mode'],'PENDING_CLOSE_UNKNOWN_CONTACT');self.assertFalse(pending['state']['held'])
            self.assertEqual(pending['state']['phase_state']['phase'],'close')
            self.assertEqual(pending['state']['phase_state']['updated_at'],.25)
            self.assertEqual(pending['action']['opening'],0.);self.assertIsNone(Planner.seen[-1]['warm'])
            self.assertNotIsInstance(Planner.seen[-1]['model'],BoundForecastExecutionModel)
            for tick in range(6,21):result,robot=self.feed(tick*.05,opening=max(.6,1.-.1*(tick-4)),close=True)
            reopened=coordinator.decide(raw,result,robot,h(1.))
            self.assertEqual(reopened['mode'],'REOPENING_UNKNOWN_CONTACT');self.assertEqual(reopened['action']['opening'],1.)
            self.assertTrue(coordinator.reopening_active)
            for c in Planner.seen[-1]['candidates']:
                self.assertAlmostEqual(c['pregrasp_offset'],.1)
                self.assertGreater(np.linalg.norm(c['object_tcp_position']+c['approach_direction']*c['pregrasp_offset']),.07)
            result,robot=self.feed(1.05,opening=.8,close=False,visible=False)
            self.assertIsNone(coordinator.decide(raw,result,robot,h(1.05,1.))['state'])
            result,robot=self.feed(1.1,opening=1.,close=False)
            restored=coordinator.decide(raw,result,robot,h(1.1,1.))
            self.assertEqual(restored['mode'],'REOPENING_UNKNOWN_CONTACT');self.assertEqual(restored['action']['opening'],1.)

    def test_public_attachment_required_for_hold_and_visible_lift_metric_not_controlled_pose(self):
        metric=PublicLiftHoldMetric();observed=False;first_attached=None
        for tick in range(23):
            t=tick*.05;opening=max(.6,1.-.1*tick)
            result,robot=self.feed(t,opening=opening,close=tick>0,moving=True)
            attachment=self.wrapper.latest_attachment
            value=metric.observe(result,robot,attachment,self.wrapper.close_visual_baseline)
            if result['mode']=='ATTACHED_PUBLIC':
                first_attached=t if first_attached is None else first_attached
                state=self.assembler.build(result,robot,h(t))
                self.assertTrue(state['held']);self.assertFalse(state['free_motion_asserted'])
            observed|=value['public_lift_hold_observed']
        self.assertIsNotNone(first_attached);self.assertTrue(observed)
        self.assertTrue(metric.passed);self.assertGreaterEqual(metric.latest['visual_lift_m'],.04)
        saved=result['attachment_estimate'];result['attachment_estimate']={}
        with self.assertRaisesRegex(ValueError,'attachment-confirmed'):self.assembler.build(result,robot,h(robot.timestamp))
        result['attachment_estimate']=saved
        result,robot=self.feed(1.15,opening=.6,close=True,moving=True,visible=False)
        metric.observe(result,robot,self.wrapper.latest_attachment,self.wrapper.close_visual_baseline)
        self.assertFalse(metric.passed)
        # Raising a model-estimated controlled pose never supplies a visual label.
        self.assertFalse(metric.latest['current_visual_target']);self.assertFalse(metric.latest['controlled_pose_used_as_visual_label'])

    def test_controlled_pose_height_alone_cannot_supply_visible_lift(self):
        latest=None
        for tick in range(10):latest,robot=self.feed(tick*.05,opening=max(.6,1.-.1*tick),close=tick>0,moving=True)
        self.assertEqual(latest['mode'],'ATTACHED_PUBLIC')
        # Keep the current measured visual center at its actual baseline, while
        # pretending the separate controlled model is arbitrarily high.
        baseline=self.wrapper.close_visual_baseline
        latest['controlled']['estimated_world_pose']=Pose([.2,-.15,100.])
        latest['object_observation']=ObjectState('target',robot.timestamp,baseline.pose,baseline.points_world,np.zeros(3),np.zeros(3))
        metric=PublicLiftHoldMetric();value=metric.observe(latest,robot,self.wrapper.latest_attachment,baseline)
        self.assertEqual(value['visual_lift_m'],0.);self.assertFalse(value['public_lift_hold_observed'])

    def test_observer_one_real_frame_and_accurate_scope(self):
        raw=RawBackend();observer=ObservedPublicGraspBackend(raw,self.root/'public',public_bridge=self.wrapper,save_frames=False)
        def track(obs):
            target,_=sample(obs['timestamp']);return target,{'backend':'synthetic_CPU_fixture'}
        with patch.object(self.wrapper.tracker,'observe',side_effect=track) as spy,\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
            observer.sample();observer.tick('control');observer.tick('control')
        self.assertEqual(raw.ticks,2);self.assertEqual(raw.captures,3);self.assertEqual(spy.call_count,3)
        scope=json.loads((self.root/'public/scope.json').read_text())
        self.assertTrue(scope['public_attachment_estimator_used']);self.assertIn('grasp_prototype',scope['mode'])
        self.assertTrue(observer.receipt()['attachment_estimator_used'])

    def test_runner_visual_loss_holds_actual_q_and_last_closed_request_without_GT_exit(self):
        raw=RawBackend();observer=ObservedPublicGraspBackend(raw,self.root/'public',public_bridge=self.wrapper,save_frames=False)
        executor=Executor(observer);helper=GoalHelper()
        def track(obs):
            if obs['timestamp']>=.4-1e-9:raise ValueError('explicit synthetic visual occlusion')
            target,_=sample(obs['timestamp']);return target,{'backend':'synthetic_CPU_fixture'}
        with patch.object(self.wrapper.tracker,'observe',side_effect=track),\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})),\
             patch('our_mani.local_mpc_control_20261103.public_grasp.build_response_planner',return_value=(self.core(),helper,{})),\
             patch('our_mani.local_mpc_control_20261103.public_grasp.ForecastJointGoalCEMPlanner',Planner),\
             patch('our_mani.local_mpc_control_20261103.public_grasp.JointGoalCEMPlanner',Planner):
            report=run_public_grasp(observer,executor,self.root,dict(public_grasp_prototype=True,simulation_seconds=1.3))
        self.assertFalse(report['public_lift_hold_observed']);self.assertEqual(report['canary_stop_reason'],'budget_exhausted')
        commands=executor.commands;first=next(i for i,c in enumerate(commands) if c['opening']==0.)
        self.assertTrue(all(c['opening']==0. for c in commands[first:]))
        rows=[json.loads(line) for line in (self.root/'public_grasp_decisions.jsonl').read_text().splitlines()]
        self.assertTrue(any(r['decision']['state'] is None and r['executed_action']['opening']==0. for r in rows))
        self.assertFalse(report['Native_success_exit_used'])
        def contains_poison(value):
            if isinstance(value,dict):return any(contains_poison(v) for v in value.values())
            if isinstance(value,list):return any(contains_poison(v) for v in value)
            return type(value) in (int,float) and value==999
        self.assertFalse(contains_poison(rows))


if __name__=='__main__':unittest.main()
