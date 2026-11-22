"""v4 observer/coordinator contracts, real frozen component construction CPU only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.local_mpc_control_20261103.public_grasp_v4 import (
    ActiveVisualPublicBridge,ObservedPublicGraspV4Backend,ActiveVisualGraspConfig,
    ActiveVisualGraspCoordinator,run_public_grasp_v4)
from our_mani.local_mpc_control_20261103.public_grasp import PublicLiftHoldMetric
from our_mani.local_mpc_20261102.terminal_mpc import TerminalCostConfig
from tests.test_local_mpc_active_visual_attachment_20261103 import sample,h
from tests.test_local_mpc_public_motion_bridge_20261103 import observation
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture
from tests.test_local_mpc_public_grasp_20261103 import RawBackend,Planner
from tests.test_local_mpc_public_pregrasp_20261103 import Executor,GoalHelper,NumericModel


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.wrapper=ActiveVisualPublicBridge()
    def setUp(self):self.wrapper.reset()
    def feed(self,t,dz=0.,*,close=False):
        obj,robot=sample(t,dz=dz)
        with patch.object(self.wrapper.tracker,'observe',return_value=(obj,{'backend':'explicit_synthetic_CPU_fixture'})),\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
            result=self.wrapper.observe(observation(t),robot,close_requested=close)
        return result,robot

    def test_legacy_positive_cannot_override_v4_unknown_or_feed_FREE_GRU(self):
        for i in range(5):result,robot=self.feed(i*.05)
        proof=PublicMotionEvidence('attachment_confirmed',.25,('rgbd_relative_pose_tracking','public_robot_pose'),Pose([0,0,-.025]))
        with patch.object(self.wrapper.estimator,'observe',return_value=dict(timestamp=.25,mode='ATTACHED_PUBLIC',evidence=proof)),\
             patch.object(self.wrapper.bridge.forecaster,'observe',wraps=self.wrapper.bridge.forecaster.observe) as spy:
            result,robot=self.feed(.25,close=True)
        self.assertEqual(result['legacy_attachment_diagnostic']['mode'],'ATTACHED_PUBLIC')
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertFalse(result['controlled_mode_available'])
        self.assertFalse(result['free_forecast_available']);self.assertFalse(result['legacy_attachment_controls_bridge'])
        self.assertEqual(spy.call_count,0);self.assertEqual(result['attachment_estimate']['protocol'],'active_vertical_visual_comotion_v4')

    def test_observer_once_per_real_tick_and_new_positive_requires_visible_metric(self):
        raw=RawBackend(budget=32)
        with TemporaryDirectory() as directory:
            observer=ObservedPublicGraspV4Backend(raw,Path(directory)/'public',public_bridge=self.wrapper,save_frames=False)
            def track(obs):
                obj,_=sample(obs['timestamp']);return obj,{'backend':'explicit_synthetic_CPU_fixture'}
            with patch.object(self.wrapper.tracker,'observe',side_effect=track) as spy,\
                 patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
                observer.sample();observer.tick('test');observer.tick('test')
            self.assertEqual(raw.ticks,2);self.assertEqual(raw.captures,3);self.assertEqual(spy.call_count,3)
            self.assertFalse(observer.receipt()['legacy_attachment_controls_bridge'])
        # Same tracked sample used for active evidence and actual visible lift.
        self.wrapper.reset();metric=PublicLiftHoldMetric(ActiveVisualGraspConfig())
        result,robot=self.feed(0.);metric.observe(result,robot,self.wrapper.latest_attachment,self.wrapper.close_visual_baseline)
        result,robot=self.feed(.05,close=True)
        self.wrapper.active.begin_probe(self.wrapper.active.last_target,robot,h(.05))
        metric.observe(result,robot,self.wrapper.latest_attachment,self.wrapper.close_visual_baseline)
        for i in range(1,6):
            result,robot=self.feed(.05+i*.05,.012*i/5,close=True)
            metric.observe(result,robot,self.wrapper.latest_attachment,self.wrapper.close_visual_baseline)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC');self.assertFalse(metric.passed)
        for i in range(11):
            result,robot=self.feed(.35+i*.05,.05,close=True)
            metric.observe(result,robot,self.wrapper.latest_attachment,self.wrapper.close_visual_baseline)
        self.assertTrue(metric.passed);self.assertFalse(metric.latest['controlled_pose_used_as_visual_label'])

    def test_coordinator_probe_deadline_is_admission_once_plus_point8_not_original_close_timer(self):
        base,state,candidates=fixture();state['opening']=.6;state['finger_open_amounts']=np.array([.7,.3]);state['requested_opening']=0.
        active=ActiveVisualAttachment();target=[None]
        class Assembler:
            robot_twist=(np.zeros(3),np.zeros(3),True)
            binding=SimpleNamespace(latest_visual_target=None)
            def build(self,*args,**kwargs):return None
            def build_pending(self,result,robot,actuator_state,**kwargs):
                value=deepcopy(state);value.update(time=robot.timestamp,actuator_state=deepcopy(actuator_state),phase_state=kwargs.get('phase_state'))
                return value
        class Oracle:
            def probe_joint_collision(self,q):return False
        assembler=Assembler();core=SimpleNamespace(config=base.config,execution_model=base.execution_model,
            planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig())
        coordinator=ActiveVisualGraspCoordinator(core,base.goal_helper,assembler,active)
        coordinator.candidates=candidates;coordinator.pending_started=-.6
        def feedback(t):
            obj=ObjectState('target',t,Pose(state['object_position']),np.zeros((3,3)),np.zeros(3),np.zeros(3))
            robot=PublicRobotSensors(t,state['tcp_position'],state['tcp_quaternion'],state['q'],state['dq'],state['finger_open_amounts'])
            active.observe(obj,robot,close_requested=True);assembler.binding.latest_visual_target=obj
            return dict(timestamp=t,mode='UNRESOLVED'),robot,h(t)
        result,robot,actuator=feedback(0.)
        first=coordinator.decide(Oracle(),result,robot,actuator)
        self.assertEqual(first['mode'],'ACTIVE_VISUAL_VERIFICATION_PROBE');self.assertEqual(first['probe']['deadline'],.8)
        identity=first['probe']['identity']
        for i in range(1,7):result,robot,actuator=feedback(i*.05)
        later=coordinator.decide(Oracle(),result,robot,actuator)
        self.assertEqual(later['mode'],'ACTIVE_VISUAL_VERIFICATION_PROBE')
        self.assertEqual(later['probe']['identity'],identity);self.assertEqual(later['probe']['deadline'],.8)
        self.assertEqual(later['plan']['budget']['sequences'],256)
        for i in range(7,17):result,robot,actuator=feedback(i*.05)
        expired=coordinator.decide(Oracle(),result,robot,actuator)
        self.assertEqual(expired['mode'],'REOPENING_UNKNOWN_CONTACT')

    def test_runner_does_not_use_Native_proxy_held_or_success_to_stop(self):
        self.wrapper.reset();raw=RawBackend(budget=10)
        with TemporaryDirectory() as directory:
            observer=ObservedPublicGraspV4Backend(raw,Path(directory)/'public',public_bridge=self.wrapper,save_frames=False)
            executor=Executor(observer);helper=GoalHelper()
            core=SimpleNamespace(config=__import__('our_mani.local_mpc_20261102.mpc',fromlist=['MPCConfig']).MPCConfig(),
                execution_model=NumericModel(),planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig())
            def track(obs):
                obj,_=sample(obs['timestamp']);return obj,{'backend':'explicit_synthetic_CPU_fixture'}
            with patch.object(self.wrapper.tracker,'observe',side_effect=track),\
                 patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})),\
                 patch('our_mani.local_mpc_control_20261103.public_grasp_v4.build_response_planner',return_value=(core,helper,{})),\
                 patch('our_mani.local_mpc_control_20261103.public_grasp_v4.ForecastJointGoalCEMPlanner',Planner),\
                 patch('our_mani.local_mpc_control_20261103.public_grasp_v4.JointGoalCEMPlanner',Planner),\
                 patch('our_mani.local_mpc_control_20261103.public_grasp_v4.VerificationProbePlanner',Planner):
                report=run_public_grasp_v4(observer,executor,directory,dict(public_grasp_active_visual_v4=True,simulation_seconds=.5))
            self.assertFalse(report['public_lift_hold_observed']);self.assertEqual(report['cycles'],5)
            self.assertFalse(report['Native_success_exit_used']);self.assertFalse(report['force_or_frictional_grasp_proof'])


if __name__=='__main__':unittest.main()
