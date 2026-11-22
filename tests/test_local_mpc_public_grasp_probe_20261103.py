"""Pure CPU bounded verification contracts; no Native or fitting."""
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch
import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.public_attachment import AttachmentConfig
from our_mani.local_mpc_control_20261103.public_grasp_probe import (
    BoundedVerificationProbe,VerificationProbeConfig,VerificationProbePlanner,
    MOTION_GATES,NONMOTION_GATES,qualify_probe_candidates)
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture
from our_mani.local_mpc_20261102.terminal_mpc import TerminalCostConfig
from our_mani.local_mpc_control_20261103.public_grasp import PublicGraspCoordinator,PublicGraspConfig


def sensors(t=1.,position=None):
    p=np.array([.2,-.15,.78]) if position is None else np.array(position,float)
    robot=PublicRobotSensors(t,p,[0,0,0,1],np.zeros(7),np.zeros(7),[.6,.6])
    target=ObjectState('target',t,Pose(p-[0,0,.015]),np.zeros((3,3)),np.zeros(3),np.zeros(3))
    gates={key:key not in MOTION_GATES for key in MOTION_GATES|NONMOTION_GATES}
    a=dict(timestamp=t,mode='UNRESOLVED',config=asdict(AttachmentConfig()),diagnostic=dict(gates=gates))
    result=dict(timestamp=t,mode='UNRESOLVED',object_observation=target,attachment_estimate=a)
    h=dict(gripper_target=0.,reference_clock=t)
    return result,robot,h


def planner_fixture():
    base,state,candidates=fixture();state['held']=False;state['requested_opening']=0.
    state['opening']=.6;state['finger_open_amounts']=np.array([.6,.6])
    state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=0.,close_entered_at=0.,reopening=False)
    result,robot,h=sensors(0.,state['tcp_position'])
    robot=PublicRobotSensors(0.,state['tcp_position'],state['tcp_quaternion'],state['q'],state['dq'],[.6,.6])
    manager=BoundedVerificationProbe();spec=manager.consider(result,robot,h)
    planner=VerificationProbePlanner(execution_model=base.execution_model,goal_helper=base.goal_helper,probe_spec=spec)
    candidates=[dict(c,native_pose_qualified=True) for c in candidates]
    return planner,state,candidates,manager


class ProbeTests(unittest.TestCase):
    def coordinator_fixture(self,enabled=True):
        planner,state,candidates,_=planner_fixture()
        class Assembler:
            robot_twist=(np.zeros(3),np.zeros(3),True)
            def build(self,*args,**kwargs):return None
            def build_pending(self,result,robot,h,**kwargs):
                if result.get('object_observation') is None:return None
                value=deepcopy(state);value['time']=robot.timestamp
                value['phase_state']=kwargs.get('phase_state')
                value['grasp_motion_mode']='PENDING_CLOSE_UNKNOWN_CONTACT'
                return value
        class Oracle:
            def probe_joint_collision(self,q):return False
        core=SimpleNamespace(config=planner.config,execution_model=planner.execution_model,
            planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig())
        coordinator=PublicGraspCoordinator(core,planner.goal_helper,Assembler(),config=PublicGraspConfig(verification_probe_enabled=enabled))
        coordinator.candidates=candidates;coordinator.pending_started=-.4;coordinator.mode='PENDING_CLOSE_UNKNOWN_CONTACT'
        result,_,h=sensors(0.,state['tcp_position'])
        robot=PublicRobotSensors(0.,state['tcp_position'],state['tcp_quaternion'],state['q'],state['dq'],[.6,.6])
        return coordinator,Oracle(),result,robot,h

    def test_admission_requires_each_original_nonmotion_gate_and_current_robot_visual(self):
        for key in NONMOTION_GATES:
            result,robot,h=sensors();result['attachment_estimate']['diagnostic']['gates'][key]=False
            probe=BoundedVerificationProbe();self.assertIsNone(probe.consider(result,robot,h));self.assertFalse(probe.used)
        for mutate in ('stale','vision','open','empty','missinggate'):
            result,robot,h=sensors()
            if mutate=='stale':h['reference_clock']-=.05
            elif mutate=='vision':result['object_observation']=None
            elif mutate=='open':h['gripper_target']=1.
            elif mutate=='empty':robot=PublicRobotSensors(1.,robot.tcp_position,robot.tcp_quaternion,robot.q,robot.dq,[0.,0.])
            else:del result['attachment_estimate']['diagnostic']['gates']['closure_actually_observed']
            self.assertIsNone(BoundedVerificationProbe().consider(result,robot,h),mutate)

    def test_fixed_trigger_anchor_not_preclose_or_chasing_target_and_once_per_attempt(self):
        probe=BoundedVerificationProbe();result,robot,h=sensors()
        spec=probe.consider(result,robot,h);self.assertIsNotNone(spec)
        np.testing.assert_array_equal(spec['anchor_position'],robot.tcp_position)
        np.testing.assert_allclose(spec['target_position']-robot.tcp_position,[0,0,.012],atol=1e-15)
        result,robot,h=sensors(1.05,robot.tcp_position+[0,0,.002])
        again=probe.consider(result,robot,h)
        np.testing.assert_array_equal(again['target_position'],spec['target_position'])
        self.assertEqual(spec['identity'],again['identity']);self.assertFalse(again['held'])
        result['attachment_estimate']['diagnostic']['gates']['stable_relative_translation']=False
        self.assertIsNone(probe.consider(result,robot,h));self.assertFalse(probe.active)
        result['attachment_estimate']['diagnostic']['gates']['stable_relative_translation']=True
        self.assertIsNone(probe.consider(result,robot,h));self.assertTrue(probe.used)
        probe.reset();self.assertIsNotNone(probe.consider(result,robot,h))

    def test_actual_bound_gate_and_configuration_cannot_expand_beyond_15mm(self):
        for kwargs in (dict(lift_m=.02),dict(maximum_displacement_m=.016),dict(maximum_rotation_rad=.05)):
            with self.assertRaises(ValueError):VerificationProbeConfig(**kwargs)
        probe=BoundedVerificationProbe();result,robot,h=sensors();probe.consider(result,robot,h)
        result,robot,h=sensors(1.05,robot.tcp_position+[0,0,.016])
        self.assertIsNone(probe.consider(result,robot,h));self.assertIn('bound',probe.reason)

    def test_mid50ms_failure_latches_even_if_next100ms_frame_recovers(self):
        probe=BoundedVerificationProbe();result,robot,h=sensors();probe.consider(result,robot,h)
        result,robot,h=sensors(1.05)
        result['attachment_estimate']['diagnostic']['gates']['fingers_consistent']=False
        probe.observe_feedback(result,robot,close_requested=True)
        self.assertFalse(probe.active);self.assertEqual(probe.abort_feedback_time,1.05)
        result,robot,h=sensors(1.1)
        probe.observe_feedback(result,robot,close_requested=True)
        self.assertIsNone(probe.consider(result,robot,h))
        self.assertFalse(probe.receipt()['midmacro_physical_hard_stop_claimed'])
        self.assertTrue(probe.used)
        probe=BoundedVerificationProbe();result,robot,h=sensors()
        result['attachment_estimate']['config']['minimum_TCP_motion_m']=.001
        self.assertIsNone(probe.consider(result,robot,h))

    def test_same_frame_attachment_confirmation_cannot_hide_actual_bound_failure(self):
        probe=BoundedVerificationProbe();result,robot,h=sensors();probe.consider(result,robot,h)
        result,robot,h=sensors(1.05,robot.tcp_position+[0.,0.,.016])
        result['mode']='ATTACHED_PUBLIC'
        probe.observe_feedback(result,robot,close_requested=True)
        self.assertFalse(probe.active);self.assertEqual(probe.abort_feedback_time,1.05)
        self.assertIn('bound_failure',probe.reason)

    def test_CEM_256_robot_model_exact_joint_dispatch_no_fake_held_or_opening_edit(self):
        planner,state,candidates,_=planner_fixture();original=deepcopy(state);calls=[];solve=planner.goal_helper.adapter.solve
        with patch.object(planner.goal_helper.adapter,'solve',side_effect=lambda q,p,r:(calls.append(q.shape) or solve(q,p,r))):
            result=planner.plan(state,candidates)
        self.assertEqual(calls,[(8,7)]);self.assertEqual(result['budget']['sequences'],256)
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80]);self.assertIsNotNone(result['action'])
        self.assertEqual(result['diagnostics']['execution_model_calls'],24)
        self.assertFalse(result['diagnostics']['object_follows_probe_assumed'])
        self.assertTrue(all(not s['held'] for s in result['predicted_states']))
        self.assertTrue(all(a['opening']==0. for a in result['sequence']))
        self.assertTrue(all(planner._inside(s['tcp_position'][None],s['tcp_quaternion'][None])[0] for s in result['predicted_states']))
        for action in result['sequence']:
            fk=planner.goal_helper.kinematics(action['joint_goal'])
            np.testing.assert_array_equal(action['tcp_position'],fk[:3,3])
        q,reason,receipt=planner.goal_helper.solve_action_target(state,result['action'])
        self.assertIsNone(reason);self.assertEqual(receipt['DLS_solves'],0)
        np.testing.assert_array_equal(q,result['action']['joint_goal'])
        self.assertGreater(result['predicted_states'][-1]['tcp_position'][2],state['tcp_position'][2]+.005)
        np.testing.assert_array_equal(state['q'],original['q']);self.assertEqual(state['held'],original['held'])
        bad=deepcopy(result['warm_start']);bad['verification_probe_identity']='different'
        with self.assertRaisesRegex(ValueError,'warm'):planner.plan(state,candidates,warm_start=bad)

    def test_bounds_apply_to_commands_and_both_micro_predictions_not_only_endpoint(self):
        planner,state,candidates,_=planner_fixture();planner._prepare_nominals(state,candidates,state['phase_state'])
        score,_,_,actions=planner._rollout_batch(state,candidates,np.arange(8),np.ones((8,8,7)),state['phase_state'],None,None)
        self.assertTrue(np.isinf(score).all());self.assertFalse(actions['joint_goal_feasible'].any())
        original=planner.execution_model.step_batch
        def overshoot(*args):
            nxt,h,info=original(*args);info['predicted_actual_q'][:,0,0]+=.4
            return nxt,h,info
        with patch.object(planner.execution_model,'step_batch',side_effect=overshoot):
            score,_,_,_=planner._rollout_batch(state,candidates,np.arange(8),np.zeros((8,8,7)),state['phase_state'],None,None)
        self.assertTrue(np.isinf(score).all())

    def test_collision_qualification_preserves_original_ids_deadlines_and_only_probe_target(self):
        planner,state,candidates,manager=planner_fixture()
        class Oracle:
            calls=[]
            def probe_joint_collision(self,q):self.calls.append(q.copy());return False
            def solve_ik(self,*args):raise AssertionError('Native IK forbidden')
        oracle=Oracle();qualified,receipt=qualify_probe_candidates(oracle,state,candidates,planner.goal_helper,manager.spec)
        self.assertEqual(len(qualified),8);self.assertEqual(len(oracle.calls),8)
        self.assertEqual([c['deadline'] for c in qualified],[c['deadline'] for c in candidates])
        self.assertEqual([c['id'] for c in qualified],[c['id'] for c in candidates])
        for c in receipt['cases']:np.testing.assert_array_equal(c['target_position'],manager.spec['target_position'])
        self.assertFalse(receipt['formal_qualification'])

    def test_coordinator_probe_is_opt_in_and_gate_loss_holds_until_original_timeout(self):
        coordinator,oracle,result,robot,h=self.coordinator_fixture()
        coordinator.warm={'incompatible_grasp_warm':True}
        decision=coordinator.decide(oracle,result,robot,h)
        self.assertEqual(decision['mode'],'PENDING_VERIFICATION_PROBE')
        self.assertFalse(decision['state']['held']);self.assertEqual(decision['action']['opening'],0.)
        self.assertEqual(decision['plan']['budget']['sequences'],256)
        self.assertEqual(coordinator.pending_started,-.4)
        # A later bad gate cannot silently switch back to moving pending MPC.
        result['attachment_estimate']['diagnostic']['gates']['stable_relative_translation']=False
        result['timestamp']=.1;result['attachment_estimate']['timestamp']=.1
        old=result['object_observation'];result['object_observation']=ObjectState('target',.1,old.pose,old.points_world,np.zeros(3),np.zeros(3))
        robot=PublicRobotSensors(.1,robot.tcp_position,robot.tcp_quaternion,robot.q,robot.dq,robot.finger_open_amounts)
        h['reference_clock']=.1
        held=coordinator.decide(oracle,result,robot,h)
        self.assertEqual(held['mode'],'PENDING_PROBE_ABORTED_HOLD');self.assertIsNone(held['state'])
        self.assertIsNone(coordinator.warm)
        # Original first-close+.8s is .4, unchanged by probe admission at 0.
        result['timestamp']=.4;result['attachment_estimate']['timestamp']=.4
        result['object_observation']=ObjectState('target',.4,old.pose,old.points_world,np.zeros(3),np.zeros(3))
        robot=PublicRobotSensors(.4,robot.tcp_position,robot.tcp_quaternion,robot.q,robot.dq,robot.finger_open_amounts)
        h['reference_clock']=.4
        reopened=coordinator.decide(oracle,result,robot,h)
        self.assertEqual(reopened['mode'],'REOPENING_UNKNOWN_CONTACT')
        if reopened['action'] is not None:self.assertEqual(reopened['action']['opening'],1.)
        disabled,oracle,result,robot,h=self.coordinator_fixture(enabled=False)
        decision=disabled.decide(oracle,result,robot,h)
        self.assertEqual(decision['mode'],'PENDING_CLOSE_UNKNOWN_CONTACT')
        self.assertFalse(disabled.verification_probe.used)

    def test_probe_visual_loss_keeps_unknown_and_recovery_cannot_restart_attempt(self):
        coordinator,oracle,result,robot,h=self.coordinator_fixture()
        decision=coordinator.decide(oracle,result,robot,h);self.assertEqual(decision['mode'],'PENDING_VERIFICATION_PROBE')
        target=result['object_observation'];result['object_observation']=None
        missing=coordinator.decide(oracle,result,robot,h)
        self.assertEqual(missing['mode'],'UNAVAILABLE');self.assertIsNone(missing['state'])
        result['object_observation']=target
        restored=coordinator.decide(oracle,result,robot,h)
        self.assertEqual(restored['mode'],'PENDING_PROBE_ABORTED_HOLD')
        self.assertIsNone(restored['action']);self.assertFalse(coordinator.verification_probe.active)


if __name__=='__main__':unittest.main()
