"""Causal empirical-servo tests; sampled checks are explicitly approximate."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.contracts import Pose,RobotState
from our_mani.reachability import TimedReachability
from our_mani.multitask.feedback_interaction import bounded_joint_segment,ObservedFeedbackController
from our_mani.multitask.online_servo_response import (OnlineServoResponseCompensator,
    ServoResponseConfig,actuator_reference_bounds,install_servo_response_compensator)
from our_mani.multitask.mode_observer import PublicCommandSegments
from tests.test_feedback_interaction import GateModel,robot
from tests import test_feedback_interaction as feedback_fixtures


class ServoTests(unittest.TestCase):
    def warm(self,comp,model,gain=.65,count=9):
        q=np.zeros(7);v=np.zeros(7)
        for i in range(count):
            measured=robot(q,v,i*.05);comp.observe(measured)
            reference=q+np.full(7,.002)
            comp.record_issued(measured,reference,kind='joint_trajectory')
            next_q=q+gain*(reference-q);v=(next_q-q)/.05;q=next_q
        return robot(q,v,count*.05)

    def test_fit_only_uses_observed_past_issued_transitions_and_minimum_samples(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model)
        measured=self.warm(comp,model,count=7)
        self.assertFalse(comp.estimate()['ready'])
        comp.observe(measured)
        self.assertFalse(comp.estimate()['ready'])
        comp.record_issued(measured,measured.joints+.002,kind='joint_trajectory')
        next_q=measured.joints+.65*.002
        following=robot(next_q,np.full(7,.65*.002/.05),measured.timestamp+.05)
        comp.observe(following);estimate=comp.estimate()
        self.assertTrue(estimate['ready']);self.assertTrue(np.allclose(estimate['median'],.65))
        self.assertEqual(estimate['observed_until'],following.timestamp)
        with self.assertRaises(ValueError):comp.observe(measured)
        comp.record_issued(following,following.joints+.002,kind='joint_trajectory')
        with self.assertRaisesRegex(ValueError,'one issued'):comp.record_issued(following,following.joints+.002,kind='joint_trajectory')

    def test_unknown_response_explicitly_falls_back_and_setpoint_is_independently_bounded(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=robot()
        q,v=bounded_joint_segment(measured,np.ones(7),model,.05,.5)
        decision=comp.prepare(measured,q,v,TimedReachability(model))
        self.assertEqual(decision['action'],'fallback_original_bounded')
        self.assertFalse(decision['actual_tracking_guaranteed'])
        bounded=actuator_reference_bounds(measured,np.full(7,100.),model,.05)
        self.assertTrue(np.allclose(bounded,.05))
        self.assertFalse(decision['servo_reference_original_hermite_gate_verified'])

    def test_compensated_predicted_actual_path_and_servo_reference_are_distinct(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=self.warm(comp,model)
        q,v=bounded_joint_segment(measured,np.ones(7),model,.05,.5)
        decision=comp.prepare(measured,q,v,TimedReachability(model))
        self.assertEqual(decision['action'],'emit_compensated_reference')
        self.assertTrue(np.linalg.norm(np.asarray(decision['servo_reference'])-measured.joints)
            >np.linalg.norm(q-measured.joints))
        self.assertTrue(all(sample['feasible'] for sample in decision['predicted_actual_samples']))
        self.assertEqual(len(decision['predicted_actual_samples']),17)
        self.assertFalse(decision['full_uncertainty_region_proved'])
        self.assertFalse(decision['servo_reference_original_hermite_gate_verified'])

    def test_synthetic_motor_loop_tracks_faster_with_all_actual_observed_caps_retained(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);gate=TimedReachability(model)
        q=np.zeros(7);v=np.zeros(7);base_q=q.copy();base_v=v.copy();emitted=0
        for i in range(40):
            measured=robot(q,v,i*.05)
            desired,dv=bounded_joint_segment(measured,np.ones(7),model,.05,.5)
            decision=comp.prepare(measured,desired,dv,gate)
            self.assertNotEqual(decision['action'],'hold_replan',decision.get('reason'))
            emitted+=decision['action']=='emit_compensated_reference'
            reference=np.asarray(decision['servo_reference']);comp.record_issued(measured,reference,kind='joint_trajectory')
            next_q=q+.65*(reference-q);next_v=(next_q-q)/.05
            self.assertTrue(np.all(abs(next_v)<=model.velocity_limits+1e-7))
            self.assertTrue(np.all(abs(next_v-v)/.05<=model.acceleration_limits+1e-6))
            q,v=next_q,next_v
            baseline=robot(base_q,base_v,i*.05)
            desired,dv=bounded_joint_segment(baseline,np.ones(7),model,.05,.5)
            checked=gate.check_joints(baseline,desired,dv,.05);self.assertTrue(checked.feasible)
            next_q=base_q+.65*(desired-base_q);base_v=(next_q-base_q)/.05;base_q=next_q
        self.assertGreater(emitted,20)
        self.assertGreater(float(q[0]-base_q[0]),.3)
        self.assertGreater(float(q[0]),2*float(base_q[0]))

    def test_observed_acceleration_or_model_violation_holds_and_invalidates_model(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=self.warm(comp,model)
        bad=robot(measured.joints,np.ones(7),measured.timestamp)
        desired,dv=bounded_joint_segment(bad,np.ones(7),model,.05,.5)
        decision=comp.prepare(bad,desired,dv,TimedReachability(model))
        self.assertEqual(decision['action'],'hold_replan')
        self.assertIn('observed_joint_acceleration_limit',decision['observed_guard']['violations'])
        self.assertFalse(comp.estimate()['ready'])

    def test_original_collision_gate_cannot_be_bypassed_by_compensation(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=self.warm(comp,model)
        model.dynamic_block=True
        desired,dv=bounded_joint_segment(measured,np.ones(7),model,.05,.5)
        decision=comp.prepare(measured,desired,dv,TimedReachability(model))
        self.assertEqual(decision['action'],'hold_replan')
        self.assertEqual(decision['desired_actual_path_original_gate']['reason'],'trajectory_collision')

    def test_response_interval_violation_with_legal_motion_still_stops_and_relearns(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=self.warm(comp,model)
        comp.observe(measured);self.assertTrue(comp.estimate()['ready'])
        comp.record_issued(measured,measured.joints+.002,kind='joint_trajectory')
        changed=robot(measured.joints+.9*.002,np.full(7,.9*.002/.05),measured.timestamp+.05)
        guard=comp.observe(changed)
        self.assertEqual(guard['violations'],['observed_empirical_response_interval_violation'])
        self.assertTrue(guard['hold_replan']);self.assertFalse(comp.estimate()['ready'])

    def test_reported_terminal_velocity_is_not_silently_replaced_by_average_velocity(self):
        model=GateModel();comp=OnlineServoResponseCompensator(model);measured=self.warm(comp,model)
        inconsistent=replace(measured,joint_velocities=measured.joint_velocities+.01)
        guard=comp.observe(inconsistent)
        self.assertFalse(guard['hold_replan'])
        self.assertFalse(comp.estimate()['ready'])
        self.assertFalse(comp.transitions[-1]['reported_velocity_matches_backward_difference'])

    def test_installer_uses_final_issued_command_for_press_mask_and_closing_edge(self):
        base,model=feedback_fixtures.FeedbackTests().controller()
        controller=ObservedFeedbackController.from_controller(base,{})
        segments=PublicCommandSegments(('plate',));events=[]
        actor=SimpleNamespace(controller=controller,robot_model=model,command_segments=segments,
            requested_opening=1.,events=events,last_diagnostics={},last_command={'command':'hold'})
        def original_act(observation,measured,remaining):
            actor.command_segments.issue('press');actor.requested_opening=0.
            actor.controller._close_edge_issued_at=measured.timestamp
            actor.controller.gripper_timing_events.append(dict(time=measured.timestamp,commanded_opening=0.))
            actor.controller.feedback_events.append(dict(observed_time=measured.timestamp,
                bounded_joints=measured.joints.tolist(),command_issued=True))
            events.append(dict(time=measured.timestamp,command='joint_trajectory',phase='press',public_command_kind='press'))
            return dict(command='joint_trajectory',joints=measured.joints,opening=0.,phase='press',public_command_kind='press')
        actor.act=original_act;actor.metrics_report=lambda:{'public_events':events}
        install_servo_response_compensator(actor,{})
        actor.act({},robot(),10.)
        self.assertIn('plate',segments.non_exogenous)
        # A later guard-rejected proposed press must not latch a previously
        # external plate or create an unissued gripper closing edge.
        segments.non_exogenous.clear()
        command=actor.act({},robot(np.zeros(7),np.ones(7),.05),9.95)
        self.assertEqual(command['command'],'hold');self.assertNotIn('plate',segments.non_exogenous)
        self.assertIsNone(controller._close_edge_issued_at)
        self.assertEqual(events[-1]['public_command_kind'],'hold')
        self.assertFalse(controller.feedback_events[-1]['command_issued'])
        self.assertTrue(controller.feedback_events[-1]['inner_command_proposed'])
        self.assertFalse(controller.gripper_timing_events[-1]['actual_opening_request_issued'])
        self.assertIsNone(controller.gripper_timing_events[-1]['commanded_opening'])
        self.assertFalse(actor.metrics_report()['online_servo_response']['actual_tracking_guaranteed'])


if __name__=='__main__':unittest.main()
