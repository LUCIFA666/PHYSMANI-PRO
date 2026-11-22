"""Observed-motor high-speed braking, unchanged collisions and issued receipts."""
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.reachability import TimedReachability
from our_mani.multitask.feedback_interaction import bounded_joint_segment, ObservedFeedbackController
from our_mani.multitask.online_servo_response import OnlineServoResponseCompensator
from our_mani.multitask.online_servo_response_braking import (
    OnlineServoResponseBrakingCompensator, install_servo_response_braking, braking_response_binding)
from our_mani.multitask.mode_observer import PublicCommandSegments
from tests.test_feedback_interaction import GateModel, robot, FeedbackTests


class BrakingTests(unittest.TestCase):
    def warm(self, comp, speed=.4, gain=.65, count=9):
        q=np.zeros(7);v=np.full(7,speed)
        for tick in range(count):
            observed=robot(q,v,tick*.05);comp.observe(observed)
            ref=q+speed*.05/gain
            comp.record_issued(observed,ref,kind='joint_trajectory')
            next_q=q+gain*(ref-q);v=(next_q-q)/.05;q=next_q
        return robot(q,v,count*.05)

    def test_high_measured_velocity_old_displacement_scaling_fails_new_anchor_passes(self):
        model=GateModel();old=OnlineServoResponseCompensator(model)
        new=OnlineServoResponseBrakingCompensator(model)
        observed=self.warm(old);matching=self.warm(new)
        self.assertTrue(np.array_equal(observed.joints,matching.joints))
        desired,terminal=bounded_joint_segment(observed,observed.joints-1.,model,.05,.5)
        gate=TimedReachability(model)
        original=old.prepare(observed,desired,terminal,gate)
        changed=new.prepare(observed,desired,terminal,gate)
        self.assertEqual(original['action'],'hold_replan')
        self.assertEqual(original['reason'],'no_empirical_interval_sampled_path_feasible')
        self.assertEqual(changed['action'],'emit_compensated_reference')
        self.assertLess(changed['selected_scale'],1.)
        self.assertTrue(all(x['feasible']for x in changed['predicted_actual_samples']))
        qnext=observed.joints+.65*(np.asarray(changed['servo_reference'])-observed.joints)
        vnext=(qnext-observed.joints)/.05
        self.assertLessEqual(max(abs(vnext-observed.joint_velocities))/.05,6.+1e-6)
        self.assertFalse(changed['actual_tracking_guaranteed'])

    def test_tracking_failure_uses_feasible_continuous_brake_at_high_velocity(self):
        model=GateModel();comp=OnlineServoResponseBrakingCompensator(model)
        observed=self.warm(comp);comp.observe(observed)
        # The desired target has a zero-duration-style stop, outside the old
        # acceleration gate. Brake from actual measured velocity instead.
        result=comp.prepare(observed,observed.joints,np.zeros(7),TimedReachability(model))
        self.assertEqual(result['action'],'emit_empirical_braking_reference')
        self.assertFalse(result['desired_actual_path_original_gate']['feasible'])
        qnext=observed.joints+.65*(np.asarray(result['servo_reference'])-observed.joints)
        vnext=(qnext-observed.joints)/.05
        self.assertGreater(min(vnext),0.)
        self.assertLessEqual(max(vnext),.4+1e-6)
        self.assertLessEqual(max(abs(vnext-observed.joint_velocities))/.05,6.+1e-6)
        self.assertTrue(all(x['feasible']for x in result['predicted_actual_samples']))

    def test_original_dynamic_collision_rejects_tracking_and_every_braking_candidate(self):
        model=GateModel();comp=OnlineServoResponseBrakingCompensator(model)
        observed=self.warm(comp);model.dynamic_block=True
        desired,terminal=bounded_joint_segment(observed,observed.joints-1.,model,.05,.5)
        result=comp.prepare(observed,desired,terminal,TimedReachability(model))
        self.assertEqual(result['action'],'unverified_hold_no_feasible_braking')
        self.assertFalse(result['acceleration_limited_stop_verified'])
        self.assertTrue(model.collision_calls)
        self.assertTrue(all(not t['passed']for t in result['braking_trials']))
        self.assertTrue(any(t['samples'][-1]['reason']=='trajectory_collision'for t in result['braking_trials']))

    def test_closed_loop_braking_does_not_repeat_current_q_acceleration_spike(self):
        model=GateModel();comp=OnlineServoResponseBrakingCompensator(model)
        observed=self.warm(comp);gate=TimedReachability(model);speeds=[]
        # Current-q target with this synthetic motor stops .4rad/s in50ms,
        # giving8rad/s^2. The actual emitted brake must stay below6 instead.
        self.assertGreater(max(abs(observed.joint_velocities))/.05,6.)
        for _ in range(30):
            result=comp.brake(observed,gate,trigger='synthetic_continuous_braking')
            self.assertNotEqual(result['action'],'unverified_hold_no_feasible_braking')
            reference=np.asarray(result['servo_reference'])
            comp.record_issued(observed,reference,kind='joint_trajectory')
            next_q=observed.joints+.65*(reference-observed.joints)
            next_v=(next_q-observed.joints)/.05
            self.assertLessEqual(max(abs(next_v-observed.joint_velocities))/.05,6.+1e-6)
            self.assertLessEqual(max(abs(next_v)),max(abs(observed.joint_velocities))+1e-6)
            speeds.append(float(max(abs(next_v))))
            observed=robot(next_q,next_v,observed.timestamp+.05)
        self.assertLess(speeds[-1],.01)
        self.assertGreater(len(model.collision_calls),30)

    def test_model_mismatch_is_separate_from_caps_and_relearns_with_bounded_brake(self):
        model=GateModel();comp=OnlineServoResponseBrakingCompensator(model)
        observed=self.warm(comp,speed=.026);comp.observe(observed)
        comp.record_issued(observed,observed.joints+.002,kind='joint_trajectory')
        following=robot(observed.joints+.9*.002,np.full(7,.9*.002/.05),observed.timestamp+.05)
        result=comp.brake(following,TimedReachability(model),trigger='test_observed_response_change')
        self.assertEqual(result['observed_cap_violations'],[])
        self.assertEqual(result['observed_model_violations'],['observed_empirical_response_interval_violation'])
        self.assertFalse(comp.estimate()['ready'])
        self.assertEqual(result['action'],'emit_original_bounded_braking_reference')
        self.assertTrue(result['braking_reference_gate']['feasible'])
        self.assertFalse(result['predicted_actual_response_available'])
        self.assertFalse(result['goal_abandoned_by_outer_controller'])

    def test_installer_preserves_goal_clock_and_records_final_reference_and_opening(self):
        base,model=FeedbackTests().controller()
        controller=ObservedFeedbackController.from_controller(base,{})
        segments=PublicCommandSegments(('plate',));events=[]
        actor=SimpleNamespace(controller=controller,robot_model=model,command_segments=segments,
            requested_opening=1.,events=events,last_diagnostics={},last_command={'command':'hold'})
        def original_act(observation,observed,remaining):
            actor.command_segments.issue('press');actor.requested_opening=0.
            controller._close_edge_issued_at=observed.timestamp
            controller.gripper_timing_events.append(dict(time=observed.timestamp,commanded_opening=0.))
            events.append(dict(time=observed.timestamp,command='joint_trajectory',phase='approach',public_command_kind='press'))
            return dict(command='joint_trajectory',joints=observed.joints, joint_velocities=np.zeros(7),opening=0.,
                phase='approach',public_command_kind='press',measured_feedback_command=True)
        actor.act=original_act;actor.metrics_report=lambda:{'public_events':events}
        install_servo_response_braking(actor,{})
        comp=actor.servo_response_compensator;observed=self.warm(comp)
        goal=SimpleNamespace(selected_at=.1,phase_budget=3.3);trajectory=object();origin=object()
        controller.goal=goal;controller.trajectory=trajectory;controller.origin=origin
        result=actor.act({},observed,10.-observed.timestamp)
        self.assertEqual(result['command'],'joint_trajectory')
        self.assertEqual(result['servo_response_action'],'emit_empirical_braking_reference')
        self.assertIs(controller.goal,goal);self.assertIs(controller.trajectory,trajectory);self.assertIs(controller.origin,origin)
        self.assertEqual(goal.selected_at,.1);self.assertEqual(goal.phase_budget,3.3)
        self.assertIsNone(controller._close_edge_issued_at)
        self.assertEqual(result['opening'],1.);self.assertEqual(actor.requested_opening,1.)
        self.assertNotIn('plate',segments.non_exogenous)
        self.assertEqual(events[-1]['public_command_kind'],'joint_trajectory')
        self.assertEqual(comp.previous_issued['issued_reference'],result['joints'].tolist())
        self.assertEqual(actor.metrics_report()['online_servo_response_braking']['command_events'][-1]['servo_reference'],result['joints'].tolist())
        self.assertEqual(controller.gripper_timing_events[-1]['commanded_opening'],1.)
        self.assertEqual(controller.gripper_timing_events[-1]['inner_opening_proposal'],0.)
        self.assertFalse(actor.metrics_report()['online_servo_response_braking']['phase_clock_reset'])

    def test_invalid_config_and_duplicate_installer_are_rejected(self):
        for config in ({'version':'old'},{'braking_acceleration_fractions':[.5]}, {'braking_acceleration_fractions':[0.,1.]}, {'unknown':1}):
            with self.assertRaises(ValueError):braking_response_binding(config)
        binding=braking_response_binding({})
        self.assertNotEqual(binding['source'],binding['base_source'])
        base,model=FeedbackTests().controller()
        actor=SimpleNamespace(controller=ObservedFeedbackController.from_controller(base,{}),
            robot_model=model,act=lambda *args:dict(command='hold'),metrics_report=lambda:{})
        install_servo_response_braking(actor,{})
        with self.assertRaises(ValueError):install_servo_response_braking(actor,{})

    def test_first_act_uninitialized_opening_does_not_issue_inner_new_close_during_brake(self):
        base,model=FeedbackTests().controller()
        controller=ObservedFeedbackController.from_controller(base,{})
        actor=SimpleNamespace(controller=controller,robot_model=model,requested_opening=None,
            events=[],last_diagnostics={},last_command={'command':'hold'})
        def original_act(observation,observed,remaining):
            actor.requested_opening=0.
            return dict(command='joint_trajectory',joints=observed.joints,joint_velocities=np.zeros(7),
                opening=0.,phase='approach',measured_feedback_command=True)
        actor.act=original_act;actor.metrics_report=lambda:{}
        install_servo_response_braking(actor,{})
        # No response estimate exists on this first frame. The inner stop is
        # rejected at high measured velocity, so an explicit bounded brake is
        # emitted while the observed open gripper remains open.
        observed=robot(np.zeros(7),np.full(7,.4))
        result=actor.act({},observed,10.)
        self.assertEqual(result['servo_response_action'],'emit_original_bounded_braking_reference')
        self.assertEqual(result['opening'],float(observed.gripper_open))
        self.assertEqual(actor.requested_opening,float(observed.gripper_open))
        self.assertEqual(result['opening'],1.)

    def test_observed_cap_failure_is_logged_separately_and_no_safe_stop_is_invented(self):
        model=GateModel();comp=OnlineServoResponseBrakingCompensator(model)
        observed=robot(np.zeros(7),np.full(7,1.1))
        result=comp.brake(observed,TimedReachability(model),trigger='actual_observed_speed_cap')
        self.assertEqual(result['action'],'unverified_hold_no_feasible_braking')
        self.assertEqual(result['observed_model_violations'],[])
        self.assertEqual(result['observed_cap_violations'],['observed_joint_velocity_limit'])
        self.assertFalse(result['acceleration_limited_stop_verified'])
        self.assertEqual(comp.metrics()['observed_guard_events'][-1]['cap_violations'],['observed_joint_velocity_limit'])


if __name__=='__main__':unittest.main()
