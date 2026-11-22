from dataclasses import replace
from types import SimpleNamespace as NS,MethodType
import unittest
import test_transport_phase_clock_v158 as original
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.cup_phase_slack_v205 import configure_clock_and_slack


class Tests(original.Tests):
    def test_existing_slack_is_applied_once_without_resetting_confirmed_epoch(self):
        controller=NS(_grant_phase_slack=lambda *args:None)
        old=configure_clock_and_slack(controller,self.clock,2.)
        self.assertAlmostEqual(old[self.index],2.3)
        self.assertAlmostEqual(self.clock.budgets[self.index],4.3)
        self.clock.confirm(self.index,5.4,5.45,9.)
        self.clock.confirm(self.index,8.4,8.45,9.9)
        self.assertEqual(self.clock.epochs[self.index]['deadline'],9.)
        self.assertEqual(self.clock.epochs[self.index]['issued_at'],5.4)
        with self.assertRaises(ValueError):configure_clock_and_slack(controller,self.clock,2.)

    def test_later_endpoint_cannot_receive_slack_beyond_first_prefix_deadline(self):
        state,forecast=self.scene(7.55)
        # Existing test fixture first starts the epoch at6.55, deadline8.85.
        c=NS(_phase_slack_tokens=set(),attempts=NS(records=[]),phase_timing_events=[],
            feedback_config=NS(phase_tracking_slack_seconds=2.),goal=None,phase_index=self.index)
        c._grant_phase_slack=MethodType(ObservedFeedbackController._grant_phase_slack,c)
        configure_clock_and_slack(c,self.clock,2.)
        self.clock.confirm(self.index,6.55,6.6,8.85)
        g,_=self.generator();c.goal=g.generate(state,forecast,self.index,2.45,state.robot.pose)[0]
        c._grant_phase_slack(state,2.45)
        self.assertLessEqual(c.goal.selected_at+c.goal.phase_budget,8.85+1e-8)
        self.assertEqual(c.phase_timing_events[0]['granted_slack_seconds'],0.)
        original_deadline=c.goal.selected_at+c.goal.phase_budget
        c._grant_phase_slack(state,2.45)
        self.assertEqual(len(c.phase_timing_events),1)
        self.assertEqual(c.goal.selected_at+c.goal.phase_budget,original_deadline)

    def test_early_endpoint_receives_only_headroom_within_same_global_capped_epoch(self):
        c=NS(_phase_slack_tokens=set(),attempts=NS(records=[]),phase_timing_events=[],
            feedback_config=NS(phase_tracking_slack_seconds=2.),goal=None,phase_index=self.index)
        c._grant_phase_slack=MethodType(ObservedFeedbackController._grant_phase_slack,c)
        configure_clock_and_slack(c,self.clock,2.)
        self.clock.confirm(self.index,6.55,6.6,9.)
        state,forecast=self.scene(6.6);g,_=self.generator()
        c.goal=g.generate(state,forecast,self.index,3.4,state.robot.pose)[0]
        c._grant_phase_slack(state,3.4)
        self.assertAlmostEqual(c.phase_timing_events[0]['granted_slack_seconds'],.1)
        self.assertAlmostEqual(c.goal.selected_at+c.goal.phase_budget,9.)


if __name__=='__main__':unittest.main()
