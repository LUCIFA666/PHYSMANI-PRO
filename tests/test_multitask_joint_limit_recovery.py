"""Actual public joint reference recovery never enters virtual planning probes."""
from dataclasses import replace
import unittest

import numpy as np

from tests.test_multitask_pipeline import make_actor,observation,robot


class PublicJointRecoveryTests(unittest.TestCase):
    def actor(self):
        actor,scene,tracker=make_actor(joint_margin=.005)
        actor.robot_model.lower=np.full(7,-1.)
        actor.robot_model.upper=np.full(7,1.)
        return actor,tracker

    def test_only_violating_references_move_inward_without_planning_or_critic(self):
        actor,tracker=self.actor();calls=[];observe=tracker.observe
        tracker.observe=lambda *args,**kwargs:(calls.append(args[0]['timestamp']) or observe(*args,**kwargs))
        actor.controller.step=lambda *args:(_ for _ in ()).throw(AssertionError('recovery must bypass candidate/critic'))
        initial=robot(0.);q=np.array([-.3,-1.0002,.2,1.0001,0.,.4,-.2])
        current=replace(initial,joints=q)
        command=actor.act(observation(0.),current,10.)
        self.assertEqual(command['phase'],'joint_limit_recovery')
        np.testing.assert_allclose(command['joints'],[-.3,-.995,.2,.995,0.,.4,-.2])
        self.assertEqual(calls,[0.])
        self.assertFalse(command['trajectory_collision_verified'])
        self.assertFalse(command['native_state_reset'])
        self.assertTrue(command['each_command_requires_one_counted_tick'])
        self.assertEqual(len(actor.controller.attempts.records),0)

    def test_previous_requested_close_is_preserved_during_hold_and_recovery(self):
        actor,_=self.actor();actor.requested_opening=0.
        # Public open boolean can still be true while the last close command
        # is physically closing. Recovery must not release that attachment.
        initial=replace(robot(0.,contact=True),joints=np.r_[-1.0002,np.zeros(6)])
        command=actor.act(observation(0.),initial,10.)
        self.assertEqual(command['opening'],0.)
        self.assertEqual(actor.metrics_report()['latest_requested_opening'],0.)

    def test_normal_q_keeps_regular_warmup_and_recovery_does_not_pause_deadline(self):
        actor,_=self.actor();first=robot(0.)
        self.assertEqual(actor.act(observation(0.),first,10.)['command'],'hold')
        self.assertFalse(actor.joint_limit_recovery_events)
        bad=replace(robot(.05),joints=np.r_[-1.0002,np.zeros(6)])
        actor.act(observation(.05),bad,9.95)
        self.assertEqual(actor.joint_limit_recovery_events[-1]['remaining_episode_seconds'],9.95)
        actor.act(observation(.10),robot(.10),9.90)
        report=actor.metrics_report()
        self.assertEqual(report['recovery_commands_emitted'],1)
        self.assertEqual(report['recovery_counted_ticks_confirmed_by_next_public_observation'],1)
        self.assertFalse(report['recovery_final_command_may_require_parent_dispatch_receipt'])
        self.assertEqual(actor.episode_start,0.)


if __name__=='__main__': unittest.main()
