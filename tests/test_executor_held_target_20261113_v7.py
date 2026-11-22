import unittest
from our_mani.executor_held_target_20261113_v7.pose_executor import HeldGoalContinuity
from our_mani.executor_timed_tasks_20261112_v1.commands import TimedWorldTarget,WorldPoseGoal

class HeldTargetTest(unittest.TestCase):
    def packet(self,t,x):return TimedWorldTarget(WorldPoseGoal(t,[x,0,1.],[0,0,0,1],0.),.5,'actual_source_goal')

    def test_approach_can_reselect_but_held_placement_cannot_jump_targets(self):
        state=[False];gate=HeldGoalContinuity(lambda:state[0])
        for i in range(6):self.assertTrue(gate.observe(self.packet(i*.1,i*.1),'approach')[0])
        self.assertEqual(len(gate.history),0)
        state[0]=True
        for i in range(5):self.assertTrue(gate.observe(self.packet(1.+i*.1,.15+i*.001),'place')[0])
        self.assertFalse(gate.observe(self.packet(1.5,.30),'place')[0])

    def test_release_starts_a_new_target_selection_cycle(self):
        state=[True];gate=HeldGoalContinuity(lambda:state[0])
        for i in range(5):gate.observe(self.packet(i*.1,0.),'place')
        self.assertFalse(gate.observe(self.packet(.5,.2),'place')[0])
        state[0]=False
        self.assertTrue(gate.observe(self.packet(.6,.2),'place')[0])
        state[0]=True
        self.assertTrue(gate.observe(self.packet(.7,.2),'place')[0])

if __name__=='__main__':unittest.main()
