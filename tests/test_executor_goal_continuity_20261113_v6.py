import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_goal_continuity_20261113_v6.continuity import GoalContinuity,endpoint_errors
from our_mani.executor_timed_tasks_20261112_v1.commands import TimedWorldTarget,WorldPoseGoal

class GoalContinuityTest(unittest.TestCase):
    def packet(self,t,p,quaternion=(0,0,0,1),h=.5):
        return TimedWorldTarget(WorldPoseGoal(t,p,quaternion,0.),h,'explicit_test_forecast')

    def test_same_phase_jump_is_quarantined_without_synthetic_observations(self):
        gate=GoalContinuity()
        for i in range(5):self.assertTrue(gate.observe(self.packet(i*.1,[i*.001,0,1.]),'place')[0])
        packet=self.packet(.5,[.15,0,1.06]);accepted,receipt=gate.observe(packet,'place')
        self.assertFalse(accepted);self.assertEqual(len(gate.history),5)
        np.testing.assert_allclose(packet.pose.position,[.15,0,1.06])
        self.assertTrue(gate.observe(self.packet(.6,[.006,0,1.]),'place')[0])

    def test_real_segment_and_timing_changes_can_replace_the_goal(self):
        gate=GoalContinuity()
        for i in range(5):gate.observe(self.packet(i*.1,[0,0,1.]),'lift')
        self.assertTrue(gate.observe(self.packet(.5,[.3,0,.8]),'place')[0])
        self.assertEqual(len(gate.history),1)
        self.assertTrue(gate.observe(self.packet(.6,[.1,0,.8],h=0.),'place')[0])

    def test_quaternion_sign_flip_does_not_switch_a_physical_target(self):
        gate=GoalContinuity()
        for i in range(5):gate.observe(self.packet(i*.1,[0,0,1.]),'place')
        self.assertTrue(gate.observe(self.packet(.5,[0,0,1.],(0,0,0,-1)),'place')[0])

    def test_static_segment_ack_requires_position_and_orientation_completion(self):
        p,r=endpoint_errors([0,0,.97,0,0,0,1],[0,0,1.],[0,0,0,1])
        self.assertGreater(p,.003)
        q=Rotation.from_rotvec([0,.1,0]).as_quat()
        p,r=endpoint_errors([0,0,1.,*q],[0,0,1.],[0,0,0,1])
        self.assertGreater(r,.03)

if __name__=='__main__':unittest.main()
