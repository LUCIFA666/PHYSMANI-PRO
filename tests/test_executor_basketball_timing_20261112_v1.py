import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
from our_mani.executor_cartesian_consumer_20261112_v1.pose_executor import PoseGoalExecutor,WorldPoseGoal

class BasketballTimingTests(unittest.TestCase):
    def make(self,held,latched):
        dispatch=object.__new__(PoseGoalExecutor)
        dispatch.backend=SimpleNamespace(clock=lambda:0.,gripper=SimpleNamespace(get_grasped_objects=lambda:[1]if held else[]))
        dispatch.commitment=SimpleNamespace(release_latched=latched)
        dispatch.capture_lookahead_seconds=.5
        dispatch._goal_history=deque([(t,np.array([.2*(t+.5),0.,.8]),np.array([0.,1.,0.,0.]))for t in(-.4,-.3,-.2,-.1,0.)],maxlen=5)
        dispatch._active_action=WorldPoseGoal(0.,[.1,0,.8],[0,1,0,0],1.)
        return dispatch

    def test_open_unheld_approach_does_not_subtract_future_lookahead(self):
        p,q,twist,receipt=self.make(False,False)._sample_target()
        np.testing.assert_allclose(p,[.1,0,.8]);np.testing.assert_array_equal(twist,np.zeros(6))
        self.assertEqual(receipt['declared_lookahead_seconds'],0.)

    def test_inflight_release_keeps_place_timing_after_detachment(self):
        p,q,twist,receipt=self.make(False,True)._sample_target()
        np.testing.assert_allclose(p,[0.,0.,.8],atol=1e-10)
        self.assertTrue(receipt['valid']);self.assertAlmostEqual(twist[0],.2,places=8)

if __name__=='__main__':unittest.main()
