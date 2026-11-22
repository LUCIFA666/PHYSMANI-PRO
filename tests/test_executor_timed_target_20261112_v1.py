import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_timed_tasks_20261112_v1.trajectory import sample
from our_mani.executor_timed_tasks_20261112_v1.commands import TimedWorldTarget,WorldPoseGoal

class TimedTargetTests(unittest.TestCase):
    def test_zero_lookahead_current_samples_are_advanced_to_next_physical_tick(self):
        history=[(t,np.array([.2*t,0,.8]),Rotation.from_euler('z',t).as_quat())for t in(0.,.1,.2,.3,.4)]
        p,q,twist,r=sample(history,0.,.05)
        np.testing.assert_allclose(p,[.09,0,.8],atol=1e-10)
        self.assertAlmostEqual(twist[0],.2,places=8)
        self.assertLess((Rotation.from_quat(q).inv()*Rotation.from_euler('z',.45)).magnitude(),1e-9)
    def test_timing_cannot_be_omitted_or_negative(self):
        goal=WorldPoseGoal(0.,[0,0,.8],[0,1,0,0],1.)
        with self.assertRaises(ValueError):TimedWorldTarget(goal,-.1,'source')
        with self.assertRaises(ValueError):TimedWorldTarget(goal,.5,'')

if __name__=='__main__':unittest.main()
