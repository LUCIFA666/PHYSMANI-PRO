import unittest
import numpy as np
from our_mani.executor_grasp_frame_20261113_v1.approach import AlignedApproach,TrackingRecovery

class ApproachRecoveryTest(unittest.TestCase):
    def test_misaligned_approach_keeps_source_goal_and_adds_hover(self):
        gate=AlignedApproach();goal=np.array([.3,.1,.9]);q=[0,0,0,1]
        p,r=gate.sample(goal,q,[.3,.1,1.1,1,0,0,0],0.,held=False)
        np.testing.assert_allclose(p,[.3,.1,1.1]);np.testing.assert_allclose(goal,[.3,.1,.9])
        self.assertFalse(r['source_world_goal_changed'])
        gate.sample(goal,q,[.3,.1,1.1,*q],.1,held=False)
        p,r=gate.sample(goal,q,[.3,.1,1.1,*q],.3,held=False)
        np.testing.assert_allclose(p,goal);self.assertEqual(r['phase'],'descend')
        p,r=gate.sample(goal,q,[.3,.1,.9,*q],.4,held=True)
        np.testing.assert_allclose(p,goal);self.assertFalse(r['applied'])

    def test_tracking_stall_produces_reference_goal_without_state_reset(self):
        gate=TrackingRecovery();reference=np.array([.6,.2]);actual=np.array([.1,.2])
        target,r=gate.update(reference,actual,1.)
        self.assertIsNone(target)
        target,r=gate.update(reference,actual,1.2)
        np.testing.assert_allclose(target,actual);np.testing.assert_allclose(reference,[.6,.2])
        self.assertFalse(r['reference_reseeded'])
        # The snapshot stays fixed while the reference unloads; it does not chase measurements.
        target,r=gate.update([.3,.2],[.12,.2],1.3)
        np.testing.assert_allclose(target,[.1,.2])
        target,r=gate.update([.12,.2],[.12,.2],1.4)
        self.assertIsNone(target)

if __name__=='__main__':unittest.main()
