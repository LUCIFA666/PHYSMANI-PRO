import unittest
import numpy as np
from our_mani.executor_reference_unload_20261113_v4.unload import LiveReferenceUnload
from our_mani.executor_grasp_frame_20261113_v1.reference import step,ReferenceState
from our_mani.local_mpc_20261102.executor import JointLimits

class LiveUnloadTest(unittest.TestCase):
    def test_unload_follows_a_second_physical_displacement_with_continuous_commands(self):
        law=LiveReferenceUnload();limits=JointLimits([-2,-2],[2,2],[1,1],acceleration=12.,jerk=1000.)
        state=ReferenceState(np.array([.2,.1]),np.zeros(2),np.zeros(2));maximum=0.
        for i in range(220):
            actual=np.zeros(2)if i<100 else np.array([-.15,.1])
            v,r=law.update(state.position,actual,limits.lower,limits.upper,i*.05)
            old=state;state,_=step(state,np.zeros(2)if v is None else v,limits,actual_q=actual)
            maximum=max(maximum,float(np.max(abs(state.velocity))))
            np.testing.assert_allclose(state.position-old.position,.05*state.velocity,atol=1e-10)
            self.assertTrue(np.all(abs(state.acceleration-old.acceleration)<=50.+1e-7))
        self.assertLess(np.max(abs(state.position-actual)),.025)
        self.assertLessEqual(maximum,1.+1e-7)

    def test_measured_limit_violation_remains_visible_and_commands_retreat_inside(self):
        law=LiveReferenceUnload();actual=np.array([2.1,-2.1])
        velocity,r=law.update([1.99,-1.99],actual,[-2,-2],[2,2],0.)
        self.assertTrue(r['active']);self.assertEqual(r['measured_joint_positions'],[2.1,-2.1])
        self.assertEqual(r['explicit_reference_unload_goal'],[1.95,-1.95])
        self.assertLess(velocity[0],0);self.assertGreater(velocity[1],0)
        np.testing.assert_allclose(actual,[2.1,-2.1])

if __name__=='__main__':unittest.main()
