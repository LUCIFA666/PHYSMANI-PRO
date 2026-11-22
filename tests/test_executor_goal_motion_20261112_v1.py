import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_capture_repair_20261112_v1.goal_motion import current_pose

class GoalMotionTests(unittest.TestCase):
    def test_future_constant_motion_is_retimed_to_current_observation(self):
        history=[];velocity=np.array([.2,-.3,0.]);origin=np.array([.1,.2,.8])
        for t in (0.,.1,.2,.3,.4):
            history.append((t,origin+velocity*(t+.5),Rotation.from_euler('z',.4*(t+.5)).as_quat()))
        p,q,r=current_pose(history,.5)
        self.assertTrue(r['valid'])
        np.testing.assert_allclose(p,origin+velocity*.4,atol=1e-12)
        self.assertLess((Rotation.from_quat(q).inv()*Rotation.from_euler('z',.4*.4)).magnitude(),1e-12)

    def test_duplicate_frames_do_not_create_velocity_or_valid_history(self):
        _,_,r=current_pose([(1.,[0,0,.8],[0,0,0,1])]*5,.5)
        self.assertFalse(r['valid'])

    def test_rotating_waypoint_uses_rigid_motion_not_linear_extrapolation(self):
        history=[]
        def pose(t):
            r=Rotation.from_euler('z',1.2*t)
            return np.array([.2,-.1,.8])+r.apply([.2,0.,.1]),r.as_quat()
        for t in (0.,.1,.2,.3,.4):
            p,q=pose(t+.5);history.append((t,p,q))
        p,q,r=current_pose(history,.5);expected_p,expected_q=pose(.4)
        self.assertTrue(r['valid']);np.testing.assert_allclose(p,expected_p,atol=1e-10)
        self.assertLess((Rotation.from_quat(q).inv()*Rotation.from_quat(expected_q)).magnitude(),1e-10)

if __name__=='__main__':unittest.main()
