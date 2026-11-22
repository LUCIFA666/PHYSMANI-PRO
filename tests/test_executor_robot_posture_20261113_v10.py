import unittest
import numpy as np
from our_mani.executor_robot_posture_20261113_v10.posture import solve_posture

class RobotPostureTest(unittest.TestCase):
    def fk(self,q):
        q=np.asarray(q);result=np.broadcast_to(np.eye(4),(*q.shape[:-1],4,4)).copy()
        result[...,:3,3]=q
        return result

    def test_solver_finds_robot_pose_without_changing_reference_or_goal(self):
        bank=np.array([[0,0,0],[.2,.1,.3],[-.2,-.1,.1]])
        current=np.zeros(3);goal=np.array([.3,.2,.4])
        q,r=solve_posture(self.fk,bank,current,goal,[0,0,0,1],[-1]*3,[1]*3)
        self.assertTrue(r['available']);np.testing.assert_allclose(q,goal,atol=.002)
        np.testing.assert_allclose(current,[0,0,0]);np.testing.assert_allclose(goal,[.3,.2,.4])
        self.assertFalse(r['Native_setters_called'])

    def test_endpoints_failing_independent_geometry_check_are_not_used(self):
        q,r=solve_posture(self.fk,np.array([[0,0,0],[.1,.1,.1]]),np.zeros(3),[.3,.2,.4],
            [0,0,0,1],[-1]*3,[1]*3,safe=lambda q:False)
        self.assertIsNone(q);self.assertFalse(r['available'])

if __name__=='__main__':unittest.main()
