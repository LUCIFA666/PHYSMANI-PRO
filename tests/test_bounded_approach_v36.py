import unittest
from types import SimpleNamespace as NS
import numpy as np
from our_mani.contracts import Pose
from our_mani.reachability import TimedReachability
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.bounded_approach_v36 import transit_waypoint,check_offset_brake


class Tests(unittest.TestCase):
    def test_horizontal_approach_before_descent_preserves_goal_and_uses_original_step(self):
        current=Pose(np.array([0.,0.,1.]),np.eye(3));target=Pose(np.array([.3,0.,.7]),np.eye(3))
        waypoint=transit_waypoint(current,target,.08)
        np.testing.assert_allclose(waypoint.position,[.3,0,1.]);np.testing.assert_allclose(target.position,[.3,0,.7])
        near=Pose(np.array([.04,0.,.7]),np.eye(3));np.testing.assert_allclose(transit_waypoint(current,near,.08).position,near.position)

    def test_backup_brake_checks_collisions_at_end_of_prefix_not_present_time(self):
        class Model:
            name='offset_test';lower=np.array([-2.]);upper=np.array([2.]);velocity_limits=np.array([2.]);acceleration_limits=np.array([6.])
            def __init__(self):self.times=[]
            def collision(self,q,t):self.times.append(t);return t>=.65
        model=Model();gate=TimedReachability(model);endpoint=NS(timestamp=.5,joints=np.array([.1]),joint_velocities=np.array([.2]))
        check=check_offset_brake(gate,endpoint,.5,.2)
        self.assertFalse(check.feasible);self.assertEqual(check.reason,'trajectory_collision');self.assertGreaterEqual(min(model.times),.5)

    def test_long_nominal_prefix_is_the_same_cubic_on_executed_interval(self):
        original=JointTrajectory(0.,1.1,np.zeros(2),np.zeros(2),np.array([.2,.1]),np.zeros(2))
        q,v=original.at(.5);prefix=JointTrajectory(0.,.5,np.zeros(2),np.zeros(2),q,v)
        self.assertGreater(np.linalg.norm(v),0.)
        for t in np.linspace(0,.5,21):
            a,b=original.at(t);c,d=prefix.at(t);np.testing.assert_allclose(a,c,atol=1e-14);np.testing.assert_allclose(b,d,atol=1e-14)


if __name__=='__main__':unittest.main()
