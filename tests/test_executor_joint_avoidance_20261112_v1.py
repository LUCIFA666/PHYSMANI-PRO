import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.executor_contract_repair_20261112_v2.reference import ReferenceState
from our_mani.executor_differential_tracking_20261112_v1.feedback import desired_velocity
from our_mani.executor_differential_tracking_20261112_v1.reference import step

class Robot:
    def __call__(self,q):
        transform=np.eye(4);transform[:3,3]=q[:3]
        transform[:3,:3]=Rotation.from_rotvec(np.array(q[3:6],copy=True)).as_matrix();return transform
    def geometric_jacobian(self,q):return np.c_[np.eye(6),np.zeros(6)]

class JointAvoidanceTests(unittest.TestCase):
    def test_redundancy_moves_away_from_bound_while_holding_cartesian_pose(self):
        limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        actual=np.zeros(7);actual[6]=2.8;state=ReferenceState(actual,np.zeros(7),np.zeros(7))
        velocity,_=desired_velocity(Robot(),actual,state,[0,0,0],[0,0,0,1],np.zeros(6),limits,joint_avoidance='barrier')
        self.assertLess(velocity[6],-1.)
        np.testing.assert_allclose(velocity[:6],0.,atol=1e-12)

    def test_native_component_limits_allow_larger_combined_speed(self):
        limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7))
        for _ in range(20):
            state,_=step(state,np.full(7,1.5),limits,actual_q=state.position,speed_norm=np.sqrt(7)*2.,joint_margin=.02)
            self.assertLessEqual(np.max(abs(state.velocity)),2.+1e-8)
        self.assertGreater(np.linalg.norm(state.velocity),3.)

if __name__=='__main__':unittest.main()
