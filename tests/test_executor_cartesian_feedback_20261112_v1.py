import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.executor_contract_repair_20261112_v2.reference import ReferenceState
from our_mani.executor_cartesian_tracking_20261112_v1.reference import step
from our_mani.executor_cartesian_tracking_20261112_v1.feedback import desired_velocity

class LinearRobot:
    def __call__(self,q):
        transform=np.eye(4);transform[:3,3]=q[:3]
        transform[:3,:3]=Rotation.from_rotvec(np.array(q[3:6],copy=True)).as_matrix()
        return transform
    def geometric_jacobian(self,q):return np.c_[np.eye(6),np.zeros(6)]

class CartesianFeedbackTests(unittest.TestCase):
    def test_insertion_waits_for_alignment_without_stopping_target_following(self):
        fk=LinearRobot();limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));actual=np.zeros(7)
        velocity,r=desired_velocity(fk,actual,state,[.01,0,-.04],[0,0,0,1],[.1,0,0,0,0,0],limits,
                                    mode='compliant_insert',near_contact=True)
        self.assertGreater(velocity[0],.1)
        self.assertAlmostEqual(velocity[2],0.,places=10)
        self.assertEqual(r['vertical_correction_speed_cap_m_s'],.025)

    def test_contact_bias_mode_bounds_reference_windup_against_blocked_motion(self):
        fk=LinearRobot();limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));actual=np.zeros(7)
        for _ in range(160):
            velocity,_=desired_velocity(fk,actual,state,[.01,0,0],[0,0,0,1],np.zeros(6),limits,mode='bounded_bias')
            state,_=step(state,velocity,limits,actual_q=actual)
        self.assertLessEqual(state.position[0],.01201)

    def test_actual_feedback_removes_constant_servo_bias(self):
        fk=LinearRobot();limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));bias=np.array([.001,0,0,0,0,0,0]);actual=np.zeros(7)
        goal=np.array([.2,.1,.8])
        for _ in range(160):
            velocity,_=desired_velocity(fk,actual,state,goal,[0,0,0,1],np.zeros(6),limits)
            state,_=step(state,velocity,limits,actual_q=actual)
            actual=state.position-bias
        self.assertLess(np.linalg.norm(actual[:3]-goal),1e-5)
        self.assertGreater(abs(state.position[0]-goal[0]),.0009)

    def test_moving_target_feedforward_tracks_without_destination_settling_wait(self):
        fk=LinearRobot();limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));actual=np.zeros(7)
        for i in range(160):
            goal=np.array([.1*i*.05,0.,0.])
            velocity,_=desired_velocity(fk,actual,state,goal,[0,0,0,1],[.1,0,0,0,0,0],limits)
            old=state;state,_=step(state,velocity,limits,actual_q=actual);actual=state.position.copy()
            self.assertLessEqual(np.linalg.norm(state.velocity),3.+1e-7)
            self.assertLessEqual(np.linalg.norm(state.acceleration),12.+1e-7)
            np.testing.assert_allclose((state.position-old.position)/.05,state.velocity,atol=1e-10)
        self.assertLess(abs(actual[0]-(goal[0]+.005)),1e-5)

if __name__=='__main__':unittest.main()
