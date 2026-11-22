import unittest
import numpy as np
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.executor_contract_repair_20261112_v4.reference import step,ReferenceState,DT

LIMITS=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)

class FeedbackGovernorTests(unittest.TestCase):
    def test_coupled_motion_limits_apply_to_whole_joint_vector(self):
        s=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));actual=np.zeros(7)
        for i in range(80):
            goal=np.full(7,1. if i//10%2==0 else -1.)
            old=s;s,receipt=step(s,goal,LIMITS,actual_q=actual)
            self.assertLessEqual(np.linalg.norm(s.velocity),3.+1e-6)
            self.assertLessEqual(np.linalg.norm(s.acceleration),12.+1e-6)
            self.assertLessEqual(np.max(abs(s.acceleration)),12.+1e-8)
            np.testing.assert_allclose((s.position-old.position)/DT,s.velocity,atol=1e-10)
            actual+=.3*(s.position-actual)

    def test_measured_state_feedback_brakes_instead_of_advancing_stale_goal(self):
        s=ReferenceState(np.ones(7)*.2,np.ones(7)*.2,np.zeros(7))
        _,receipt=step(s,np.ones(7),LIMITS,actual_q=np.zeros(7))
        self.assertEqual(receipt['tracking_progress_scale'],0.)

    def test_joint_bound_is_approached_with_a_braking_reserve(self):
        p=np.zeros(7);p[0]=2.9;s=ReferenceState(p,np.zeros(7),np.zeros(7));goal=p.copy();goal[0]=3.
        for _ in range(80):
            before=s;s,_=step(s,goal,LIMITS,actual_q=s.position)
            self.assertLessEqual(s.velocity[0],2*(3.-before.position[0])+1e-8)
            self.assertLessEqual(s.position[0],3.)
        self.assertGreater(s.position[0],2.999)

    def test_tabletop_barrier_preserves_clearance_while_target_lies_beyond_surface(self):
        initial=np.zeros(7);initial[0]=.5;s=ReferenceState(initial,np.zeros(7),np.zeros(7));actual=initial.copy()
        goal=np.zeros(7);goal[0]=-1.;A=np.zeros((1,7));A[0,0]=1.
        for _ in range(160):
            minimum=.95*actual[0]+.05*.1
            b=np.array([(minimum-actual[0]-(s.position[0]-actual[0]+s.velocity[0]*DT))/DT**2])
            s,receipt=step(s,goal,LIMITS,actual_q=actual,barrier_A=A,barrier_b=b)
            self.assertGreaterEqual(s.position[0],.1-1e-7)
            self.assertGreaterEqual(receipt['minimum_barrier_residual'],-1e-6)
            actual+=.6*(s.position-actual)

if __name__=='__main__':unittest.main()
