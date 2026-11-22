import unittest
import numpy as np
from our_mani.executor_release_retreat_20261113_v17.retreat import ReleaseRetreat

class ReleaseRetreatTest(unittest.TestCase):
    def sample(self,helper,t,*,held=False,opening=1.,done=False,z=.9):
        return helper.sample([.3,.2,.8],[0,0,0,1],[.1,0,0,0,0,0],
            [.3,.2,z,0,0,0,1],t,held=held,opening=opening,done=done)

    def test_initial_open_gripper_does_not_start_a_release_retreat(self):
        r=self.sample(ReleaseRetreat(),0.)
        self.assertFalse(r[3]['active']);self.assertEqual(r[0][2],.8)

    def test_opening_retreat_never_returns_to_the_low_grasp_goal(self):
        helper=ReleaseRetreat();self.sample(helper,0.,held=True,opening=0.)
        self.sample(helper,.05)
        positions=[self.sample(helper,t)[0]for t in [.1,.2,.4,.7]]
        self.assertTrue(all(p[2]>=.9 for p in positions))
        self.assertTrue(all(a[2]<=b[2]for a,b in zip(positions,positions[1:])))
        self.assertAlmostEqual(positions[-1][2],.96)
        self.assertAlmostEqual(positions[-1][0],.3+.1*(.7-.05))

    def test_gripper_completion_alone_does_not_certify_clearance(self):
        helper=ReleaseRetreat();self.sample(helper,0.,held=True,opening=0.);self.sample(helper,.05)
        self.assertTrue(self.sample(helper,.5,done=True,z=.91)[3]['active'])
        self.assertTrue(self.sample(helper,.6,done=True,z=.96)[3]['completed'])

    def test_repeated_queries_at_one_timestamp_do_not_advance_the_waypoint(self):
        helper=ReleaseRetreat();self.sample(helper,0.,held=True,opening=0.);self.sample(helper,.05)
        np.testing.assert_allclose(self.sample(helper,.1)[0],self.sample(helper,.1)[0])

    def test_new_static_source_phase_does_not_erase_release_motion(self):
        helper=ReleaseRetreat();self.sample(helper,0.,held=True,opening=0.)
        result=helper.sample([0,0,1.2],[0,0,0,1],np.zeros(6),
            [.3,.2,.9,0,0,0,1],.05,held=False,opening=1.,done=False)
        self.assertAlmostEqual(result[2][0],.1)

if __name__=='__main__':unittest.main()
