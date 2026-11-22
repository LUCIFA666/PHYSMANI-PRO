import unittest
from our_mani.executor_release_settle_20261113_v18.settle import VerticalReleaseSettle

class ReleaseSettleTest(unittest.TestCase):
    def test_real_time_and_both_velocities_must_settle(self):
        s=VerticalReleaseSettle();s.start(.9)
        self.assertFalse(s.qualify(0.,-.04,0.,0.))
        self.assertFalse(s.qualify(.1,0.,-.04,0.))
        self.assertFalse(s.qualify(.2,0.,0.,.01))
        self.assertFalse(s.qualify(.3,0.,0.,0.))
        self.assertFalse(s.qualify(.3,0.,0.,0.))
        self.assertTrue(s.qualify(.4,0.,0.,0.))

    def test_a_new_disturbance_restarts_the_stability_interval(self):
        s=VerticalReleaseSettle();s.start(.9);s.qualify(0.,0.,0.,0.)
        s.qualify(.09,.02,0.,0.)
        self.assertFalse(s.qualify(.1,0.,0.,0.));self.assertTrue(s.qualify(.2,0.,0.,0.))

    def test_nonfinite_and_inactive_feedback_cannot_qualify(self):
        s=VerticalReleaseSettle();self.assertFalse(s.qualify(1.,0.,0.,0.))
        s.start(.9);self.assertFalse(s.qualify(2.,float('nan'),0.,0.))
        s.reset();self.assertFalse(s.qualify(3.,0.,0.,0.))

if __name__=='__main__':unittest.main()
