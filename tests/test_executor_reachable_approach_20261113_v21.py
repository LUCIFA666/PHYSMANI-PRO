import unittest
from our_mani.executor_reachable_approach_20261113_v21.approach import GuidedApproach

class ReachableApproachTest(unittest.TestCase):
    def test_home_height_does_not_hold_an_unreachable_orientation_target(self):
        a=GuidedApproach();goal=[.36,.27,.94];actual=[.35,.26,1.47,0,0,0,1]
        a.sample(goal,[1,0,0,0],actual,0.,held=False)
        out=a.sample(goal,[1,0,0,0],actual,.5,held=False)
        self.assertEqual(a.phase,'hover')
        self.assertAlmostEqual(a.hover_height,1.14)
        self.assertAlmostEqual(out[0][2],1.43)

    def test_new_lower_hover_target_remains_rate_limited(self):
        a=GuidedApproach();actual=[.3,.2,1.47,0,0,0,1];q=[1,0,0,0]
        a.sample([.3,.2,1.1],q,actual,0.,held=False)
        before=a.sample([.3,.2,1.1],q,actual,.05,held=False)[0][2]
        after=a.sample([.3,.2,.8],q,actual,.1,held=False)[0][2]
        self.assertAlmostEqual(a.hover_height,1.0)
        self.assertLessEqual(abs(after-before),.004000001)

if __name__=='__main__':unittest.main()
