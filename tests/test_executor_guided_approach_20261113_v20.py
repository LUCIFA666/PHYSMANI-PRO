import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from our_mani.executor_guided_approach_20261113_v20.approach import GuidedApproach
from our_mani.executor_guided_approach_20261113_v20.pose_executor import PoseGoalExecutor,ContactExecutor

class GuidedApproachTest(unittest.TestCase):
    def test_switch_to_descent_has_no_twenty_centimetre_waypoint_jump(self):
        a=GuidedApproach();goal=[.3,.2,.8];actual=[.3,.2,1.,0,0,0,1];q=[0,0,0,1]
        rows=[a.sample(goal,q,actual,t,held=False)[0]for t in [0.,.05,.1,.15,.2,.25]]
        self.assertEqual(a.phase,'descend')
        self.assertTrue(all(abs(x[2]-y[2])<=.004000001 for x,y in zip(rows,rows[1:])))
        self.assertGreater(rows[-1][2],.98)

    def test_repeated_queries_do_not_advance_descent(self):
        a=GuidedApproach();args=([.3,.2,.8],[0,0,0,1],[.3,.2,1.,0,0,0,1])
        for t in [0.,.05,.1,.15]:a.sample(*args,t,held=False)
        first=a.sample(*args,.2,held=False)[0];second=a.sample(*args,.2,held=False)[0]
        np.testing.assert_array_equal(first,second)

    def test_realigning_does_not_jump_up_twenty_centimetres(self):
        a=GuidedApproach();q=[0,0,0,1];goal=[.3,.2,.8]
        for t in np.arange(0.,1.,.05):a.sample(goal,q,[.3,.2,1.,*q],float(t),held=False)
        before=a.commanded_z
        after=a.sample(goal,q,[.4,.2,.85,*q],1.,held=False)[0]
        self.assertLessEqual(abs(after[2]-before),.004000001)
        self.assertEqual(a.phase,'hover')

    def test_held_transport_retains_its_terminal_goal(self):
        a=GuidedApproach();p=np.array([.3,.2,.8]);out=a.sample(p,[0,0,0,1],[.3,.2,1.,0,0,0,1],0.,held=True)
        np.testing.assert_array_equal(out[0],p);self.assertFalse(out[1]['applied'])

    def test_posture_lookahead_stops_at_the_contact_endpoint(self):
        driver=PoseGoalExecutor.__new__(PoseGoalExecutor);driver.guided_approach=True
        driver.backend=SimpleNamespace(gripper=SimpleNamespace(get_grasped_objects=lambda:[]))
        driver.approach=SimpleNamespace(vertical_velocity=-.08,active_target_z=.8)
        driver.executor=SimpleNamespace(config=SimpleNamespace(reference_frequency=6.))
        with patch.object(ContactExecutor,'_sample_target',return_value=(np.array([.3,.2,.804]),[0,0,0,1],np.zeros(6),{})):
            p,q,v,m=driver._sample_target()
        self.assertAlmostEqual(p[2]+(1./6.+.025)*v[2],.8)

if __name__=='__main__':unittest.main()
