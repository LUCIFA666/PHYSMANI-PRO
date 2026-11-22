"""Control waypoints cannot overwrite public object observations or gates."""
import unittest
import numpy as np
from our_mani.perception_repair_20261111_v1.coordinator_v3 import ControlledApproachReference
from tests.test_highspeed_public_motion_20261111_v1 import forecast


class ControlledDescentContracts(unittest.TestCase):
    def test_speed_matching_holds_height_and_matched_descent_is_slow_and_bounded(self):
        p=np.array([.2,0.,.775]);v=np.array([0.,.8,0.]);f=forecast(0.,p,v);before=f.positions.copy()
        hold=ControlledApproachReference(f,0.,.07,[0,0,1],0.)
        descend=ControlledApproachReference(f,0.,.07,[0,0,1],.07)
        ph,vh,_=hold.at('target',.2);pd,vd,_=descend.at('target',.2)
        self.assertAlmostEqual(ph.position[2],.845);self.assertAlmostEqual(pd.position[2],.831)
        self.assertAlmostEqual(vh[2],0.);self.assertAlmostEqual(vd[2],-.07)
        ending=ControlledApproachReference(f,0.,.01,[0,0,1],.07)
        p_end,v_end,_=ending.at('target',.5)
        self.assertGreaterEqual(p_end.position[2],.775);self.assertEqual(v_end[2],0.)
        np.testing.assert_array_equal(f.positions,before)
        self.assertFalse(descend.metadata['is_object_motion_forecast'])
        self.assertFalse(descend.metadata['used_as_NN_observation'])


if __name__=='__main__':unittest.main()
