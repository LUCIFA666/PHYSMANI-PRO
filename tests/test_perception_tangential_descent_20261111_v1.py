"""Descent synchronization ignores its intended vertical motion, not capture gates."""
import unittest
import numpy as np
from tests.test_highspeed_tracking_20261111_v1 import setup
from our_mani.highspeed_control_20261111_v1.hybrid_tracking import HybridTrackingConfig
from our_mani.perception_repair_20261111_v1.coordinator_v4 import TangentialMatchedApproachPlanner


class TangentialDescentContracts(unittest.TestCase):
    def test_planned_vertical_motion_does_not_cancel_itself(self):
        p,s,c,f=setup();c['pregrasp_offset']=.07;c['required_opening']=1.;c['allowed_relative_speed']=.14
        s.update(held=False,opening=1.,phase_state=dict(phase='align',metrics=dict(angle=0.,relative_speed=.16)),
            tcp_velocity=np.array([.2,0.,-.12]),object_linear_velocity=np.array([.2,0.,0.]))
        planner=TangentialMatchedApproachPlanner(p.kinematics,p.joint_response,p.limits,config=HybridTrackingConfig(maximum_tcp_speed=1.1))
        r=planner.plan(s,c,f,pregrasp=False)
        self.assertTrue(r['measured_velocity_matched_for_descent']);self.assertEqual(r['desired_descent_speed_mps'],.12)
        self.assertTrue(r['original_capture_gates_unchanged']);self.assertEqual(c['allowed_relative_speed'],.14)
        s['tcp_velocity']=np.array([.5,0.,-.12])
        r=planner.plan(s,c,f,pregrasp=False)
        self.assertFalse(r['measured_velocity_matched_for_descent']);self.assertEqual(r['desired_descent_speed_mps'],0.)


if __name__=='__main__':unittest.main()
