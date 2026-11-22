"""An executor cannot silently shorten the geometry used by the phase gates."""
import unittest
import numpy as np
from tests.test_highspeed_tracking_20261111_v1 import setup
from our_mani.highspeed_control_20261111_v1.hybrid_tracking import HybridTrackingConfig
from our_mani.perception_repair_20261111_v1.coordinator_v2 import CandidateConsistentTrackingPlanner


class PregraspGeometryContracts(unittest.TestCase):
    def test_reference_preserves_candidate_offset_for_both_phases(self):
        p,s,c,f=setup();c['pregrasp_offset']=.07
        s['tcp_position']=s['tcp_position']+np.array([.4,0,0])
        planner=CandidateConsistentTrackingPlanner(p.kinematics,p.joint_response,p.limits,
            config=HybridTrackingConfig(local_mpc_radius=.001,execution_pregrasp_offset=.04))
        for pregrasp in (True,False):
            result=planner.plan(s,c,f,pregrasp=pregrasp)
            pose,_,_=f.at('target',s['time'])
            local=c['object_tcp_position']+(c['approach_direction']*.07 if pregrasp else 0.)
            np.testing.assert_allclose(result['target_position'],pose.apply(local),atol=1e-12)
            self.assertEqual(result['reference_pregrasp_offset'],.07)
            self.assertTrue(result['pregrasp_reference_matches_phase_candidate'])
        self.assertEqual(c['pregrasp_offset'],.07)


if __name__=='__main__':unittest.main()
