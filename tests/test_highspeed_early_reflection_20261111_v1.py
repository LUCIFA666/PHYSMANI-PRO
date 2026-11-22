"""Two opposing measured extrema permit an explicit reflection hypothesis."""
import unittest
import numpy as np
from our_mani.highspeed_control_20261111_v1.public_motion_v3 import EarlyPublicReflectionObserver
from tests.test_highspeed_public_motion_20261111_v1 import forecast


class EarlyReflectionContracts(unittest.TestCase):
    def test_two_measured_extrema_allow_preview_without_future_input(self):
        p=np.array([.2,0.,.8]);v=np.array([.525,0.,0.]);rows=[]
        for i in range(60):
            rows.append((p.copy(),v.copy()));p=p+v*.05
            if p[0]>=.35 or p[0]<=.05:v[0]=-v[0]
        o=EarlyPublicReflectionObserver()
        for i in range(26):o.observe(i*.05,rows[i][0])
        self.assertEqual(len(o.events[0]),2)
        out,e=o.fuse(forecast(25*.05,*rows[25]))
        self.assertTrue(e[0]['reflection_model_is_hypothesis'])
        np.testing.assert_allclose(out.positions[:5,0,0],[rows[i][0][0] for i in range(25,30)],atol=1e-12)


if __name__=='__main__':unittest.main()
