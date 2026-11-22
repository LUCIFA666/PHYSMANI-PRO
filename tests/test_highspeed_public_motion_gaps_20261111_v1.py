"""A missing image cannot become a new measured turn or velocity."""
import unittest
import numpy as np
from our_mani.highspeed_control_20261111_v1.public_motion_v2 import GapAwarePublicReflectionObserver
from tests.test_highspeed_public_motion_20261111_v1 import forecast


class PublicMotionGapContracts(unittest.TestCase):
    def test_gap_keeps_observed_extrema_without_fabricating_a_turn(self):
        p=np.array([.2,0.,.8]);v=np.array([.525,0.,0.]);rows=[]
        for i in range(80):
            rows.append((p.copy(),v.copy()));p=p+v*.05
            if p[0]>=.35 or p[0]<=.05:v[0]=-v[0]
        observer=GapAwarePublicReflectionObserver()
        for i in range(45):observer.observe(i*.05,rows[i][0])
        count=len(observer.events[0]);observer.observe(46*.05,rows[46][0])
        self.assertEqual(len(observer.events[0]),count)
        f=forecast(46*.05,*rows[46]);out,e=observer.fuse(f)
        self.assertFalse(e)  # Direction is unknown immediately after a gap.
        np.testing.assert_array_equal(out.positions,f.positions)
        observer.observe(47*.05,rows[47][0]);f=forecast(47*.05,*rows[47]);out,e=observer.fuse(f)
        self.assertTrue(e);self.assertFalse(e[0]['gap_interpolation'])
        np.testing.assert_allclose(out.positions[:5,0,0],[rows[i][0][0] for i in range(47,52)],atol=1e-12)


if __name__=='__main__':unittest.main()
