import unittest
import numpy as np
from our_mani.multitask.geometry_bounds_v223 import box_corners,world_bounds


class Tests(unittest.TestCase):
    def test_planar_workspace_retains_exact_zero_thickness(self):
        p=box_corners([-.5,.5,-.5,.5,0.,0.]);m=np.eye(4);m[2,3]=.75
        r=world_bounds(p,m)
        self.assertEqual(r['min'],[-.5,-.5,.75]);self.assertEqual(r['max'],[.5,.5,.75])
        self.assertEqual(len(np.unique(p,axis=0)),4)

    def test_rotated_box_uses_full_original_transform(self):
        p=box_corners([-.01,.01,-.02,.02,-.03,.03]);m=np.eye(4)
        m[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];m[:3,3]=[.2,.3,.75]
        r=world_bounds(p,m);np.testing.assert_allclose(r['min'],[.17,.28,.74]);np.testing.assert_allclose(r['max'],[.23,.32,.76])

    def test_unordered_or_nonfinite_bounds_still_rejected(self):
        for b in ([1,0,0,1,0,1],[0,1,0,1,0,float('nan')]):
            with self.assertRaises(ValueError):box_corners(b)


if __name__=='__main__':unittest.main()
