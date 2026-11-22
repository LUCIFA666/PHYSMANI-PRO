import unittest
import numpy as np
from our_mani.multitask.block_prototype_diagnostic_v215 import box_corners,world_bounds


class Tests(unittest.TestCase):
    def test_original_box_bounds_transform_uses_full_rotation_and_translation(self):
        box=box_corners([-.01,.01,-.02,.02,-.03,.03])
        m=np.eye(4);m[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];m[:3,3]=[.2,.3,.75]
        result=world_bounds(box,m)
        np.testing.assert_allclose(result['min'],[.17,.28,.74])
        np.testing.assert_allclose(result['max'],[.23,.32,.76])

    def test_degenerate_or_nonfinite_prototype_diagnostic_is_rejected(self):
        for bounds in ([0,0,0,1,0,1],[0,1,0,1,0,float('nan')]):
            with self.assertRaises(ValueError):box_corners(bounds)

    def test_geometry_measurement_does_not_mutate_original_arrays(self):
        points=np.array([[0.,0.,0.],[1.,2.,3.]]);matrix=np.eye(4);matrix[:3,3]=[1,2,3]
        before=points.copy();m=matrix.copy();result=world_bounds(points,matrix)
        np.testing.assert_array_equal(points,before);np.testing.assert_array_equal(matrix,m)
        self.assertEqual(result['max'],[2.,4.,6.])


if __name__=='__main__':unittest.main()
