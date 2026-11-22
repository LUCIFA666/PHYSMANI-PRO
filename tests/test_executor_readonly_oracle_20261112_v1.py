import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_task_goals_20261112_v1.oracle import future_pose

class ReadOnlyOracleTests(unittest.TestCase):
    def test_future_rotation_preserves_height_and_rotates_about_frame_origin(self):
        base=np.eye(4);base[:3,3]=[.2,.3,.8]
        waypoint=np.eye(4);waypoint[:3,3]=[.4,.3,.9]
        p=future_pose(SimpleNamespace(get_matrix=lambda:base),SimpleNamespace(get_matrix=lambda:waypoint),1.,90.)
        np.testing.assert_allclose(p[:3],[.2,.5,.9],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(p[3:]).apply([1.,0.,0.]),[0.,1.,0.],atol=1e-12)
        np.testing.assert_array_equal(base[:3,3],[.2,.3,.8])

if __name__=='__main__':unittest.main()
