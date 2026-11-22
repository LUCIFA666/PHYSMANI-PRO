import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_capture_repair_20261112_v1.precision import refine

class PosePrecisionTests(unittest.TestCase):
    def test_refinement_removes_pose_residual_without_changing_world_goal(self):
        def fk(q):
            result=np.eye(4);result[:3,3]=q[:3]
            result[:3,:3]=Rotation.from_rotvec(q[3:6]).as_matrix()
            return result
        target=np.array([.2,-.1,.8]);rotation=Rotation.from_euler('xyz',[.2,-.3,.4]);quat=rotation.as_quat()
        seed=np.r_[target+.001,rotation.as_rotvec()+.002,0.]
        original_target=target.copy();original_quat=quat.copy()
        q,r=refine(fk,seed,target,quat,np.full(7,-2.),np.full(7,2.))
        self.assertGreater(r['before_position_m'],.001)
        self.assertLess(r['after_position_m'],1e-6)
        self.assertLess(r['after_rotation_rad'],1e-6)
        self.assertTrue(np.all(abs(q)<=2.))
        np.testing.assert_array_equal(target,original_target)
        np.testing.assert_array_equal(quat,original_quat)

if __name__=='__main__':unittest.main()
