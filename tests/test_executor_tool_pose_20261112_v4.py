import unittest
import numpy as np
from our_mani.executor_contract_repair_20261112_v4.tool_pose import housing_clearance


class ToolPoseTests(unittest.TestCase):
    def check(self,height,quaternion=(0,0,0,1)):
        tcp=np.eye(4);housing=np.eye(4);housing[2,3]=.09
        vertices=np.array([[x,y,z]for x in (-.03,.03)for y in (-.1,.1)for z in (-.05,.05)])
        return housing_clearance((0,0,height),quaternion,tcp_reference=tcp,housing_reference=housing,
            vertices=vertices,table_matrix=np.eye(4),table_bounds=(-1,1,-1,1,0,.75),margin=.02)

    def test_tool_offset_is_accounted_for_without_IK_or_goal_correction(self):
        safe=self.check(.76);unsafe=self.check(.69)
        self.assertTrue(safe['admissible']);self.assertFalse(unsafe['admissible'])
        self.assertFalse(unsafe['Native_IK_called'])
        self.assertAlmostEqual(unsafe['tool_housing_clearance_m'],-.02)

    def test_same_tip_position_can_have_an_unsafe_housing_orientation(self):
        self.assertTrue(self.check(.76)['admissible'])
        self.assertFalse(self.check(.76,(1,0,0,0))['admissible'])

if __name__=='__main__':unittest.main()
