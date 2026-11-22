import unittest
import numpy as np
from our_mani.executor_grasp_frame_20261113_v1.recovery import joint_recovery

class RecoveryTests(unittest.TestCase):
    def test_outside_measurement_is_retained_and_reference_goal_moves_inside(self):
        actual=np.array([1.001,0.]);before=actual.copy()
        target,receipt=joint_recovery([.999,0.],actual,[-1,-1],[1,1])
        np.testing.assert_array_equal(actual,before)
        np.testing.assert_allclose(target,[.95,0.])
        self.assertTrue(receipt['active']);self.assertFalse(receipt['actual_state_clipped'])
        self.assertEqual(receipt['violating_joints'],[0])
