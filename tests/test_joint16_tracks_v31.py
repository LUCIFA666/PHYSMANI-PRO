import unittest
import numpy as np
from our_mani.multitask.joint16_tracks_v31 import block_arrays,precontact_end


class Tests(unittest.TestCase):
    def test_first_closed_gripper_masks_before_lift_without_using_grasp_stage_as_contact(self):
        self.assertEqual(precontact_end([b'reach',b'grasp',b'grasp',b'grasp',b'lift'],[1,1,1,0,0],5),3)
        self.assertEqual(precontact_end(['reach']*8,[1]*8,6),6)

    def test_adapter_has_no_observed_rotation_or_native_metric_label(self):
        a=dict(times=np.arange(8)*.05,positions=np.zeros((8,3)),velocities=np.ones((8,3)),confidence=np.ones(8),valid=np.ones(8,bool),truth_metric_only=np.full((8,3),99.))
        b=block_arrays(a,6)
        self.assertFalse(b['rotation_observable'].any());self.assertNotIn('truth_metric_only',b)
        self.assertEqual(b['positions'].shape,(6,1,3));np.testing.assert_array_equal(b['linear_velocities'][:,0],a['velocities'][:6])
        with self.assertRaises(ValueError):block_arrays(a,4)


if __name__=='__main__':unittest.main()
