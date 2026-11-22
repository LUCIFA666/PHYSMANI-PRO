import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.placement_clearance_v50 import vertical_shift


class Tests(unittest.TestCase):
    def test_entire_rotated_held_mesh_clears_entire_bin_without_changing_XY_or_rotation(self):
        bin_pose=Pose(np.array([0.,0.,.85]));bin_vertices=np.array([[-.1,-.1,-.1],[.1,.1,.13]])
        tcp=Pose(np.array([0.,0.,.99]));local=Pose(np.array([-.04,0.,-.01]));held=np.array([[-.05,-.02,-.015],[.05,.02,.015]])
        shift,row=vertical_shift(bin_pose,bin_vertices,tcp,local,held)
        self.assertGreater(shift,0.);bottom=(local.apply(held)@tcp.rotation.T+tcp.position+np.array([0,0,shift]))[:,2].min()
        self.assertGreater(bottom,bin_pose.apply(bin_vertices)[:,2].max())
        np.testing.assert_array_equal(tcp.position,[0,0,.99]);self.assertAlmostEqual(shift,.015+row['numerical_margin_m'])

    def test_already_clear_TRAIN_intention_keeps_height(self):
        p=Pose(np.array([0,0,1.1]));shift,_=vertical_shift(Pose(np.zeros(3)),np.array([[0,0,.8],[0,0,.9]]),p,Pose(np.zeros(3)),np.array([[0,0,-.01],[0,0,.01]]))
        self.assertEqual(shift,0.)


if __name__=='__main__':unittest.main()
