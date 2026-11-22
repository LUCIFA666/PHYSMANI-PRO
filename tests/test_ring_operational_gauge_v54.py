import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.ring_operational_gauge_v54 import choose


class Tests(unittest.TestCase):
    def group(self):
        z=[Rotation.from_euler('z',i*90,degrees=True).as_matrix() for i in range(4)]
        flip=Rotation.from_euler('x',180,degrees=True).as_matrix()
        return np.asarray(z+[r@flip for r in z])

    def test_action_representative_keeps_world_symmetry_center_and_can_avoid_upside_down_TCP(self):
        pose=Pose(np.array([.2,0.,.8]));center=np.array([.01,.02,0.]);tcp=Pose(np.zeros(3),self.group()[4])
        selected,row=choose(pose,self.group(),center,tcp=tcp,local_grasp_rotation=np.eye(3))
        np.testing.assert_allclose(selected.position+selected.rotation@center,pose.position+pose.rotation@center)
        self.assertLess(row['orientation_travel_rad'],1e-12);self.assertFalse(row['full_orientation_promoted'])

    def test_later_frames_choose_previous_public_branch_instead_of_chasing_robot_orientation(self):
        pose=Pose(np.array([.2,0.,.8]),self.group()[4]);previous=Pose(np.array([.2,0.,.8]))
        selected,row=choose(pose,self.group(),np.zeros(3),previous=previous,tcp=pose,local_grasp_rotation=np.eye(3))
        np.testing.assert_allclose(selected.rotation,np.eye(3),atol=1e-12);self.assertTrue(row['previous_pose_used'])


if __name__=='__main__':unittest.main()
