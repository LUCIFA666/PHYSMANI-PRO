"""Carry uses public rigid offset and original orientation, not free GRU."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.highspeed_control_20261111_v1.coordinator_v3 import carry_reference,ControlledPoseReference


class CarryReferenceContracts(unittest.TestCase):
    def test_world_tcp_offset_is_preserved_at_new_object_goal(self):
        obj=Pose([.2,-.1,.8],Rotation.from_rotvec([.2,.1,-.3]).as_matrix())
        tcp=np.array([.21,-.12,.79]);goal=np.array([.3,.2,1.])
        quat=Rotation.from_rotvec([3.1,.01,.02]).as_quat()
        reference,candidate=carry_reference(tcp,obj,goal,quat)
        pose,v,w=reference.at('target',5.)
        np.testing.assert_allclose(pose.apply(candidate['object_tcp_position']),goal+tcp-obj.position,atol=1e-14)
        R=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
        np.testing.assert_allclose(R,Rotation.from_quat(quat).as_matrix(),atol=1e-14)
        np.testing.assert_array_equal(v,np.zeros(3));np.testing.assert_array_equal(w,np.zeros(3))
        self.assertIsInstance(reference,ControlledPoseReference)


if __name__=='__main__':unittest.main()
