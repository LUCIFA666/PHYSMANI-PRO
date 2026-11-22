import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_control_lead_20261113_v9.lead import lead_pose
from our_mani.executor_control_lead_20261113_v9.feedback import desired_velocity

class LeadTest(unittest.TestCase):
    def test_world_rigid_motion_lead_preserves_the_input_and_moves_along_the_circle(self):
        p=np.array([1.,0,0]);q=[0,0,0,1];v=[0,1,0,0,0,1]
        after,rotation,velocity=lead_pose(p,q,v,.05)
        np.testing.assert_allclose(after,[np.cos(.05),np.sin(.05),0],atol=1e-10)
        np.testing.assert_allclose(velocity[:3],[-np.sin(.05),np.cos(.05),0],atol=1e-10)
        np.testing.assert_allclose(p,[1,0,0]);self.assertAlmostEqual(Rotation.from_quat(rotation).magnitude(),.05)

    def test_future_control_rotation_does_not_replace_current_alignment_guard(self):
        class FK:
            def __call__(self,q):
                m=np.eye(4);m[:3,3]=q;return m
            def geometric_jacobian(self,q):return np.vstack([np.eye(3),np.zeros((3,3))])
        fk=FK();state=SimpleNamespace(position=np.zeros(3));limits=SimpleNamespace(lower=-np.ones(3),upper=np.ones(3))
        rotation=Rotation.from_rotvec([0,0,.05]).as_quat()
        _,feedback=desired_velocity(fk,np.zeros(3),state,[0,0,-.1],rotation,np.zeros(6),limits,
            mode='compliant_insert',near_contact=True,alignment_reference=([0,0,-.1],[0,0,0,1]))
        self.assertLess(feedback['desired_Cartesian_twist'][2],0.)
        self.assertAlmostEqual(feedback['current_alignment_rotation_error_rad'],0.)

if __name__=='__main__':unittest.main()
