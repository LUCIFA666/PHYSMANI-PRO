import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_grasp_frame_20261113_v1.grasp_frame import matrix,capture_delta,apply_delta
from our_mani.executor_grasp_frame_20261113_v1.nominal_robot import Robot
from types import SimpleNamespace
from our_mani.executor_grasp_frame_20261113_v1.phases import LatchedHeldContact

class GraspFrameTests(unittest.TestCase):
    def test_contact_stroke_is_not_used_as_the_nominal_grasp_frame(self):
        contact=np.array([.2,.1,.78]);actual=np.array([.2,.1,.81,0,0,0,1.])
        delta=capture_delta(contact+[0,0,.02],[0,0,0,1],actual)
        self.assertAlmostEqual(delta[2,3],.01)
    def test_contact_stage_survives_rotating_fixture_height_changes(self):
        phase=LatchedHeldContact()
        self.assertFalse(phase.observe(held=True,goal_z=1.))
        self.assertTrue(phase.observe(held=True,goal_z=.94))
        self.assertTrue(phase.observe(held=True,goal_z=.99))
        self.assertFalse(phase.observe(held=False,goal_z=.99))

    def test_oracle_progress_view_uses_nominal_frame_without_moving_real_robot(self):
        nominal=matrix([.2,.3,.9],[0,1,0,0]);delta=matrix([.02,0,0],Rotation.from_euler('z',.2).as_quat())
        actual=nominal@delta
        tip=SimpleNamespace(get_matrix=lambda:actual.copy())
        native=SimpleNamespace(arm=SimpleNamespace(get_tip=lambda:tip),gripper=object())
        viewed=Robot(native,lambda:delta)
        np.testing.assert_allclose(viewed.arm.get_tip().get_matrix(),nominal,atol=1e-12)
        np.testing.assert_allclose(native.arm.get_tip().get_matrix(),actual,atol=1e-12)

    def test_misaligned_capture_preserves_desired_object_placement(self):
        object_now=matrix([.2,.3,.8],Rotation.from_euler('z',.7).as_quat())
        nominal_grasp=matrix([0,0,-.03],Rotation.from_euler('x',np.pi).as_quat())
        mismatch=matrix([.013,0,0],Rotation.from_euler('y',.35).as_quat())
        actual_grasp=nominal_grasp@mismatch
        nominal_now=object_now@nominal_grasp;actual_now=object_now@actual_grasp
        pose=np.r_[actual_now[:3,3],Rotation.from_matrix(actual_now[:3,:3]).as_quat()]
        delta=capture_delta(nominal_now[:3,3],Rotation.from_matrix(nominal_now[:3,:3]).as_quat(),pose)
        object_target=matrix([.3,-.2,1.],Rotation.from_euler('z',1.2).as_quat())
        nominal_target=object_target@nominal_grasp
        p,q,_=apply_delta(nominal_target[:3,3],Rotation.from_matrix(nominal_target[:3,:3]).as_quat(),np.zeros(6),delta)
        np.testing.assert_allclose(matrix(p,q),object_target@actual_grasp,atol=1e-12)

    def test_velocity_compensation_includes_rotating_grasp_offset(self):
        delta=matrix([.02,0,0],[0,0,0,1])
        p,q,velocity=apply_delta([0,0,1],[0,0,0,1],[0,0,0,0,0,2.],delta)
        np.testing.assert_allclose(velocity,[0,.04,0,0,0,2.],atol=1e-12)

if __name__=='__main__':unittest.main()
