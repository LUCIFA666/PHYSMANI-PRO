from types import SimpleNamespace
import unittest

import numpy as np
import torch

from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics, SCHEMA, export_native_kinematic_snapshot, twist_exp
from our_mani.local_mpc_p2_20261103.torch_fk import TorchPoEKinematics


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def planar(q):
    pose = np.eye(4)
    pose[:3, :3] = rz(q[0]+q[1])
    pose[:3, 3] = rz(q[0]) @ [.7, 0., 0.] + rz(q[0]+q[1]) @ [.4, 0., 0.]
    return pose


def planar_snapshot():
    reference = np.array([.3, -.5])
    origin = rz(reference[0]) @ [.7, 0., 0.]
    axes = np.array([[0., 0., 1.], [0., 0., 1.]])
    origins = np.array([[0., 0., 0.], origin])
    return dict(schema=SCHEMA, q_ref=reference,
                space_screws_vw=np.c_[-np.cross(axes, origins), axes],
                tcp_reference_matrix=planar(reference), twist_order="v_omega")


def serial(q):
    value, frames = np.eye(4), []
    for i, angle in enumerate(q):
        fixed = np.eye(4)
        if i % 2:
            fixed[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        value = value @ fixed
        frames.append(value.copy())
        rotation, translation = np.eye(4), np.eye(4)
        rotation[:3, :3], translation[0, 3] = rz(angle), .1 + .01*i
        value = value @ rotation @ translation
    tool = np.eye(4); tool[:3, 3] = [.02, -.01, .09]
    return frames, value @ tool


class GetterOnlyArm:
    def __init__(self):
        self.q = np.array([.3, -.4, .2, .1, -.2, .5, -.1])
        self.dq, self.servo = np.ones(7)*.01, self.q+.02
        self.clock = 0
        self.leak = False
        self.joints = [SimpleNamespace(get_joint_type=lambda: SimpleNamespace(name="REVOLUTE"),
                          get_matrix=lambda i=i: serial(self.q)[0][i]) for i in range(7)]
        self._ik_target = SimpleNamespace(get_pose=lambda: [0., 0., 0., 0., 0., 0., 1.])

    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.dq.copy()
    def get_joint_target_positions(self): return self.servo.copy()
    def get_tip(self):
        def matrix():
            if self.leak: self.clock += 1
            return serial(self.q)[1]
        return SimpleNamespace(get_matrix=matrix)
    def set_joint_positions(self, *args, **kwargs): raise AssertionError("setter prohibited")
    def get_jacobian(self): raise AssertionError("Native Jacobian prohibited")


class PoETests(unittest.TestCase):
    def test_arbitrary_nonzero_reference_and_batch_matches_direct_two_link_fk(self):
        model = PoEKinematics(planar_snapshot())
        q = np.random.default_rng(3).uniform(-2, 2, (4, 9, 2))
        actual = model(q)
        expected = np.stack([planar(row) for row in q.reshape(-1, 2)]).reshape(4, 9, 4, 4)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(model(model.q_ref), planar(model.q_ref), atol=1e-12)

    def test_world_geometric_jacobian_is_tcp_linear_velocity_not_space_v(self):
        model = PoEKinematics(planar_snapshot())
        q = np.array([.6, -.2]); geometric = model.geometric_jacobian(q)
        for i in range(2):
            shifted = q.copy(); shifted[i] += 1e-6
            finite = (model(shifted)[:3, 3]-model(q)[:3, 3])/1e-6
            np.testing.assert_allclose(geometric[:3, i], finite, atol=6e-7)
        self.assertGreater(np.linalg.norm(geometric[:3]-model.space_jacobian(q)[:3]), .1)

    def test_pure_getter_native_export_matches_independent_spatial_seven_joint_chain(self):
        arm = GetterOnlyArm()
        snapshot = export_native_kinematic_snapshot(arm, clock=lambda: (arm.clock, 0., 0))
        self.assertTrue(snapshot['getters_only'])
        self.assertFalse(snapshot['setters_called'])
        self.assertFalse(snapshot['Native_FK_validation_passed'])
        model = PoEKinematics(snapshot)
        for q in np.random.default_rng(4).uniform(-1, 1, (10, 7)):
            np.testing.assert_allclose(model(q), serial(q)[1], atol=1e-12)
        arm.leak = True
        with self.assertRaises(RuntimeError):
            export_native_kinematic_snapshot(arm, clock=lambda: (arm.clock, 0., 0))

    def test_torch_batches_match_numpy_and_have_finite_zero_and_nonzero_joint_gradients(self):
        snapshot = planar_snapshot()
        numpy_model, model = PoEKinematics(snapshot), TorchPoEKinematics(snapshot)
        q = torch.tensor([[.3, -.5], [.8, -.4]], dtype=torch.float64, requires_grad=True)
        result = model(q)
        np.testing.assert_allclose(result.detach(), numpy_model(q.detach()), atol=1e-12)
        result[..., :3, 3].square().sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()), 0.)
        self.assertTrue(torch.autograd.gradcheck(model, (q.detach().requires_grad_(),), eps=1e-6, atol=1e-5))

    def test_invalid_rotations_and_shapes_rejected(self):
        snapshot = planar_snapshot(); snapshot['tcp_reference_matrix'] = np.zeros((4, 4))
        with self.assertRaises(ValueError): PoEKinematics(snapshot)
        with self.assertRaises(ValueError): PoEKinematics(planar_snapshot())([0, 1, 2])
        with self.assertRaises(ValueError): twist_exp([0, 0, 0, 0, 0, 2], .1)


if __name__ == '__main__': unittest.main()
