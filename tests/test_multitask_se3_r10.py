"""Mathematical and causal checks for the untrained R10 dynamics architecture."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from our_mani.multitask.context_se3 import (
    HISTORY_DIM, STATE_DIM, SE3ContextNetwork, finite_difference_twist, history_inputs, so3_exp)


class SE3DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exponential_agrees_with_independent_so3_reference_and_zero_gradient(self):
        values = np.array([[0., 0., 0.], [1e-9, -2e-9, 1e-9], [.4, -.7, 1.2], [3.1, .1, -.2]])
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        result = so3_exp(tensor)
        np.testing.assert_allclose(result.detach().numpy(), Rotation.from_rotvec(values).as_matrix(), atol=1e-12)
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_past_pose_twist_recovers_world_velocity_and_angular_rate(self):
        times = np.arange(6) * .05
        velocity = np.array([.3, -.7, .1])
        angular = np.array([.7, -.2, .3])
        initial = Rotation.from_rotvec([.2, .4, -.1]).as_matrix()
        rotations = Rotation.from_rotvec(times[:, None] * angular).as_matrix() @ initial
        measured_v, measured_w = finite_difference_twist(times, times[:, None] * velocity, rotations)
        np.testing.assert_allclose(measured_v, velocity, atol=1e-12)
        np.testing.assert_allclose(measured_w, angular, atol=1e-12)

    def test_zero_residual_is_exact_constant_world_twist_for_full_rollout(self):
        torch.manual_seed(19)
        model = SE3ContextNetwork().eval()
        position = torch.tensor([[.2, -.3, 1.1], [-.1, .2, .8]])
        rotation = torch.tensor(Rotation.from_rotvec([[.3, .2, -.1], [-.2, .1, .4]]).as_matrix(), dtype=torch.float32)
        velocity = torch.tensor([[.1, -.2, .3], [.3, .2, -.1]])
        angular = torch.tensor([[.4, -.5, .6], [.3, .1, -.2]])
        state = model.normalized(position, rotation, velocity, angular, torch.tensor([0, 2]))
        history = torch.zeros((2, 20, HISTORY_DIM)); history[:, :5, :STATE_DIM] = state[:, None]
        with torch.no_grad():
            output = model(history, torch.tensor([5, 5]), state, steps=22)
        times = np.arange(23) * .05
        expected_p = position.numpy()[:, None] + times[None, :, None] * velocity.numpy()[:, None]
        expected_r = np.stack([Rotation.from_rotvec(times[:, None] * w).as_matrix() @ r
                               for w, r in zip(angular.numpy(), rotation.numpy())])
        np.testing.assert_allclose(output['positions'], expected_p, atol=2e-6)
        np.testing.assert_allclose(output['rotations'], expected_r, atol=2e-6)
        products = output['rotations'].transpose(-1, -2) @ output['rotations']
        np.testing.assert_allclose(products, np.broadcast_to(np.eye(3), products.shape), atol=3e-6)

    def test_padding_and_other_body_do_not_change_a_context(self):
        torch.manual_seed(7)
        model = SE3ContextNetwork().eval()
        history = torch.randn((2, 20, HISTORY_DIM))
        lengths = torch.tensor([5, 12])
        with torch.no_grad():
            before = model.encode_context(history, lengths)
            altered = history.clone(); altered[0, 5:] = 1000.; altered[1] *= -20.
            after = model.encode_context(altered, lengths)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before[1], after[1]))
        self.assertTrue(torch.allclose(before.reshape(2, -1, 8).sum(-1), torch.ones((2, 2))))

    def test_mode_switch_or_missing_frame_cannot_be_silently_encoded(self):
        model = SE3ContextNetwork()
        rows = [dict(timestamp=i * .05, position=np.zeros(3), rotation=np.eye(3),
                     velocity=np.zeros(3), angular_velocity=np.zeros(3), confidence=1., mode=0)
                for i in range(5)]
        history_inputs(rows, model)
        rows[-1] = dict(rows[-1], mode=2)
        with self.assertRaisesRegex(ValueError, 'modes'):
            history_inputs(rows, model)
        rows[-1] = dict(rows[-1], mode=0, timestamp=.3)
        with self.assertRaisesRegex(ValueError, 'missing'):
            history_inputs(rows, model)

    def test_unobservable_rotation_cannot_learn_an_angular_residual(self):
        model = SE3ContextNetwork()
        with torch.no_grad():
            model.acceleration[-1].bias[3:] = 1.
        state = model.normalized(torch.zeros((1, 3)), torch.eye(3)[None],
            torch.zeros((1, 3)), torch.tensor([[0., .2, 0.]]), torch.tensor([0]), torch.tensor([False]))
        history = torch.zeros((1, 20, HISTORY_DIM)); history[:, :5, :STATE_DIM] = state[:, None]
        result = model(history, torch.tensor([5]), state, steps=22)
        torch.testing.assert_close(result['angular_velocities'], torch.tensor([[[0., .2, 0.]]]).expand(1, 23, 3))


if __name__ == '__main__':
    unittest.main()
