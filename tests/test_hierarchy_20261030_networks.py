import unittest

import torch
from torch.nn import functional as F

from our_mani.hierarchy_20261030.networks import (
    CandidatePointTransformer, CausalSceneSkillEncoder, HierarchyNetwork,
    PretrainedSkillPolicy, matrix_to_quaternion_xyzw, rotation_6d_to_matrix,
)


class HierarchyNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(30)
        self.center = (.2, 0., .8)
        self.scale = (.5, .6, .5)

    @staticmethod
    def history(batch=2, steps=3):
        return torch.rand(batch, steps, 4, 4, 64, 64), torch.randn(batch, steps, 15), torch.tensor([1, steps][:batch])

    @staticmethod
    def forecast(batch=2, steps=5):
        trajectory = torch.randn(batch, steps, 20)
        trajectory[..., -1] = torch.arange(steps) * .05
        return trajectory, torch.ones(batch, steps, dtype=torch.bool)

    def test_scene_padding_and_unobserved_future_do_not_change_features(self):
        model = CausalSceneSkillEncoder().eval()
        rgbd, robot, lengths = self.history()
        other_images, other_robot = rgbd.clone(), robot.clone()
        other_images[0, 1:] = float('nan')
        other_robot[0, 1:] = float('nan')
        with torch.no_grad():
            first = model(rgbd, robot, lengths)
            second = model(other_images, other_robot, lengths)
            truncated = model(rgbd[:1, :1], robot[:1, :1], torch.tensor([1]))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            torch.testing.assert_close(first[key][:1], truncated[key], rtol=2e-5, atol=2e-6)
        self.assertEqual(first['scene_features'].shape, (2, 128))
        self.assertEqual(first['scene_latent'].shape, (2, 64))
        torch.testing.assert_close(first['scene_latent'].reshape(2, 8, 8).sum(-1), torch.ones(2, 8))
        torch.testing.assert_close(first['skill_probabilities'].sum(-1), torch.ones(2))
        self.assertTrue(bool((first['skill_latent'].norm(dim=-1) <= 1.00001).all()))

    def test_scene_rejects_invalid_lengths_shapes_and_visible_nan(self):
        model = CausalSceneSkillEncoder()
        rgbd, robot, lengths = self.history()
        with self.assertRaises(ValueError):
            model(rgbd, robot, torch.tensor([0, 3]))
        with self.assertRaises(ValueError):
            model(rgbd[:, :, :3], robot, lengths)
        with self.assertRaises(ValueError):
            model(rgbd, robot[..., :14], lengths)
        rgbd[0, 0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(rgbd, robot, lengths)

    def test_candidate_ignores_padding_and_selects_grid_and_time_argmax(self):
        model = CandidatePointTransformer(self.center, self.scale).eval()
        trajectory, mask = self.forecast()
        mask[0, 3:] = False
        latent, robot = torch.randn(2, 64), torch.randn(2, 15)
        changed = trajectory.clone()
        changed[0, 3:] = float('nan')
        with torch.no_grad():
            first = model(trajectory, mask, latent, robot)
            second = model(changed, mask, latent, robot)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(first['point_logits'].shape, (2, 4096))
        self.assertEqual(first['time_logits'].shape, (2, 23))
        torch.testing.assert_close(first['point_probabilities'].sum(-1), torch.ones(2))
        normalized = (first['candidate_xyz'] - model.workspace_center) / model.workspace_scale
        self.assertTrue(bool((normalized.abs() <= 1.00001).all()))
        first['point_logits'] = torch.full_like(first['point_logits'], -100.)
        first['point_logits'][:, 177] = 100.
        first['point_probabilities'] = first['point_logits'].softmax(-1)
        first['time_logits'] = torch.full_like(first['time_logits'], -100.)
        first['time_logits'][:, 7] = 100.
        first['time_probabilities'] = first['time_logits'].softmax(-1)
        choice = model.choose(first)
        torch.testing.assert_close(choice['goal_xyz'], first['candidate_xyz'][:, 177])
        torch.testing.assert_close(choice['goal_time'], torch.full((2,), .35))
        self.assertIn('not_calibrated', model.probability_semantics)

    def test_candidate_rejects_invalid_forecasts_and_workspace(self):
        with self.assertRaises(ValueError):
            CandidatePointTransformer(self.center, (.5, 0., .5))
        model = CandidatePointTransformer(self.center, self.scale)
        trajectory, mask = self.forecast()
        latent, robot = torch.randn(2, 64), torch.randn(2, 15)
        mask[0] = False
        with self.assertRaises(ValueError):
            model(trajectory, mask, latent, robot)
        mask[:] = True
        trajectory[0, 1, -1] = 0.
        with self.assertRaises(ValueError):
            model(trajectory, mask, latent, robot)
        with self.assertRaises(ValueError):
            model(trajectory[..., :19], mask, latent, robot)

    def test_skill_without_trajectory_is_explicit_and_decodes_native_action(self):
        model = PretrainedSkillPolicy(self.center, self.scale).eval()
        features, latent, robot = torch.randn(2, 128), torch.randn(2, 64), torch.randn(2, 15)
        goals, times = torch.tensor(self.center).expand(2, 3), torch.tensor([0., .5])
        output = model(features, latent, robot, goals, times)
        self.assertFalse(output['trajectory_consumed'])
        self.assertEqual(output['action'].shape, (2, 8))
        torch.testing.assert_close(output['action'][:, :3], output['xyz'])
        torch.testing.assert_close(output['action'][:, 3:7], output['quaternion_xyzw'])
        torch.testing.assert_close(output['quaternion_xyzw'].norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(output['action'][:, 7], output['gripper_logits'].argmax(-1).float())
        rotation = output['rotation_matrix']
        torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand(2, 3, 3))
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2))
        trajectory, mask = self.forecast()
        with self.assertRaises(ValueError):
            model(features, latent, robot, goals, times, trajectory=trajectory)
        with self.assertRaises(ValueError):
            model(features, latent, robot, goals, torch.tensor([-.1, .5]))

    def test_skill_real_trajectory_padding_is_ignored_and_feedback_changes_action(self):
        model = PretrainedSkillPolicy(self.center, self.scale).eval()
        features, latent, robot = torch.randn(2, 128), torch.randn(2, 64), torch.randn(2, 15)
        goals, times = torch.tensor(self.center).expand(2, 3), torch.tensor([0., .5])
        trajectory, mask = self.forecast()
        mask[0, 3:] = False
        changed = trajectory.clone()
        changed[0, 3:] = float('nan')
        first = model(features, latent, robot, goals, times, trajectory, mask)
        second = model(features, latent, robot, goals, times, changed, mask)
        torch.testing.assert_close(first['action'], second['action'], rtol=0, atol=0)
        self.assertTrue(first['trajectory_consumed'])
        revised_robot = robot.clone()
        revised_robot[:, :3] += .25
        feedback = model(features, latent, revised_robot, goals, times, trajectory, mask)
        self.assertGreater(float((first['xyz'] - feedback['xyz']).abs().sum().detach()), 1e-6)

    def test_legal_rotations_including_degeneracy_and_finite_gradients(self):
        values = torch.randn(32, 6, requires_grad=True)
        rotation = rotation_6d_to_matrix(values)
        quaternion = matrix_to_quaternion_xyzw(rotation)
        torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand(32, 3, 3), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(32), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(32))
        x, y, z, w = quaternion.unbind(-1)
        reconstructed = torch.stack((
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
            2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
        ), -1).reshape(32, 3, 3)
        torch.testing.assert_close(reconstructed, rotation, atol=2e-5, rtol=2e-5)
        (quaternion.square().mean() + rotation.mean()).backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        zero_rotation = rotation_6d_to_matrix(torch.zeros(2, 6))
        torch.testing.assert_close(zero_rotation, torch.eye(3).expand(2, 3, 3))
        torch.testing.assert_close(matrix_to_quaternion_xyzw(zero_rotation), torch.tensor([0., 0., 0., 1.]).expand(2, 4))

    def test_same_scene_skill_latent_conditions_both_independent_modules(self):
        encoder = CausalSceneSkillEncoder()
        candidate = CandidatePointTransformer(self.center, self.scale)
        skill = PretrainedSkillPolicy(self.center, self.scale)
        rgbd, robot_history, lengths = self.history()
        scene = encoder(rgbd, robot_history, lengths)
        latent = scene['skill_latent']
        latent.retain_grad()
        robot = torch.randn(2, 15)
        trajectory, mask = self.forecast()
        point = candidate(trajectory, mask, latent, robot)
        point_loss = F.cross_entropy(point['point_logits'], torch.tensor([10, 21]))
        point_gradient = torch.autograd.grad(point_loss, latent, retain_graph=True)[0]
        self.assertGreater(float(point_gradient.abs().sum()), 1e-8)
        action = skill(scene['scene_features'], latent, robot, torch.tensor(self.center).expand(2, 3), torch.ones(2) * .5)
        skill_loss = action['xyz'].square().mean() + F.cross_entropy(action['gripper_logits'], torch.tensor([0, 1]))
        skill_gradient = torch.autograd.grad(skill_loss, latent, retain_graph=True)[0]
        self.assertGreater(float(skill_gradient.abs().sum()), 1e-8)
        (point_loss + skill_loss).backward()
        self.assertGreater(float(encoder.skill_embeddings.weight.grad.abs().sum()), 1e-8)
        self.assertGreater(float(encoder.classifier.weight.grad.abs().sum()), 1e-8)

    def test_composition_requires_real_forecast_and_pretraining_does_not_call_candidate(self):
        model = HierarchyNetwork(self.center, self.scale).eval()
        rgbd, robot, lengths = self.history()
        original = model.candidate_network.forward
        def forbidden(*args, **kwargs):
            raise AssertionError('candidate must not be consumed during independent skill pretraining')
        model.candidate_network.forward = forbidden
        result = model.forward_skill(rgbd, robot, lengths, torch.tensor(self.center).expand(2, 3), torch.zeros(2))
        self.assertFalse(result['candidate_network_consumed'])
        self.assertFalse(result['skill']['trajectory_consumed'])
        self.assertEqual(result['goal_source'], 'offline_expert_conditioning')
        model.candidate_network.forward = original
        trajectory, mask = self.forecast()
        result = model(rgbd, robot, lengths, trajectory, mask)
        self.assertTrue(result['skill']['trajectory_consumed'])
        self.assertEqual(result['goal_source'], 'predicted_candidate_imitation')
        with self.assertRaises(TypeError):
            model(rgbd, robot, lengths)


if __name__ == '__main__':
    unittest.main()
