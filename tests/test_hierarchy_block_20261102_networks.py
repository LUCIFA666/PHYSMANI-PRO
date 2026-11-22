import inspect
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from our_mani.hierarchy_20261030.networks import CausalSceneSkillEncoder
from our_mani.hierarchy_block_20261102.networks import (
    BlockHierarchyNetwork, GeometryTokenEncoder, quaternion_xyzw_to_matrix,
)


class BlockHierarchyNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(1102)
        cls.encoder_state = CausalSceneSkillEncoder().state_dict()

    def setUp(self):
        torch.manual_seed(2)
        self.model = BlockHierarchyNetwork([0., 0., .8], [.5, .6, .5], self.encoder_state)
        self.model.eval()
        self.rgbd = torch.randn(2, 5, 4, 4, 64, 64)
        self.robot_history = torch.randn(2, 5, 15) * .1
        self.robot_history[..., 3:7] = torch.tensor([0., 0., 0., 1.])
        self.robot_history[..., 7] = 1
        self.lengths = torch.tensor([5, 5])
        self.robot = self.robot_history[:, -1].clone()
        self.points = torch.randn(2, 19, 7) * .1
        self.points[..., 2] += .8
        self.points[..., 3:6] = self.points[..., 3:6].sigmoid()
        self.points[..., 6] = .6
        self.point_mask = torch.ones(2, 19, dtype=torch.bool)
        self.trajectory = torch.randn(2, 6, 23, 20)
        self.trajectory[..., -1] = torch.arange(23) * .05
        self.forecast_mask = torch.zeros(2, 6, 23, dtype=torch.bool)
        self.forecast_mask[:, 0] = True
        self.goal_xyz = torch.tensor([[.1, 0., .8], [-.1, .1, .9]])
        self.goal_rotation = torch.eye(3).expand(2, -1, -1).clone()
        self.goal_time = torch.tensor([.3, 2.4])

    def encode(self, *, rgbd=None, robots=None, lengths=None, points=None, mask=None):
        return self.model.encode(self.rgbd if rgbd is None else rgbd,
                                  self.robot_history if robots is None else robots,
                                  self.lengths if lengths is None else lengths,
                                  self.points if points is None else points,
                                  self.point_mask if mask is None else mask)

    def candidate(self, encoded, trajectory=None, mask=None):
        return self.model.forward_candidate(encoded, self.robot,
                                             self.trajectory if trajectory is None else trajectory,
                                             self.forecast_mask if mask is None else mask)

    def skill(self, encoded, goal_rotation=None, goal_time=None):
        return self.model.forward_skill(encoded, self.robot, self.goal_xyz,
                                         self.goal_rotation if goal_rotation is None else goal_rotation,
                                         self.goal_time if goal_time is None else goal_time)

    def test_strict_11_class_warmstart_and_default_classifier_freeze(self):
        for name, value in self.encoder_state.items():
            torch.testing.assert_close(self.model.encoder.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(self.model.skill_embeddings.weight,
                                   self.encoder_state["skill_embeddings.weight"][:4], rtol=0, atol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.geometry.parameters()))
        self.model.train()
        self.assertFalse(self.model.encoder.training)
        self.assertTrue(self.model.geometry.training)
        missing = dict(self.encoder_state)
        missing.pop("classifier.weight")
        with self.assertRaises(RuntimeError):
            BlockHierarchyNetwork([0, 0, .8], [.5, .6, .5], missing)

    def test_point_geometry_keeps_world_location_and_dynamic_masked_sets(self):
        encoder = self.model.geometry
        self.assertIsInstance(encoder, GeometryTokenEncoder)
        with torch.no_grad():
            original = encoder(self.points, self.point_mask)
            translated = self.points.clone()
            translated[..., 0] += .15
            moved = encoder(translated, self.point_mask)
            permutation = torch.randperm(self.points.shape[1])
            permuted = encoder(self.points[:, permutation], self.point_mask[:, permutation])
            padded = torch.cat((self.points, torch.full((2, 7, 7), float("nan"))), dim=1)
            padded_mask = torch.cat((self.point_mask, torch.zeros(2, 7, dtype=torch.bool)), dim=1)
            masked = encoder(padded, padded_mask)
        self.assertEqual(tuple(original.shape), (2, 32, 128))
        self.assertGreater(float((original - moved).norm()), 1e-3)
        torch.testing.assert_close(original, permuted, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(original, masked, atol=1e-5, rtol=1e-5)
        self.assertGreater(float((original[:, 0] - original[:, 1]).norm()), 1e-3)
        self.assertFalse(any(isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d))
                             for module in encoder.modules()))
        with self.assertRaises(ValueError):
            encoder(self.points, self.point_mask & False)
        with self.assertRaises(ValueError):
            encoder(self.points, self.point_mask.float())

    def test_shared_group_simnorm_e_and_both_heads_joint_gradient(self):
        encoded = self.encode()
        self.assertEqual(tuple(encoded["skill_logits"].shape), (2, 4))
        torch.testing.assert_close(encoded["skill_probabilities"].sum(-1), torch.ones(2))
        torch.testing.assert_close(encoded["skill_latent"].reshape(2, 8, 8).sum(-1), torch.ones(2, 8))
        expected_e = encoded["skill_probabilities"] @ self.model.simnorm(self.model.skill_embeddings.weight)
        torch.testing.assert_close(encoded["skill_latent"], expected_e)
        captured = []
        candidate_hook = self.model.candidate.register_forward_pre_hook(
            lambda module, args: captured.append(args[4]))
        policy_hook = self.model.skill.register_forward_pre_hook(
            lambda module, args: captured.append(args[2]))
        candidate = self.candidate(encoded)
        policy = self.skill(encoded)
        candidate["positions"].retain_grad()
        candidate["times"].retain_grad()
        position_targets = torch.tensor([.2, -.15, 1.0])[None, None]
        time_targets = torch.arange(1, 5, dtype=torch.float32)[None]
        loss = ((candidate["positions"] - position_targets).square().sum()
                + (candidate["times"] - time_targets).square().sum()
                + F.cross_entropy(candidate["logits"], torch.tensor([0, 1]))
                + (policy["positions"] - self.goal_xyz[:, None]).square().sum()
                + F.cross_entropy(policy["gripper_logits"].reshape(-1, 2), torch.arange(8) % 2))
        loss.backward()
        self.assertIs(captured[0], encoded["skill_latent"])
        self.assertIs(captured[1], encoded["skill_latent"])
        self.assertGreater(float(self.model.skill_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.geometry.point_mlp[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.geometry.xyz_encoder[0].weight.grad.abs().sum()), 0)
        self.assertTrue(bool((candidate["positions"].grad.abs().sum(-1) > 0).all()))
        self.assertTrue(bool((candidate["times"].grad.abs() > 0).all()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.encoder.parameters()))
        for module in (self.model.geometry, self.model.candidate, self.model.skill, self.model.skill_embeddings):
            for parameter in module.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        candidate_hook.remove()
        policy_hook.remove()

    def test_policy_and_candidate_each_use_geometry_and_shared_e(self):
        for branch in ("candidate", "skill"):
            self.model.zero_grad(set_to_none=True)
            encoded = self.encode()
            prediction = self.candidate(encoded) if branch == "candidate" else self.skill(encoded)
            loss = prediction["positions"].square().mean()
            loss.backward()
            self.assertGreater(float(self.model.skill_embeddings.weight.grad.abs().sum()), 0, branch)
            self.assertGreater(float(self.model.geometry.point_mlp[0].weight.grad.abs().sum()), 0, branch)

    def test_real_forecast_unknown_padding_clock_and_no_fabricated_fallback(self):
        encoded = self.encode()
        unknown = self.trajectory.clone()
        unknown[~self.forecast_mask] = float("nan")
        with torch.no_grad():
            first = self.candidate(encoded)
            second = self.candidate(encoded, unknown)
        for key in ("positions", "rotation_matrices", "times", "logits"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.candidate(encoded, mask=self.forecast_mask & False)
        bad_clock = self.trajectory.clone()
        bad_clock[0, 0, 22, -1] += .01
        with self.assertRaises(ValueError):
            self.candidate(encoded, bad_clock)
        partial = self.forecast_mask.clone()
        partial[0, 0, 10] = False
        with self.assertRaises(ValueError):
            self.candidate(encoded, mask=partial)

    def test_joint_candidate_output_contract_rotations_and_unclipped_positive_time(self):
        with torch.no_grad():
            self.model.candidate.time_head.weight.zero_()
            self.model.candidate.time_head.bias.fill_(2.4)
            output = self.candidate(self.encode())
        shapes = dict(positions=(2, 4, 3), rotation_matrices=(2, 4, 3, 3), quaternions=(2, 4, 4),
                      times=(2, 4), logits=(2, 4), probabilities=(2, 4), selected_index=(2,),
                      goal_position=(2, 3), goal_rotation=(2, 3, 3), goal_quaternion=(2, 4), goal_time=(2,))
        for name, shape in shapes.items():
            self.assertEqual(tuple(output[name].shape), shape, name)
        torch.testing.assert_close(output["probabilities"].sum(-1), torch.ones(2))
        rotation = output["rotation_matrices"]
        torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                                   torch.eye(3).expand(2, 4, 3, 3), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2, 4), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(quaternion_xyzw_to_matrix(output["quaternions"]), rotation,
                                   atol=1e-5, rtol=1e-5)
        self.assertTrue(bool((output["times"] > 1.1).all()))
        rows, index = torch.arange(2), output["selected_index"]
        for selected, all_values in (("goal_position", "positions"), ("goal_rotation", "rotation_matrices"),
                                     ("goal_quaternion", "quaternions"), ("goal_time", "times")):
            torch.testing.assert_close(output[selected], output[all_values][rows, index])
        self.assertIn("not_calibrated_success", self.model.candidate.score_semantics)
        self.assertIn("nominal_H5", self.model.metadata["goal_time_clock"])

    def test_decode_world_deltas_and_current_times_relative_rotations_without_accumulation(self):
        robot = torch.zeros(1, 15)
        robot[0, :3] = torch.tensor([1., 2., 3.])
        robot[0, 3:7] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        output = dict(delta_xyz=torch.tensor([[[.1, 0., 0.]]] * 4).transpose(0, 1),
                      relative_rotation_matrices=rx90.expand(1, 4, 3, 3),
                      gripper_logits=torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]]))
        actions = self.model.decode_skill(robot, output)
        self.assertEqual(tuple(actions.shape), (1, 4, 8))
        torch.testing.assert_close(actions[0, :, :3], torch.tensor([1.1, 2., 3.]).expand(4, 3))
        torch.testing.assert_close(quaternion_xyzw_to_matrix(actions[..., 3:7]),
                                   (rz90 @ rx90).expand(1, 4, 3, 3), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actions[0, :, 7], torch.tensor([0., 1., 0., 1.]))

    def test_causal_history_padding_invariance_and_feedback_reencoding(self):
        lengths = torch.tensor([3, 5])
        rgbd = torch.cat((self.rgbd, torch.zeros(2, 2, 4, 4, 64, 64)), dim=1)
        robots = torch.cat((self.robot_history, torch.zeros(2, 2, 15)), dim=1)
        valid = torch.arange(7)[None] < lengths[:, None]
        changed_images, changed_robots = rgbd.clone(), robots.clone()
        changed_images[~valid] = float("nan")
        changed_robots[~valid] = float("nan")
        with torch.no_grad():
            a = self.encode(rgbd=rgbd, robots=robots, lengths=lengths)
            b = self.encode(rgbd=changed_images, robots=changed_robots, lengths=lengths)
        for name in ("scene_features", "skill_latent", "skill_probabilities"):
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        with torch.no_grad():
            original = self.encode()
            changed_history = self.robot_history.clone()
            changed_history[:, :4, 8] += 1.0
            feedback = self.encode(robots=changed_history)
            before, after = self.skill(original), self.skill(feedback)
        self.assertGreater(float((original["scene_features"] - feedback["scene_features"]).norm()), 1e-4)
        self.assertGreater(float((before["delta_xyz"] - after["delta_xyz"]).norm()), 1e-7)

    def test_unfreeze_configuration_trains_visual_path_but_not_legacy_E(self):
        self.model.set_encoder_visual_frozen(False)
        self.model.train()
        self.assertTrue(self.model.encoder.training)
        self.assertFalse(self.model.metadata["classifier_visual_frozen"])
        encoded = self.encode()
        loss = F.cross_entropy(encoded["skill_logits"], torch.tensor([0, 1]))
        loss = loss + self.skill(encoded)["positions"].square().mean()
        loss.backward()
        self.assertGreater(float(self.model.encoder.camera_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.encoder.history_encoder.weight_ih_l0.grad.abs().sum()), 0)
        self.assertIsNone(self.model.encoder.skill_embeddings.weight.grad)

    def test_cached_frozen_features_match_encode_and_still_train_geometry_and_shared_E(self):
        with torch.no_grad():
            old = self.model.encoder(self.rgbd, self.robot_history, self.lengths)
        cached_scene = old["scene_features"].clone().requires_grad_()
        cached_latent = old["scene_latent"].clone().requires_grad_()
        cached_logits = old["skill_logits"][:, :4].clone().requires_grad_()
        cached = self.model.encode_frozen_features(cached_scene, cached_latent, cached_logits,
                                                   self.points, self.point_mask)
        fresh = self.encode()
        self.assertTrue(self.model.encoder_visual_frozen)
        for name in fresh:
            torch.testing.assert_close(cached[name], fresh[name], rtol=0, atol=0)
        candidate, skill = self.candidate(cached), self.skill(cached)
        loss = candidate["positions"].square().mean() + skill["positions"].square().mean()
        loss.backward()
        for value in (cached_scene, cached_latent, cached_logits):
            self.assertIsNone(value.grad)
        self.assertGreater(float(self.model.geometry.point_mlp[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.skill_embeddings.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.encoder.parameters()))
        with self.assertRaises(ValueError):
            self.model.encode_frozen_features(cached_scene, cached_latent, old["skill_logits"],
                                               self.points, self.point_mask)
        self.model.set_encoder_visual_frozen(False)
        self.assertFalse(self.model.encoder_visual_frozen)
        with self.assertRaises(ValueError):
            self.model.encode_frozen_features(cached_scene, cached_latent, cached_logits,
                                               self.points, self.point_mask)

    def test_labels_not_inputs_and_invalid_goal_robot_or_points_are_rejected(self):
        for method in (self.model.encode, self.model.forward_candidate, self.model.forward_skill):
            names = tuple(inspect.signature(method).parameters)
            self.assertFalse(any("task" in name or "label" in name or "stage" in name for name in names))
        with self.assertRaises(TypeError):
            self.model.encode(self.rgbd, self.robot_history, self.lengths, self.points,
                               self.point_mask, task_id=torch.tensor([0, 0]))
        encoded = self.encode()
        with self.assertRaises(ValueError):
            self.skill(encoded, goal_rotation=torch.zeros(2, 3, 3))
        with self.assertRaises(ValueError):
            self.skill(encoded, goal_time=torch.tensor([-.1, 1.]))
        with self.assertRaises(ValueError):
            self.model.forward_candidate(encoded, torch.zeros(2, 15), self.trajectory, self.forecast_mask)
        invalid_points = self.points.clone()
        invalid_points[0, 0, 2] = float("nan")
        with self.assertRaises(ValueError):
            self.encode(points=invalid_points)


if __name__ == "__main__":
    unittest.main()
