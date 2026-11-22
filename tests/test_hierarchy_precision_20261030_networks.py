import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from our_mani.hierarchy_20261030.candidate_multibody import MultiBodyCandidateTransformer
from our_mani.hierarchy_20261030.networks import CausalSceneSkillEncoder, PretrainedSkillPolicy
from our_mani.hierarchy_precision_20261030.networks import PrecisionHeads


class PrecisionHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(1026)
        cls.center, cls.scale, cls.roles = [0., 0., .8], [.5, .6, .5], ("target", "ring")
        cls.encoder_state = CausalSceneSkillEncoder().state_dict()
        cls.policy_state = PretrainedSkillPolicy(cls.center, cls.scale).state_dict()
        cls.candidate_state = MultiBodyCandidateTransformer(
            cls.center, cls.scale, role_names=cls.roles).state_dict()

    def setUp(self):
        torch.manual_seed(30)
        self.model = PrecisionHeads(self.center, self.scale, self.roles,
                                    self.encoder_state, self.policy_state, self.candidate_state)
        self.model.eval()
        self.rgbd = torch.randn(2, 4, 4, 64, 64)
        self.scene = torch.randn(2, 128)
        self.probabilities = torch.randn(2, 11).softmax(-1)
        self.robot = torch.randn(2, 15)
        self.trajectory = torch.randn(2, 2, 23, 20)
        self.trajectory[..., -1] = torch.arange(23) * .05
        self.mask = torch.ones(2, 2, 23, dtype=torch.bool)
        self.goal = torch.tensor([[.1, .2, .9], [-.1, 0., .8]])
        self.seconds = torch.tensor([.2, 2.45])

    def candidate(self, trajectory=None, mask=None):
        return self.model.forward_candidate(self.rgbd, self.scene, self.probabilities, self.robot,
                                            self.trajectory if trajectory is None else trajectory,
                                            self.mask if mask is None else mask)

    def skill(self):
        return self.model.forward_skill(self.rgbd, self.scene, self.probabilities,
                                        self.robot, self.goal, self.seconds)

    def test_exact_copied_heads_and_no_classifier_in_optimizer(self):
        for name, value in self.candidate_state.items():
            torch.testing.assert_close(self.model.candidate.state_dict()[name], value, rtol=0, atol=0)
        for name, value in self.policy_state.items():
            torch.testing.assert_close(self.model.policy.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(self.model.skill_embeddings.weight,
                                   self.encoder_state["skill_embeddings.weight"], rtol=0, atol=0)
        self.assertNotEqual(self.model.skill_embeddings.weight.data_ptr(),
                            self.encoder_state["skill_embeddings.weight"].data_ptr())
        self.assertFalse(any(isinstance(module, CausalSceneSkillEncoder) for module in self.model.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.candidate.time_head.parameters()))
        optimizer = torch.optim.AdamW(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertNotIn(id(self.model.candidate.time_head.weight), optimized)
        self.assertIn(id(self.model.skill_embeddings.weight), optimized)

    def test_geometry_distinguishes_identical_pixels_at_different_positions(self):
        frames = torch.zeros(2, 4, 4, 64, 64)
        frames[0, 0, :, 8:16, 8:16] = 1
        frames[1, 0, :, 40:48, 40:48] = 1
        torch.testing.assert_close(frames[0].sum(), frames[1].sum())
        scene = self.model.geometry(frames, torch.zeros(2, 128))
        self.assertEqual(tuple(scene.shape), (2, 128))
        self.assertGreater(float((scene[0] - scene[1]).detach().norm()), 1e-4)
        self.assertFalse(any(isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d))
                             for module in self.model.geometry.modules()))
        cells = self.model.geometry.camera_encoder(torch.zeros(8, 6, 64, 64))
        self.assertEqual(tuple(cells.shape), (8, 32, 8, 8))

    def test_same_embedding_reaches_both_heads_and_both_losses_train_it(self):
        self.scene.requires_grad_()
        self.probabilities.requires_grad_()
        captured = []
        candidate_hook = self.model.candidate.register_forward_pre_hook(
            lambda module, args: captured.append(args[2].detach().clone()))
        skill_hook = self.model.policy.register_forward_pre_hook(
            lambda module, args: captured.append(args[1].detach().clone()))
        prediction = self.candidate()
        candidate_loss = (F.cross_entropy(prediction["point_logits"], torch.tensor([0, 10]))
                          + prediction["goal_time"].square().mean())
        candidate_loss.backward()
        self.assertGreater(float(self.model.skill_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.geometry.spatial_projection[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.scene.grad)
        self.assertIsNone(self.probabilities.grad)
        self.assertIsNone(self.model.candidate.time_head.weight.grad)
        self.model.zero_grad(set_to_none=True)
        action = self.skill()
        (action["xyz"].square().mean() + F.cross_entropy(action["gripper_logits"],
                                                        torch.tensor([0, 1]))).backward()
        self.assertGreater(float(self.model.skill_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.geometry.spatial_projection[0].weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(self.model.skill_embeddings.weight.grad).all())
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        expected = self.probabilities.detach() @ F.normalize(self.model.skill_embeddings.weight.detach(), dim=-1)
        torch.testing.assert_close(captured[0], expected)
        candidate_hook.remove()
        skill_hook.remove()

    def test_unknown_forecast_values_are_masked_and_valid_clock_is_preserved(self):
        self.mask[:, 1] = False
        original = self.trajectory.clone()
        with torch.no_grad():
            first = self.candidate()
            changed = self.trajectory.clone()
            changed[:, 1] = float("nan")
            second = self.candidate(changed)
        for name in ("point_logits", "candidate_xyz", "goal_time"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        torch.testing.assert_close(original, self.trajectory, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.candidate(mask=self.mask & False)
        bad_clock = self.trajectory.clone()
        bad_clock[:, 0, 20, -1] += .01
        with self.assertRaises(ValueError):
            self.candidate(bad_clock)
        partial = self.mask.clone()
        partial[:, 0, 20] = False
        with self.assertRaises(ValueError):
            self.candidate(mask=partial)

    def test_remaining_time_is_continuous_unclipped_and_explicitly_nominal(self):
        with torch.no_grad():
            for parameter in self.model.time_regressor.parameters():
                parameter.zero_()
            self.model.time_regressor[-1].bias.fill_(2.4)
        prediction = self.candidate()
        self.assertTrue(bool((prediction["goal_time"] > 1.1).all()))
        self.assertIn("legacy_time_logits", prediction)
        self.assertNotIn("time_logits", prediction)
        self.assertTrue(bool((prediction["legacy_discrete_goal_time"] <= 1.1).all()))
        loss = (prediction["goal_time"] - torch.tensor([2.45, 1.6])).square().mean()
        loss.backward()
        self.assertGreater(float(self.model.time_regressor[-1].bias.grad.abs().sum()), 0)
        self.assertIn("nominal_H5", self.model.metadata["goal_time_clock"])
        self.assertIn("not_measured_object_interception", self.model.metadata["goal_time_semantics"])

    def test_skill_accepts_latched_goal_time_beyond_forecast_and_decodes_legal_action(self):
        output = self.skill()
        self.assertFalse(output["trajectory_consumed"])
        self.assertEqual(tuple(output["action"].shape), (2, 8))
        self.assertTrue(torch.isfinite(output["action"]).all())
        rotation = output["rotation_matrix"]
        torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                                   torch.eye(3).expand(2, -1, -1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output["quaternion_xyzw"].norm(dim=-1), torch.ones(2))

    def test_labels_are_not_inputs_and_invalid_observations_or_probabilities_fail(self):
        for method in (self.model.encode, self.model.forward_candidate, self.model.forward_skill):
            names = tuple(inspect.signature(method).parameters)
            self.assertFalse(any("label" in name or "stage" in name or "task" in name for name in names))
        with self.assertRaises(TypeError):
            self.model.forward_candidate(self.rgbd, self.scene, self.probabilities, self.robot,
                                         self.trajectory, self.mask, skill_labels=torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            self.model.encode(self.rgbd, self.scene, torch.ones_like(self.probabilities))
        with self.assertRaises(ValueError):
            self.model.encode(self.rgbd, self.scene, torch.tensor([0, 1]))
        bad = self.rgbd.clone()
        bad[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model.encode(bad, self.scene, self.probabilities)

    def test_mismatched_workspace_and_missing_head_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PrecisionHeads([1, 0, .8], self.scale, self.roles,
                           self.encoder_state, self.policy_state, self.candidate_state)
        missing = dict(self.policy_state)
        missing.pop("xyz_head.weight")
        with self.assertRaises(RuntimeError):
            PrecisionHeads(self.center, self.scale, self.roles,
                           self.encoder_state, missing, self.candidate_state)


if __name__ == "__main__":
    unittest.main()
