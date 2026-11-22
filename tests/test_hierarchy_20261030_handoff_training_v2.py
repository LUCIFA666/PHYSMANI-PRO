import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from our_mani.hierarchy_20261030.data import file_sha256
from our_mani.hierarchy_20261030.handoff_training_v2 import (
    EpisodeHandoffSkillWindows, episode_handoff_goal, evaluate_handoff,
    frozen_encoder_sha256, initialize_handoff_models,
)
from our_mani.hierarchy_20261030.networks import CausalSceneSkillEncoder
from our_mani.hierarchy_20261030.training import CausalSkillWindows, skill_loss


class EpisodeHandoffTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        n = 6
        path = root / "episode.npz"
        actions = np.zeros((n, 8), dtype=np.float32)
        actions[:, :3] = np.arange(n)[:, None] * np.array([.02, -.03, .01]) + [.1, .2, .8]
        actions[:, 6] = 1.
        actions[:, 7] = [1., 1., 0., 0., 0., 1.]
        np.savez(path, rgb=np.arange(n, dtype=np.uint8)[:, None, None, None, None]
                 * np.ones((n, 4, 64, 64, 3), dtype=np.uint8),
                 depth=np.ones((n, 4, 64, 64), dtype=np.float32),
                 robot=np.zeros((n, 15), dtype=np.float32), expert_actions=actions,
                 skill_labels=np.array([0, 0, 1, 2, 3, 3]), expert_command_run=np.arange(n))
        row = dict(split="train", task="pick_moving_target_on_the_table", raw_episode_id="episode0",
                   parent_source_group="train:pick_moving_target_on_the_table:episode0",
                   npz_path="episode.npz", cache_sha256=file_sha256(path), frames=n)
        return {"episodes": [row]}, actions

    def test_whole_episode_goal_is_first_closed_command_across_skill_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, actions = self.fixture(root)
            data = EpisodeHandoffSkillWindows(root, manifest, "train", history_frames=3)
            for index in range(len(data)):
                np.testing.assert_array_equal(data[index]["goal_xyz"], actions[2, :3])
                self.assertEqual(data[index]["goal_time"], .5)
            self.assertNotEqual(len(set(tuple(data[i]["action"][:3]) for i in range(len(data)))), 1)
            self.assertTrue(any(not np.array_equal(data[i]["goal_xyz"], data[i]["action"][:3]) for i in range(len(data))))
            self.assertEqual(data.handoff_records[0]["first_closed_command_frame"], 2)
            with self.assertRaises(ValueError):
                episode_handoff_goal(np.column_stack((actions[:, :7], np.ones(len(actions)))))

    def test_causal_scene_inputs_and_parent_command_weights_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.fixture(root)
            data = EpisodeHandoffSkillWindows(root, manifest, "train", history_frames=3)
            original = CausalSkillWindows(root, manifest, "train", history_frames=3)
            np.testing.assert_array_equal(data.weights, original.weights)
            for index in range(len(data)):
                for key in ("rgbd", "robot_history", "lengths", "weight", "episode", "frame"):
                    np.testing.assert_array_equal(data[index][key], original[index][key])
            before = data[1]
            data.episodes[0]["expert_actions"][4:, :3] = 99.
            data.episodes[0]["skill_labels"][:] = 10
            after = data[1]
            for key in ("rgbd", "robot_history", "lengths"):
                np.testing.assert_array_equal(before[key], after[key])

    def test_only_reinitialized_embedding_and_policy_update(self):
        torch.manual_seed(7)
        source = CausalSceneSkillEncoder().state_dict()
        source["skill_embeddings.weight"].fill_(17.)
        encoder, policy = initialize_handoff_models(source, [.2, 0., .8], [.5, .6, .5])
        other_encoder, other_policy = initialize_handoff_models(source, [.2, 0., .8], [.5, .6, .5])
        self.assertEqual([name for name, value in encoder.named_parameters() if value.requires_grad], ["skill_embeddings.weight"])
        self.assertFalse(torch.equal(encoder.skill_embeddings.weight, source["skill_embeddings.weight"]))
        torch.testing.assert_close(encoder.skill_embeddings.weight, other_encoder.skill_embeddings.weight, rtol=0, atol=0)
        for left, right in zip(policy.parameters(), other_policy.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        frozen = frozen_encoder_sha256(encoder)
        old_embedding = encoder.skill_embeddings.weight.detach().clone()
        old_policy = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        rgbd = torch.randn(2, 1, 4, 4, 64, 64)
        robot = torch.randn(2, 1, 15)
        encoded = encoder(rgbd, robot, torch.ones(2, dtype=torch.long))
        target = torch.tensor([[.3, .1, .9, 0., 0., 0., 1., 0.], [.1, -.1, .8, 0., 0., 0., 1., 1.]])
        prediction = policy(encoded["scene_features"], encoded["skill_latent"], robot[:, 0],
                            torch.tensor([[.2, 0., .8]]).expand(2, 3), torch.full((2,), .5))
        self.assertFalse(prediction["trajectory_consumed"])
        optimizer = torch.optim.AdamW([encoder.skill_embeddings.weight, *policy.parameters()], lr=3e-4)
        optimizer.zero_grad()
        skill_loss(prediction, target, torch.tensor([.5, .6, .5])).mean().backward()
        optimizer.step()
        self.assertEqual(frozen_encoder_sha256(encoder), frozen)
        self.assertFalse(torch.equal(encoder.skill_embeddings.weight, old_embedding))
        self.assertTrue(any(not torch.equal(value, old_policy[key]) for key, value in policy.state_dict().items()))
        self.assertTrue(all(value.grad is None for name, value in encoder.named_parameters() if name != "skill_embeddings.weight"))

    def test_metrics_report_episode_goal_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.fixture(root)
            data = EpisodeHandoffSkillWindows(root, manifest, "train", history_frames=3)
            encoder, policy = initialize_handoff_models(CausalSceneSkillEncoder().state_dict(), [.2, 0., .8], [.5, .6, .5])
            result = evaluate_handoff(encoder, policy, DataLoader(data, batch_size=2), torch.device("cpu"), torch.tensor([.5, .6, .5]))
            self.assertEqual(result["goal_source"], "episode_first_closed_expert_TCP_target")
            self.assertTrue(result["goal_latched_for_whole_episode"])
            self.assertNotIn("copy_segment_entry_goal_position_error_m", result)
            self.assertIn("copy_episode_handoff_goal_position_error_m", result)
            self.assertFalse(result["native_success_measured"])


if __name__ == "__main__":
    unittest.main()
