import copy
import unittest

import numpy as np
import torch

from our_mani.hierarchy_20261030.candidate_multibody import MultiBodyCandidateTransformer, candidate_imitation_loss
from our_mani.hierarchy_20261030.candidate_training import (
    DATA_SCHEMA, MOTION_SCHEMA, ROLE_NAMES, TASK_FAMILIES,
    check_full_source_coverage, encode_candidate_batch, select_candidate_anchors,
    validate_motion_arrays,
)
from our_mani.hierarchy_20261030.networks import CausalSceneSkillEncoder


class CandidateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def forecasts(self, n=10):
        trajectory = np.full((n, 6, 23, 20), np.nan, np.float32)
        mask = np.zeros((n, 6, 23), dtype=bool)
        history = np.zeros((n, 6), dtype=np.int64)
        for frame in range(4, n):
            trajectory[frame, 0] = 0.
            trajectory[frame, 0, :, -1] = np.arange(23) * .05
            mask[frame, 0] = True; history[frame, 0] = 5
        return trajectory, mask, history

    def test_first_close_boundary_missing_predictions_and_zero_weight_ledger(self):
        actions = np.zeros((10, 8), np.float32)
        actions[:, 6] = 1.; actions[:8, 7] = 1.
        _, mask, history = self.forecasts()
        frames = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        keep = frames
        eligible, decisions, first_close = select_candidate_anchors(actions, frames, mask[keep], history[keep])
        self.assertEqual(first_close, 8)
        self.assertEqual(eligible, [(6, 5)])
        reasons = {item["frame"]: item["zero_weight_reason"] for item in decisions}
        self.assertEqual(reasons[0], "scene_history_fewer_than_five_frames")
        self.assertEqual(reasons[4], "motion_frame_missing")
        self.assertEqual(reasons[8], "at_or_after_first_expert_close")
        no_forecast = np.zeros_like(mask)
        self.assertEqual(select_candidate_anchors(actions, np.arange(10), no_forecast, history)[0], [])

    def test_real_forecast_history_and_independent_clocks_are_required(self):
        trajectory, mask, history = self.forecasts()
        arrays = dict(timestamps=np.arange(10) * .05, role_names=np.asarray(ROLE_NAMES),
                      predicted_trajectory=trajectory, trajectory_mask=mask)
        audit = dict(frames=[dict(timestamp=i * .05,
                     motion_status={role: dict(available=bool(mask[i, j].all()), history_frames=int(history[i, j]))
                                    for j, role in enumerate(ROLE_NAMES)}) for i in range(10)])
        validate_motion_arrays(arrays, audit, 10)
        other = copy.deepcopy(audit)
        other["frames"][4]["motion_status"]["target"]["history_frames"] = 4
        with self.assertRaises(ValueError):
            validate_motion_arrays(arrays, other, 10)
        corrupted = dict(arrays, predicted_trajectory=trajectory.copy())
        corrupted["predicted_trajectory"][4, 0, :, -1] += 1.
        with self.assertRaises(ValueError):
            validate_motion_arrays(corrupted, audit, 10)

    def source_documents(self):
        rows = []
        for split in ("train", "validation"):
            for task in TASK_FAMILIES:
                count = 100 if split == "train" else 18 if task == "insert_onto_rotating_peg" else 19 if task == "insert_onto_rotating_peg_high_speed" else 20
                for episode in range(count):
                    rows.append(dict(split=split, task=task, raw_episode_id=f"{split}_episode{episode}",
                                     parent_source_group=f"{split}:{task}:{episode}", source_path=f"/{split}/{task}/{episode}"))
        data = dict(schema=DATA_SCHEMA, complete=True, smoke_only=False, TEST_read=False,
                    source_debug_target_positions_read=False, episodes=rows)
        motion = dict(schema=MOTION_SCHEMA, complete=True, smoke_only=False, expected_cases=477,
                      TEST_read=False, source_debug_target_positions_read=False,
                      frontend="blind_parallel_fixed_common_TRAIN_priors", source_task_selects_frontend=False,
                      identity={"data_manifest_sha256": "test-source"}, cases=copy.deepcopy(rows))
        return data, motion

    def test_exact_all_parent_coverage_and_blind_frontend_are_mandatory(self):
        data, motion = self.source_documents()
        self.assertEqual(len(check_full_source_coverage(data, motion, "test-source")), 477)
        missing = copy.deepcopy(motion); missing["cases"].pop()
        with self.assertRaises(ValueError):
            check_full_source_coverage(data, missing, "test-source")
        smoke = dict(motion, smoke_only=True)
        with self.assertRaises(ValueError):
            check_full_source_coverage(data, smoke, "test-source")
        task_oracle = dict(motion, source_task_selects_frontend=True)
        with self.assertRaises(ValueError):
            check_full_source_coverage(data, task_oracle, "test-source")

    def batch(self):
        trajectory = torch.full((2, 6, 23, 20), float("nan"))
        mask = torch.zeros(2, 6, 23, dtype=torch.bool)
        trajectory[:, 0] = 0.; trajectory[:, 0, :, -1] = torch.arange(23) * .05; mask[:, 0] = True
        return dict(rgbd=torch.rand(2, 5, 4, 4, 64, 64), robot_history=torch.rand(2, 5, 15),
                    lengths=torch.tensor([3, 5]), trajectory=trajectory, trajectory_mask=mask,
                    target_xyz=torch.tensor([[.2, 0., .8], [.2, 0., .8]]))

    def test_goal_labels_do_not_enter_frozen_scene_inputs(self):
        encoder = CausalSceneSkillEncoder().eval()
        for value in encoder.parameters():
            value.requires_grad_(False)
        batch = self.batch()
        first = encode_candidate_batch(encoder, batch)
        altered = {**batch, "target_xyz": torch.full((2, 3), 999.), "expert_stage": torch.full((2,), 10)}
        altered["rgbd"] = batch["rgbd"].clone(); altered["robot_history"] = batch["robot_history"].clone()
        altered["rgbd"][0, 3:] = float("nan"); altered["robot_history"][0, 3:] = float("nan")
        second = encode_candidate_batch(encoder, altered)
        for key in ("scene_features", "skill_latent", "robot"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_padding_cannot_change_choice_and_no_known_prediction_is_rejected(self):
        torch.manual_seed(30)
        model = MultiBodyCandidateTransformer([.2, 0., .8], [.5, .6, .5], role_names=ROLE_NAMES).eval()
        batch = self.batch()
        latent, robot, scene = torch.randn(2, 64), torch.randn(2, 15), torch.randn(2, 128)
        with torch.no_grad():
            first = model(batch["trajectory"], batch["trajectory_mask"], latent, robot, scene)
            changed = batch["trajectory"].clone(); changed[:, 1:] = 123456.
            second = model(changed, batch["trajectory_mask"], latent, robot, scene)
        torch.testing.assert_close(first["goal_xyz"], second["goal_xyz"], atol=0, rtol=0)
        torch.testing.assert_close(first["point_logits"], second["point_logits"], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model(batch["trajectory"], torch.zeros_like(batch["trajectory_mask"]), latent, robot, scene)
        losses, outside = candidate_imitation_loss(model, first, batch["target_xyz"])
        self.assertTrue(torch.isfinite(losses).all())
        self.assertFalse(outside.any())


if __name__ == "__main__":
    unittest.main()
