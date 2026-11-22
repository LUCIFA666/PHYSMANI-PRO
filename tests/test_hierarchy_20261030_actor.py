"""CPU actor causality/dataflow tests with explicit, isolated test modules."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from our_mani.contracts import Pose, RobotState
from our_mani.perception import CameraCalibration
from our_mani.hierarchy_20261030 import actor as a
from our_mani.hierarchy_20261030.evaluation import validate_decision
from our_mani.hierarchy_20261030.evaluation import pointer


class TestEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, rgbd, robots, lengths):
        self.calls.append((rgbd.clone(), robots.clone(), lengths.clone()))
        current = robots[:, lengths.item() - 1, :1]
        e = current.expand(1, 64).clone()
        logits = torch.zeros(1, 11)
        logits[:, int(current.item() * 20) % 11] = 1.
        return {"scene_features": current.expand(1, 128).clone(), "skill_latent": e, "skill_logits": logits}


class TestCandidate(nn.Module):
    role_names = a.ROLES

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, trajectory, mask, e, robot, scene):
        self.calls.append({"e": e, "mask": mask.clone(), "trajectory": trajectory.clone()})
        return {"goal_xyz": torch.tensor([[.1, .2, .3]]) + e[:, :3], "goal_time": torch.tensor([.9])}


class TestSkill(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, scene, e, robot, goal, goal_time, trajectory=None, trajectory_mask=None):
        assert trajectory is None and trajectory_mask is None
        self.calls.append({"e": e, "goal": goal.clone(), "goal_time": goal_time.clone(), "robot": robot.clone()})
        action = torch.cat((robot[:, :3] + e[:, :3] * .1, torch.tensor([[0., 0., 0., 1., 1.]])), -1)
        return {"action": action, "trajectory_consumed": False}


class TestMotion:
    roles = a.ROLES

    def __init__(self):
        self.calls = []
        self.available = True

    def reset(self):
        self.calls.clear()

    def observe(self, observation, robot):
        assert "observation/instance_masks" not in observation
        assert "observation/sensor_protocol" not in observation
        self.calls.append((observation["timestamp"], robot.copy()))
        tokens = np.zeros((23, 20), np.float32)
        tokens[:, -1] = np.arange(23, dtype=np.float32) * np.float32(.05)
        forecasts = {"target": {"trajectory": tokens, "observed_until": observation["timestamp"]}} if self.available else {}
        return {"timestamp": observation["timestamp"], "motion": {"forecasts": forecasts,
            "status": {"target": {"available": self.available}}, "checkpoint_sha256": "f" * 64,
            "native_or_expert_poses_used": False, "trajectories_fabricated": False}}


def observation(t, color=40):
    return {"timestamp": t,
        "observation/images": {camera: np.full((256, 256, 3), color, np.uint8) for camera in a.CAMERAS},
        "observation/depths": {camera: np.full((256, 256), .8, np.float32) for camera in a.CAMERAS},
        "observation/calibrations": {camera: CameraCalibration(np.eye(3), Pose(np.zeros(3))) for camera in a.CAMERAS},
        "observation/sensor_protocol": {"version": "test_public_camera_metadata"}}


def robot(t):
    return RobotState(t, Pose(np.array([t, .2, .8])), np.zeros(3), np.zeros(3), np.arange(7) / 10,
                      np.zeros(7), True, False, True)


class ActorTests(unittest.TestCase):
    @staticmethod
    def build():
        encoder, candidate, skill, motion = TestEncoder(), TestCandidate(), TestSkill(), TestMotion()
        instance = a.LearnedHierarchyActor(encoder, candidate, skill, motion,
            model_identity={"forecaster_sha256": "f" * 64})
        return instance, encoder, candidate, skill, motion

    @staticmethod
    def warmup(instance):
        for index in range(5):
            t = index * .05
            instance.observe(observation(t), robot(t))

    def test_exact_preprocessing_and_robot15_match_training(self):
        obs = observation(.2)
        yy, xx = np.indices((256, 256))
        rgb = np.stack((xx, yy, (xx + yy) % 256), -1).astype(np.uint8)
        depth = (.3 + (xx + yy) / 500).astype(np.float32)
        obs["observation/images"]["front"] = rgb
        obs["observation/depths"]["front"] = depth
        timestamp, encoded = a.encode_public_frame(obs)
        expected_rgb = np.asarray(Image.fromarray(rgb).resize((64, 64), Image.Resampling.BILINEAR)).transpose(2, 0, 1) / 255. - .5
        expected_depth = np.log1p(np.asarray(Image.fromarray(depth).resize((64, 64), Image.Resampling.NEAREST)))
        np.testing.assert_array_equal(encoded[0, :3], expected_rgb.astype(np.float32))
        np.testing.assert_array_equal(encoded[0, 3], expected_depth)
        measured = a.robot_vector(robot(.2), .2)
        np.testing.assert_array_equal(measured, np.r_[robot(.2).pose.xyzw(), 1., np.arange(7) / 10].astype(np.float32))
        self.assertEqual(timestamp, .2)

    def test_handoff_same_exact_e_two_consumers_and_fixed_half_second_goal(self):
        actor, encoder, candidate, skill, motion = self.build()
        self.warmup(actor)
        decision = actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        validate_decision(decision, timestamp=.2)
        self.assertIs(candidate.calls[0]["e"], skill.calls[0]["e"])
        self.assertEqual(decision["receipt"]["event"], "candidate_handoff")
        self.assertEqual(decision["receipt"]["forecast_generated_at"], .2)
        self.assertEqual(len(motion.calls), 5)
        self.assertEqual(encoder.calls[0][2].tolist(), [5])
        self.assertEqual(skill.calls[0]["goal_time"].tolist(), [.5])
        self.assertEqual(candidate.calls[0]["mask"].shape, (1, 6, 23))
        self.assertTrue(bool(candidate.calls[0]["mask"][0, 0].all()))
        self.assertFalse(bool(candidate.calls[0]["mask"][0, 1:].any()))
        self.assertEqual(decision["receipt"]["goal_source"], "predicted_candidate_episode_handoff")

    def test_goal_remains_fixed_across_skill_changes_and_missing_closed_body_forecast(self):
        actor, encoder, candidate, skill, motion = self.build()
        self.warmup(actor)
        first = actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        motion.available = False
        motion.observe = lambda *_: (_ for _ in ()).throw(AssertionError("unused tracking called after skill takeover"))
        second = actor.act(observation(.4, 120), robot(.4), remaining_seconds=9.6)
        validate_decision(second, timestamp=.4)
        self.assertNotEqual(first["receipt"]["inferred_skill_id"], second["receipt"]["inferred_skill_id"])
        self.assertEqual(first["receipt"]["goal_xyz"], second["receipt"]["goal_xyz"])
        self.assertEqual(first["receipt"]["goal_selected_at"], second["receipt"]["goal_selected_at"])
        self.assertEqual(len(candidate.calls), 1)
        self.assertEqual(len(skill.calls), 2)
        self.assertFalse(second["receipt"]["real_forecast_consumed"])
        self.assertIsNone(second["receipt"]["candidate_skill_latent_sha256"])
        self.assertIsNone(second["receipt"]["forecast_generated_at"])
        self.assertFalse(second["receipt"]["public_tracks_valid"])
        self.assertIsNone(actor.latest_motion)
        self.assertFalse(np.array_equal(first["action"], second["action"]))

    def test_unknown_before_real_history_or_forecast_never_calls_candidate_or_skill(self):
        actor, encoder, candidate, skill, motion = self.build()
        initial = actor.act(observation(0.), robot(0.), remaining_seconds=10.)
        self.assertIsNone(initial["action"])
        self.assertIn("five_causal", initial["receipt"]["reason"])
        for index in range(1, 5):
            actor.observe(observation(index * .05), robot(index * .05))
        motion.available = False
        missing = actor.act(observation(.25), robot(.25), remaining_seconds=9.75)
        validate_decision(missing, timestamp=.25)
        self.assertIsNone(missing["action"])
        self.assertEqual(len(candidate.calls), 0)
        self.assertEqual(len(skill.calls), 0)
        self.assertIsNone(actor.goal_xyz)

    def test_future_frames_and_duplicate_timestamps_cannot_change_past_decision(self):
        left, _, _, _, lm = self.build()
        right, _, _, _, rm = self.build()
        self.warmup(left); self.warmup(right)
        right.observe(observation(.2, 220), robot(.2))
        first = left.act(observation(.2), robot(.2), remaining_seconds=9.8)
        second = right.act(observation(.2), robot(.2), remaining_seconds=9.8)
        np.testing.assert_array_equal(first["action"], second["action"])
        self.assertEqual(len(lm.calls), 5)
        self.assertEqual(len(rm.calls), 5)
        with self.assertRaisesRegex(a.ActorContractError, "backwards"):
            right.observe(observation(.1), robot(.1))
        left.observe(observation(.3, 50), robot(.3))
        np.testing.assert_array_equal(first["action"], second["action"])

    def test_gt_task_and_instance_mask_channels_are_rejected(self):
        actor, *_ = self.build()
        for forbidden in ("task_id", "expert_debug", "observation/instance_masks", "expert_stage"):
            obs = observation(0.); obs[forbidden] = 1
            with self.assertRaisesRegex(a.ActorContractError, "only causal public"):
                actor.observe(obs, robot(0.))

    def test_cached_old_forecast_refused_at_first_handoff(self):
        actor, *_ = self.build()
        self.warmup(actor)
        actor.latest_motion["motion"]["forecasts"]["target"]["observed_until"] = .15
        with self.assertRaisesRegex(a.ActorContractError, "cached older"):
            actor.act(observation(.2), robot(.2), remaining_seconds=9.8)

    def test_missing_real_checkpoint_refuses_factory_before_any_native_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(a.EvaluationBlocked):
                a.create_actor(root=root, freeze_manifest={}, device="cpu")
            manifest = {"protocol": {"inference_device": "cpu"},
                        "components": {"skill_policy": {"artifact": {"path": "absent_skill.pt", "sha256": "0" * 64}}}}
            with patch.object(a, "validate_freeze", return_value=manifest), patch.object(torch, "load", side_effect=AssertionError("missing artifact was loaded")):
                with self.assertRaises(FileNotFoundError):
                    a.create_actor(root=root, freeze_manifest=manifest, device="cpu")

    def test_production_freeze_validation_refuses_a_real_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = {"path": "missing_actual_checkpoint.pt", "sha256": "0" * 64}
            manifest = {"schema": "hierarchy_runtime_freeze_v1_20261030", "frozen": True,
                "tasks": ["pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed",
                          "insert_onto_rotating_peg", "insert_onto_rotating_peg_high_speed"],
                "readiness": {key: True for key in ("scene_classifier_trained", "skill_policy_trained", "candidate_trained", "public_tracking_ready", "forecaster_ready")},
                "components": {"scene_classifier": {"artifact": missing, "evidence": missing, "trained": True}},
                "protocol": {"inference_device": "cpu"}}
            # Exercise the actual production validator, not a patched checker.
            with patch.object(torch, "load", side_effect=AssertionError("missing checkpoint was loaded")):
                with self.assertRaises(FileNotFoundError):
                    a.create_actor(root=root, freeze_manifest=manifest, device="cpu")

    def test_serialized_unfinished_candidate_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill_path, candidate_path = root / "skill.pt", root / "candidate.pt"
            torch.save({"version": a.SKILL_VERSION, "encoder_state_dict": {"fixture": torch.ones(1)},
                "skill_state_dict": {"fixture": torch.ones(1)}, "metadata": {"stage": "skill", "smoke_only": False,
                "goal_latched_for_whole_episode": True, "goal_time_seconds": .5,
                "skill_policy_trajectory_consumed": False, "scene_input_uses_expert_labels": False,
                "workspace_center": [0., 0., .8], "workspace_half_width": [1., 1., 1.]}}, skill_path)
            candidate_metadata = {"stage": "candidate", "training_complete": False,
                "role_names": list(a.ROLES), "skill_checkpoint_sha256": a.sha(skill_path),
                "fixed_goal_time_seconds": .5, "time_target_index": 10, "score_semantics": a.SCORE_SEMANTICS}
            torch.save({"version": a.CANDIDATE_VERSION, "candidate_state_dict": {"fixture": torch.ones(1)},
                        "metadata": candidate_metadata}, candidate_path)
            for name in ("forecaster.pt", "segmentation.pt"):
                (root / name).write_bytes(b"unopened fixture")
            manifest = {"protocol": {"inference_device": "cpu"}, "components": {
                "scene_classifier": {"artifact": pointer(root, skill_path)},
                "skill_policy": {"artifact": pointer(root, skill_path)},
                "candidate_network": {"artifact": pointer(root, candidate_path)},
                "forecaster": {"artifact": pointer(root, root / "forecaster.pt")},
                "segmentation": {"artifact": pointer(root, root / "segmentation.pt")}}}
            with patch.object(a, "validate_freeze", return_value=manifest), patch.object(a, "LearnedHierarchyActor", side_effect=AssertionError("unfinished candidate reached runtime")):
                with self.assertRaisesRegex(a.EvaluationBlocked, "completed real six-body candidate training"):
                    a.create_actor(root=root, freeze_manifest=manifest, device="cpu")


if __name__ == "__main__":
    unittest.main()
