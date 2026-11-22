"""CPU-only causal actor contracts; explicit test modules never enter production."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from our_mani.contracts import Pose, RobotState
from our_mani.perception import CameraCalibration
from our_mani.hierarchy_block_20261102 import actor_v2 as a
from our_mani.hierarchy_block_20261102.data import encode_public_history, geometry_points
from our_mani.hierarchy_block_20261102.evaluation_v2 import pointer, validate_decision


class ExplicitTestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodings, self.candidates, self.skills = [], [], []

    def encode(self, rgbd, robots, lengths, points, mask):
        current = robots[:, lengths.item() - 1, :1]
        encoded = {"scene_features": current.expand(1, 128).clone(),
            "skill_latent": current.expand(1, 64).clone(), "skill_logits": torch.cat((current, -current, current*0, current*0), -1)}
        self.encodings.append(dict(encoded=encoded, rgbd=rgbd.clone(), robots=robots.clone(),
                                   points=points.clone(), mask=mask.clone()))
        return encoded

    def forward_candidate(self, encoded, robot, trajectory, mask):
        self.candidates.append(dict(encoded=encoded, e=encoded["skill_latent"], mask=mask.clone()))
        return dict(goal_position=torch.tensor([[.1, .2, .8]]), goal_rotation=torch.eye(3)[None],
                    goal_quaternion=torch.tensor([[0., 0., 0., 1.]]), goal_time=torch.tensor([2.8]))

    def forward_skill(self, encoded, robot, goal, rotation, time):
        self.skills.append(dict(encoded=encoded, e=encoded["skill_latent"], goal=goal.clone(),
                               rotation=rotation.clone(), time=time.clone()))
        return dict(e=encoded["skill_latent"])

    def decode_skill(self, robot, policy):
        action = torch.cat((robot[:, :3] + policy["e"][:, :3] * .1,
                            torch.tensor([[0., 0., 0., 1., 1.]])), -1)
        actions = action[:, None].repeat(1, 4, 1)
        actions[:, 1:, :3] += 1.  # Only the first chunk may be requested.
        return actions


class ExplicitTestMotion:
    roles = a.ROLES

    def __init__(self):
        self.calls, self.available = [], True
        self.role = "target"

    def reset(self):
        self.calls.clear()

    def observe(self, observation, robot):
        assert "observation/sensor_protocol" not in observation
        self.calls.append(observation["timestamp"])
        tokens = np.zeros((23, 20), np.float32)
        tokens[:, -1] = np.arange(23, dtype=np.float32)*np.float32(.05)
        forecasts = {self.role: {"trajectory": tokens, "observed_until": observation["timestamp"]}} if self.available else {}
        return {"timestamp": observation["timestamp"], "motion": {"forecasts": forecasts,
            "status": {self.role: {"available": self.available}}, "checkpoint_sha256": "f"*64,
            "native_or_expert_poses_used": False, "trajectories_fabricated": False}}


def observation(t, color=40, depth=.8):
    return {"timestamp": t,
        "observation/images": {camera: np.full((256,256,3), color, np.uint8) for camera in a.CAMERAS},
        "observation/depths": {camera: np.full((256,256), depth, np.float32) for camera in a.CAMERAS},
        "observation/calibrations": {camera: CameraCalibration(np.eye(3), Pose(np.zeros(3))) for camera in a.CAMERAS}}


def robot(t):
    return RobotState(t, Pose(np.array([t,.2,.8])), np.zeros(3), np.zeros(3), np.arange(7)/10,
                      np.zeros(7), True, False, True)


class BlockActorTests(unittest.TestCase):
    def build(self):
        model, motion = ExplicitTestNetwork(), ExplicitTestMotion()
        return a.BlockHierarchyActor(model, motion, model_identity={"forecaster_sha256": "f"*64}), model, motion

    @staticmethod
    def warmup(actor):
        for i in range(5):
            actor.observe(observation(i*.05), robot(i*.05))

    def test_handoff_exact_shared_tensor_first_chunk_and_unclipped_goal_time(self):
        actor, model, motion = self.build(); self.warmup(actor)
        result = actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        validate_decision(result, timestamp=.2)
        self.assertIs(model.candidates[0]["e"], model.skills[0]["e"])
        self.assertIs(model.candidates[0]["encoded"], model.skills[0]["encoded"])
        self.assertAlmostEqual(result["receipt"]["goal_time_seconds"], 2.8, places=6)
        self.assertAlmostEqual(model.skills[0]["time"].item(), 2.8, places=6)
        np.testing.assert_allclose(result["action"][:3], [.22,.22,.82])
        self.assertEqual(len(motion.calls), 5)
        self.assertEqual(result["receipt"]["executed_action_chunk_index"], 0)
        self.assertAlmostEqual(result["receipt"]["request_position_error_m"], np.sqrt(3)*.02, places=6)

    def test_fixed_pose_countdown_and_latest_geometry_feedback_without_reopening_forecast(self):
        actor, model, motion = self.build(); self.warmup(actor)
        first = actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        motion.observe = lambda *_: (_ for _ in ()).throw(AssertionError("post-handoff tracker called"))
        second = actor.act(observation(.7, 120, .9), robot(.7), remaining_seconds=9.3)
        validate_decision(second, timestamp=.7)
        self.assertEqual(first["receipt"]["goal_xyz"], second["receipt"]["goal_xyz"])
        self.assertEqual(first["receipt"]["goal_quaternion_xyzw"], second["receipt"]["goal_quaternion_xyzw"])
        self.assertAlmostEqual(second["receipt"]["deadline_remaining_seconds"], 2.3, places=6)
        self.assertEqual(second["receipt"]["goal_command_time_semantics"], "remaining_until_first_closed_advice")
        self.assertEqual(second["receipt"]["expert_grasp_lookahead_seconds"], .5)
        self.assertAlmostEqual(second["receipt"]["implied_object_target_remaining_seconds"], 2.8, places=6)
        self.assertFalse(second["receipt"]["time_is_reachability_guarantee"])
        self.assertEqual(len(model.candidates), 1)
        self.assertFalse(torch.equal(model.encodings[0]["points"], model.encodings[1]["points"]))
        self.assertFalse(second["receipt"]["real_forecast_consumed"])
        third = actor.act(observation(3.2), robot(3.2), remaining_seconds=6.8)
        self.assertEqual(third["receipt"]["deadline_remaining_seconds"], 0.)

    def test_peg_forecast_cannot_substitute_for_missing_target_or_short_history(self):
        actor, model, motion = self.build()
        result = actor.act(observation(0.), robot(0.), remaining_seconds=10.)
        self.assertIsNone(result["action"])
        motion.role = "peg0"; self.warmup(actor)
        result = actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        self.assertIsNone(result["action"])
        self.assertIn("target", result["receipt"]["reason"])
        self.assertEqual(len(model.candidates), 0)
        self.assertEqual(len(model.skills), 0)

    def test_shared_offline_geometry_and_window_encoding_are_exact(self):
        actor, model, _ = self.build(); self.warmup(actor)
        actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        rgb = np.full((5,4,64,64,3), 40, np.uint8)
        depth = np.full((5,4,64,64), .8, np.float32)
        expected = encode_public_history(rgb, depth, np.stack([a.robot_vector(robot(i*.05), i*.05) for i in range(5)]))
        np.testing.assert_array_equal(model.encodings[0]["rgbd"][0].numpy(), expected["rgbd"])
        points, mask = geometry_points(rgb[-1], depth[-1], np.repeat(np.eye(3)[None],4,0), np.repeat(np.eye(4)[None],4,0))
        np.testing.assert_array_equal(model.encodings[0]["points"][0].numpy(), points)
        np.testing.assert_array_equal(model.encodings[0]["mask"][0].numpy(), mask)

    def test_unknown_geometry_and_stale_forecast_never_generate_action(self):
        actor, model, _ = self.build(); self.warmup(actor)
        result = actor.act(observation(.25, depth=0.), robot(.25), remaining_seconds=9.75)
        self.assertIsNone(result["action"])
        self.assertIn("calibrated", result["receipt"]["reason"])
        self.assertEqual(model.encodings, [])
        actor.observe(observation(.3), robot(.3))
        actor.latest_motion["motion"]["forecasts"]["target"]["observed_until"] = .25
        with self.assertRaisesRegex(a.ActorContractError, "cached older"):
            actor.act(observation(.3), robot(.3), remaining_seconds=9.7)

    def test_task_labels_and_privileged_channels_rejected_and_duplicates_causal(self):
        actor, model, _ = self.build()
        for forbidden in ("task_id", "speed", "expert_stage", "observation/instance_masks", "observation/state"):
            obs = observation(0.); obs[forbidden] = "forbidden"
            with self.assertRaisesRegex(a.ActorContractError, "only causal public"):
                actor.observe(obs, robot(0.))
        self.warmup(actor)
        actor.observe(observation(.2, 220), robot(.2))
        actor.act(observation(.2), robot(.2), remaining_seconds=9.8)
        self.assertAlmostEqual(model.encodings[0]["rgbd"][0,-1,0,0,0,0].item(), 40/255-.5)
        with self.assertRaisesRegex(a.ActorContractError, "backwards"):
            actor.observe(observation(.1), robot(.1))

    def test_real_missing_or_unfinished_checkpoint_refused_before_pipeline_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(a.EvaluationBlocked):
                a.create_actor(root=root, freeze_manifest={})
            path = root/"incomplete.pt"
            torch.save({"version": a.VERSION, "model_state_dict": {"test": torch.tensor(1)},
                        "metadata": {"training_complete": False}}, path)
            manifest = {"components": {name: {"artifact": pointer(root,path)} for name in ("scene_classifier","candidate_network","skill_policy")}}
            with patch.object(a, "validate_freeze"), patch("our_mani.hierarchy_20261030.tracking.BlindPublicMotionPipeline", side_effect=AssertionError("untrained pipeline started")):
                with self.assertRaisesRegex(a.EvaluationBlocked,"120-epoch"):
                    a.create_actor(root=root, freeze_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
