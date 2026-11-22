"""Source-blind causal inference with one learned episode handoff goal.

Production construction verifies completed, SHA-bound checkpoints. The public
constructor takes explicit modules to support CPU dataflow tests; it never
creates substitute models or trajectories when an artifact is unavailable.
"""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .evaluation import ActorContractError, EvaluationBlocked, checked, sha, validate_freeze
from .networks import CausalSceneSkillEncoder, PretrainedSkillPolicy
from .candidate_multibody import MultiBodyCandidateTransformer


SKILL_VERSION = "hierarchy_episode_handoff_skill_pretraining_20261030_v2"
CANDIDATE_VERSION = "hierarchy_multibody_candidate_pretraining_20261030_v1"
ROLES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
CAMERAS = ("front", "8", "16", "36")
HISTORY_FRAMES = 5
GOAL_TIME_SECONDS = .5
SCORE_SEMANTICS = "expert_handoff_point_imitation_probability_not_native_success_probability"
PUBLIC_KEYS = frozenset(("timestamp", "observation/images", "observation/depths", "observation/calibrations",
                         "observation/camera_timestamps", "observation/sensor_protocol"))
REQUIRED_KEYS = frozenset(("timestamp", "observation/images", "observation/depths", "observation/calibrations"))


def tensor_sha256(value):
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def robot_vector(robot, timestamp):
    """Match data_v1: measured TCP xyzw7, open scalar1, joint positions7."""
    if abs(float(robot.timestamp) - timestamp) > 1e-8:
        raise ActorContractError("robot and public RGBD timestamps differ")
    joints = np.asarray(robot.joints, dtype=np.float32)
    if joints.shape != (7,):
        raise ActorContractError("seven measured robot joint positions required")
    value = np.r_[robot.pose.xyzw(), float(bool(robot.gripper_open)), joints].astype(np.float32)
    if value.shape != (15,) or not np.isfinite(value).all():
        raise ActorContractError("finite measured TCP7/gripper1/joints7 required")
    return value


def encode_public_frame(observation):
    """Exactly data.py resizing followed by training.py RGB/log-depth features."""
    if set(observation) - PUBLIC_KEYS or not REQUIRED_KEYS <= set(observation):
        raise ActorContractError("only causal public RGBD/calibration channels may enter the actor")
    timestamp = float(observation["timestamp"])
    if not np.isfinite(timestamp):
        raise ActorContractError("finite public observation timestamp required")
    images, depths = observation["observation/images"], observation["observation/depths"]
    if set(images) != set(CAMERAS) or set(depths) != set(CAMERAS):
        raise ActorContractError("exact four-camera public RGBD inventory required")
    result = np.empty((4, 4, 64, 64), dtype=np.float32)
    for index, camera in enumerate(CAMERAS):
        rgb = np.asarray(images[camera])
        depth = np.asarray(depths[camera], dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3) or depth.shape != (256, 256):
            raise ActorContractError("original 256x256 uint8 RGB and metric depth required")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ActorContractError("finite nonnegative metric depth required")
        small_rgb = np.asarray(Image.fromarray(rgb).resize((64, 64), Image.Resampling.BILINEAR))
        small_depth = np.asarray(Image.fromarray(depth).resize((64, 64), Image.Resampling.NEAREST), dtype=np.float32)
        result[index, :3] = small_rgb.transpose(2, 0, 1) / 255.0 - .5
        result[index, 3] = np.log1p(small_depth)
    return timestamp, result


class LearnedHierarchyActor:
    def __init__(self, encoder, candidate, skill_policy, motion_pipeline, *, device="cpu", model_identity=None):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.candidate = candidate.to(self.device).eval()
        self.skill_policy = skill_policy.to(self.device).eval()
        if tuple(candidate.role_names) != ROLES or tuple(motion_pipeline.roles) != ROLES:
            raise ActorContractError("fixed source-blind six-role forecast order required")
        self.motion_pipeline = motion_pipeline
        self.model_identity = dict(model_identity or {})
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.last_timestamp = None
        self.latest_motion = None
        self.goal_xyz = None
        self.goal_time = None
        self.goal_selected_at = None
        self.closed = False
        self.motion_pipeline.reset()

    def observe(self, observation, robot):
        if self.closed:
            raise ActorContractError("closed actor requires a new episode instance")
        timestamp, rgbd = encode_public_frame(observation)
        measured_robot = robot_vector(robot, timestamp)
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp:
                raise ActorContractError("public observation time moved backwards")
            if timestamp == self.last_timestamp:
                return
        if self.goal_xyz is None:
            # Sensor protocol metadata is not an input to the public tracker.
            tracker_observation = {key: value for key, value in observation.items()
                                   if key != "observation/sensor_protocol"}
            motion = self.motion_pipeline.observe(tracker_observation, measured_robot.copy())
            if float(motion["timestamp"]) != timestamp:
                raise ActorContractError("public motion pipeline returned stale or future scene evidence")
        else:
            # The fixed goal has handed control to the feedback skill network.
            # No unused ICP/GRU work and no stale prediction reported as current.
            motion = None
        self.history.append((timestamp, rgbd.copy(), measured_robot.copy()))
        self.latest_motion = motion
        self.last_timestamp = timestamp

    def _history_tensors(self):
        length = len(self.history)
        images = np.zeros((1, HISTORY_FRAMES, 4, 4, 64, 64), dtype=np.float32)
        robots = np.zeros((1, HISTORY_FRAMES, 15), dtype=np.float32)
        for index, (_, rgbd, measured_robot) in enumerate(self.history):
            images[0, index] = rgbd
            robots[0, index] = measured_robot
        return (torch.from_numpy(images).to(self.device), torch.from_numpy(robots).to(self.device),
                torch.tensor([length], dtype=torch.long, device=self.device))

    def _unknown(self, reason):
        return {"action": None, "receipt": {"event": "perception_unknown", "reason": reason,
            "public_tracks_valid": False, "real_forecast_consumed": False,
            "forecast_generated_at": None, "old_stage_rule_calls": 0, "task_metadata_used": False,
            "source_task_or_speed_used": False, "native_labels_consumed": False,
            "current_timestamp": self.last_timestamp, "history_frames": len(self.history),
            "motion_status": {} if self.latest_motion is None else self.latest_motion["motion"].get("status", {})}}

    def _forecast_tensors(self):
        motion = self.latest_motion["motion"]
        if motion.get("native_or_expert_poses_used") is not False or motion.get("trajectories_fabricated") is not False:
            raise ActorContractError("public motion provenance does not exclude privileged or fabricated tracks")
        expected = self.model_identity.get("forecaster_sha256")
        if expected is not None and motion.get("checkpoint_sha256") != expected:
            raise ActorContractError("motion checkpoint differs from the frozen actor identity")
        forecast = motion.get("forecasts", {})
        if set(forecast) - set(ROLES):
            raise ActorContractError("unknown forecast role in the fixed blind registry")
        # Zeros are explicit masked padding for unavailable bodies, never an
        # asserted predicted trajectory or a substitute forecast.
        trajectory = np.zeros((1, len(ROLES), 23, 20), dtype=np.float32)
        mask = np.zeros((1, len(ROLES), 23), dtype=bool)
        consumed_roles = []
        expected_times = np.arange(23, dtype=np.float32) * np.float32(.05)
        for index, role in enumerate(ROLES):
            if role not in forecast:
                continue
            entry = forecast[role]
            tokens = np.asarray(entry["trajectory"], dtype=np.float32)
            if (tokens.shape != (23, 20) or not np.isfinite(tokens).all()
                    or not np.allclose(tokens[:, -1], expected_times, rtol=0, atol=1e-6)):
                raise ActorContractError("invalid per-body real forecast shape/clock")
            if abs(float(entry["observed_until"]) - self.last_timestamp) > 1e-8:
                raise ActorContractError("candidate handoff cannot consume a cached older body forecast")
            if motion.get("status", {}).get(role, {}).get("available") is not True:
                raise ActorContractError("forecast availability differs from the actual public motion status")
            trajectory[0, index] = tokens
            mask[0, index] = True
            consumed_roles.append(role)
        if not consumed_roles:
            return None
        return (torch.from_numpy(trajectory).to(self.device), torch.from_numpy(mask).to(self.device), consumed_roles)

    @torch.inference_mode()
    def act(self, observation, robot, *, remaining_seconds):
        if not np.isfinite(remaining_seconds) or not 0 < remaining_seconds <= 10.0 + 1e-8:
            raise ActorContractError("positive remaining time inside the fixed ten-second budget required")
        self.observe(observation, robot)  # Existing timestamp is deduplicated.
        if len(self.history) < HISTORY_FRAMES:
            return self._unknown("fewer_than_five_causal_scene_frames")
        handoff = self.goal_xyz is None
        forecast = None
        if handoff:
            forecast = self._forecast_tensors()
            if forecast is None:
                return self._unknown("no_current_real_body_forecast")
        images, robots, lengths = self._history_tensors()
        scene = self.encoder(images, robots, lengths)
        # This exact Tensor object enters both consumers at handoff. Subsequent
        # steps recompute it from current feedback without reopening the goal.
        shared_e = scene["skill_latent"]
        shared_sha = tensor_sha256(shared_e)
        current_robot = robots[:, len(self.history) - 1]
        if handoff:
            tokens, mask, roles = forecast
            proposed = self.candidate(tokens, mask, shared_e, current_robot, scene["scene_features"])
            goal_xyz = proposed["goal_xyz"].detach().clone()
            if goal_xyz.shape != (1, 3) or not bool(torch.isfinite(goal_xyz).all()):
                raise ActorContractError("learned candidate must supply a finite TCP goal xyz")
            # Both candidate supervision and the v2 skill policy freeze 0.5s.
            goal_time = torch.full((1,), GOAL_TIME_SECONDS, dtype=torch.float32, device=self.device)
        else:
            roles = []
            goal_xyz, goal_time = self.goal_xyz, self.goal_time
        # v2 never trained the optional trajectory encoder: do not invoke it.
        skill = self.skill_policy(scene["scene_features"], shared_e, current_robot, goal_xyz, goal_time)
        if skill.get("trajectory_consumed") is not False:
            raise ActorContractError("v2 skill policy cannot consume its untrained trajectory branch")
        action = skill["action"].detach().to(device="cpu", dtype=torch.float32).numpy()
        if (action.shape != (1, 8) or not np.isfinite(action).all()
                or not np.isclose(np.linalg.norm(action[0, 3:7]), 1., atol=1e-5)
                or action[0, 7] not in (0., 1.)):
            raise ActorContractError("learned skill policy returned an invalid native action")
        if handoff:
            self.goal_xyz, self.goal_time = goal_xyz, goal_time
            self.goal_selected_at = self.last_timestamp
        receipt = {"event": "candidate_handoff" if handoff else "skill_execution",
            "public_tracks_valid": bool(handoff and self.latest_motion is not None
                                        and self.latest_motion["motion"].get("forecasts")),
            "real_forecast_consumed": handoff, "forecast_generated_at": self.last_timestamp if handoff else None,
            "forecast_roles_consumed": roles, "shared_skill_latent_sha256": shared_sha,
            "candidate_skill_latent_sha256": shared_sha if handoff else None,
            "policy_skill_latent_sha256": shared_sha, "old_stage_rule_calls": 0,
            "task_metadata_used": False, "source_task_or_speed_used": False,
            "native_labels_consumed": False, "inferred_skill_id": int(scene["skill_logits"].argmax(-1).item()),
            "goal_source": "predicted_candidate_episode_handoff", "episode_goal_fixed": True,
            "goal_xyz": self.goal_xyz[0].detach().cpu().tolist(),
            "goal_time_seconds": GOAL_TIME_SECONDS, "goal_selected_at": self.goal_selected_at,
            "candidate_score_semantics": SCORE_SEMANTICS, "skill_trajectory_consumed": False,
            "current_timestamp": self.last_timestamp, "history_frames": len(self.history)}
        return {"action": action[0].copy(), "receipt": receipt}

    def close(self):
        if not self.closed:
            self.history.clear()
            if callable(getattr(self.motion_pipeline, "close", None)):
                self.motion_pipeline.close()
            self.closed = True


def create_actor(*, root, freeze_manifest, device="cpu"):
    """Evaluation factory: every real dependency must exist and be trained."""
    root = Path(root).resolve()
    validate_freeze(root, freeze_manifest)
    if freeze_manifest.get("protocol", {}).get("inference_device") != "cpu" or torch.device(device) != torch.device("cpu"):
        raise EvaluationBlocked("this runtime freeze requires explicit CPU inference; no device fallback is allowed")
    components = freeze_manifest["components"]
    skill_path = checked(root, components["skill_policy"]["artifact"])
    scene_path = checked(root, components["scene_classifier"]["artifact"])
    if sha(scene_path) != sha(skill_path):
        raise EvaluationBlocked("the v2 best_skill checkpoint must supply both runtime encoder/E and skill weights")
    candidate_path = checked(root, components["candidate_network"]["artifact"])
    forecaster_path = checked(root, components["forecaster"]["artifact"])
    segmentation = components.get("segmentation", {})
    segment_path = checked(root, segmentation.get("artifact"))
    skill = torch.load(skill_path, map_location="cpu", weights_only=True)
    metadata = skill.get("metadata", {})
    if (skill.get("version") != SKILL_VERSION or metadata.get("stage") != "skill"
            or metadata.get("smoke_only") is not False or metadata.get("goal_latched_for_whole_episode") is not True
            or metadata.get("goal_time_seconds") != GOAL_TIME_SECONDS
            or metadata.get("skill_policy_trajectory_consumed") is not False
            or metadata.get("scene_input_uses_expert_labels") is not False
            or not skill.get("encoder_state_dict") or not skill.get("skill_state_dict")):
        raise EvaluationBlocked("completed v2 episode-handoff encoder/skill checkpoint required")
    candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
    candidate_metadata = candidate_payload.get("metadata", {})
    if (candidate_payload.get("version") != CANDIDATE_VERSION
            or candidate_metadata.get("training_complete") is not True
            or candidate_metadata.get("stage") != "candidate"
            or tuple(candidate_metadata.get("role_names", [])) != ROLES
            or candidate_metadata.get("skill_checkpoint_sha256") != sha(skill_path)
            or candidate_metadata.get("fixed_goal_time_seconds") != GOAL_TIME_SECONDS
            or candidate_metadata.get("time_target_index") != 10
            or candidate_metadata.get("score_semantics") != SCORE_SEMANTICS
            or not candidate_payload.get("candidate_state_dict")):
        raise EvaluationBlocked("completed real six-body candidate training bound to this v2 encoder/E is required")
    center = np.asarray(metadata.get("workspace_center"), dtype=np.float32)
    scale = np.asarray(metadata.get("workspace_half_width"), dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise EvaluationBlocked("valid checkpoint workspace metadata required")
    if (not np.array_equal(center, np.asarray(candidate_metadata.get("workspace_center"), dtype=np.float32))
            or not np.array_equal(scale, np.asarray(candidate_metadata.get("workspace_half_width"), dtype=np.float32))):
        raise EvaluationBlocked("candidate and skill checkpoint workspaces differ")
    encoder = CausalSceneSkillEncoder()
    policy = PretrainedSkillPolicy(center.tolist(), scale.tolist())
    candidate = MultiBodyCandidateTransformer(center.tolist(), scale.tolist(), role_names=ROLES)
    encoder.load_state_dict(skill["encoder_state_dict"], strict=True)
    policy.load_state_dict(skill["skill_state_dict"], strict=True)
    candidate.load_state_dict(candidate_payload["candidate_state_dict"], strict=True)
    for module in (encoder, policy, candidate):
        module.requires_grad_(False)
    from .tracking import BlindPublicMotionPipeline
    pipeline = BlindPublicMotionPipeline(root, segment_checkpoint=segment_path, motion_checkpoint=forecaster_path)
    if pipeline.checkpoint_sha256 != sha(forecaster_path):
        raise EvaluationBlocked("blind public pipeline forecaster does not match the freeze")
    return LearnedHierarchyActor(encoder, candidate, policy, pipeline, device=device,
        model_identity={"skill_checkpoint_sha256": sha(skill_path), "candidate_checkpoint_sha256": sha(candidate_path),
                        "forecaster_sha256": sha(forecaster_path), "segmentation_sha256": sha(segment_path)})
