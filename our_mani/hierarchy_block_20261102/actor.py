"""Task-blind block inference with one goal-pose handoff and feedback skills.

The constructor accepts explicit test modules; the production factory requires
complete SHA-bound real checkpoints. No task/speed/stage/native label enters
model inputs. Geometry is rebuilt from every latest public RGBD frame.
"""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path

import numpy as np
import torch

from .evaluation import ActorContractError, EvaluationBlocked, checked, sha, validate_freeze
from .data import geometry_points, resize_public_rgbd, encode_public_history

VERSION = "hierarchy_block_training_20261102_v1"
ROLES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
CAMERAS = ("front", "8", "16", "36")
HISTORY_FRAMES = 5
SCORE_SEMANTICS = "expert_joint_handoff_pose_time_imitation_not_calibrated_success"
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


def cached_public_frame(observation):
    if set(observation) - PUBLIC_KEYS or not REQUIRED_KEYS <= set(observation):
        raise ActorContractError("only causal public RGBD/calibration channels may enter the actor")
    timestamp = float(observation["timestamp"])
    if not np.isfinite(timestamp):
        raise ActorContractError("finite public observation timestamp required")
    images, depths = observation["observation/images"], observation["observation/depths"]
    if set(images) != set(CAMERAS) or set(depths) != set(CAMERAS):
        raise ActorContractError("exact four-camera public RGBD inventory required")
    for camera in CAMERAS:
        rgb, depth = np.asarray(images[camera]), np.asarray(depths[camera], np.float32)
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3) or depth.shape != (256, 256):
            raise ActorContractError("original 256x256 uint8 RGB and metric depth required")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ActorContractError("finite nonnegative metric depth required")
    rgb, depth = resize_public_rgbd(np.stack([images[camera] for camera in CAMERAS]),
                                   np.stack([depths[camera] for camera in CAMERAS]))
    return timestamp, rgb, depth


def encode_public_frame(observation):
    """The exact shared offline/online resize and RGB/log-depth encoding."""
    timestamp, rgb, depth = cached_public_frame(observation)
    encoded = encode_public_history(rgb[None], depth[None], np.zeros((1, 15), np.float32))
    return timestamp, encoded["rgbd"][0]


def public_geometry(observation):
    """Use identical cached pixels, public signed K and camera-to-world pose."""
    from our_mani.perception import CameraCalibration
    _, rgb, depth = cached_public_frame(observation)
    intrinsics, extrinsics = [], []
    calibrations = observation["observation/calibrations"]
    if set(calibrations) != set(CAMERAS):
        raise ActorContractError("exact four public camera calibrations required")
    for camera in CAMERAS:
        calibration = calibrations[camera]
        if not isinstance(calibration, CameraCalibration):
            raise ActorContractError("public CameraCalibration required")
        intrinsics.append(calibration.intrinsics)
        transform = np.eye(4)
        transform[:3, :3] = calibration.camera_to_world.rotation
        transform[:3, 3] = calibration.camera_to_world.position
        extrinsics.append(transform)
        camera_time = float(observation.get("observation/camera_timestamps", {}).get(camera, observation["timestamp"]))
        if abs(camera_time + calibration.timestamp_offset - float(observation["timestamp"])) > 1e-8:
            raise ActorContractError("current synchronized public geometry required")
    return geometry_points(rgb, depth, np.stack(intrinsics), np.stack(extrinsics))


class BlockHierarchyActor:
    def __init__(self, model, motion_pipeline, *, device="cpu", model_identity=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if tuple(motion_pipeline.roles) != ROLES:
            raise ActorContractError("fixed blind six-role forecast registry required")
        self.motion_pipeline = motion_pipeline
        self.model_identity = dict(model_identity or {})
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.last_timestamp = self.latest_motion = self.latest_geometry = None
        self.goal_position = self.goal_rotation = self.goal_quaternion = None
        self.selected_goal_time = self.goal_selected_at = None
        self.closed = False
        motion_pipeline.reset()

    def observe(self, observation, robot):
        if self.closed:
            raise ActorContractError("closed actor requires a new episode")
        timestamp, rgb, depth = cached_public_frame(observation)
        measured = robot_vector(robot, timestamp)
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp:
                raise ActorContractError("public observation time moved backwards")
            if timestamp == self.last_timestamp:
                return
        geometry = public_geometry(observation)
        if self.goal_position is None:
            public = {key: value for key, value in observation.items() if key != "observation/sensor_protocol"}
            motion = self.motion_pipeline.observe(public, measured.copy())
            if float(motion["timestamp"]) != timestamp:
                raise ActorContractError("public motion timestamp differs from current causal scene")
        else:
            motion = None
        self.history.append((timestamp, rgb.copy(), depth.copy(), measured.copy()))
        self.latest_motion, self.latest_geometry, self.last_timestamp = motion, geometry, timestamp

    def _history_tensors(self):
        encoded = encode_public_history(np.stack([row[1] for row in self.history]),
            np.stack([row[2] for row in self.history]), np.stack([row[3] for row in self.history]))
        return (torch.from_numpy(encoded["rgbd"][None]).to(self.device),
                torch.from_numpy(encoded["robot_history"][None]).to(self.device),
                torch.tensor([encoded["lengths"]], dtype=torch.long, device=self.device))

    def _unknown(self, reason):
        return {"action": None, "receipt": {"event": "perception_unknown", "reason": reason,
            "public_tracks_valid": False, "real_forecast_consumed": False,
            "old_stage_rule_calls": 0, "task_metadata_used": False, "native_labels_consumed": False,
            "source_task_or_speed_used": False, "current_timestamp": self.last_timestamp,
            "history_frames": len(self.history)}}

    def _forecast_tensors(self):
        motion = self.latest_motion["motion"]
        if motion.get("native_or_expert_poses_used") is not False or motion.get("trajectories_fabricated") is not False:
            raise ActorContractError("privileged or fabricated public motion is forbidden")
        expected = self.model_identity.get("forecaster_sha256")
        if expected is not None and motion.get("checkpoint_sha256") != expected:
            raise ActorContractError("motion checkpoint differs from the frozen identity")
        forecasts = motion.get("forecasts", {})
        if set(forecasts) - set(ROLES):
            raise ActorContractError("unexpected blind forecast role")
        # Only role zero is the moving block. Peg/ring false positives can never
        # substitute for an unavailable target trajectory.
        if "target" not in forecasts:
            return None
        trajectory = np.zeros((1, len(ROLES), 23, 20), np.float32)
        mask = np.zeros((1, len(ROLES), 23), bool)
        roles = []
        times = np.arange(23, dtype=np.float32) * np.float32(.05)
        for index, role in enumerate(ROLES):
            if role not in forecasts:
                continue
            entry = forecasts[role]
            tokens = np.asarray(entry["trajectory"], np.float32)
            if tokens.shape != (23, 20) or not np.isfinite(tokens).all() or not np.allclose(tokens[:, -1], times, rtol=0, atol=1e-6):
                raise ActorContractError("real forecast must retain its 23-step 50ms clock")
            if abs(float(entry["observed_until"]) - self.last_timestamp) > 1e-8:
                raise ActorContractError("cached older forecast cannot establish a handoff")
            if motion.get("status", {}).get(role, {}).get("available") is not True:
                raise ActorContractError("forecast status and actual trajectory differ")
            trajectory[0, index], mask[0, index] = tokens, True
            roles.append(role)
        return torch.from_numpy(trajectory).to(self.device), torch.from_numpy(mask).to(self.device), roles

    @torch.inference_mode()
    def act(self, observation, robot, *, remaining_seconds):
        if not np.isfinite(remaining_seconds) or not 0 < remaining_seconds <= 10. + 1e-8:
            raise ActorContractError("positive remaining time within ten-second budget required")
        self.observe(observation, robot)
        if len(self.history) < HISTORY_FRAMES:
            return self._unknown("fewer_than_five_causal_scene_frames")
        if not self.latest_geometry[1].any():
            return self._unknown("no_current_valid_calibrated_RGBD_points")
        handoff = self.goal_position is None
        forecast = self._forecast_tensors() if handoff else None
        if handoff and forecast is None:
            return self._unknown("no_current_real_target_forecast")
        images, robots, lengths = self._history_tensors()
        points, point_mask = self.latest_geometry
        encoded = self.model.encode(images, robots, lengths,
            torch.from_numpy(points[None]).to(self.device), torch.from_numpy(point_mask[None]).to(self.device))
        shared_e = encoded["skill_latent"]
        fingerprint = tensor_sha256(shared_e)
        current_robot = robots[:, len(self.history) - 1]
        if handoff:
            trajectory, mask, roles = forecast
            proposed = self.model.forward_candidate(encoded, current_robot, trajectory, mask)
            position = proposed["goal_position"].detach().clone()
            rotation = proposed["goal_rotation"].detach().clone()
            quaternion = proposed["goal_quaternion"].detach().clone()
            goal_time = proposed["goal_time"].detach().clone().reshape(1)
            if (position.shape != (1, 3) or rotation.shape != (1, 3, 3) or quaternion.shape != (1, 4)
                    or not all(bool(torch.isfinite(value).all()) for value in (position, rotation, quaternion, goal_time))
                    or bool((goal_time < 0).any())
                    or not torch.allclose(torch.linalg.vector_norm(quaternion, dim=-1), torch.ones(1, device=self.device), atol=1e-5)):
                raise ActorContractError("candidate must return a finite learned goal pose and nonnegative physical time")
            selected_at = self.last_timestamp
        else:
            roles = []
            position, rotation, quaternion = self.goal_position, self.goal_rotation, self.goal_quaternion
            goal_time, selected_at = self.selected_goal_time, self.goal_selected_at
        deadline = (goal_time - (self.last_timestamp - selected_at)).clamp_min(0.)
        policy = self.model.forward_skill(encoded, current_robot, position, rotation, deadline)
        if encoded["skill_latent"] is not shared_e or tensor_sha256(shared_e) != fingerprint:
            raise ActorContractError("candidate and skill must consume the same unmodified encoded latent")
        actions = self.model.decode_skill(current_robot, policy).detach().cpu().numpy()
        if (actions.shape != (1, 4, 8) or not np.isfinite(actions).all()
                or not np.allclose(np.linalg.norm(actions[0, :, 3:7], axis=-1), 1., atol=1e-5)
                or not np.isin(actions[0, :, 7], (0., 1.)).all()):
            raise ActorContractError("feedback skill must return four finite native pose/gripper actions")
        action = actions[0, 0].copy()
        if handoff:
            self.goal_position, self.goal_rotation, self.goal_quaternion = position, rotation, quaternion
            self.selected_goal_time, self.goal_selected_at = goal_time, selected_at
        measured = current_robot[0].detach().cpu().numpy()
        receipt = {"event": "candidate_handoff" if handoff else "skill_execution",
            "public_tracks_valid": handoff, "real_forecast_consumed": handoff,
            "forecast_generated_at": self.last_timestamp if handoff else None, "forecast_roles_consumed": roles,
            "shared_skill_latent_sha256": fingerprint, "candidate_skill_latent_sha256": fingerprint if handoff else None,
            "policy_skill_latent_sha256": fingerprint, "old_stage_rule_calls": 0, "task_metadata_used": False,
            "source_task_or_speed_used": False, "native_labels_consumed": False,
            "current_timestamp": self.last_timestamp, "history_frames": len(self.history),
            "inferred_skill_id": int(encoded["skill_logits"].argmax(-1).item()),
            "episode_goal_fixed": True, "goal_source": "predicted_candidate_episode_handoff_pose",
            "goal_xyz": position[0].cpu().tolist(), "goal_quaternion_xyzw": quaternion[0].cpu().tolist(),
            "goal_time_seconds": float(goal_time.item()), "goal_selected_at": selected_at,
            "deadline_remaining_seconds": float(deadline.item()),
            "goal_command_time_semantics": "remaining_until_first_closed_advice",
            "expert_grasp_lookahead_seconds": .5,
            "implied_object_target_remaining_seconds": max(float(goal_time.item()) + .5 - (self.last_timestamp-selected_at), 0.),
            "time_is_reachability_guarantee": False,
            "time_contract": "nominal advice clock; the original expert grasp position looks 0.5s farther ahead; no feasibility or actual interception-time guarantee",
            "current_tcp_xyzw": measured[:7].tolist(), "current_gripper_open": float(measured[7]),
            "requested_tcp_xyzw": action[:7].tolist(), "requested_gripper_open": float(action[7]),
            "request_position_error_m": float(np.linalg.norm(action[:3] - measured[:3])),
            "request_position_error_semantics": "requested_target_minus_current_TCP_not_expert_prediction_accuracy",
            "goal_position_error_from_current_tcp_m": float(np.linalg.norm(position[0].cpu().numpy() - measured[:3])),
            "candidate_score_semantics": SCORE_SEMANTICS, "executed_action_chunk_index": 0,
            "predicted_action_chunks": 4, "skill_trajectory_consumed": False,
            "valid_geometry_points": int(point_mask.sum())}
        return {"action": action, "receipt": receipt}

    def close(self):
        if not self.closed:
            self.history.clear()
            if callable(getattr(self.motion_pipeline, "close", None)):
                self.motion_pipeline.close()
            self.closed = True


def create_actor(*, root, freeze_manifest, device="cpu"):
    root = Path(root).resolve()
    validate_freeze(root, freeze_manifest)
    if torch.device(device) != torch.device("cpu"):
        raise EvaluationBlocked("frozen CPU inference required; no implicit device fallback")
    components = freeze_manifest["components"]
    paths = [checked(root, components[name]["artifact"]) for name in ("scene_classifier", "candidate_network", "skill_policy")]
    if len({sha(path) for path in paths}) != 1:
        raise EvaluationBlocked("one complete block checkpoint must supply encoder/candidate/skill")
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    config = meta.get("model_config", {})
    if (payload.get("version") != VERSION or meta.get("training_complete") is not True
            or meta.get("completed_epochs") != 120 or meta.get("TEST_read") is not False
            or meta.get("score_semantics") != SCORE_SEMANTICS
            or config != {"num_candidates": 4, "action_horizon": 4, "freeze_encoder_visual": True}
            or not payload.get("model_state_dict")):
        raise EvaluationBlocked("completed 120-epoch TRAIN-only block checkpoint required")
    center, scale = np.asarray(meta.get("workspace_center"), np.float32), np.asarray(meta.get("workspace_half_width"), np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise EvaluationBlocked("finite TRAIN-only workspace metadata required")
    from .networks import BlockHierarchyNetwork
    model = BlockHierarchyNetwork(center.tolist(), scale.tolist(), num_candidates=4, action_horizon=4, freeze_encoder_visual=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    # load_state_dict also restores buffers. Metadata cannot silently disagree
    # with the physical coordinate system actually used by loaded modules.
    for name, value in model.named_buffers():
        expected = center if name.endswith("workspace_center") else scale if name.endswith("workspace_scale") else None
        if expected is not None and not np.array_equal(value.detach().cpu().numpy(), expected):
            raise EvaluationBlocked("loaded model workspace buffer differs from checkpoint metadata: " + name)
    model.requires_grad_(False)
    forecaster = checked(root, components["forecaster"]["artifact"])
    segment = checked(root, components["segmentation"]["artifact"])
    from our_mani.hierarchy_20261030.tracking import BlindPublicMotionPipeline
    pipeline = BlindPublicMotionPipeline(root, segment_checkpoint=segment, motion_checkpoint=forecaster)
    if pipeline.checkpoint_sha256 != sha(forecaster):
        raise EvaluationBlocked("actual blind forecaster differs from frozen artifact")
    return BlockHierarchyActor(model, pipeline, device=device, model_identity={"checkpoint_sha256": sha(paths[0]),
        "forecaster_sha256": sha(forecaster), "segmentation_sha256": sha(segment)})

