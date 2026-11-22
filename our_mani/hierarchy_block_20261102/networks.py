"""Geometry-aware point proposals and receding-horizon block skills.

This package consumes only causal public RGB-D/calibration/proprioception and
the unchanged external GRU/context/SE3 forecaster's real trajectories. Candidate
scores are expert-imitation mixture weights, never calibrated task-success
probabilities. No native task identity, expert stage or future label is a model
input. Physical execution, contact and outcome checks remain external.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from our_mani.hierarchy_20261030.networks import (
    CausalSceneSkillEncoder, SimNorm, _finite, _floating, _same_device, _workspace,
    matrix_to_quaternion_xyzw, rotation_6d_to_matrix,
)


ROLE_NAMES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
PICK_SKILL_LABELS = ("pick_reach", "pick_pre_grasp", "pick_grasp", "pick_lift")
MODEL_DIM = 128
SKILL_DIM = 64
FORECAST_STEPS = 23
FORECAST_DT = .05


class GeometryTokenEncoder(nn.Module):
    """Keep calibrated world geometry as distinct learned-query spatial tokens.

    ``point_features`` is [B,N,7] = world XYZ metres, RGB, log(1+metric_depth).
    Calibration/backprojection and deterministic point sampling are upstream;
    no camera labels or source task labels enter this encoder. Invalid points
    are explicit masked padding and may contain NaN. At least one observed point
    per item is necessary; no synthetic fallback scene is fabricated.
    """

    def __init__(self, center: Sequence[float], scale: Sequence[float], *,
                 num_tokens: int = 32, model_dim: int = MODEL_DIM, heads: int = 4):
        super().__init__()
        if num_tokens < 1 or model_dim < 1 or heads < 1 or model_dim % heads:
            raise ValueError("positive token count and divisible attention dimensions required")
        center_tensor, scale_tensor = _workspace(center, scale)
        self.register_buffer("workspace_center", center_tensor)
        self.register_buffer("workspace_scale", scale_tensor)
        self.num_tokens, self.model_dim = num_tokens, model_dim
        self.point_mlp = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, model_dim))
        self.xyz_encoder = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, model_dim))
        self.queries = nn.Parameter(torch.randn(1, num_tokens, model_dim) * .02)
        self.query_norm = nn.LayerNorm(model_dim)
        self.point_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, dropout=0., batch_first=True)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.SiLU(),
                                         nn.Linear(2 * model_dim, model_dim))
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, point_features: Tensor, point_mask: Tensor) -> Tensor:
        _floating("point_features", point_features, (None, None, 7))
        batch, points = point_features.shape[:2]
        if batch < 1 or points < 1:
            raise ValueError("nonempty observed point sets required")
        if (not isinstance(point_mask, Tensor) or point_mask.dtype != torch.bool
                or point_mask.shape != (batch, points)):
            raise ValueError("aligned boolean point_mask[B,N] required")
        _same_device(point_features, point_mask)
        if not bool(point_mask.any(-1).all()):
            raise ValueError("each scene needs at least one real observed point")
        _finite("valid calibrated point features", point_features[point_mask])
        clean = torch.where(point_mask[..., None], point_features, torch.zeros_like(point_features))
        xyz = (clean[..., :3] - self.workspace_center) / self.workspace_scale
        inputs = torch.cat((xyz, clean[..., 3:]), -1)
        point_tokens = self.point_norm(self.point_mlp(inputs) + self.xyz_encoder(xyz))
        queries = self.queries.expand(batch, -1, -1)
        attended, _ = self.cross_attention(self.query_norm(queries), point_tokens, point_tokens,
                                           key_padding_mask=~point_mask, need_weights=False)
        tokens = self.attention_norm(queries + attended)
        return self.output_norm(tokens + self.feedforward(tokens))


def quaternion_xyzw_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert finite, nonzero xyzw quaternions to proper rotation matrices."""
    if not isinstance(quaternion, Tensor) or quaternion.ndim < 1:
        raise ValueError("xyzw quaternion tensor required")
    _floating("quaternion_xyzw", quaternion, (*quaternion.shape[:-1], 4))
    _finite("quaternion_xyzw", quaternion)
    if not bool((quaternion.norm(dim=-1) > 1e-8).all()):
        raise ValueError("zero quaternion cannot describe the observed TCP rotation")
    x, y, z, w = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), -1).reshape(
                            *quaternion.shape[:-1], 3, 3)


def _robot_rotation(robot: Tensor, batch: int) -> Tensor:
    _floating("robot", robot, (batch, 15))
    _finite("robot", robot)
    return quaternion_xyzw_to_matrix(robot[:, 3:7])


def _rotation_matrix(name: str, value: Tensor, batch: int) -> None:
    _floating(name, value, (batch, 3, 3))
    _finite(name, value)
    identity = torch.eye(3, device=value.device, dtype=value.dtype).expand(batch, -1, -1)
    if (not torch.allclose(value.transpose(-1, -2) @ value, identity, atol=1e-4, rtol=1e-4)
            or not torch.allclose(torch.linalg.det(value), torch.ones_like(value[:, 0, 0]),
                                  atol=1e-4, rtol=1e-4)):
        raise ValueError(f"{name} must contain proper SO(3) rotations")


def _conditioning(geometry_tokens: Tensor, robot: Tensor, skill_latent: Tensor,
                  scene_features: Tensor) -> tuple[int, Tensor]:
    _floating("geometry_tokens", geometry_tokens, (None, None, MODEL_DIM))
    batch, count = geometry_tokens.shape[:2]
    if batch < 1 or count < 1:
        raise ValueError("nonempty observed geometry tokens required")
    _floating("skill_latent", skill_latent, (batch, SKILL_DIM))
    _floating("scene_features", scene_features, (batch, MODEL_DIM))
    current_rotation = _robot_rotation(robot, batch)
    _same_device(geometry_tokens, robot, skill_latent, scene_features)
    for name, value in (("geometry_tokens", geometry_tokens), ("skill_latent", skill_latent),
                        ("scene_features", scene_features)):
        _finite(name, value)
    return batch, current_rotation


class JointCandidateTransformer(nn.Module):
    """Continuous K-way joint pose/time proposals conditioned on real forecasts.

    Every proposal query produces its own XYZ, world rotation, positive relative
    seconds and expert-imitation logit. All K proposals remain differentiable;
    argmax fields are runtime conveniences, never a required training reduction.
    The relative-time target uses the upstream declared nominal H5 frame clock,
    not a measured capture/interception timestamp or a native success label.
    """

    score_semantics = "expert_joint_handoff_pose_time_imitation_not_calibrated_success"

    def __init__(self, center, scale, *, num_candidates: int = 4):
        super().__init__()
        if num_candidates < 1:
            raise ValueError("positive candidate count required")
        center_tensor, scale_tensor = _workspace(center, scale)
        self.register_buffer("workspace_center", center_tensor)
        self.register_buffer("workspace_scale", scale_tensor)
        self.register_buffer("forecast_times", torch.arange(FORECAST_STEPS) * FORECAST_DT)
        self.num_candidates, self.role_names = num_candidates, ROLE_NAMES
        self.queries = nn.Parameter(torch.randn(1, num_candidates, MODEL_DIM) * .02)
        self.trajectory_projection = nn.Linear(20, MODEL_DIM)
        self.time_projection = nn.Linear(1, MODEL_DIM)
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), MODEL_DIM)
        self.geometry_type = nn.Parameter(torch.randn(1, 1, MODEL_DIM) * .02)
        self.robot_projection = nn.Linear(15, MODEL_DIM)
        self.skill_projection = nn.Linear(SKILL_DIM, MODEL_DIM)
        self.history_projection = nn.Linear(MODEL_DIM, MODEL_DIM)
        layer = nn.TransformerEncoderLayer(MODEL_DIM, 4, 2 * MODEL_DIM, dropout=0.,
                                            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(MODEL_DIM)
        self.position_head = nn.Linear(MODEL_DIM, 3)
        self.rotation_head = nn.Linear(MODEL_DIM, 6)
        self.time_head = nn.Linear(MODEL_DIM, 1)
        self.score_head = nn.Linear(MODEL_DIM, 1)
        nn.init.normal_(self.rotation_head.weight, std=.01)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, trajectory: Tensor, trajectory_mask: Tensor,
                geometry_tokens: Tensor, robot: Tensor, skill_latent: Tensor,
                scene_features: Tensor) -> dict[str, Tensor]:
        batch, current_rotation = _conditioning(geometry_tokens, robot, skill_latent, scene_features)
        _floating("trajectory", trajectory, (batch, len(ROLE_NAMES), FORECAST_STEPS, 20))
        if (not isinstance(trajectory_mask, Tensor) or trajectory_mask.dtype != torch.bool
                or trajectory_mask.shape != trajectory.shape[:3]):
            raise ValueError("boolean forecast mask[B,6,23] required")
        _same_device(trajectory, trajectory_mask, geometry_tokens)
        valid_bodies = trajectory_mask.all(-1)
        if bool((trajectory_mask.any(-1) != valid_bodies).any()):
            raise ValueError("real per-body forecasts must contain the full 23-step clock")
        if not bool(valid_bodies.any(-1).all()):
            raise ValueError("at least one real frozen-GRU forecast required per candidate decision")
        _finite("valid forecast tokens", trajectory[trajectory_mask])
        expected = self.forecast_times.to(trajectory).expand(batch, len(ROLE_NAMES), -1)
        if not torch.allclose(trajectory[..., -1][trajectory_mask], expected[trajectory_mask], atol=1e-6, rtol=0):
            raise ValueError("real forecast clock must remain 0..1.1 seconds at 0.05-second spacing")
        clean = torch.where(trajectory_mask[..., None], trajectory, torch.zeros_like(trajectory))
        roles = self.role_embedding.weight[None, :, None, :]
        motion_tokens = (self.trajectory_projection(clean) + self.time_projection(clean[..., -1:]) + roles)
        motion_tokens = motion_tokens.reshape(batch, len(ROLE_NAMES) * FORECAST_STEPS, MODEL_DIM)
        prefix = torch.cat((self.queries.expand(batch, -1, -1),
                            self.robot_projection(robot)[:, None],
                            self.skill_projection(skill_latent)[:, None],
                            self.history_projection(scene_features)[:, None],
                            geometry_tokens + self.geometry_type), dim=1)
        padding = torch.cat((torch.zeros(batch, prefix.shape[1], dtype=torch.bool, device=robot.device),
                             ~trajectory_mask.reshape(batch, -1)), dim=1)
        values = self.transformer(torch.cat((prefix, motion_tokens), dim=1), src_key_padding_mask=padding)
        queries = self.output_norm(values[:, :self.num_candidates])
        positions = self.workspace_center + self.workspace_scale * self.position_head(queries).tanh()
        relative_rotations = rotation_6d_to_matrix(self.rotation_head(queries))
        rotations = current_rotation[:, None] @ relative_rotations
        quaternions = matrix_to_quaternion_xyzw(rotations)
        times = F.softplus(self.time_head(queries).squeeze(-1)) + 1e-6
        logits = self.score_head(queries).squeeze(-1)
        probabilities = logits.softmax(-1)
        index = logits.argmax(-1)
        rows = torch.arange(batch, device=logits.device)
        return dict(positions=positions, rotation_matrices=rotations, quaternions=quaternions,
                    times=times, logits=logits, probabilities=probabilities,
                    selected_index=index, goal_position=positions[rows, index],
                    goal_rotation=rotations[rows, index], goal_quaternion=quaternions[rows, index],
                    goal_time=times[rows, index])


class FeedbackSkillTransformer(nn.Module):
    """Four short-horizon anchor-relative TCP proposals from current feedback.

    All delta positions are world-frame offsets from the *current observed*
    anchor TCP. Each relative rotation is independently composed as
    R_world = R_current @ R_delta; predictions are never cumulatively integrated.
    The caller executes only the first 0.5-second segment, then obtains fresh
    observations and recomputes the causal GRU, geometry and action proposals.
    """

    def __init__(self, center, scale, *, action_horizon: int = 4):
        super().__init__()
        if action_horizon < 1:
            raise ValueError("positive action horizon required")
        center_tensor, scale_tensor = _workspace(center, scale)
        self.register_buffer("workspace_center", center_tensor)
        self.register_buffer("workspace_scale", scale_tensor)
        self.register_buffer("step_end_seconds", (torch.arange(action_horizon) + 1) * .5)
        self.action_horizon = action_horizon
        self.queries = nn.Parameter(torch.randn(1, action_horizon, MODEL_DIM) * .02)
        self.step_projection = nn.Linear(1, MODEL_DIM)
        self.robot_projection = nn.Linear(15, MODEL_DIM)
        self.skill_projection = nn.Linear(SKILL_DIM, MODEL_DIM)
        self.history_projection = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.goal_projection = nn.Linear(3 + 9 + 1, MODEL_DIM)
        layer = nn.TransformerDecoderLayer(MODEL_DIM, 4, 2 * MODEL_DIM, dropout=0.,
                                            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, 2)
        self.output_norm = nn.LayerNorm(MODEL_DIM)
        self.delta_head = nn.Linear(MODEL_DIM, 3)
        self.rotation_head = nn.Linear(MODEL_DIM, 6)
        self.gripper_head = nn.Linear(MODEL_DIM, 2)
        nn.init.normal_(self.delta_head.weight, std=.01)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.rotation_head.weight, std=.01)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, geometry_tokens: Tensor, robot: Tensor, skill_latent: Tensor,
                scene_features: Tensor, goal_xyz: Tensor, goal_rotation_matrix: Tensor,
                goal_time: Tensor) -> dict[str, Tensor]:
        batch, current_rotation = _conditioning(geometry_tokens, robot, skill_latent, scene_features)
        _floating("goal_xyz", goal_xyz, (batch, 3))
        _finite("goal_xyz", goal_xyz)
        _rotation_matrix("goal_rotation_matrix", goal_rotation_matrix, batch)
        _floating("goal_time", goal_time, (batch,))
        _finite("goal_time", goal_time)
        if not bool((goal_time >= 0).all()):
            raise ValueError("nonnegative remaining goal seconds required")
        _same_device(robot, goal_xyz, goal_rotation_matrix, goal_time)
        relative_goal = (goal_xyz - robot[:, :3]) / self.workspace_scale
        relative_goal_rotation = current_rotation.transpose(-1, -2) @ goal_rotation_matrix
        goal = torch.cat((relative_goal, relative_goal_rotation.reshape(batch, 9), goal_time[:, None]), -1)
        memory = torch.cat((geometry_tokens,
                            self.robot_projection(robot)[:, None],
                            self.skill_projection(skill_latent)[:, None],
                            self.history_projection(scene_features)[:, None],
                            self.goal_projection(goal)[:, None]), dim=1)
        queries = self.queries.expand(batch, -1, -1) + self.step_projection(self.step_end_seconds[:, None])[None]
        values = self.output_norm(self.transformer(queries, memory))
        # No arbitrary output clipping: supervision in physical metres determines
        # the useful movement scale, and native planning owns executability.
        delta_xyz = self.delta_head(values) * self.workspace_scale
        rotation_6d = self.rotation_head(values)
        relative_rotations = rotation_6d_to_matrix(rotation_6d)
        rotations = current_rotation[:, None] @ relative_rotations
        return dict(delta_xyz=delta_xyz, relative_rotation_6d=rotation_6d,
                    relative_rotation_matrices=relative_rotations,
                    positions=robot[:, None, :3] + delta_xyz,
                    rotation_matrices=rotations,
                    quaternions=matrix_to_quaternion_xyzw(rotations),
                    gripper_logits=self.gripper_head(values))


class BlockHierarchyNetwork(nn.Module):
    """Independent action latent, observed geometry, candidate and feedback skill.

    The unchanged 11-class causal encoder can be strictly warm-started. Only its
    four pick logits condition this block-only model. A separate trainable table
    is copied from the first four old embeddings, group-SimNormed, then mixed by
    the current soft probabilities. Exactly this same e conditions both heads.
    The classifier's visual/history path can stay frozen; the geometric branch
    is always separately trainable and retains calibrated world positions.
    """

    def __init__(self, center, scale, encoder_state_dict: Mapping[str, Tensor] | None = None,
                 *, num_candidates: int = 4, action_horizon: int = 4,
                 freeze_encoder_visual: bool = True):
        super().__init__()
        self.encoder = CausalSceneSkillEncoder()
        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.skill_embeddings = nn.Embedding(len(PICK_SKILL_LABELS), SKILL_DIM)
        with torch.no_grad():
            self.skill_embeddings.weight.copy_(self.encoder.skill_embeddings.weight[:4])
        self.simnorm = SimNorm(8)
        self.geometry = GeometryTokenEncoder(center, scale, num_tokens=32)
        self.candidate = JointCandidateTransformer(center, scale, num_candidates=num_candidates)
        self.skill = FeedbackSkillTransformer(center, scale, action_horizon=action_horizon)
        self.set_encoder_visual_frozen(freeze_encoder_visual)
        self.metadata = dict(
            schema="hierarchy_block_20261102_geometry_joint_pose_time_feedback_v1",
            source_task_or_speed_input=False, expert_stage_input=False,
            geometry_input="causal_public_calibrated_world_XYZ_m_RGB_log1p_depth",
            geometry_tokens=32, geometry_model_dim=MODEL_DIM,
            geometry_reduction="learned_query_spatial_cross_attention_not_global_scene_pooling",
            forecaster="external_original_frozen_GRU_context_SE3_unchanged",
            role_names=ROLE_NAMES, forecast_clock="0..1.1s_at_0.05s_per_real_body",
            shared_skill_latent="four_pick_probabilities_times_group8_SimNorm_embeddings",
            skill_labels=PICK_SKILL_LABELS, encoder_initial_classes=11,
            classifier_visual_frozen=self.freeze_encoder_visual,
            frozen_feature_cache="detached_scene128_scene_latent64_pick_logits4_only_geometry_and_E_recomputed",
            candidate_count=num_candidates,
            candidate_score_semantics=self.candidate.score_semantics,
            candidate_selection="all_K_differentiable_for_training_argmax_only_runtime_convenience",
            goal_time_clock="nominal_H5_frame_index_times_0.05_not_recorded_interception_timestamps",
            goal_time_range="positive_continuous_seconds_not_clipped_to_forecast_horizon",
            policy_goal_source="expert_conditioning_permitted_offline_predicted_joint_candidate_required_at_runtime",
            action_horizon=action_horizon, action_segment_seconds=.5,
            action_translation="world_offset_from_current_observed_anchor_TCP_not_cumulative",
            action_rotation="R_world_equals_R_current_times_R_delta_each_step_not_cumulative",
            runtime_replanning="execute_first_segment_only_then_fresh_observation_and_causal_GRU_encode",
            native_action="world_xyz_quaternion_xyzw_gripper_0closed_1open",
            precision_or_success_claim="none_requires_fixed_cohort_training_and_evaluation",
        )

    @property
    def policy(self):
        return self.skill

    @property
    def encoder_visual_frozen(self) -> bool:
        return self.freeze_encoder_visual

    def set_encoder_visual_frozen(self, frozen: bool) -> None:
        if not isinstance(frozen, bool):
            raise ValueError("explicit boolean encoder freeze configuration required")
        self.freeze_encoder_visual = frozen
        self.encoder.requires_grad_(not frozen)
        # The legacy 11-row table is not a second downstream skill latent.
        self.encoder.skill_embeddings.requires_grad_(False)
        self.encoder.train(self.training and not frozen)
        if hasattr(self, "metadata"):
            self.metadata["classifier_visual_frozen"] = frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder_visual:
            self.encoder.eval()
        return self

    def encode(self, rgbd_history: Tensor, robot_history: Tensor, lengths: Tensor,
               point_features: Tensor, point_mask: Tensor) -> dict[str, Tensor]:
        if self.freeze_encoder_visual:
            with torch.no_grad():
                scene = self.encoder(rgbd_history, robot_history, lengths)
        else:
            scene = self.encoder(rgbd_history, robot_history, lengths)
        return self._assemble_encoding(scene["scene_features"], scene["scene_latent"],
                                       scene["skill_logits"][:, :len(PICK_SKILL_LABELS)],
                                       point_features, point_mask)

    def _assemble_encoding(self, scene_features: Tensor, scene_latent: Tensor, logits: Tensor,
                           point_features: Tensor, point_mask: Tensor) -> dict[str, Tensor]:
        _floating("scene_features", scene_features, (None, MODEL_DIM))
        batch = scene_features.shape[0]
        _floating("scene_latent", scene_latent, (batch, SKILL_DIM))
        _floating("pick skill_logits", logits, (batch, len(PICK_SKILL_LABELS)))
        _same_device(scene_features, scene_latent, logits, point_features, point_mask)
        for name, value in (("scene_features", scene_features), ("scene_latent", scene_latent),
                            ("pick skill_logits", logits)):
            _finite(name, value)
        geometry_tokens = self.geometry(point_features, point_mask)
        if geometry_tokens.shape[0] != batch:
            raise ValueError("current point sets and causal histories must share their batch")
        probabilities = logits.softmax(-1)
        e = probabilities @ self.simnorm(self.skill_embeddings.weight)
        return dict(scene_features=scene_features, scene_latent=scene_latent,
                    geometry_tokens=geometry_tokens, skill_logits=logits,
                    skill_probabilities=probabilities, skill_latent=e)

    def encode_frozen_features(self, scene_features: Tensor, scene_latent: Tensor,
                               skill_logits: Tensor, point_features: Tensor,
                               point_mask: Tensor) -> dict[str, Tensor]:
        """Reuse input-only classifier features; freshly encode geometry and E.

        Caches must come from this exact frozen encoder, with causal history and
        source/frame provenance verified upstream. Only the first four raw
        classifier logits are accepted, never class labels. The three cached
        tensors are detached defensively. Unfreezing the classifier makes this
        path invalid, preventing a stale cache from silently blocking training.
        """
        if not self.encoder_visual_frozen:
            raise ValueError("frozen-feature encoding requires a frozen visual/history/classifier path")
        for name, value in (("scene_features", scene_features), ("scene_latent", scene_latent),
                            ("skill_logits", skill_logits)):
            if not isinstance(value, Tensor):
                raise ValueError(f"{name} must be a cached floating tensor")
        return self._assemble_encoding(scene_features.detach(), scene_latent.detach(), skill_logits.detach(),
                                       point_features, point_mask)

    def forward_candidate(self, encoded: Mapping[str, Tensor], robot: Tensor,
                          trajectory: Tensor, trajectory_mask: Tensor) -> dict[str, Tensor]:
        return self.candidate(trajectory, trajectory_mask, encoded["geometry_tokens"], robot,
                              encoded["skill_latent"], encoded["scene_features"])

    def forward_skill(self, encoded: Mapping[str, Tensor], robot: Tensor,
                      goal_xyz: Tensor, goal_rotation_matrix: Tensor,
                      goal_time: Tensor) -> dict[str, Tensor]:
        return self.skill(encoded["geometry_tokens"], robot, encoded["skill_latent"],
                          encoded["scene_features"], goal_xyz, goal_rotation_matrix, goal_time)

    def decode_skill(self, robot: Tensor, output: Mapping[str, Tensor]) -> Tensor:
        """Decode every anchor-relative proposal; caller executes only row zero."""
        delta = output["delta_xyz"]
        _floating("delta_xyz", delta, (None, self.skill.action_horizon, 3))
        batch = delta.shape[0]
        current_rotation = _robot_rotation(robot, batch)
        relative_rotation = output["relative_rotation_matrices"]
        _floating("relative_rotation_matrices", relative_rotation,
                  (batch, self.skill.action_horizon, 3, 3))
        logits = output["gripper_logits"]
        _floating("gripper_logits", logits, (batch, self.skill.action_horizon, 2))
        _same_device(robot, delta, relative_rotation, logits)
        for name, value in (("delta_xyz", delta), ("relative_rotation_matrices", relative_rotation),
                            ("gripper_logits", logits)):
            _finite(name, value)
        _rotation_matrix("decoded relative rotations", relative_rotation.reshape(-1, 3, 3),
                         batch * self.skill.action_horizon)
        rotations = current_rotation[:, None] @ relative_rotation
        quaternions = matrix_to_quaternion_xyzw(rotations)
        opening = logits.argmax(-1).to(delta.dtype)
        return torch.cat((robot[:, None, :3] + delta, quaternions, opening[..., None]), dim=-1)
