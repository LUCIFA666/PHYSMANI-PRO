"""Spatially aware fine-tuning heads for the separately frozen scene classifier.

Only observations, classifier probabilities and real frozen-GRU forecasts enter
these heads. Expert action labels and remaining-time labels belong to the loss,
never to ``encode`` or ``forward_candidate``. The continuous time output uses the
nominal H5 frame-index * 0.05 s clock; it is not a measured interception time.
No claim about achieved position precision is made by this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from our_mani.hierarchy_20261030.candidate_multibody import MultiBodyCandidateTransformer
from our_mani.hierarchy_20261030.networks import (
    SKILL_LABELS, PretrainedSkillPolicy, _finite, _floating, _same_device, _workspace,
)


class SpatialGeometryResidual(nn.Module):
    """Shared-camera CoordConv, retaining all 4 x 8 x 8 spatial cells.

    The four camera positions and every final spatial cell have distinct input
    columns in the projection. There is no global spatial pooling. Coordinates
    identify image locations; they do not stand in for calibrated world XYZ.
    """

    def __init__(self):
        super().__init__()
        axis = torch.linspace(-1., 1., 64)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("image_coordinates", torch.stack((xx, yy))[None])
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(4 * 32 * 8 * 8, 128), nn.SiLU(), nn.LayerNorm(128),
        )
        # Start with a small, nonzero correction while preserving the pretrained
        # scene representation as a residual path.
        self.residual_scale = nn.Parameter(torch.tensor(.1))

    def forward(self, rgbd_current: Tensor, frozen_scene: Tensor) -> Tensor:
        _floating("rgbd_current", rgbd_current, (None, 4, 4, 64, 64))
        batch = rgbd_current.shape[0]
        if batch < 1:
            raise ValueError("nonempty current observations required")
        _floating("frozen_scene", frozen_scene, (batch, 128))
        _same_device(rgbd_current, frozen_scene)
        _finite("rgbd_current", rgbd_current)
        _finite("frozen_scene", frozen_scene)
        frames = rgbd_current.reshape(batch * 4, 4, 64, 64)
        coordinates = self.image_coordinates.to(frames).expand(batch * 4, -1, -1, -1)
        cells = self.camera_encoder(torch.cat((frames, coordinates), dim=1))
        # Flatten, rather than average, preserves camera and image-cell order.
        spatial_features = self.spatial_projection(cells.reshape(batch, 4 * 32 * 8 * 8))
        return frozen_scene.detach() + self.residual_scale * spatial_features


def _require_workspace(module: nn.Module, center: Tensor, scale: Tensor, name: str) -> None:
    if (not torch.equal(module.workspace_center.detach().cpu(), center)
            or not torch.equal(module.workspace_scale.detach().cpu(), scale)):
        raise ValueError(f"{name} checkpoint workspace differs from the requested TRAIN workspace")


class PrecisionHeads(nn.Module):
    """Fine-tune new geometry, one shared skill table and the two loaded heads.

    The original classifier is deliberately not a child module. Callers compute
    its scene features and class probabilities under ``no_grad`` and pass them
    here; these inputs are also detached defensively. ``roles`` must preserve the
    role order recorded in the loaded candidate checkpoint's metadata.

    ``forward_candidate`` requires true multi-body forecasts [B,R,23,20] and an
    aligned boolean mask. Their 0..1.1 s clock is unchanged. Its ``goal_time`` is
    a separate positive continuous regression and may exceed that horizon.
    ``forward_skill`` uses the supplied latched goal, without an untrained dummy
    trajectory or an expert stage input.
    """

    def __init__(self, center: Sequence[float], scale: Sequence[float], roles: Sequence[str],
                 encoder_state_dict: Mapping[str, Tensor],
                 skill_state_dict: Mapping[str, Tensor],
                 candidate_state_dict: Mapping[str, Tensor]):
        super().__init__()
        center_tensor, scale_tensor = _workspace(center, scale)
        if (not isinstance(roles, Sequence) or isinstance(roles, (str, bytes))
                or not roles or any(not isinstance(role, str) or not role for role in roles)
                or len(roles) != len(set(roles))):
            raise ValueError("ordered unique public role names required")
        if not all(isinstance(state, Mapping) for state in
                   (encoder_state_dict, skill_state_dict, candidate_state_dict)):
            raise ValueError("three explicit pretrained state dictionaries required")
        self.role_names = tuple(roles)
        self.geometry = SpatialGeometryResidual()
        table = encoder_state_dict.get("skill_embeddings.weight")
        _floating("pretrained skill table", table, (len(SKILL_LABELS), 64))
        _finite("pretrained skill table", table)
        if not bool((table.norm(dim=-1) > 0).all()):
            raise ValueError("pretrained skill embeddings must have nonzero rows")
        self.skill_embeddings = nn.Embedding(len(SKILL_LABELS), 64)
        with torch.no_grad():
            self.skill_embeddings.weight.copy_(table)
        self.candidate = MultiBodyCandidateTransformer(center, scale, role_names=self.role_names)
        self.candidate.load_state_dict(candidate_state_dict, strict=True)
        self.policy = PretrainedSkillPolicy(center, scale)
        self.policy.load_state_dict(skill_state_dict, strict=True)
        _require_workspace(self.candidate, center_tensor, scale_tensor, "candidate")
        _require_workspace(self.policy, center_tensor, scale_tensor, "skill")
        self.candidate.time_head.requires_grad_(False)
        self.time_regressor = nn.Sequential(
            nn.Linear(128 + 64 + 15, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.metadata = dict(
            version="hierarchy_precision_20261030_spatial_residual_v1",
            observation_inputs=("current_rgbd", "frozen_scene_features", "frozen_skill_probabilities", "robot"),
            label_inputs=False,
            frozen_classifier_embedded=False,
            frozen_classifier_inputs_detached=True,
            spatial_grid=(4, 32, 8, 8),
            spatial_pooling="none_flatten_camera_and_cell_positions",
            skill_embedding="one_trainable_probability_weighted_normalized_table_shared_by_both_heads",
            role_names=self.role_names,
            candidate_probability_semantics=self.candidate.probability_semantics,
            forecaster="external_frozen_GRU_context_unchanged_real_forecasts_required",
            forecast_relative_seconds="0..1.1_in_0.05_steps_per_body_unchanged",
            goal_time_supervision="remaining_until_first_expert_closed_command_advice",
            goal_time_clock="nominal_H5_frame_index_times_0.05_seconds_not_recorded_timestamps",
            goal_time_semantics="nominal_remaining_expert_advice_time_not_measured_object_interception_time",
            goal_time_range="positive_continuous_seconds_no_forecast_horizon_clipping",
            legacy_discrete_time_head="loaded_frozen_diagnostic_only_not_supervised",
            skill_trajectory_consumed=False,
            achieved_precision="unmeasured_requires_training_and_held_out_evaluation",
        )

    def encode(self, rgbd_current: Tensor, frozen_scene: Tensor,
               skill_probabilities: Tensor) -> dict[str, Tensor]:
        scene = self.geometry(rgbd_current, frozen_scene)
        _floating("skill_probabilities", skill_probabilities, (scene.shape[0], len(SKILL_LABELS)))
        _same_device(scene, skill_probabilities)
        _finite("skill_probabilities", skill_probabilities)
        if (not bool(((skill_probabilities >= 0) & (skill_probabilities <= 1)).all())
                or not torch.allclose(skill_probabilities.sum(-1), torch.ones_like(skill_probabilities[:, 0]),
                                      atol=1e-5, rtol=1e-5)):
            raise ValueError("normalized soft classifier probabilities required")
        embeddings = F.normalize(self.skill_embeddings.weight, dim=-1, eps=1e-8)
        latent = skill_probabilities.detach() @ embeddings
        return dict(scene_features=scene, skill_latent=latent)

    def forward_candidate(self, rgbd_current: Tensor, frozen_scene: Tensor,
                          skill_probabilities: Tensor, robot: Tensor,
                          trajectory: Tensor, trajectory_mask: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(rgbd_current, frozen_scene, skill_probabilities)
        output = self.candidate(trajectory, trajectory_mask, encoded["skill_latent"],
                                robot, encoded["scene_features"])
        # Preserve the copied categorical head only as explicitly named legacy
        # diagnostics. Its forecast-bounded time is not the new target clock.
        for name in ("time_logits", "time_probabilities", "time_index", "time_probability"):
            output["legacy_" + name] = output.pop(name)
        output["legacy_discrete_goal_time"] = output["goal_time"]
        timing_features = torch.cat((encoded["scene_features"], encoded["skill_latent"], robot), -1)
        raw_seconds = self.time_regressor(timing_features).squeeze(-1)
        output["goal_time"] = F.softplus(raw_seconds) + 1e-6
        return output

    def forward_skill(self, rgbd_current: Tensor, frozen_scene: Tensor,
                      skill_probabilities: Tensor, robot: Tensor,
                      goal_xyz: Tensor, goal_time: Tensor) -> dict[str, Tensor | bool]:
        encoded = self.encode(rgbd_current, frozen_scene, skill_probabilities)
        return self.policy(encoded["scene_features"], encoded["skill_latent"],
                           robot, goal_xyz, goal_time)
