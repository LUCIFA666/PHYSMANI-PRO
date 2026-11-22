"""Independent scene, candidate-point, and closed-loop skill networks.

Only current/past scene frames are accepted by the scene encoder. Task IDs and
expert stage labels are training targets, never inputs to these networks.
Candidate logits imitate expert target choices; they are NOT calibrated native
skill-success probabilities. Trajectories must come from the real forecaster.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SKILL_LABELS = (
    "pick_reach", "pick_pre_grasp", "pick_grasp", "pick_lift",
    "insert_wp0", "insert_wp1", "insert_wp1_grasp", "insert_wp2",
    "insert_wp3", "insert_wp4", "insert_wp4_open",
)


class SimNorm(nn.Module):
    def __init__(self, group_size: int = 8):
        super().__init__()
        if group_size < 1:
            raise ValueError("positive SimNorm group size required")
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % self.group_size:
            raise ValueError("latent dimension must be divisible by SimNorm group size")
        return x.reshape(*x.shape[:-1], -1, self.group_size).softmax(-1).reshape_as(x)


def _floating(name: str, value: Tensor, shape: tuple[int | None, ...]) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if value.ndim != len(shape) or any(n is not None and value.shape[i] != n for i, n in enumerate(shape)):
        raise ValueError(f"invalid {name} shape: expected {shape}, received {tuple(value.shape)}")


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite {name}")


def _same_device(*values: Tensor) -> None:
    if len({value.device for value in values}) != 1:
        raise ValueError("network inputs must share a device")


def _workspace(center: Sequence[float], scale: Sequence[float]) -> tuple[Tensor, Tensor]:
    center, scale = torch.as_tensor(center, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)
    if center.shape != (3,) or scale.shape != (3,):
        raise ValueError("workspace center and scale must be xyz triples")
    _finite("workspace center", center)
    _finite("workspace scale", scale)
    if not bool((scale > 0).all()):
        raise ValueError("workspace scale must contain positive xyz half-widths")
    return center.clone(), scale.clone()


def _trajectory(trajectory: Tensor, mask: Tensor, batch: int) -> tuple[Tensor, Tensor]:
    _floating("trajectory", trajectory, (batch, None, 20))
    if trajectory.shape[1] < 1 or not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != trajectory.shape[:2]:
        raise ValueError("trajectory requires an aligned boolean mask and at least one step")
    _same_device(trajectory, mask)
    if not bool(mask.any(-1).all()):
        raise ValueError("every trajectory must contain a valid step")
    _finite("valid trajectory", trajectory[mask])
    for row, valid in zip(trajectory, mask):
        times = row[valid, -1]
        if not bool((times >= 0).all()) or not bool((times[1:] > times[:-1]).all()):
            raise ValueError("valid trajectory times must be nonnegative and strictly increasing")
    clean = torch.where(mask[..., None], trajectory, torch.zeros_like(trajectory))
    return clean, mask


def rotation_6d_to_matrix(values: Tensor) -> Tensor:
    """Convert two xyz column vectors into SO(3), including degenerate inputs."""
    _floating("rotation_6d", values, (*values.shape[:-1], 6))
    _finite("rotation_6d", values)
    first, second = values[..., :3], values[..., 3:]
    first_norm = first.norm(dim=-1, keepdim=True)
    canonical = torch.zeros_like(first)
    canonical[..., 0] = 1
    b1 = torch.where(first_norm > 1e-6, first / first_norm.clamp_min(1e-6), canonical)
    orthogonal = second - (second * b1).sum(-1, keepdim=True) * b1
    second_norm = orthogonal.norm(dim=-1, keepdim=True)
    axis = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(values.dtype)
    fallback = axis - (axis * b1).sum(-1, keepdim=True) * b1
    fallback = F.normalize(fallback, dim=-1, eps=1e-6)
    b2 = torch.where(second_norm > 1e-6, orthogonal / second_norm.clamp_min(1e-6), fallback)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_quaternion_xyzw(matrix: Tensor) -> Tensor:
    """Stable, differentiable SO(3) conversion; output order is x,y,z,w."""
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                           1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    magnitude = squared.clamp_min(1e-8).sqrt()
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
    ), -2) / (2 * magnitude.clamp_min(.1)[..., :, None])
    index = magnitude.argmax(-1)
    selected = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    xyzw = torch.cat((selected[..., 1:], selected[..., :1]), -1)
    return F.normalize(xyzw, dim=-1, eps=1e-8)


class CausalSceneSkillEncoder(nn.Module):
    """Four shared-camera CNNs followed by a length-masked past-only GRU.

    rgbd: [B,T,4 cameras,4 channels,64,64], normalized by the data contract.
    robot_history: [B,T,15]. Valid frames occupy each sequence's prefix.
    """
    def __init__(self, num_skills: int = 11, scene_dim: int = 128, latent_dim: int = 64,
                 skill_dim: int = 64, simnorm_group: int = 8):
        super().__init__()
        if num_skills != len(SKILL_LABELS):
            raise ValueError("this version requires the frozen eleven action classes")
        if min(scene_dim, latent_dim, skill_dim) < 1 or latent_dim % simnorm_group:
            raise ValueError("invalid scene/latent/skill dimensions")
        self.scene_dim, self.latent_dim, self.skill_dim = scene_dim, latent_dim, skill_dim
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.camera_fusion = nn.Sequential(nn.Linear(4 * 64, 96), nn.SiLU())
        self.robot_encoder = nn.Sequential(nn.Linear(15, 32), nn.SiLU())
        self.history_encoder = nn.GRU(128, scene_dim, batch_first=True)
        self.feature_norm = nn.LayerNorm(scene_dim)
        self.scene_projection = nn.Sequential(nn.Linear(scene_dim, latent_dim), SimNorm(simnorm_group))
        self.classifier = nn.Linear(latent_dim, num_skills)
        self.skill_embeddings = nn.Embedding(num_skills, skill_dim)
        nn.init.normal_(self.skill_embeddings.weight, std=.1)

    def forward(self, rgbd: Tensor, robot_history: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        _floating("rgbd", rgbd, (None, None, 4, 4, 64, 64))
        batch, steps = rgbd.shape[:2]
        if batch < 1 or steps < 1:
            raise ValueError("nonempty scene history required")
        _floating("robot_history", robot_history, (batch, steps, 15))
        _same_device(rgbd, robot_history)
        if not isinstance(lengths, Tensor) or lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("integer lengths[B] required")
        if not bool(((lengths >= 1) & (lengths <= steps)).all()):
            raise ValueError("history lengths outside available frames")
        valid = torch.arange(steps, device=rgbd.device)[None] < lengths.to(rgbd.device)[:, None]
        _finite("valid rgbd", rgbd[valid])
        _finite("valid robot_history", robot_history[valid])
        frames = rgbd[valid]
        camera_features = self.camera_encoder(frames.reshape(-1, 4, 64, 64)).reshape(-1, 4 * 64)
        visual = self.camera_fusion(camera_features)
        proprioception = self.robot_encoder(robot_history[valid])
        encoded = rgbd.new_zeros(batch, steps, 128)
        encoded[valid] = torch.cat((visual, proprioception), -1)
        packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        scene_features = self.feature_norm(hidden[-1])
        scene_latent = self.scene_projection(scene_features)
        skill_logits = self.classifier(scene_latent)
        probabilities = skill_logits.softmax(-1)
        table = F.normalize(self.skill_embeddings.weight, dim=-1, eps=1e-8)
        skill_latent = probabilities @ table
        return dict(scene_features=scene_features, scene_latent=scene_latent,
                    skill_logits=skill_logits, skill_probabilities=probabilities,
                    skill_latent=skill_latent)


class CandidatePointTransformer(nn.Module):
    """Imitate an expert spatial target on a 16^3 grid plus bounded cell offsets.

    workspace_scale is an xyz half-width from TRAIN-only statistics. The
    location/time distributions are factorized and are not success predictors.
    trajectory[..., :19] is normalized SE3 state; [..., 19] is relative seconds.
    """
    probability_semantics = "expert_target_imitation_not_calibrated_skill_success"

    def __init__(self, workspace_center: Sequence[float], workspace_scale: Sequence[float],
                 skill_dim: int = 64, model_dim: int = 128, heads: int = 4,
                 layers: int = 2, dropout: float = 0., time_values: Sequence[float] | None = None):
        super().__init__()
        if model_dim < 1 or heads < 1 or model_dim % heads or layers < 1:
            raise ValueError("invalid Transformer dimensions")
        center, scale = _workspace(workspace_center, workspace_scale)
        self.register_buffer("workspace_center", center)
        self.register_buffer("workspace_scale", scale)
        times = torch.arange(23, dtype=torch.float32) * .05 if time_values is None else torch.as_tensor(time_values, dtype=torch.float32)
        if times.shape != (23,) or not bool(torch.isfinite(times).all()) or not bool((times >= 0).all()) or not bool((times[1:] > times[:-1]).all()):
            raise ValueError("23 strictly increasing nonnegative time values required")
        self.register_buffer("time_values", times.clone())
        axis = (torch.arange(16, dtype=torch.float32) + .5) / 8 - 1
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("normalized_grid_centers", grid)
        self.skill_dim = skill_dim
        self.trajectory_projection = nn.Linear(20, model_dim)
        self.time_projection = nn.Linear(1, model_dim)
        self.skill_projection = nn.Linear(skill_dim, model_dim)
        self.robot_projection = nn.Linear(15, model_dim)
        self.query = nn.Parameter(torch.randn(1, 1, model_dim) * .02)
        layer = nn.TransformerEncoderLayer(model_dim, heads, 2 * model_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(model_dim)
        self.point_head = nn.Linear(model_dim, 16 ** 3)
        self.offset_head = nn.Linear(model_dim, 16 ** 3 * 3)
        self.time_head = nn.Linear(model_dim, 23)

    def forward(self, trajectory: Tensor, trajectory_mask: Tensor, skill_latent: Tensor, robot: Tensor) -> dict[str, Tensor]:
        batch = trajectory.shape[0]
        trajectory, mask = _trajectory(trajectory, trajectory_mask, batch)
        _floating("skill_latent", skill_latent, (batch, self.skill_dim))
        _floating("robot", robot, (batch, 15))
        _finite("skill_latent", skill_latent)
        _finite("robot", robot)
        _same_device(trajectory, skill_latent, robot)
        tokens = self.trajectory_projection(trajectory) + self.time_projection(trajectory[..., -1:])
        prefix = torch.cat((self.query.expand(batch, -1, -1), self.skill_projection(skill_latent)[:, None], self.robot_projection(robot)[:, None]), 1)
        padding = torch.cat((torch.zeros(batch, 3, device=mask.device, dtype=torch.bool), ~mask), 1)
        pooled = self.output_norm(self.transformer(torch.cat((prefix, tokens), 1), src_key_padding_mask=padding)[:, 0])
        logits = self.point_head(pooled)
        offsets = self.offset_head(pooled).reshape(batch, 16 ** 3, 3).tanh() / 16
        candidate_xyz = self.workspace_center + self.workspace_scale * (self.normalized_grid_centers[None] + offsets)
        time_logits = self.time_head(pooled)
        output = dict(point_logits=logits, point_probabilities=logits.softmax(-1),
                      xyz_offsets=offsets, candidate_xyz=candidate_xyz,
                      time_logits=time_logits, time_probabilities=time_logits.softmax(-1))
        output.update(self.choose(output))
        return output

    def choose(self, output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        point_index = output["point_logits"].argmax(-1)
        time_index = output["time_logits"].argmax(-1)
        batch = point_index.shape[0]
        rows = torch.arange(batch, device=point_index.device)
        return dict(goal_xyz=output["candidate_xyz"][rows, point_index],
                    goal_time=self.time_values[time_index], point_index=point_index,
                    time_index=time_index,
                    point_probability=output["point_probabilities"][rows, point_index],
                    time_probability=output["time_probabilities"][rows, time_index])


class PretrainedSkillPolicy(nn.Module):
    """Independent feedback policy, callable with or without a real trajectory.

    Missing trajectories are explicitly unconsumed; no dummy forecast is made.
    Goals supplied by expert demonstrations are permitted only for offline skill
    pretraining. Runtime callers must supply goals from the candidate network.
    """
    def __init__(self, workspace_center: Sequence[float], workspace_scale: Sequence[float],
                 scene_dim: int = 128, skill_dim: int = 64, hidden: int = 256):
        super().__init__()
        center, scale = _workspace(workspace_center, workspace_scale)
        self.register_buffer("workspace_center", center)
        self.register_buffer("workspace_scale", scale)
        self.scene_dim, self.skill_dim = scene_dim, skill_dim
        self.feedback_encoder = nn.Sequential(nn.Linear(scene_dim + skill_dim + 15 + 3 + 1, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
        self.trajectory_encoder = nn.Sequential(nn.Linear(20, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.xyz_head = nn.Linear(hidden, 3)
        self.rotation_head = nn.Linear(hidden, 6)
        self.gripper_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.rotation_head.weight)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, scene_features: Tensor, skill_latent: Tensor, robot: Tensor,
                goal_xyz: Tensor, goal_time: Tensor,
                trajectory: Tensor | None = None, trajectory_mask: Tensor | None = None) -> dict[str, Tensor | bool]:
        batch = scene_features.shape[0]
        for name, value, shape in (("scene_features", scene_features, (batch, self.scene_dim)),
                                   ("skill_latent", skill_latent, (batch, self.skill_dim)),
                                   ("robot", robot, (batch, 15)), ("goal_xyz", goal_xyz, (batch, 3))):
            _floating(name, value, shape)
            _finite(name, value)
        _floating("goal_time", goal_time, (batch,))
        _finite("goal_time", goal_time)
        if not bool((goal_time >= 0).all()):
            raise ValueError("goal_time must be nonnegative relative seconds")
        _same_device(scene_features, skill_latent, robot, goal_xyz, goal_time)
        if (trajectory is None) != (trajectory_mask is None):
            raise ValueError("trajectory and trajectory_mask must be supplied together")
        normalized_goal = (goal_xyz - self.workspace_center) / self.workspace_scale
        hidden = self.feedback_encoder(torch.cat((scene_features, skill_latent, robot, normalized_goal, goal_time[:, None]), -1))
        consumed = trajectory is not None
        if consumed:
            clean, mask = _trajectory(trajectory, trajectory_mask, batch)
            _same_device(scene_features, clean)
            summary = clean.sum(1) / mask.sum(1, keepdim=True)
            hidden = hidden + self.trajectory_encoder(summary)
        xyz = self.workspace_center + self.workspace_scale * self.xyz_head(hidden).tanh()
        rotation_6d = self.rotation_head(hidden)
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)
        quaternion = matrix_to_quaternion_xyzw(rotation_matrix)
        gripper_logits = self.gripper_head(hidden)
        opening = gripper_logits.argmax(-1).to(xyz.dtype)
        action = torch.cat((xyz, quaternion, opening[:, None]), -1)
        return dict(xyz=xyz, rotation_6d=rotation_6d, rotation_matrix=rotation_matrix,
                    quaternion_xyzw=quaternion, gripper_logits=gripper_logits,
                    action=action, trajectory_consumed=consumed)


class HierarchyNetwork(nn.Module):
    """Compose independent modules, requiring a real forecast for full inference."""
    def __init__(self, workspace_center: Sequence[float], workspace_scale: Sequence[float],
                 scene_encoder: CausalSceneSkillEncoder | None = None,
                 candidate_network: CandidatePointTransformer | None = None,
                 skill_policy: PretrainedSkillPolicy | None = None):
        super().__init__()
        self.scene_encoder = scene_encoder if scene_encoder is not None else CausalSceneSkillEncoder()
        self.candidate_network = candidate_network if candidate_network is not None else CandidatePointTransformer(workspace_center, workspace_scale)
        self.skill_policy = skill_policy if skill_policy is not None else PretrainedSkillPolicy(workspace_center, workspace_scale)

    @property
    def encoder(self) -> CausalSceneSkillEncoder:
        return self.scene_encoder

    @property
    def candidate(self) -> CandidatePointTransformer:
        return self.candidate_network

    @property
    def skill(self) -> PretrainedSkillPolicy:
        return self.skill_policy

    @staticmethod
    def _current_robot(robot_history: Tensor, lengths: Tensor) -> Tensor:
        rows = torch.arange(robot_history.shape[0], device=robot_history.device)
        return robot_history[rows, lengths.to(robot_history.device) - 1]

    def pretrain_skill(self, rgbd: Tensor, robot_history: Tensor, lengths: Tensor,
                       goal_xyz: Tensor, goal_time: Tensor,
                       trajectory: Tensor | None = None, trajectory_mask: Tensor | None = None) -> dict[str, object]:
        scene = self.scene_encoder(rgbd, robot_history, lengths)
        robot = self._current_robot(robot_history, lengths)
        skill = self.skill_policy(scene["scene_features"], scene["skill_latent"], robot,
                                  goal_xyz, goal_time, trajectory, trajectory_mask)
        return dict(scene=scene, skill=skill, goal_source="offline_expert_conditioning",
                    candidate_network_consumed=False)

    def forward(self, rgbd: Tensor, robot_history: Tensor, lengths: Tensor,
                trajectory: Tensor, trajectory_mask: Tensor) -> dict[str, object]:
        scene = self.scene_encoder(rgbd, robot_history, lengths)
        robot = self._current_robot(robot_history, lengths)
        candidate = self.candidate_network(trajectory, trajectory_mask, scene["skill_latent"], robot)
        skill = self.skill_policy(scene["scene_features"], scene["skill_latent"], robot,
                                  candidate["goal_xyz"], candidate["goal_time"], trajectory, trajectory_mask)
        return dict(scene=scene, candidate=candidate, skill=skill,
                    goal_source="predicted_candidate_imitation",
                    candidate_probability_semantics=self.candidate_network.probability_semantics)

    def forward_skill(self, rgbd: Tensor, robot_history: Tensor, lengths: Tensor,
                      goal_xyz: Tensor, goal_time: Tensor,
                      trajectory: Tensor | None = None, trajectory_mask: Tensor | None = None) -> dict[str, object]:
        """Explicit alias for offline teacher-goal skill pretraining."""
        return self.pretrain_skill(rgbd, robot_history, lengths, goal_xyz, goal_time,
                                   trajectory, trajectory_mask)
