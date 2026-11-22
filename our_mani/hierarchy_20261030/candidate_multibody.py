"""A source-blind, multi-body handoff-point Transformer.

Each real body forecast retains its own clock and a fixed public prior-role
token. Equal times on distinct bodies are valid; no artificial clock offsets
or body-averaged trajectories are manufactured. Output scores are imitation
probabilities, not calibrated native success probabilities.
"""
from __future__ import annotations

import torch
from torch import nn

from .networks import CandidatePointTransformer, _finite, _floating, _same_device


class MultiBodyCandidateTransformer(CandidatePointTransformer):
    def __init__(self, workspace_center, workspace_scale, *, role_names,
                 skill_dim=64, scene_dim=128, model_dim=128, heads=4, layers=2):
        if not role_names or len(role_names) != len(set(role_names)):
            raise ValueError("fixed unique public prior-role names required")
        super().__init__(workspace_center, workspace_scale, skill_dim, model_dim,
                         heads, layers, dropout=0.)
        self.role_names = tuple(role_names)
        self.scene_dim = scene_dim
        self.role_embedding = nn.Embedding(len(role_names), model_dim)
        self.scene_projection = nn.Linear(scene_dim, model_dim)

    def forward(self, trajectory, trajectory_mask, skill_latent, robot, scene_features):
        _floating("multi-body trajectory", trajectory, (None, len(self.role_names), 23, 20))
        batch, bodies, steps, _ = trajectory.shape
        if (not isinstance(trajectory_mask, torch.Tensor)
                or trajectory_mask.dtype != torch.bool
                or trajectory_mask.shape != trajectory.shape[:3]):
            raise ValueError("boolean [B,N,23] forecast mask required")
        _same_device(trajectory, trajectory_mask, skill_latent, robot, scene_features)
        _floating("skill_latent", skill_latent, (batch, self.skill_dim))
        _floating("robot", robot, (batch, 15))
        _floating("scene_features", scene_features, (batch, self.scene_dim))
        for name, value in (("skill_latent", skill_latent), ("robot", robot),
                            ("scene_features", scene_features)):
            _finite(name, value)
        if not bool(trajectory_mask.reshape(batch, -1).any(-1).all()):
            raise ValueError("at least one real forecast is required for every handoff")
        _finite("valid forecast tokens", trajectory[trajectory_mask])
        expected_times = self.time_values.to(trajectory)
        for item in range(batch):
            for body in range(bodies):
                valid = trajectory_mask[item, body]
                if bool(valid.any()):
                    if not bool(valid.all()):
                        raise ValueError("this version accepts complete per-body 23-step real forecasts")
                    if not torch.allclose(trajectory[item, body, :, -1], expected_times, atol=1e-6, rtol=0):
                        raise ValueError("per-body forecast clock must match 0..1.1s at 50ms")
        clean = torch.where(trajectory_mask[..., None], trajectory, torch.zeros_like(trajectory))
        roles = self.role_embedding.weight[None, :, None, :]
        tokens = self.trajectory_projection(clean) + self.time_projection(clean[..., -1:]) + roles
        tokens = tokens.reshape(batch, bodies * steps, -1)
        prefix = torch.cat((self.query.expand(batch, -1, -1),
                            self.skill_projection(skill_latent)[:, None],
                            self.robot_projection(robot)[:, None],
                            self.scene_projection(scene_features)[:, None]), 1)
        flat_mask = trajectory_mask.reshape(batch, -1)
        padding = torch.cat((torch.zeros(batch, 4, device=flat_mask.device, dtype=torch.bool), ~flat_mask), 1)
        pooled = self.output_norm(self.transformer(torch.cat((prefix, tokens), 1),
                                                  src_key_padding_mask=padding)[:, 0])
        logits = self.point_head(pooled)
        offsets = self.offset_head(pooled).reshape(batch, 16**3, 3).tanh()/16
        candidates = self.workspace_center + self.workspace_scale * (self.normalized_grid_centers[None] + offsets)
        time_logits = self.time_head(pooled)
        output = dict(point_logits=logits, point_probabilities=logits.softmax(-1),
                      xyz_offsets=offsets, candidate_xyz=candidates,
                      time_logits=time_logits, time_probabilities=time_logits.softmax(-1))
        output.update(self.choose(output))
        return output

    def target_cell(self, xyz):
        normalized = (xyz-self.workspace_center)/self.workspace_scale
        cell_xyz = ((normalized+1)*8).floor().long().clamp(0, 15)
        index = cell_xyz[:, 0]*256 + cell_xyz[:, 1]*16 + cell_xyz[:, 2]
        outside = (normalized.abs() > 1).any(-1)
        return index, outside


def candidate_imitation_loss(model, prediction, target_xyz):
    """Spatial expert handoff label, with a declared fixed 0.5s time contract."""
    index, outside = model.target_cell(target_xyz)
    rows = torch.arange(len(index), device=index.device)
    classification = nn.functional.cross_entropy(prediction["point_logits"], index, reduction="none")
    selected_xyz = prediction["candidate_xyz"][rows, index]
    residual = ((selected_xyz-target_xyz)/(model.workspace_scale/8)).square().mean(-1)
    time_target = torch.full_like(index, 10)
    timing = nn.functional.cross_entropy(prediction["time_logits"], time_target, reduction="none")
    return classification + .1*residual + .1*timing, outside
