"""Reuse the causal scene encoder's feature path, never its action policy.

The eight labels denote operations, shared across normal/high-speed episodes.
They do not identify a particular cup/peg or resolve an unseen instruction.
"""
from __future__ import annotations

import torch
from torch import nn

from ..hierarchy_20261030.networks import CausalSceneSkillEncoder
from . import VERSION

OPERATIONS = (
    'beat_the_rotating_buzz', 'insert_onto_rotating_peg',
    'moving_basketball_in_hoop', 'pick_moving_target_on_the_table',
    'place_cups_on_rotating_frame', 'push_moving_button',
    'put_rubbish_in_moving_bin', 'remove_cups_from_rotating_frame',
)


def semantic_label(task: str) -> int:
    if not isinstance(task, str):
        raise ValueError('explicit offline task string required')
    family = task.removesuffix('_high_speed')
    if family not in OPERATIONS or task not in (family, family + '_high_speed'):
        raise ValueError('exact declared normal/high task version required')
    return OPERATIONS.index(family)


class SemanticTaskEncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct only to reuse the exact tested public visual/history path.
        # No state dict is loaded. Its 11-way head, E table and old policy are
        # not installed in this model or used as semantic supervision.
        source = CausalSceneSkillEncoder(scene_dim=128)
        self.camera_encoder = source.camera_encoder
        self.camera_fusion = source.camera_fusion
        self.robot_encoder = source.robot_encoder
        self.history_encoder = source.history_encoder
        self.feature_norm = source.feature_norm
        self.task_projection = nn.Linear(128, 16)
        self.operation_classifier = nn.Linear(16, len(OPERATIONS))

    def forward(self, *, rgbd, robot_history, lengths):
        """Past-prefix inputs only: B,T,4 cameras,RGBD,64,64 and robot15."""
        if (not isinstance(rgbd, torch.Tensor) or not rgbd.is_floating_point()
                or rgbd.ndim != 6 or tuple(rgbd.shape[2:]) != (4, 4, 64, 64)
                or min(rgbd.shape[:2]) < 1):
            raise ValueError('floating RGBD[B,T,4,4,64,64] required')
        batch, steps = rgbd.shape[:2]
        if (not isinstance(robot_history, torch.Tensor)
                or not robot_history.is_floating_point()
                or robot_history.shape != (batch, steps, 15)
                or robot_history.device != rgbd.device):
            raise ValueError('same-device public measured robot[B,T,15] required')
        if (not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,)
                or lengths.dtype not in (torch.int32, torch.int64)
                or lengths.device != rgbd.device
                or not bool(((lengths >= 1) & (lengths <= steps)).all())):
            raise ValueError('same-device valid prefix lengths[B] required')
        valid = torch.arange(steps, device=rgbd.device)[None] < lengths[:, None]
        if (not bool(torch.isfinite(rgbd[valid]).all())
                or not bool(torch.isfinite(robot_history[valid]).all())):
            raise ValueError('valid public prefix contains nonfinite values')
        frames = rgbd[valid]
        camera = self.camera_encoder(frames.reshape(-1, 4, 64, 64)).reshape(-1, 256)
        visual = self.camera_fusion(camera)
        robot = self.robot_encoder(robot_history[valid])
        encoded = rgbd.new_zeros(batch, steps, 128)
        encoded[valid] = torch.cat((visual, robot), -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        scene = self.feature_norm(hidden[-1])
        task_e = self.task_projection(scene).reshape(batch, 2, 8).softmax(-1).reshape(batch, 16)
        logits = self.operation_classifier(task_e)
        return dict(task_e=task_e, operation_logits=logits,
                    operation_probabilities=logits.softmax(-1))

    def spec(self):
        return dict(version=VERSION, semantic_operations=list(OPERATIONS),
                    evaluated_task_versions=16, e_dimension=16,
                    simnorm_groups=2, simnorm_group_size=8,
                    public_input='past-prefix RGBD and measured robot proprioception',
                    metadata_task_is_forward_input=False,
                    Native_debug_or_future_input=False,
                    old_action_policy_or_E_table_loaded=False,
                    motion_context_c_separate=True,
                    current_uniform_e_P4_substitution_allowed=False,
                    scene_instruction_ambiguity_resolved=False,
                    all16_closed_loop_ready=False)


def operation_loss(output, labels):
    logits = output['operation_logits']
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
            or labels.shape != logits.shape[:1] or labels.device != logits.device
            or not bool(((labels >= 0) & (labels < len(OPERATIONS))).all())):
        raise ValueError('offline int64 operation labels[B] in0..7 required')
    return nn.functional.cross_entropy(logits, labels)
