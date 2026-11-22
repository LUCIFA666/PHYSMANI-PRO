"""Instruction conditioning through the existing candidate task-token port.

Motion c remains a separate physical input. Eight-operation / two-speed heads
are auxiliary supervised outputs, not task IDs or measured speed inputs.
Only Cube action weights can be initialized from the completed baseline.
"""
from copy import deepcopy

import torch
from torch import nn

from . import SCHEMA, VERSION
from .text import FrozenCLIPTextEncoder
from ..candidate_transformer_cube_20261104_v1.model import CandidateConfig
from ..cube_candidate_matched_20261104_v2 import FRAME
from ..local_mpc_p4_20261103.model import CandidateTransformerPilot
from ..semantic_task_e_20261104_v1.model import OPERATIONS


class InstructionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 128), nn.GELU())
        self.to_task_e = nn.Linear(128, 16)
        # Exact uniform task_e at initialization preserves the fitted baseline.
        nn.init.zeros_(self.to_task_e.weight)
        nn.init.zeros_(self.to_task_e.bias)
        self.operation_head = nn.Linear(16, len(OPERATIONS))
        self.speed_head = nn.Linear(16, 2)

    def forward(self, instruction_features):
        x = instruction_features
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
                or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 1024
                or not bool(torch.isfinite(x).all())
                or not torch.allclose(x.norm(dim=-1), x.new_ones(x.shape[0]), atol=1e-4, rtol=0)):
            raise ValueError('Finite L2-normalized FP32 CLIP text[B,1024] required')
        task_e = self.to_task_e(self.hidden(x)).reshape(-1, 2, 8).softmax(-1).flatten(1)
        return dict(task_e=task_e, operation_logits=self.operation_head(task_e),
                    speed_logits=self.speed_head(task_e))


class InstructionConditionedCubeCandidate(nn.Module):
    """Trainable language adapter + matched physical candidate architecture.

    This is a NEW development family, never accepted by the frozen runtime.
    Encoding text is separate so it can be done once per instruction rather
    than once per simulation tick. Do not cache task_e during joint training.
    """
    def __init__(self, config=CandidateConfig()):
        super().__init__()
        if type(config) is not CandidateConfig:
            raise TypeError('Explicit CandidateConfig required')
        self.config = config
        self.candidate = CandidateTransformerPilot(**config.to_dict())
        self.language = InstructionAdapter()
        self.warm_start = None

    def forward(self, *, future_tokens, robot, geometry, context_c, instruction_features):
        semantic = self.language(instruction_features)
        output = self.candidate(future_tokens, robot, geometry, context_c, semantic['task_e'])
        return dict(output, **semantic)

    def spec(self):
        physics = self.candidate.spec()
        physics['context_and_task']['task_embedding_source'] = 'new_family_instruction_adapter'
        physics['context_and_task']['automatic_task_classification_learned'] = False
        physics['output_pose_frame'] = 'causal public forecast reference coordinates at actual contact'
        physics['output_rotation_frame'] = 'TCP orientation relative to the same public forecast basis'
        physics['explicit_prior_semantics'].update(
            approach_direction='public forecast-frame backoff toward pregrasp',
            pregrasp_reference='predicted public centre + public reference rotation @ (contact offset + backoff)')
        return dict(schema=SCHEMA, version=VERSION, model_config=self.config.to_dict(),
                    target_frame=FRAME, candidate_physics=physics,
                    text_features=dict(dimension=1024, encoder='frozen_CLIP_RN50_text', l2_normalized=True),
                    task_e=dict(dimension=16, simnorm_groups=2, group_size=8,
                                source='instruction_only_trainable_adapter', initial_value=0.125),
                    auxiliary_operation_labels=list(OPERATIONS),
                    auxiliary_speed_labels=['ordinary', 'high_speed'],
                    physical_motion_context_c_separate=True,
                    measured_velocity_from_language=False, Native_target_state_input=False,
                    task_ID_or_scene_name_forward_input=False,
                    language_conditioned_action_training_required=True,
                    auxiliary_heads_trained_by_construction=False,
                    supported_action_training_scope='Cube_only',
                    all16_closed_loop_ready=False, production_runtime_compatible=False,
                    warm_start=deepcopy(self.warm_start))

    @classmethod
    def from_matched_checkpoint(cls, checkpoint, *, expected_sha256, forecast_sha256):
        """Read-only, hash-checked warm initialization, not language training."""
        from ..cube_corrected_live_20261104_v1.matched_candidate_runtime import MatchedCorrectedCandidateInference
        source = MatchedCorrectedCandidateInference(
            checkpoint, expected_sha256=expected_sha256, forecast_sha256=forecast_sha256)
        result = cls(source._model.config)
        result.candidate.load_state_dict(source._model.state_dict(), strict=True)
        result.warm_start = dict(checkpoint_sha256=expected_sha256,
                                 forecast_checkpoint_sha256=forecast_sha256,
                                 old_sources_or_weights_modified=False,
                                 instruction_action_optimizer_steps=0)
        return result


class LanguageConditionedCubePolicy(nn.Module):
    """Raw instruction + causal public features -> experimental candidates.

    No operation argmax automatically dispatches an existing executor. A new
    matched checkpoint and closed-loop evaluation are required for deployment.
    """
    def __init__(self, text_encoder, candidate):
        super().__init__()
        if type(text_encoder) is not FrozenCLIPTextEncoder or type(candidate) is not InstructionConditionedCubeCandidate:
            raise TypeError('Explicit frozen text tower and new candidate family required')
        self.text_encoder, self.candidate = text_encoder, candidate

    def forward(self, *, instructions, future_tokens, robot, geometry, context_c):
        text = self.text_encoder(instructions)
        return self.candidate(future_tokens=future_tokens, robot=robot, geometry=geometry,
                              context_c=context_c, instruction_features=text)
