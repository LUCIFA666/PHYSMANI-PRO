"""Large scene/history encoder with the same e16 output contract.

This new module does not change or load the earlier skill/action policies.
All valid frames are past/current observations. Metadata labels stay outside
forward. The output encodes an operation, not a target-instance identity.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..semantic_task_e_20261104_v1.model import OPERATIONS
from . import VERSION


@dataclass(frozen=True)
class EncoderConfig:
    width: int = 1024
    layers: int = 16
    heads: int = 16
    feedforward_multiplier: int = 4
    cnn_base: int = 64
    spatial_grid: int = 2
    maximum_history: int = 5
    dropout: float = .1

    def __post_init__(self):
        integer = (self.width, self.layers, self.heads, self.feedforward_multiplier,
                   self.cnn_base, self.spatial_grid, self.maximum_history)
        if (any(type(value) is not int or value < 1 for value in integer)
                or self.width % self.heads or self.width < 16
                or self.spatial_grid > 4 or self.maximum_history > 20
                or not isinstance(self.dropout, (int, float))
                or not 0 <= self.dropout < 1):
            raise ValueError('explicit valid transformer/CNN architecture required')


class LargeSemanticTaskEncoderV1(nn.Module):
    def __init__(self, config=EncoderConfig()):
        super().__init__()
        if type(config) is not EncoderConfig:
            raise ValueError('exact data-only EncoderConfig required')
        self.config = config
        blocks, incoming = [], 4
        for outgoing in (config.cnn_base, 2*config.cnn_base,
                         4*config.cnn_base, 8*config.cnn_base):
            blocks.extend((nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1),
                           nn.GELU(), nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.GELU()))
            incoming = outgoing
        self.camera_encoder = nn.Sequential(*blocks,
            nn.AdaptiveAvgPool2d((config.spatial_grid, config.spatial_grid)))
        self.visual_projection = nn.Linear(incoming, config.width)
        self.robot_projection = nn.Sequential(nn.Linear(15, config.width), nn.GELU(),
                                              nn.Linear(config.width, config.width))
        # Camera identity is not embedded. Same spatial cell shares its token
        # across cameras; camera appearance shortcuts still require validation.
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, 1,
            config.spatial_grid**2, config.width))
        self.time_embedding = nn.Parameter(torch.zeros(1, config.maximum_history, 1, config.width))
        self.modality_embedding = nn.Parameter(torch.zeros(2, config.width))
        self.query = nn.Parameter(torch.zeros(1, 1, config.width))
        layer = nn.TransformerEncoderLayer(config.width, config.heads,
            dim_feedforward=config.width*config.feedforward_multiplier,
            dropout=float(config.dropout), activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers,
                                                 enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.width)
        self.task_projection = nn.Linear(config.width, 16)
        self.operation_classifier = nn.Linear(16, len(OPERATIONS))
        for value in (self.spatial_embedding, self.time_embedding,
                      self.modality_embedding, self.query):
            nn.init.normal_(value, std=.02)

    def forward(self, *, rgbd, robot_history, lengths):
        if (not isinstance(rgbd, torch.Tensor) or not rgbd.is_floating_point()
                or rgbd.ndim != 6 or rgbd.shape[2:4] != (4, 4)
                or rgbd.shape[-2:] not in ((64, 64), (128, 128), (256, 256))
                or rgbd.shape[0] < 1 or not 1 <= rgbd.shape[1] <= self.config.maximum_history):
            raise ValueError('public RGBD[B,T,4,4,R,R],R64/128/256 required')
        batch, steps, _, _, height, width = rgbd.shape
        if (not isinstance(robot_history, torch.Tensor)
                or not robot_history.is_floating_point()
                or robot_history.shape != (batch, steps, 15)
                or robot_history.device != rgbd.device):
            raise ValueError('same-device measured robot[B,T,15] required')
        if (not isinstance(lengths, torch.Tensor) or lengths.dtype not in (torch.int32, torch.int64)
                or lengths.shape != (batch,) or lengths.device != rgbd.device
                or not bool(((lengths >= 1) & (lengths <= steps)).all())):
            raise ValueError('same-device valid prefix lengths[B] required')
        valid = torch.arange(steps, device=rgbd.device)[None] < lengths[:, None]
        if (not bool(torch.isfinite(rgbd[valid]).all())
                or not bool(torch.isfinite(robot_history[valid]).all())):
            raise ValueError('valid public prefix has nonfinite values')
        # Official HDF gripper_open is altered using Native grasp state.
        # This unavailable channel is an explicit zero placeholder, never a
        # contact/held observation or an expert-opening replacement.
        if not bool((robot_history[valid][:, -1] == 0).all()):
            raise ValueError('unobserved gripper channel must be explicit zero')
        count, grid = int(valid.sum()), self.config.spatial_grid**2
        camera = self.camera_encoder(rgbd[valid].reshape(count*4, 4, height, width))
        camera = camera.flatten(2).transpose(1, 2)
        camera = self.visual_projection(camera).reshape(count, 4, grid, self.config.width)
        # BF16 autocast may make encoder outputs BF16 while raw public tensors
        # remain FP32. Indexed writes require matching dtypes.
        visual = camera.new_zeros(batch, steps, 4, grid, self.config.width)
        visual[valid] = camera
        visual = visual + self.spatial_embedding + self.time_embedding[:, :steps, :, None]
        visual = visual + self.modality_embedding[0]
        visual = visual.reshape(batch, steps, 4*grid, self.config.width)
        valid_robot = self.robot_projection(robot_history[valid])
        robot = valid_robot.new_zeros(batch, steps, self.config.width)
        robot[valid] = valid_robot
        robot = robot + self.time_embedding[:, :steps, 0] + self.modality_embedding[1]
        tokens = torch.cat((visual, robot[:, :, None]), 2).reshape(batch, -1, self.config.width)
        masked = (~valid)[:, :, None].expand(batch, steps, 4*grid+1).reshape(batch, -1)
        masked = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=rgbd.device), masked), 1)
        tokens = torch.cat((self.query.expand(batch, -1, -1), tokens), 1)
        scene = self.final_norm(self.transformer(tokens, src_key_padding_mask=masked)[:, 0])
        # Keep the public SimNorm/probability contract in FP32 even while the
        # CNN/transformer/linear operations use BF16 autocast.
        task_e = self.task_projection(scene).float().reshape(batch, 2, 8).softmax(-1).reshape(batch, 16)
        logits = self.operation_classifier(task_e)
        return dict(task_e=task_e, operation_logits=logits,
                    operation_probabilities=logits.float().softmax(-1))

    def spec(self):
        return dict(version=VERSION, architecture=asdict(self.config),
                    operations=list(OPERATIONS), semantic_operations=8, evaluated_versions=16,
                    task_e_dimension=16, simnorm_groups=2, simnorm_group_size=8,
                    public_input='causal RGBD/proprioceptive prefixes only',
                    metadata_task_input=False, Native_or_future_input=False,
                    proprioception='measured q7+TCP pose7+unobserved zero placeholder1',
                    gripper_open_input_observed=False,
                    legacy_action_policy_loaded=False, motion_c_separate=True,
                    target_instance_identity_inferred=False,
                    current_uniform_e_P4_substitution_allowed=False,
                    all16_closed_loop_ready=False)


def parameter_count(config):
    """Exact shapes on meta device; no data, checkpoint, or real NN forward."""
    with torch.device('meta'):
        model = LargeSemanticTaskEncoderV1(config)
    return sum(value.numel() for value in model.parameters())
