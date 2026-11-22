"""Small task branch; visual perception and motion context are separate.

Inputs are shared, observation-derived scene features. Task labels supervise
training and are never forward inputs. No old encoder, policy, or weights are
loaded by this component. It is a new, untrained interface prototype.
"""
import torch
from torch import nn

from . import VERSION


class CompactTaskLatentHead(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        if type(classes) is not int or classes not in (8, 16):
            raise ValueError('Explicit eight operation or sixteen version labels required')
        self.classes = classes
        self.projection = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 16))
        self.classifier = nn.Linear(16, classes)

    def forward(self, scene_features):
        if (not isinstance(scene_features, torch.Tensor)
                or not scene_features.is_floating_point()
                or scene_features.ndim != 2 or scene_features.shape[1] != 512
                or scene_features.shape[0] < 1
                or not bool(torch.isfinite(scene_features).all())):
            raise ValueError('Finite shared public scene features [B,512] required')
        raw = self.projection(scene_features).float()
        # Classification gradients act on raw features, independently of the
        # SimNorm distribution passed to the candidate/operation modules.
        logits = self.classifier(raw)
        return dict(task_e=raw.reshape(-1, 2, 8).softmax(-1).reshape(-1, 16),
                    raw_task_latent=raw, operation_logits=logits,
                    operation_probabilities=logits.float().softmax(-1))

    def spec(self):
        return dict(version=VERSION, input_width=512, hidden_width=128,
                    task_e_dimension=16, simnorm_groups=2, simnorm_group_size=8,
                    classes=self.classes, parameters=sum(p.numel() for p in self.parameters()),
                    visual_backbone_included=False, motion_context_c_separate=True,
                    task_label_is_forward_input=False, trained=False,
                    compatible_with_old_candidate_weights_without_adaptation=False)
