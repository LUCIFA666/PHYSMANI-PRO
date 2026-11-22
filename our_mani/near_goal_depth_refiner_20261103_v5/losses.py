"""One v5 loss change: real UNKNOWN pixels learn an abstention surrogate.

The original known-label CE and Dice are imported unchanged. UNKNOWN is
never assigned a background target. Uniform2 is a reject-training surrogate,
not a claim that the unknown object's probability is statistically one half.
"""
import math

import torch

from ..near_goal_depth_refiner_20261103.model import binary_segmentation_loss
from ..near_goal_depth_refiner_20261103.data import FEATURE_NAMES


PROTOCOL_SHA256 = '472c14d9b10ecbd06a67f9bdd545543ec235f906ea571e322c5c660775b94710'
UNKNOWN_WEIGHT = 1.0
REAL_CROP_FEATURE_INDEX = 8
if FEATURE_NAMES[REAL_CROP_FEATURE_INDEX] != 'inside_original_image':
    raise ValueError('The immutable public feature8 must be real crop validity')


def unknown_abstention_loss(logits, labels, inside_original_image):
    """One KL(Uniform2 || p) mean per view, over real UNKNOWN pixels only."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 2
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or not isinstance(labels, torch.Tensor) or labels.shape != logits.shape[:1]+logits.shape[2:]
            or labels.is_floating_point() or labels.dtype == torch.bool
            or labels.device != logits.device
            or not isinstance(inside_original_image, torch.Tensor)
            or inside_original_image.dtype != torch.bool or inside_original_image.shape != labels.shape
            or inside_original_image.device != logits.device):
        raise ValueError('Finite2logits, integer semantic and aligned real-crop bool required')
    if torch.any((labels != 0) & (labels != 1) & (labels != 255)):
        raise ValueError('Only known0/rawCube1/UNKNOWN255 labels are allowed')
    unknown = (labels == 255) & inside_original_image
    if not torch.any(unknown):
        # Connected zero without summing possibly large irrelevant logits.
        return logits[:, 0, 0, 0]*0
    logp = logits.log_softmax(1)
    kl = -.5*(logp[:, 0]+logp[:, 1])-math.log(2.)
    # The exact expression is >=0; floor only possible floating roundoff.
    kl = kl.clamp_min(0.)
    count = unknown.sum((1, 2)).clamp_min(1)
    return (kl*unknown).sum((1, 2))/count


def per_view_components(logits, labels, inside_original_image):
    """Unmodified known loss plus exactly one separately normalized U term."""
    known = binary_segmentation_loss(logits, labels)
    abstention = unknown_abstention_loss(logits, labels, inside_original_image)
    if not torch.any((labels == 255) & inside_original_image):
        total = known
    else:
        total = known+UNKNOWN_WEIGHT*abstention
    return dict(known=known, abstention=abstention, total=total)


def loss_batch(model, batch, device):
    """One original four-view forward; equal view and shot weighting."""
    if not isinstance(batch, dict) or set(batch) != {'features', 'labels'}:
        raise ValueError('Only cached public features and semantic supervision allowed')
    x, y = batch['features'].to(device), batch['labels'].to(device)
    if (x.ndim != 5 or x.shape[1:] != (4, 9, 64, 64) or x.dtype != torch.float32
            or not torch.isfinite(x).all() or y.shape != (len(x), 4, 64, 64) or len(x) == 0):
        raise ValueError('Exact nonempty current float32 Bx4x9x64x64 feature batch required')
    crop = x[:, :, REAL_CROP_FEATURE_INDEX]
    if not torch.all((crop == 0) | (crop == 1)):
        raise ValueError('Immutable inside_original_image feature must remain binary')
    count = len(x)
    flat = x.reshape(count*4, 9, 64, 64)
    logits = model(flat)
    components = per_view_components(logits, y.reshape(count*4, 64, 64),
                                     crop.reshape(count*4, 64, 64) == 1)
    return components['total'].reshape(count, 4).mean(1).mean()
