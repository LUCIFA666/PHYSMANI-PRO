"""Original pose/time supervision plus optional instruction auxiliary labels."""
import math

import torch
from torch.nn import functional as F

from ..cube_matched_candidate_20261104_v1.losses import candidate_loss


def instruction_candidate_loss(output, target_position, target_rotation, target_dt,
                               target_time_valid, *, operation_labels=None,
                               speed_labels=None, operation_weight=0.1, speed_weight=0.1):
    physical = candidate_loss(output, target_position, target_rotation, target_dt, target_time_valid)
    loss = physical['per_example']
    result = dict(physical)
    for name, labels, weight in (('operation', operation_labels, operation_weight),
                                 ('speed', speed_labels, speed_weight)):
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Finite nonnegative auxiliary weight required')
        if labels is None:
            if weight != 0:
                raise ValueError(name + ' labels required unless its auxiliary weight is zero')
            continue
        logits = output[name + '_logits']
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
                or labels.shape != (len(loss),) or labels.device != logits.device
                or not bool(((labels >= 0) & (labels < logits.shape[-1])).all())):
            raise ValueError('Offline int64 ' + name + ' labels[B] required')
        value = F.cross_entropy(logits, labels, reduction='none')
        result[name + '_per_example'] = value
        loss = loss + weight * value
    # Caller must retain whole-parent split and parent-balanced aggregation.
    result['per_example'] = loss
    result['candidate_success_probability_trained'] = False
    return result
