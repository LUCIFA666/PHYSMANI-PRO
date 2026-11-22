"""Future Native positions are offline labels only, never recurrent inputs."""
import math
import torch
from torch.nn import functional as F


def physical_motion_loss(output, target_positions, target_valid, target_times, *,
                         position_scale_m=.02, residual_regularization_weight=1e-4):
    predicted = output['positions']
    if (predicted.ndim != 3 or predicted.shape[1:] != (17, 3)
            or target_positions.shape != predicted.shape or target_positions.dtype != predicted.dtype
            or target_valid.shape != predicted.shape[:2] or target_valid.dtype != torch.bool
            or target_times.shape != predicted.shape[:2] or output['times'].shape != target_times.shape
            or any(v.device != predicted.device for v in (target_positions, target_valid, target_times))):
        raise ValueError('Aligned positions[B,17,3], bool label mask and actual label times required')
    if (isinstance(position_scale_m,bool) or isinstance(residual_regularization_weight,bool)
            or not math.isfinite(position_scale_m) or not math.isfinite(residual_regularization_weight)
            or not position_scale_m > 0 or residual_regularization_weight < 0
            or not bool(torch.isfinite(target_times).all())
            or not torch.allclose(target_times.to(torch.float64), output['times'].to(torch.float64), atol=1e-6, rtol=0)):
        raise ValueError('Same actual future timestamps and valid fixed loss scales required')
    valid = target_valid.clone()
    valid[:, 0] = False  # Current pose is anchored to actual public input, not replaced by GT.
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target_positions[valid]).all()):
        raise ValueError('Finite predictions and all available future labels required')
    target = torch.where(valid[..., None], target_positions, predicted.detach())
    error = predicted - target
    huber = F.smooth_l1_loss(error / position_scale_m, torch.zeros_like(error), reduction='none').mean(-1)
    count = valid.sum(1)
    per_example_position = (huber * valid).sum(1) / count.clamp_min(1)
    acceleration = output['residual_accelerations']
    regularization = acceleration.square().mean((1, 2))
    per_example = per_example_position + residual_regularization_weight * regularization
    per_example = torch.where(count > 0, per_example, per_example * 0.)
    cv_error = torch.linalg.vector_norm(output['cv_positions'] - target, dim=-1)
    position_error = torch.linalg.vector_norm(error, dim=-1)
    return dict(per_example=per_example, position_loss=per_example_position,
        residual_acceleration_regularization=regularization, valid_future_count=count,
        position_error_m=torch.where(valid, position_error, 0.),
        cv_position_error_m=torch.where(valid, cv_error, 0.), valid_mask=valid,
        targets_are_runtime_inputs=False, orientation_label_loss_used=False,
        accuracy_threshold_or_confidence_interval=False)
