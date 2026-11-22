"""Original centre/scope loss plus original V5 per-view pixel objective.

Scope-negative does not mean foreground-negative. Centre targets and mask
targets remain independent; neither supplies a model input. All labelled
positive centre rows remain in the centre loss, even if proposal is rejected.
"""
import torch

from ..current_cube_rgbd_centre_20261103_v1.model import centre_presence_loss
from ..near_goal_depth_refiner_20261103_v5.losses import per_view_components

PIXEL_LOSS_WEIGHT = 1.


def prototype_loss(output, *, centre_target_world, centre_valid, presence_target,
                   pixel_labels, inside_original_image):
    centre=centre_presence_loss(output,centre_target_world,centre_valid,presence_target)
    logits=output['pixel_logits']
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=6 or logits.shape[1:]!=(2,4,2,64,64)
            or not isinstance(pixel_labels,torch.Tensor) or pixel_labels.shape!=logits.shape[:3]+(64,64)
            or not isinstance(inside_original_image,torch.Tensor)
            or inside_original_image.shape!=pixel_labels.shape or inside_original_image.dtype!=torch.bool):
        raise ValueError('Separate aligned dual-scale pixel labels and real-crop mask required')
    n=len(logits)
    terms=per_view_components(logits.float().reshape(n*8,2,64,64),
        pixel_labels.reshape(n*8,64,64),inside_original_image.reshape(n*8,64,64))
    known=terms['known'].reshape(n,8).mean(1).mean()
    unknown=terms['abstention'].reshape(n,8).mean(1).mean()
    pixel=terms['total'].reshape(n,8).mean(1).mean()
    return dict(total=centre['total']+PIXEL_LOSS_WEIGHT*pixel,
        centre=centre['centre'],availability=centre['presence'],
        pixel_known=known,pixel_unknown_abstention=unknown,pixel_total=pixel,
        centre_samples=centre['centre_samples'],known_presence_samples=centre['known_presence_samples'],
        unknown_presence_samples=centre['unknown_samples'],pixel_views=n*8,
        proposal_rejection_does_not_shrink_centre_loss_denominator=True,
        model_qualified=False,uncertainty_qualified=False)
