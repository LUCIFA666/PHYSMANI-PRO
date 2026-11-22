"""Original matched masked-time mathematics, without an untrained score loss."""
from ..cube_candidate_matched_20261104_v2.losses import candidate_loss as _matched_loss


def candidate_loss(output,target_position,target_rotation,target_dt,target_time_valid):
    return _matched_loss(output,target_position,target_rotation,target_dt,target_time_valid,
        compatibility_logits=None,ranking_weight=0.)
