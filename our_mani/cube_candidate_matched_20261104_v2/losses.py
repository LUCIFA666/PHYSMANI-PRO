"""Pose/time demonstration compatibility, never physical failure probabilities."""
import math
import torch
from ..local_mpc_p4_20261103.model import parallel_gripper_rotation_distance,_proper_rotation

SCORE_SEMANTICS='relative_compatibility_with_one_observed_successful_demonstration_not_physical_success_probability'


def candidate_loss(output,target_position,target_rotation,target_dt,target_time_valid,*,compatibility_logits=None,ranking_weight=0.):
    """Full-pose supervision, bounded-horizon true-time mask, optional rank term.

    With no ranking head this returns coverage loss and explicitly no ranking
    qualification. Rank targets softly describe distance to *this* demonstrated
    successful grasp; they do not label seven unexecuted alternatives failures.
    A new model/runtime and explicit consumer are needed to use such a head.
    """
    p=output['positions'];r=output['rotation_matrices'];tau=output['times'];n=len(p)
    if p.shape!=(n,8,3)or r.shape!=(n,8,3,3)or tau.shape!=(n,8):raise ValueError('Exact K8 proposal tensors required')
    if target_position.shape!=(n,3)or target_rotation.shape!=(n,3,3)or target_dt.shape!=(n,)or target_time_valid.shape!=(n,)or target_time_valid.dtype is not torch.bool or target_time_valid.device!=p.device:
        raise ValueError('Exact positive pose, actual tau and Boolean time-valid mask required')
    tensors=(p,r,tau,target_position,target_rotation,target_dt)
    if any(t.device!=p.device or t.dtype!=p.dtype or not bool(torch.isfinite(t).all())for t in tensors):
        raise ValueError('Shared finite floating dtype/device required')
    if isinstance(ranking_weight,bool)or not math.isfinite(ranking_weight)or ranking_weight<0:raise ValueError('Explicit finite nonnegative rank weight required')
    if bool((target_dt<=0).any()):raise ValueError('Actual positive contact tau must be retained')
    if bool((target_time_valid&((target_dt<.05-1e-7)|(target_dt>.8+1e-7))).any()):
        raise ValueError('Far actual tau cannot be time-valid or clipped')
    if bool(((tau<.05-1e-7)|(tau>.8+1e-7)).any()):raise ValueError('Deployment time remains bounded')
    _proper_rotation(r,(n,8,3,3),'proposal rotation')
    _proper_rotation(target_rotation,(n,3,3),'target rotation')
    pe=torch.linalg.vector_norm(p-target_position[:,None],dim=-1)
    re=parallel_gripper_rotation_distance(r,target_rotation[:,None])
    te=(tau-target_dt[:,None]).abs()
    energy=(pe/.03).square()+(re/.2).square()+torch.where(target_time_valid[:,None],(te/.2).square(),torch.zeros_like(te))
    index=energy.argmin(1)
    coverage=energy.gather(1,index[:,None]).squeeze(1)
    target_distribution=torch.softmax(-energy.detach(),dim=1)
    ranking=torch.zeros_like(coverage)
    if compatibility_logits is not None:
        if compatibility_logits.shape!=(n,8)or compatibility_logits.device!=p.device or compatibility_logits.dtype!=p.dtype or not bool(torch.isfinite(compatibility_logits).all()):
            raise ValueError('Finite matching new-family compatibility logits required')
        ranking=-(target_distribution*torch.log_softmax(compatibility_logits,dim=1)).sum(1)
    elif ranking_weight!=0.:
        raise ValueError('Nonzero rank weight cannot silently omit a ranking head')
    return dict(per_example=coverage+ranking_weight*ranking,coverage_per_example=coverage,
        ranking_per_example=ranking,matching_index=index,pairwise_energy=energy,
        position_error_m=pe,rotation_error_rad=re,time_error_seconds=te,
        target_time_valid=target_time_valid,demonstration_rank_target=target_distribution,
        ranking_head_present=compatibility_logits is not None,score_semantics=SCORE_SEMANTICS,
        unmatched_candidates_are_negative=False,calibrated_physical_success_probability=False,
        far_time_was_clipped=False)
