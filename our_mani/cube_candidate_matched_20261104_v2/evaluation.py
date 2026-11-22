"""Metrics distinguish coverage, explicit selected slots, and actual live arrival."""
import numpy as np


def stats(values):
    a=np.asarray(values,float)
    if not np.isfinite(a).all():raise ValueError('Finite evaluated metrics required')
    return dict(count=len(a),mean=None if not len(a)else float(a.mean()),p95=None if not len(a)else float(np.quantile(a,.95)),maximum=None if not len(a)else float(a.max()))


def candidate_metrics(loss_result,*,selected_index=None):
    """No automatic choice from GT. selected_index must come from saved MPC.

    Supplying argmax(rank logits) is a separate proposed selector diagnostic,
    not MPC execution; callers must label its source rather than substituting.
    """
    arrays={k:loss_result[k].detach().cpu().numpy()for k in('pairwise_energy','position_error_m','rotation_error_rad','time_error_seconds','target_time_valid','matching_index')}
    n=len(arrays['matching_index']);valid=arrays['target_time_valid'].astype(bool)
    def at(indices,mask):
        rows=np.flatnonzero(mask);slots=np.asarray(indices)[mask]
        return dict(anchors=len(rows),position_error_m=stats(arrays['position_error_m'][rows,slots]),rotation_error_rad=stats(arrays['rotation_error_rad'][rows,slots]),
            time_valid_anchors=int(valid[rows].sum()),time_error_seconds=stats(arrays['time_error_seconds'][rows[valid[rows]],slots[valid[rows]]]),
            pose_only_anchors=int((~valid[rows]).sum()),joint_energy=stats(arrays['pairwise_energy'][rows,slots]))
    report=dict(anchor_denominator=n,oracle_coverage=at(arrays['matching_index'],np.ones(n,bool)),time_supervised=int(valid.sum()),pose_only_far=int((~valid).sum()),
        selected=dict(status='not_evaluated',missing=n),scope='Offline observed-demonstration compatibility; not execution success')
    if selected_index is not None:
        selected=np.asarray(selected_index)
        if selected.shape!=(n,)or selected.dtype.kind not in'iu'or np.any((selected< -1)|(selected>7)):
            raise ValueError('Saved selected index0..7 or explicit missing-1 required')
        mask=selected>=0
        report['selected']=dict(status='evaluated'if mask.any()else'not_evaluated',missing=int((~mask).sum()),**at(selected,mask))
    return report


def live_arrival_metrics(rows):
    """All actual selected/deadline receipts, missing arrivals remain denominator.

    Compiler must join exact physical timestamp to saved chosen candidate with
    no nearest-frame interpolation. No new model/counterfactual is evaluated.
    """
    errors=[];angles=[];time_errors=[];unexecuted=missing=0;nativeheld=0
    for row in rows:
        if row.get('actual_selected_dispatched')is not True:
            unexecuted+=1;continue
        if row.get('exact_physics_time_arrival_available')is not True:
            missing+=1;continue
        if row.get('GT_used_only_posthoc')is not True:raise ValueError('Arrival truth must be report-only')
        if row['TCP_to_same_time_selected_reference_point_m']<0 or row['TCP_to_same_time_selected_reference_angle_rad']<0:
            raise ValueError('Arrival distance/angle are nonnegative; time error remains signed')
        errors.append(float(row['TCP_to_same_time_selected_reference_point_m']))
        angles.append(float(row['TCP_to_same_time_selected_reference_angle_rad']))
        time_errors.append(float(row['actual_arrival_time_minus_selected_deadline_s']))
        nativeheld+=int(row.get('Native_held_at_arrival')is True)
    return dict(selected_receipt_denominator=len(rows),not_dispatched=unexecuted,missing_arrival=missing,evaluated=len(errors),
        actual_arrival_position_m=stats(errors),actual_arrival_angle_rad=stats(angles),actual_minus_deadline_s=stats(time_errors),
        Native_held_at_arrival=nativeheld,held_is_not_force_contact=True,offline_GT_oracle_not_used_for_live_choice=True,
        actual_closed_loop_task_success_requires_original_separate_judge=True)
