"""Compile corrected public forecasts against Native-success/local-only labels.

Same public features, reference frame and far-time mask as frozen v2; only
private acceptance and dataset schema change. No model runs in this compiler.
"""
from collections import Counter
import numpy as np
from ..cube_candidate_matched_20261104_v2.data import (
    public_features,public_reference_target,load_partition,PARTITION,PUBLIC_ROW_KEYS,CLOCK_TOLERANCE)
from . import SCHEMA,FRAME
from .labels import actual_local_grasp_contacts

def compile_contact_episode(public_rows,initial,ticks,*,parent_id,episode_id,expected_forecast_sha256,partition_path=PARTITION):
    """Compile fit/selection only. No contact => empty arrays and honest not_ready."""
    partition=load_partition(partition_path)
    matches=[r for r in partition['rows']if r['parent_id']==parent_id]
    if len(matches)!=1 or matches[0]['role']not in('fit','selection'):
        raise ValueError('Heldout/unknown parents cannot supply candidate fit or selection supervision')
    if not public_rows:raise ValueError('Every recorded public frame required')
    if len(public_rows)!=len(ticks)+1:raise ValueError('Keep initial and all failed/unknown physical frames')
    # Explicit whitelist strips only the same-episode envelope, never private fields.
    if any(set(row)!=PUBLIC_ROW_KEYS|{'episode_id'}for row in public_rows):
        raise ValueError('Exact public row plus episode identity; private/unknown fields forbidden')
    projected=[{key:row[key]for key in PUBLIC_ROW_KEYS}for row in public_rows]
    # Build all possible causal feature rows without reading private contact labels.
    features=[];rejected=Counter()
    for index,row in enumerate(public_rows):
        if row.get('episode_id')!=episode_id:raise ValueError('No cross-pass public feature joins')
        try:value,provenance=public_features(projected[index],projected[:index+1],expected_forecast_sha256=expected_forecast_sha256)
        except (KeyError,ValueError)as error:
            features.append(None);rejected[str(error)]+=1;continue
        features.append((value,provenance))
    contacts=actual_local_grasp_contacts(initial,ticks,episode_id=episode_id)
    primary=[c for c in contacts if c['original_Native_task_success_observed']]
    selected_contacts=(primary or contacts)[:1]
    values=[];ledger=[]
    first_close=next((state['time']for state in ticks if state['requested_opening']<.5),float('inf'))
    for index,(row,state)in enumerate(zip(public_rows,[initial,*ticks])):
        now=float(row['public_robot']['timestamp'])
        if abs(now-state['time'])>CLOCK_TOLERANCE:raise ValueError('Public/private actual timestamp mismatch; no nearest-frame matching')
        if features[index]is None or now>=first_close-1e-8:continue
        value,provenance=features[index]
        for contact in selected_contacts:
            if contact['time']<=now+1e-8:continue
            target=public_reference_target(row['result']['forecast'],contact,anchor_time=now)
            values.append(dict(**value,**{k:target[k]for k in('target_position','target_rotation','target_dt','target_time_valid')}))
            ledger.append(dict(parent_id=parent_id,episode_id=episode_id,role=matches[0]['role'],public_index=index,
                timestamp=now,contact_time=contact['time'],attempt_id=contact['attempt_id'],target_frame=FRAME,
                feature_provenance=provenance,acceptance_class=contact['acceptance_class'],original_goal_ever_in_segment=contact['original_goal_ever_in_segment'],target_time_valid=bool(target['target_time_valid']),public_reference_basis=target['public_reference_basis'],Native_object_orientation_used_in_target=False))
    arrays={k:np.stack([v[k]for v in values])for k in values[0]}if values else{}
    return arrays,dict(schema=SCHEMA,parent_id=parent_id,episode_id=episode_id,role=matches[0]['role'],
        candidate_training_ready=bool(values),public_frames_total=len(public_rows),private_frames_total=len(ticks)+1,
        accepted_contact_count=len(contacts),selected_contact_acceptance_classes=[c['acceptance_class']for c in selected_contacts],acceptance_counts=dict(Counter(c['acceptance_class']for c in contacts)),original_Native_goal_required_for_local_label=False,
        Native_task_success_any_observed=any(row['native_success']for row in [initial,*ticks]),
        primary_Native_success_positive_has_no_extra_hold_gate=True,supervised_anchors=len(values),time_supervised_anchors=sum(bool(v['target_time_valid'])for v in values),pose_only_far_anchors=sum(not bool(v['target_time_valid'])for v in values),rejected_feature_reasons=dict(rejected),
        all_failed_and_unknown_frames_retained_in_episode_denominator=True,
        unmatched_candidates_are_negative=False,target_frame=FRAME,ledger=ledger,supervision_scope='All available causal visible FREE anchors strictly before first close in Native-success or separately marked local-only accepted episodes; far real tau retained and time loss masked, never clamped',
        frozen_forecast_checkpoint_sha256=expected_forecast_sha256,
        other_contacts_or_failed_attempts_not_success_probability_labels=True)

def combine_episode_results(results):
    """Preserve every declared episode report, including zero-positive failures.

    The caller passes every preassigned same-role episode, not just successes.
    source-parent weighting is computed only after preserving this denominator.
    """
    if not results:raise ValueError('All fixed role episodes required')
    reports=[report for _,report in results]
    roles={r['role']for r in reports}
    if len(roles)!=1 or not roles<=set(('fit','selection')):raise ValueError('Never mix roles or include heldout')
    partition=load_partition();expected=[r['parent_id']for r in partition['rows']if r['role']in roles]
    actual=[r['parent_id']for r in reports]
    if len(set(actual))!=len(actual)or set(actual)!=set(expected):raise ValueError('Every fixed whole parent including failures required')
    counts={report['parent_id']:len(arrays.get('target_dt',[]))for arrays,report in results}
    parents=[p for p in expected if counts[p]]
    nonempty=[arrays for arrays,_ in results if len(arrays.get('target_dt',[]))]
    combined={k:np.concatenate([a[k]for a in nonempty])for k in nonempty[0]}if nonempty else{}
    if combined:
        ids=np.concatenate([np.full(counts[report['parent_id']],parents.index(report['parent_id']),dtype=np.int64)for arrays,report in results if counts[report['parent_id']]])
        combined['parent_index']=ids
        combined['sample_weight']=np.asarray([1./counts[parents[index]]for index in ids],np.float32)
    return combined,dict(schema=SCHEMA,role=next(iter(roles)),declared_parent_denominator=len(expected),
        supervised_parents=parents,failed_or_no_contact_parents=[p for p in expected if not counts[p]],
        candidate_training_ready=bool(combined),supervised_anchors=sum(counts.values()),
        episodes=reports,target_frame=FRAME,all_fixed_episodes_preserved=True,
        no_failed_or_unmatched_candidate_negative_labels=True)
