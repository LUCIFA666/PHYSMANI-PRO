"""Private TRAIN labels only: local grasp does not require final goal transfer.

Changing acceptance is explicit v5: Native task success is authoritative. It does not alter Native Task.success or
claim force contact. Unexecuted K7 alternatives remain unknown.
"""
from ..cube_candidate_matched_20261104_v2.data import _states,HoldMetric

def actual_local_grasp_contacts(initial,ticks,*,episode_id):
    """A contact observation is not a force certificate; no expert-target label."""
    rows=_states(initial,ticks,episode_id);segments=[];active=[];attempt=0;previous=initial
    for row in rows[1:]:
        if previous['requested_opening']>=.5 and row['requested_opening']<.5:attempt+=1
        if row['held'] and row['requested_opening']<.5:
            if not attempt:raise ValueError('Attached label without a preceding explicit close edge')
            active.append(row)
        elif active:
            segments.append((attempt,active));active=[]
        previous=row
    if active:segments.append((attempt,active))
    accepted=[]
    for attempt_id,segment in segments:
        metric=HoldMetric(.5,.04,.02);metric.update(initial)
        for row in segment:metric.update(row)
        goal_ever=any(row['native_success'] and row['held']for row in segment)
        if not goal_ever and not metric.passed:continue
        acceptance_class='native_task_success_contact'if goal_ever else'local_lift_hold_without_task_success'
        success_time=next((row['time']for row in segment if row['native_success'] and row['held']),None)
        first=segment[0]
        accepted.append(dict(episode_id=episode_id,attempt_id=attempt_id,time=first['time'],
            physics_time=first['physics_time'],object_position=first['object_position'],
            tcp_position=first['tcp_position'],tcp_quaternion=first['tcp_quaternion'],
            label_event='first_actual_50ms_attachment_observation_not_force_contact',
            acceptance=('original_Native_task_success_during_actual_held_segment_no_extra_hold_gate'if goal_ever else'local_only_original_P1_lifted_hold_same_segment'),
            acceptance_class=acceptance_class,original_goal_ever_in_segment=goal_ever,original_Native_task_success_observed=goal_ever,
            Native_task_success_observation_time=success_time,extra_hold_gate_applied=not goal_ever,force_contact_claimed=False,
            hold=metric.receipt()))
    return accepted
