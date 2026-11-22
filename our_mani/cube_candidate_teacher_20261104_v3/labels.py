"""Private TRAIN labels only: local grasp does not require final goal transfer.

Changing acceptance is explicit v3. It does not alter Native Task.success or
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
        if not metric.passed:continue
        goal_ever=any(row['native_success']for row in segment)
        acceptance_class='lift_hold_with_goal'if goal_ever else'lift_hold_without_goal'
        first=segment[0]
        accepted.append(dict(episode_id=episode_id,attempt_id=attempt_id,time=first['time'],
            physics_time=first['physics_time'],object_position=first['object_position'],
            tcp_position=first['tcp_position'],tcp_quaternion=first['tcp_quaternion'],
            label_event='first_actual_50ms_attachment_observation_not_force_contact',
            acceptance='original_P1_lifted_hold_in_same_uninterrupted_closed_attachment_segment',
            acceptance_class=acceptance_class,original_goal_ever_in_segment=goal_ever,full_task_success_claimed=False,
            hold=metric.receipt()))
    return accepted
