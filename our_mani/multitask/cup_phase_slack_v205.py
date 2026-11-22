"""Apply the already declared feedback slack once to a confirmed transport epoch."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import math
from .cup_transport_native_v161 import binding as original_binding, TASKS
from .evaluation import file_sha


def binding(root, protocol, task):
    original = original_binding(root, protocol, task)
    slack = float(protocol['policy_overlay']['configurations']['feedback']['phase_tracking_slack_seconds'])
    if slack != 2.: raise ValueError('restore only the existing frozen two-second feedback allowance')
    base = original['transport_original_maximum_budget_seconds']
    return dict(schema='r10_cup_confirmed_transport_existing_feedback_slack_v205', task=task,
        sources_sha256={'our_mani/multitask/cup_phase_slack_v205.py':file_sha(__file__)},
        previous_extra_prefix_phase_cap_seconds=base, existing_declared_feedback_slack_seconds=slack,
        proposed_maximum_before_global_suffix_cap_seconds={k:v+slack for k,v in base.items()},
        first_actual_prefix_epoch_owns_one_absolute_deadline=True,
        later_endpoint_slack_capped_by_same_deadline=True,
        previous_prefix_policy_phase_cap_increased=True, original_global10s_and_suffix_unchanged=True,
        original_fitting_native_physics_motion_collision_grasp_success_gates_unchanged=True,
        formal_VAL_TEST=False)


def configure_clock_and_slack(controller, clock, slack):
    if clock.epochs or clock.events or not math.isfinite(slack) or not 0<=slack<=2.:
        raise ValueError('configure the existing allowance only before first transport dispatch')
    previous_budgets = deepcopy(clock.budgets)
    clock.budgets = {index:budget+slack for index,budget in previous_budgets.items()}
    original = controller._grant_phase_slack
    def grant(instance, state, remaining_seconds):
        goal = instance.goal
        if goal is None or goal.phase!='transport' or instance.phase_index not in clock.epochs:
            return original(state, remaining_seconds)
        token=(goal.candidate_id,goal.selected_at,len(instance.attempts.records))
        if token in instance._phase_slack_tokens: return
        instance._phase_slack_tokens.add(token)
        now = float(state.robot.timestamp)
        original_remaining = max(0.,goal.selected_at+goal.phase_budget-now)
        reserve = max(0.,float(goal.suffix_budget))
        available = max(0.,float(remaining_seconds)-original_remaining-reserve)
        deadline = clock.epochs[instance.phase_index]['deadline']
        fixed_available = max(0.,deadline-(goal.selected_at+goal.phase_budget))
        requested = instance.feedback_config.phase_tracking_slack_seconds
        granted = min(requested,available,fixed_available)
        instance.phase_timing_events.append(dict(candidate_id=goal.candidate_id,selected_at=goal.selected_at,
            observed_time=now,original_phase_budget_seconds=goal.phase_budget,
            requested_slack_seconds=requested,granted_slack_seconds=granted,
            total_task_remaining_seconds=float(remaining_seconds),suffix_reserve_seconds=reserve,
            original_phase_deadline=goal.selected_at+goal.phase_budget,
            actual_phase_deadline=goal.selected_at+goal.phase_budget+granted,
            original_confirmed_transport_deadline=deadline,
            total_budget_extended=False,nominal_trajectory_completed=False,
            source='original_declared_feedback_slack_capped_by_first_confirmed_prefix_epoch_and_original_suffix'))
        if granted>0: instance.goal=replace(goal,phase_budget=goal.phase_budget+granted)
    controller._grant_phase_slack=MethodType(grant,controller)
    return previous_budgets


def install(actor, protocol, root, task):
    expected=binding(root,protocol,task)
    if protocol.get('cup_phase_slack_v205',{}).get(task)!=expected:
        raise ValueError('freeze actual confirmed phase allowance policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:
        raise ValueError('install before first control step')
    clock=m.transport_phase_clock
    if c.generator.transport_clock is not clock:
        raise ValueError('candidate/prefix/acknowledgement must share the same actual clock object')
    old=configure_clock_and_slack(c,clock,expected['existing_declared_feedback_slack_seconds'])
    if {str(k):v for k,v in old.items()}!=expected['previous_extra_prefix_phase_cap_seconds']:
        raise ValueError('unexpected previous prefix phase budgets')
    before=actor.metrics_report
    def metrics(instance):return dict(before(),cup_phase_slack_v205=dict(binding=expected,
        actual_epoch_deadlines=deepcopy(clock.epochs),earlier_base_only_clock_binding_superseded=True))
    actor.metrics_report=MethodType(metrics,actor)
