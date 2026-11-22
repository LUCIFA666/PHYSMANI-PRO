"""Renew completed cup reference streams against fresh external SE3 observations."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from ..contracts import Reachability
from .evaluation import file_sha

TASKS = ('place_cups_on_rotating_frame', 'place_cups_on_rotating_frame_high_speed')


def eligible(goal):
    return (goal is not None and goal.phase in ('transport', 'align', 'hang')
        and goal.template.phase == goal.phase and goal.template.reference_time_basis == 'current'
        and goal.template.reference_kind == 'observed_se3'
        and goal.template.reference_role == 'selected_spoke' and goal.template.reference_mode == 'external')


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('explicit cup placement current-reference policy only')
    return dict(schema='r10_cup_current_external_SE3_completed_stream_renewal_v377', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/cup_current_tracking_v377.py', 'our_mani/multitask/task_reference_v37.py')},
        interaction=protocol['learned_artifacts'][task]['interaction'],
        renewed_phases=['transport', 'align', 'hang'],
        original_roll_current_reference_full_actual_and_reference_collision_gates=True,
        current_observation_mode_forecast_cutoff_and_target_twist_required=True,
        same_candidate_attempt_template_phase_epoch_deadline_and_suffix=True,
        phase_completion_release_and_native_success_unchanged=True,
        extra_camera_or_physics_queries=False, global10s_unchanged=True, formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('cup_current_tracking_v377', {}).get(task) != expected:
        raise ValueError('freeze actual cup external stream renewal policy')
    m, c = actor.continuous_reference_manager, actor.controller
    if c.goal is not None or c.feedback_step_calls or m.confirmations:
        raise ValueError('install before first control step')
    old = m.check_stream
    events = []
    def check(instance, robot, duration):
        goal, plan = c.goal, instance.plan
        if (eligible(goal) and plan is not None and not instance.invalidated
                and robot.timestamp >= plan.end-1e-8
                and not (instance.capture_stopped(robot) and instance.capture_at_current_endpoint(robot))):
            committed = instance.roll_current_reference(robot)
            events.append(dict(time=float(robot.timestamp), phase=goal.phase, committed=committed,
                phase_epoch=goal.selected_at, phase_deadline=goal.selected_at+goal.phase_budget,
                original_candidate_id=goal.candidate_id, candidate_unchanged=c.goal is goal,
                original_complete_dual_gate_required=True, total_budget_extended=False))
            if not committed:
                return Reachability(False, 'fresh_current_cup_phase_renewal_original_gates_rejected')
        return old(robot, duration)
    m.check_stream = MethodType(check, m)
    before = actor.metrics_report
    def metrics(instance):
        return dict(before(), cup_current_tracking_v377=dict(binding=expected, events=deepcopy(events)))
    actor.metrics_report = MethodType(metrics, actor)
