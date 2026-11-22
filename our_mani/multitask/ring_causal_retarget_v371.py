"""Causally reissue qualified moving-peg alignment and insertion intentions."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from .reference_time import COMMAND_GOAL
from .causal_retarget_interaction import CausalRetargetManager
from .evaluation import file_sha

TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')


def eligible(goal):
    return (goal.phase in ('align', 'insert') and goal.template.phase == goal.phase
        and goal.template.reference_time_basis == COMMAND_GOAL
        and goal.template.reference_kind == 'observed_se3'
        and goal.template.reference_role == 'selected_peg'
        and goal.template.reference_mode == 'external')


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('explicit original TRAIN-qualified ring tasks only')
    return dict(schema='r10_ring_external_SE3_command_goal_causal_reissue_v371', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/ring_causal_retarget_v371.py',
            'our_mani/multitask/causal_retarget_interaction.py')},
        interaction=protocol['learned_artifacts'][task]['interaction'],
        eligibility='align/insert; external selected_peg; observed_SE3; COMMAND_GOAL',
        original_template_attempt_phase_epoch_deadline_and_suffix_unchanged=True,
        original_full_dual_joint_and_current_dynamic_collision_gates_required=True,
        actual_reissue_requires_original_outer_dispatch_confirmation=True,
        original_phase_completion_grasp_success_and_global10s_unchanged=True,
        extra_camera_physics_or_native_future_state=False, formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('ring_causal_retarget_v371', {}).get(task) != expected:
        raise ValueError('freeze actual ring external-reference reissue policy')
    manager = actor.continuous_reference_manager.retarget
    if not isinstance(manager, CausalRetargetManager) or manager.epochs or manager.events:
        raise ValueError('install before first retarget control step')
    original = manager._eligible
    def allow(instance, goal):
        return original(goal) or eligible(goal)
    manager._eligible = MethodType(allow, manager)
    before = actor.metrics_report
    def metrics(instance):
        return dict(before(), ring_causal_retarget_v371=deepcopy(expected))
    actor.metrics_report = MethodType(metrics, actor)
