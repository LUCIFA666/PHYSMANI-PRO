"""Refresh declared moving command frames without changing completion geometry."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
from .reference_time import COMMAND_GOAL
from .causal_retarget_interaction import CausalRetargetManager, retarget_binding
from .evaluation import file_sha

TASK_PHASES = {task:('approach','acquire','follow','release') for task in
    ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')}
SCHEMA = 'r10_declared_external_command_frame_refresh_v1'


def eligible(goal, phases):
    t = goal.template
    return (goal.phase in phases and t.phase == goal.phase
        and t.reference_time_basis == COMMAND_GOAL and t.reference_kind == 'observed_se3'
        and t.reference_mode == 'external' and t.reference_role in ('wand','frame'))


def binding(root, config, task):
    phases = TASK_PHASES[task]
    sources = ('our_mani/multitask/command_reference_service.py',
        'our_mani/multitask/causal_retarget_interaction.py')
    return dict(schema=SCHEMA,task=task,phases=list(phases),maximum_reissues_per_phase=16,
        interaction=config['learned_artifacts'][task]['interaction'],
        sources_sha256={p:file_sha(Path(root)/p) for p in sources},
        original_COMMAND_GOAL_and_fresh_canonical_completion_geometry_retained=True,
        original_attempt_epoch_phase_deadline_suffix_and_dispatch_confirmation_retained=True,
        original_full_reference_actual_and_dynamic_collision_checks=True,
        additional_physics_or_camera_calls=0,native_future_truth_used=False,formal_VAL_TEST=False)


def install(actor, config, root, task):
    expected = binding(root,config,task)
    if config.get('command_reference_service') != expected:
        raise ValueError('source-bound command frame refresh required')
    m = actor.continuous_reference_manager.retarget
    if not isinstance(m,CausalRetargetManager) or m.events or m.epochs:
        raise ValueError('install on actual fresh reference manager')
    c = actor.controller
    before_templates = c.library.templates
    original = m._eligible
    phases = tuple(expected['phases'])
    def allow(instance,goal):
        return original(goal) or eligible(goal,phases)
    m._eligible = MethodType(allow,m)
    m.config = replace(m.config,maximum_reissues_per_phase=expected['maximum_reissues_per_phase'])
    m.binding = retarget_binding(m.config)
    if any(a is not b for a,b in zip(c.library.templates,before_templates)):
        raise ValueError('completion templates must remain identical')
    previous = actor.metrics_report
    def metrics(instance):
        return dict(previous(),command_reference_service=deepcopy(expected))
    actor.metrics_report = MethodType(metrics,actor)
    return m
