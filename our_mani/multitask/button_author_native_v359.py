"""Original author button stages, checked3D transit, and shared16 Context."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from .button_intent_native_v101 import ButtonIntentNativeBackend as OriginalBackend
from .button_intent_native_v101 import model_preflight as original_preflight
from .button_intent_library_v71 import TASKS
from .button_author_transit_v358 import BoundedApproach, SCHEMA
from . import reference_stop_guard_v274 as stop
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('original button command experiment only')
    return dict(schema='r10_button_original_author_stage_checked3D_shared16_v359', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/button_author_native_v359.py',
            'our_mani/multitask/button_author_transit_v358.py',
            'our_mani/multitask/joint16_checkpoint_v172.py')},
        approach_to_press_guard='original author current public target distance <=0.3m',
        extra_fitted_action_error_intersection_replaced=True,
        original_author_stage_transition_is_not_completed_skill=True,
        actual_press_success_owned_only_by_original_native_benchmark=True,
        original_command_fit_caps_and_source_support_unchanged=True,
        checked_prefix_phases=['approach', 'press'], segment_seconds=.5,
        maximum_TCP_proposal_step_m=.2, reference_acceleration4_actual_criterion6=True,
        original_Cartesian_collision_grasp_physics_and_global10s_unchanged=True,
        formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('button_author_v359', {}).get(task) != expected:
        raise ValueError('freeze original author stage and transit policy')
    c, m = actor.controller, actor.continuous_reference_manager
    if c.feedback_step_calls or c.goal is not None or m.confirmations:
        raise ValueError('install before first control step')
    if c.library.model['source_stage_guard']['approach_to_press_distance_m'] != .3:
        raise ValueError('original author approach guard required')
    raw = deepcopy(m.approach_transit.normalized)
    raw.update(version=SCHEMA, segment_seconds=.5, maximum_TCP_step_m=.2)
    m.approach_transit = BoundedApproach(m, raw)
    stop.install(actor, protocol, root, task)
    before = actor.metrics_report
    def metrics(instance):
        return dict(before(), button_author_v359=dict(binding=expected,
            actual_prefix_configuration=deepcopy(m.approach_transit.normalized)))
    actor.metrics_report = MethodType(metrics, actor)


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    task = prepared['args'].task
    expected = binding(root, prepared['config'], task)
    if prepared['config'].get('button_author_v359', {}).get(task) != expected:
        raise ValueError('actual button author stage binding differs')
    files = {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()}
    stopped = stop.binding(root, prepared['config'], task)
    if prepared['config'].get('reference_stop_guard_v274', {}).get(task) != stopped:
        raise ValueError('freeze next-reference stopping policy')
    files.update({str(Path(root)/name): sha for name, sha in stopped['sources_sha256'].items()})
    checked_subset(root, prepared['files'], files)
    shared = None
    if prepared['args'].method == 'context_geometry':
        from .joint16_checkpoint_v172 import verified_checkpoint
        shared = verified_checkpoint(root, prepared)
    return dict(result, button_author_v359=expected, joint16_development_checkpoint=shared,
        reference_stop_guard_v274=stopped)


class ButtonIntentNativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
