"""Allow fully checked partial align transit after the unchanged held-ring repair."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from .high_peg_target_memory_native_v340 import model_preflight as original_preflight
from .high_peg_target_memory_native_v340 import SharedJoint16NativeBackend as OriginalBackend
from .bounded_alignment_v352 import BoundedApproach, SCHEMA
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def binding(root, protocol, task):
    if task != 'insert_onto_rotating_peg_high_speed':
        raise ValueError('explicit highpeg alignment policy only')
    return dict(schema='r10_highpeg_checked_partial_alignment_v353', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/high_peg_alignment_native_v353.py',
            'our_mani/multitask/bounded_alignment_v352.py')},
        additional_partial_transit_phase='align',
        full_task_candidate_selection_and_phase_completion_unchanged=True,
        actual6_reference4_joint_and_original_cartesian_limits_unchanged=True,
        fresh_actual_and_reference_full_collision_and_offset_stop_checks=True,
        current_observation_and_causal_forecast_required=True,
        original_fitted_suffix_budget_and_global10s_retained=True,
        held_geometry_and_target_identity_policy_unchanged=True,
        successful_task_or_completed_skill_claimed=False, formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('high_peg_alignment_v353', {}).get(task) != expected:
        raise ValueError('freeze actual partial align policy')
    m = actor.continuous_reference_manager
    if actor.controller.feedback_step_calls or m.confirmations or actor.controller.goal is not None:
        raise ValueError('install before first control step')
    old = m.approach_transit
    raw = deepcopy(old.normalized)
    if raw['maximum_TCP_step_m'] != .2 or raw['segment_seconds'] != .5:
        raise ValueError('retain preceding ring transit segment geometry')
    raw['version'] = SCHEMA
    m.approach_transit = BoundedApproach(m, raw)
    before = actor.metrics_report
    def metrics(instance):
        return dict(before(), high_peg_alignment_v353=dict(binding=expected,
            actual_prefix_configuration=deepcopy(m.approach_transit.normalized)))
    actor.metrics_report = MethodType(metrics, actor)


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('high_peg_alignment_v353', {}).get(prepared['args'].task) != expected:
        raise ValueError('actual checked alignment binding differs')
    checked_subset(root, prepared['files'],
        {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
    return dict(result, high_peg_alignment_v353=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
