"""Complete the same checked transit path for the fitted highpeg insert phase."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from .high_peg_retarget_native_v372 import model_preflight as original_preflight
from .high_peg_retarget_native_v372 import SharedJoint16NativeBackend as OriginalBackend
from .bounded_task_transit import BoundedApproach, SCHEMA
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def binding(root, protocol, task):
    if task != 'insert_onto_rotating_peg_high_speed':
        raise ValueError('explicit highpeg original fitted insertion policy only')
    return dict(schema='r10_highpeg_all_original_fitted_phases_checked_transit_v394', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/high_peg_insert_native_v394.py',
            'our_mani/multitask/bounded_task_transit.py')},
        interaction=protocol['learned_artifacts'][task]['interaction'],
        enabled_phases=['approach', 'acquire', 'lift', 'align', 'insert'],
        additional_partial_transit_phase='insert',
        original_manager_native_and_reference_acceleration_limits_rad_s2=6.,
        inherited353_reference4_metadata_is_not_the_actual_highpeg_manager=True,
        current_target_twist_IK_full_actual_reference_Cartesian_collision_and_offset_stop_required=True,
        original_fitted_completion_grasp_causal_reissue_phase_deadline_suffix_and10s_unchanged=True,
        native_success_or_motor_tracking_guarantee_claimed=False, formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('high_peg_insert_v394', {}).get(task) != expected:
        raise ValueError('freeze actual insert transit policy')
    m, c = actor.continuous_reference_manager, actor.controller
    if c.goal is not None or c.feedback_step_calls or m.confirmations:
        raise ValueError('install before first native control step')
    if list(c.library.phase_order) != expected['enabled_phases']:
        raise ValueError('actual admitted initial sequence phases differ')
    raw = deepcopy(m.approach_transit.normalized)
    if raw['maximum_TCP_step_m'] != .2 or raw['segment_seconds'] != .5:
        raise ValueError('retain preceding transit segment geometry')
    if not np.allclose(m.native_gate.model.acceleration_limits, 6.):
        raise ValueError('retain original actual six-unit gate')
    if not np.allclose(getattr(m, 'reference_reserve_gate', m.native_gate).model.acceleration_limits, 6.):
        raise ValueError('actual highpeg reference gate differs from its frozen binding')
    raw['version'] = SCHEMA
    m.approach_transit = BoundedApproach(m, raw, phases=tuple(c.library.phase_order))
    before = actor.metrics_report
    def metrics(instance):
        return dict(before(), high_peg_insert_v394=dict(binding=expected,
            actual_reference_acceleration_limits=m.approach_transit.reference_gate.model.acceleration_limits.tolist(),
            actual_native_acceleration_limits=m.native_gate.model.acceleration_limits.tolist()))
    actor.metrics_report = MethodType(metrics, actor)


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('high_peg_insert_v394', {}).get(prepared['args'].task) != expected:
        raise ValueError('actual insert transit binding differs')
    checked_subset(root, prepared['files'], {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
    return dict(result, high_peg_insert_v394=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
