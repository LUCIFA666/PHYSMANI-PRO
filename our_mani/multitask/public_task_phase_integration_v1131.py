"""Scoped estimated public ring-seating/Buzz-approach candidate over1121.

Keep the complete1121 lifetime, actuator, public identity and preclose witness.
Only ring/Buzz enter new instance scopes. This is an estimated policy candidate;
neither trained stage completion nor nominal CAD seating proves Native success,
physical seating, collision freedom or an RML path's final done state.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

from . import retained_preclose_policy_integration_v1121 as parent

SCHEMA = 'r10_estimated_public_ring_seating_and_buzz_approach_integration_v1131'
SOURCE = 'our_mani/multitask/public_task_phase_integration_v1131.py'
RING_MODULE = 'our_mani.multitask.public_ring_seating_policy_v1129'
BUZZ_MODULE = 'our_mani.multitask.public_buzz_approach_curve_v1130'
RING_TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')
BUZZ_TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
NEW_SOURCES = (SOURCE, RING_MODULE.replace('.', '/')+'.py', BUZZ_MODULE.replace('.', '/')+'.py')
SUBSTRATE_INPUTS = {
    'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json':
        'd497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e',
    'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json':
        '5ea2190dae2b986dc6f7d1b208c0372e2750da68e610639963b697d055e720c0',
    'artifacts/framework/multitask_r10/TRAIN_author_aligned_full16_v1122_setup/source_overlay_lock.json':
        'a949256d1183d41f682fda02956b77965b8d4e847e255f93818ec17e810abc2e',
}


def binding(root, native_repo):
    """Source graph only; each owned task scope records its precise TRAIN rule."""
    root = Path(root)
    checksum = lambda name: hashlib.sha256((root/name).read_bytes()).hexdigest()
    inputs = {name:checksum(name) for name in SUBSTRATE_INPUTS}
    if inputs != SUBSTRATE_INPUTS:
        raise ValueError('original TRAIN source/model/route and parent overlay bytes required')
    return dict(schema=SCHEMA, sources_sha256={name:checksum(name) for name in NEW_SOURCES},
        parent=parent.binding(root, native_repo), inherited_source_graph_locks=inputs,
        estimated_public_task_phase_candidate=True,
        ring_tasks=list(RING_TASKS), buzz_tasks=list(BUZZ_TASKS),
        ring_adapter='nominal_CAD_seating_insert_done_and_final_dispatch_release_guard',
        buzz_adapter='frozen_TRAIN_approach_curve_public_cursor',
        non_ring_or_buzz='complete_original1121_route',
        final_ring_guard_applies_to_actual_executor_skill_policy=True,
        actual_native_actuator_and_outcome_loop_inherited=True,
        additional_native_getters_or_success_calls=0,
        native_or_physical_seating_or_collision_certificate=False,
        author_literal_native_trajectory_reproduction_claimed=False,
        initial_RML_path_done_used_for_control=False,
        final_RML_path_done_observed_or_inferred=False,
        formal_VAL_TEST_qualification=False)


def _source_policy(dispatched):
    """Read owned Python wrapper storage; never discover an environment object."""
    try:
        source = object.__getattribute__(dispatched, '__dict__').get('source_policy')
    except (AttributeError, TypeError):
        source = None
    if source is None:
        raise TypeError('owned1121 source-policy witness required for ring/Buzz adaptation')
    return source


@contextmanager
def _receipt_scope(dispatched, task, source_binding):
    original = dispatched.receipt

    def receipt():
        value = dict(original())
        value['public_task_phase_integration_v1131'] = dict(schema=SCHEMA, task=task,
            source_binding=deepcopy(source_binding), estimated_public_policy=True,
            native_success_owned_by_original_parent=True,
            extra_native_getters_or_success_calls=0,
            native_or_physical_seating_certificate=False,
            initial_RML_path_done_used_for_control=False,
            final_RML_path_done_observed_or_inferred=False,
            formal_VAL_TEST_qualification=False)
        return value

    with patch.object(dispatched, 'receipt', receipt):
        yield


@contextmanager
def integrated_components(root, config, task, actor):
    """Compose task adapters inside the complete parent1121 owned lifetime."""
    with parent.integrated_components(root, config, task, actor) as (frontend, dispatched, executor):
        if task not in RING_TASKS and task not in BUZZ_TASKS:
            yield frontend, dispatched, executor
            return
        source = _source_policy(dispatched)
        source_binding = binding(root, config['baseline_repo'])
        if task in RING_TASKS:
            ring = importlib.import_module(RING_MODULE)
            with ring.integrated_ring_seating_scope(root, config, task, frontend, dispatched, executor) as triple:
                guarded_frontend, guarded, guarded_executor = triple
                if (guarded_frontend is not frontend or guarded_executor is not executor
                        or executor.skill_policy is not guarded or _source_policy(guarded) is not source):
                    raise RuntimeError('ring final-dispatch guard must own the same executor and source policy')
                with _receipt_scope(guarded, task, source_binding):
                    yield guarded_frontend, guarded, guarded_executor
        else:
            buzz = importlib.import_module(BUZZ_MODULE)
            rule = buzz.derive_buzz_approach_rule(root, config, task, source)
            with buzz.owned_buzz_approach_scope(source, rule):
                with _receipt_scope(dispatched, task, source_binding):
                    yield frontend, dispatched, executor
