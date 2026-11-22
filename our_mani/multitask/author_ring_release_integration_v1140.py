"""Author wp4 release with the independently confirmed semantic continuation.

Enter1121 directly. Ring keeps the original702/580 insert and terminal opening
rules; only the1137 public semantic resolver class is reused. The1129 seating
prototype is never installed. Buzz retains1130, other routes remain1121.
No extra drop steps, Native observations or success calls are introduced.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import patch

from . import retained_preclose_policy_integration_v1121 as parent
from . import committed_public_role_integration_v1137 as semantic
from . import public_buzz_approach_curve_v1130 as buzz
from . import physmani_ring_compensation_v580 as author_stage
from . import physmani_ring_release_v590 as author_release
from . import rml_policy_components_v702 as original_components

SCHEMA = 'r10_author_ring_release_and_committed_role_integration_v1140'
SOURCE = 'our_mani/multitask/author_ring_release_integration_v1140.py'
RING_TASKS = semantic.TASKS
BUZZ_TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
COMPARISON_LOCK = 'artifacts/framework/multitask_r10/TRAIN_committed_role_full16_v1138_setup/source_overlay_lock.json'
COMPARISON_LOCK_SHA = '5570bd9b6bdf923d108d818d6a4edeb4b8eba0cb208aaf802aea77edce94ca80'
SOURCES = tuple(dict.fromkeys((SOURCE, *semantic.SOURCES, parent.SOURCE,
    'our_mani/multitask/public_task_phase_integration_v1131.py',
    'our_mani/multitask/public_buzz_approach_curve_v1130.py',
    'our_mani/multitask/physmani_ring_compensation_v580.py',
    'our_mani/multitask/physmani_ring_release_v590.py',
    'our_mani/multitask/physmani_terminal_opening_v620.py',
    'our_mani/multitask/physmani_ring_release_contact_v657.py',
    'our_mani/multitask/physmani_geometry_goals_v593.py',
    'our_mani/multitask/rml_policy_components_v702.py')))


def _sha(root, name):
    return hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()


def binding(root, native_repo):
    if _sha(root, COMPARISON_LOCK) != COMPARISON_LOCK_SHA:
        raise ValueError('unchanged1138 comparison source lock required')
    return dict(schema=SCHEMA, sources_sha256={name:_sha(root,name) for name in SOURCES},
        parent=parent.binding(root,native_repo),
        comparison_source_overlay_lock=dict(path=COMPARISON_LOCK,sha256=COMPARISON_LOCK_SHA),
        semantic_class_reuse=dict(source=semantic.SOURCE, class_name='PublicCommitmentContinuation',
            source_sha256=_sha(root,semantic.SOURCE), original_guards_unchanged=True,
            original_integrated_components_called=False, original1131_parent_scope_entered=False),
        inherited_author_rules={task:dict(stage=author_stage.binding(root,native_repo,task),
            release=author_release.binding(root,native_repo,task)) for task in RING_TASKS},
        changed_factor='author_wp4_release_with_confirmed_semantic_commitment',
        ring_seating_v1129_installed=False, new_numeric_thresholds=False,
        author_drop_steps_added=0, counted_clock_and_native_outcome_unchanged=True,
        buzz_adapter='unchanged1130_frozen_TRAIN_approach_curve', other_routes='exact1121_objects_and_receipts',
        additional_native_getters_observe_predict_resolver_or_success_calls=0,
        release_success_or_physical_drop_hit_guaranteed=False, formal_VAL_TEST_qualification=False)


def ring_rule_binding(root, config, task):
    if task not in RING_TASKS:raise ValueError('two original ring tasks required')
    # The original constructor checks these same config rules. Keep their
    # complete public source bindings, without installing a different policy.
    component = original_components.binding(root,config,task)
    native = config['baseline_repo']
    return dict(schema=SCHEMA, task=task, source_sha256=_sha(root,SOURCE),
        inherited_author_stage_binding=author_stage.binding(root,native,task),
        inherited_author_release_binding=author_release.binding(root,native,task),
        inherited_component_binding=component,
        changed_factor='author_wp4_release_with_confirmed_semantic_commitment',
        insert_completion='unchanged580_current_public_wp4_position_guard_3cm',
        terminal_dispatch='unchanged1121_author_requested_opening_and_original_source_rules',
        public_semantic_continuation='unchanged1137_class_guards_only',
        ring_seating_v1129_installed=False, new_numeric_thresholds=False,
        extra_drop_steps=0, counted_clock_and_native_success_unchanged=True,
        physical_seating_or_release_success_guaranteed=False)


def _source(dispatched):
    try:result=object.__getattribute__(dispatched,'__dict__').get('source_policy')
    except (AttributeError,TypeError):result=None
    if result is None:raise TypeError('owned1121 source-policy witness required')
    return result


def _semantic_receipt(continuation):
    result=continuation.receipt()
    # The reused class was originally hosted by1131. Its old integration label
    # must not imply that the deliberately removed1129 scope is installed here.
    result.pop('later_ready_collision_forecast_and1129_guard_unchanged',None)
    return dict(result, class_only_reuse_from_v1137=True,
        later_ready_collision_forecast_unchanged=True, ring_seating_v1129_installed=False,
        receipt_projection='only_host_integration_label_changed_original_guards_and_counters_retained')


@contextmanager
def _ring_scope(source, actor, dispatched, rule, *, events=None, max_events=1000):
    resolver=actor.role_resolver; marker='_committed_public_role_continuation_v1137'
    if marker in vars(resolver):raise RuntimeError('owned resolver continuation scopes cannot nest')
    continuation=semantic.PublicCommitmentContinuation(source,actor,resolver.resolve,max_events=max_events)
    original_receipt=dispatched.receipt; original_resolve=resolver.resolve
    def receipt():
        return dict(original_receipt(),
            committed_public_role_continuation_v1137=_semantic_receipt(continuation),
            author_ring_release_integration_v1140=dict(schema=SCHEMA,rule=deepcopy(rule),
                ring_seating_v1129_installed=False, author_drop_steps_added=0,
                original_done_qualifier_and_executor_preserved=True,
                native_success_or_physical_drop_hit_guaranteed=False))
    original_error=None
    try:
        with patch.object(resolver,marker,continuation,create=True), \
             patch.object(resolver,'resolve',continuation.resolve), \
             patch.object(dispatched,'receipt',receipt):
            yield
    except BaseException as error:
        original_error=error;raise
    finally:
        before=_semantic_receipt(continuation)
        continuation.reset('owned_context_exit')
        continuation.restored=(resolver.resolve==original_resolve and dispatched.receipt==original_receipt
            and marker not in vars(resolver))
        if events is not None:events.update(schema=SCHEMA,restored=continuation.restored,
            witness_cleared=continuation.witness is None,continuation_receipt=before,
            ring_seating_v1129_installed=False)
        if not continuation.restored and original_error is None:
            raise RuntimeError('owned semantic continuation methods did not restore')


@contextmanager
def integrated_components(root, config, task, actor, *, events=None, max_events=1000):
    with parent.integrated_components(root,config,task,actor) as (frontend,dispatched,executor):
        if task in RING_TASKS:
            source=_source(dispatched)
            if executor.skill_policy is not dispatched:
                raise RuntimeError('original1121 final dispatch witness required')
            rule=ring_rule_binding(root,config,task)
            with _ring_scope(source,actor,dispatched,rule,events=events,max_events=max_events):
                yield frontend,dispatched,executor
        elif task in BUZZ_TASKS:
            source=_source(dispatched)
            rule=buzz.derive_buzz_approach_rule(root,config,task,source)
            with buzz.owned_buzz_approach_scope(source,rule):
                yield frontend,dispatched,executor
        else:
            yield frontend,dispatched,executor
