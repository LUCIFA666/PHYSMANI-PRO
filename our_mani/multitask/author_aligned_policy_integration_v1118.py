"""Scoped public-policy integration for a new, separately locked experiment.

The frozen actor, frontend and trained phase intents supply the same public
inputs. New behavior is explicit: original author actuator, action-conditioned
public attachment association, and the remove acquire transition. This module
never observes the environment or asks it for object identities or outcomes.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib

from . import rml_policy_components_v702 as original_components
from .action_conditioned_mode_observer_v1115b import ActionConditionedPublicModeObserver
from .author_aligned_remove_phase_v1116 import owned_remove_phase_scope, TASKS as REMOVE_TASKS
from .author_aligned_geometry_integration_v1114b import (
    construct as actuator_construct, protocol_pose_admission, binding as actuator_binding,
)
from .author_aligned_geometry_executor_v1114 import AuthorAlignedGeometryExecutor

SCHEMA = 'r10_author_actuator_public_intent_and_remove_phase_integration_v1118'
SOURCE = 'our_mani/multitask/author_aligned_policy_integration_v1118.py'
NEW_SOURCES = tuple('our_mani/multitask/' + name for name in (
    'author_aligned_geometry_executor_v1114.py',
    'author_aligned_geometry_integration_v1114b.py',
    'action_conditioned_mode_observer_v1115.py',
    'action_conditioned_mode_observer_v1115b.py',
    'author_aligned_remove_phase_v1116.py',
    'author_aligned_policy_integration_v1118.py',
))


def binding(root, native_repo):
    root = Path(root)
    return dict(schema=SCHEMA,
        sources_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in NEW_SOURCES},
        actuator=actuator_binding(root, native_repo),
        original_components_sha256=hashlib.sha256((root/'our_mani/multitask/rml_policy_components_v702.py').read_bytes()).hexdigest(),
        public_attachment_candidate_distance_changed=False,
        identity_association='confirmed_own_public_close_intent_in_original_candidate_set',
        native_held_identity_verified=False, native_object_identity_read=False,
        extra_native_observation_getter_or_success_calls=0,
        remove_completion='original_public_capture_without_postcapture_spoke_SE3_or_duration',
        author_any_held_semantics_exactly_reproduced=False,
        formal_VAL_TEST_qualification_granted=False)


def _synchronize(observer, policy, actor):
    # These are existing public policy/command records. In particular the
    # requested command is not robot.gripper_open, which can lag its dispatch.
    selected = policy.selected if type(policy.selected) is dict else {}
    index = policy.phase_index
    order = getattr(policy, 'phase_order', ())
    phase = order[index] if 0 <= index < len(order) else selected.get('phase')
    if getattr(policy, 'finished', False):
        phase = 'verify'
    return observer.synchronize_intent_context(phase=phase, phase_index=index,
        manipulated_ids=selected.get('manipulated_ids'),
        requested_opening=actor.requested_opening, selected_at=selected.get('selected_at'))


@contextmanager
def integrated_components(root, config, task, actor):
    """Yield a new triple, retaining all instance adapters through run/receipt.

    Only the fresh actor/frontend/policy are adapted. No frozen source, global
    class, environment method or task predicate is replaced here. The parent
    owns evaluator scopes, source locks, original success calls and tick budget.
    """
    frontend, policy, _unused_legacy_executor = original_components.construct(root, config, task, actor)
    if frontend.actor is not actor or policy.frontend is not frontend:
        raise ValueError('one fresh original frontend/actor/policy required')
    original_observer = actor.mode_observer
    observer = ActionConditionedPublicModeObserver(original_observer.body_ids,
        original_observer.graspable_ids, original_observer.maximum_distance)
    with ExitStack() as scopes:
        scopes.enter_context(patch.object(actor, 'mode_observer', observer))
        if task in REMOVE_TASKS:
            scopes.enter_context(owned_remove_phase_scope(policy))
        original_observe, original_select = frontend.observe, policy.select

        def observe(observation, robot):
            _synchronize(observer, policy, actor)
            return original_observe(observation, robot)

        def select():
            result = original_select()
            _synchronize(observer, policy, actor)
            return result

        scopes.enter_context(patch.object(frontend, 'observe', observe))
        scopes.enter_context(patch.object(policy, 'select', select))

        def confirmed(source_policy, opening, proof, proposed, *, confirmed_after):
            if source_policy is not policy:
                raise RuntimeError('confirmed opening escaped its owned public policy')
            observer.on_confirmed_opening(selected=policy.selected, opening=opening,
                proof=proof, proposed=proposed, confirmed_after=confirmed_after,
                previous_states=actor.previous_states,
                confirmed_witness=policy.close_witness, phase_index=policy.phase_index)

        if all(callable(getattr(policy, name, None)) for name in ('qualify_opening', 'confirm_opening')):
            frontend, dispatched, executor = actuator_construct(config, frontend, policy,
                on_confirmed_opening=confirmed)
        else:
            # The original button policy has no skill close witness. It keeps
            # the author gripper request and the same public pose integrity gate.
            if task not in ('push_moving_button', 'push_moving_button_high_speed'):
                raise ValueError('a nonbutton public skill must own close witness callbacks')
            dispatched = policy
            executor = AuthorAlignedGeometryExecutor(config, observe_public_frame=frontend.observe,
                admit_pose_action=protocol_pose_admission(policy), skill_policy=None,
                admit_joint_target=None, collision_diagnostic_limit=0)

        original_receipt = dispatched.receipt

        def receipt():
            result = dict(original_receipt())
            result['author_aligned_policy_integration_v1118'] = dict(schema=SCHEMA,
                task=task, maximum_attachment_candidate_distance_m=observer.maximum_distance,
                public_mode=deepcopy(observer.receipt()),
                original_native_actuator_and_task_outcome=True,
                native_held_identity_verified=False, extra_native_getters_or_success_calls=0,
                remove_phase_changed=task in REMOVE_TASKS, formal_VAL_TEST_qualified=False)
            return result

        scopes.enter_context(patch.object(dispatched, 'receipt', receipt))
        yield frontend, dispatched, executor
