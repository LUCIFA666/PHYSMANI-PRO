"""A new remove acquisition transition using own confirmed public capture.

Only acquire completion changes. The old source policy's public capture keeps
its shared held-presence bit, own confirmed close, unique public identity,
role/provenance and measured-attachment membership requirements. Once these
pass, no current external-spoke canonical pose or fitted minimum duration is
an additional acquire-completion gate. Other stages follow the original MRO.

This remains more conservative than the author's any-held-object branch. It
does not establish the native held object's identity or grant qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import hashlib

from .physmani_remove_intercept_v667 import InterceptRemoveIntents

SCHEMA = 'r10_author_aligned_remove_public_capture_transition_v1116'
SOURCE = 'our_mani/multitask/author_aligned_remove_phase_v1116.py'
TASKS = ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')
SOURCES = (
    'our_mani/multitask/physmani_skill_policy_v548.py',
    'our_mani/multitask/physmani_remove_stages_v611.py',
    'our_mani/multitask/physmani_terminal_opening_v620.py',
    'our_mani/multitask/physmani_remove_wp0_v637.py',
    'our_mani/multitask/physmani_remove_source_roles_v653.py',
    'our_mani/multitask/physmani_remove_intercept_v667.py',
    'our_mani/multitask/physmani_geometry_goals_v593.py',
)


def binding(root):
    checksum = lambda name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
    return dict(schema=SCHEMA, tasks=list(TASKS), source_sha256=checksum(SOURCE),
        original_sources_sha256={name: checksum(name) for name in SOURCES},
        acquire_completion='original_public_capture_with_own_confirmed_close',
        post_capture_current_spoke_position_gate=False,
        post_capture_current_spoke_orientation_gate=False,
        acquire_fitted_minimum_duration_gate=False,
        original_nonacquire_methods_unchanged=True,
        additional_native_getters_or_success_calls=0,
        author_any_held_semantics_exactly_reproduced=False,
        selected_pose_opening_and_selected_at_modified=False,
        native_capture_identity_verified=False, qualification_granted=False)


class _AcquisitionTrace:
    def __init__(self, task, max_events):
        if task not in TASKS or type(max_events) is not int or not 1 <= max_events <= 10000:
            raise ValueError('explicit remove task and bounded acquisition trace required')
        self.task, self.max_events = task, max_events
        self.events = []
        self.dropped_events = self.capture_calls = self.delegated_nonacquire_calls = 0

    def append(self, row):
        if len(self.events) < self.max_events:
            self.events.append(row)
        else:
            self.dropped_events += 1

    def acquire_done(self, policy, goal):
        # Invoke the original public predicate once. No target construction,
        # new perception, source-duration query or native API is introduced.
        self.capture_calls += 1
        try:
            result = policy._public_capture(goal)
        except BaseException as error:
            self.append(dict(event='original_public_capture_exception',
                             exception_type=type(error).__name__, native_outcome_inferred=False))
            raise
        if type(result) is not bool:
            raise TypeError('original public capture must return an explicit bool')
        state = policy.frontend.state
        self.append(dict(event='acquire_public_capture_transition',
            time=float(state.robot.timestamp), phase_index=policy.phase_index,
            reference_id=goal['reference_id'], manipulated_ids=list(goal['manipulated_ids']),
            public_held_presence=bool(state.robot.contact), public_capture=result,
            acquire_done=result, current_spoke_geometry_not_queried=True,
            fitted_minimum_duration_not_queried=True, native_success_claimed=False,
            native_held_identity_claimed=False))
        return result

    def receipt(self):
        return dict(schema=SCHEMA, task=self.task, events=deepcopy(self.events),
            capture_calls=self.capture_calls,
            delegated_nonacquire_calls=self.delegated_nonacquire_calls,
            dropped_events=self.dropped_events,
            preserved_public_capture_requirements=[
                'shared_public_held_presence', 'own_confirmed_close_for_current_phase',
                'selected_single_manipulated_identity_matches_witness',
                'unique_valid_public_attached_identity', 'original_public_mode_provenance',
                'current_manipulated_role_binding', 'measured_attachment_membership'],
            removed_acquire_completion_gates=[
                'current_external_spoke_canonical_2cm_position',
                'current_external_spoke_canonical_0.1rad_orientation',
                'fitted_acquire_minimum_duration'],
            retained_difference_from_author_any_held='A generic shared held=True alone cannot pass. The original public identity/own-close evidence is still required; it is public association, never a native held-ID query.',
            original_nonacquire_done_delegated=True,
            selected_pose_opening_and_selected_at_modified=False,
            extra_native_getters_or_task_success_calls=0,
            native_task_outcome_unchanged=True, formal_qualification_granted=False)


def _receipt_with_trace(original, trace):
    result = dict(original)
    result['author_aligned_remove_phase_v1116'] = trace.receipt()
    return result


class AuthorAlignedRemoveIntents(InterceptRemoveIntents):
    """Full original remove MRO; root's new factory may construct this class."""
    def __init__(self, *args, phase_trace_max_events=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_phase_trace_v1116 = _AcquisitionTrace(self.task, phase_trace_max_events)

    def _done(self, goal):
        if goal['phase'] == 'acquire':
            return self._remove_phase_trace_v1116.acquire_done(self, goal)
        self._remove_phase_trace_v1116.delegated_nonacquire_calls += 1
        return super()._done(goal)

    def receipt(self):
        return _receipt_with_trace(super().receipt(), self._remove_phase_trace_v1116)


@contextmanager
def owned_remove_phase_scope(policy, *, events=None, max_events=1000):
    """Adapt one fresh original instance; restore both methods on every exit.

    Enter before an executor's opening-witness facade is created, and retain
    this scope through the actual case and receipt collection. No class/global
    method is patched. The yielded object is the same original policy instance.
    """
    if not isinstance(policy, InterceptRemoveIntents) or isinstance(policy, AuthorAlignedRemoveIntents):
        raise TypeError('one original InterceptRemoveIntents instance required')
    marker = '_owned_remove_phase_scope_v1116'
    if marker in vars(policy):
        raise RuntimeError('remove acquisition phase scopes cannot nest')
    trace = _AcquisitionTrace(policy.task, max_events)
    original_done, original_receipt = policy._done, policy.receipt

    def done(owned, goal):
        if owned is not policy:
            raise RuntimeError('remove phase adapter escaped its owned policy')
        if goal['phase'] == 'acquire':
            return trace.acquire_done(owned, goal)
        trace.delegated_nonacquire_calls += 1
        return original_done(goal)

    def receipt(owned):
        if owned is not policy:
            raise RuntimeError('remove receipt escaped its owned policy')
        return _receipt_with_trace(original_receipt(), trace)

    try:
        with patch.object(policy, marker, trace, create=True), \
             patch.object(policy, '_done', MethodType(done, policy)), \
             patch.object(policy, 'receipt', MethodType(receipt, policy)):
            yield policy
    finally:
        restored = (policy._done == original_done and policy.receipt == original_receipt
                    and marker not in vars(policy))
        if events is not None:
            events.update(schema=SCHEMA, restored=restored, phase_receipt=trace.receipt())
