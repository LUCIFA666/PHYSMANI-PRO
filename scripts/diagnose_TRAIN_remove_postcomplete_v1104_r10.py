"""Passive remove-cup observer scope for the new post-completion TRAIN runner.

No scheduler, resource acquisition, simulator construction or default activation
lives here. The shared v1103 runner owns its current completion gate, same-process
resource lease, original v743/v744 entry, environment qualification and cleanup.
The unchanged v892 observer executes pinned bodies once with value-preserving
taps; this adapter never queries a public/native getter or repeats a predicate.
Unvisited leaves and unmaterialized compound expressions remain unknown.
"""
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
from pathlib import Path
import sys
import threading
from unittest.mock import patch

SCHEMA = 'r10_postcomplete_TRAIN_remove_capture_scope_v1104'
OBSERVER = 'our_mani/multitask/train_remove_capture_ast_v892.py'
OBSERVER_SHA = '5ec7af7c4d9022dd3954ab13fed3ab34620f028728aec9a496bb369c3676d670'
CHECKPOINT = dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
    sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
FIXED_TRAIN = {
    'remove_cups_from_rotating_frame': ('episode102', 'episode104', 'episode111', 'episode112'),
    'remove_cups_from_rotating_frame_high_speed': ('episode1', 'episode102', 'episode104', 'episode116'),
}
EXPECTED_METHODS = {
    ('SkillGeometryIntents', name) for name in ('_completion_target', '_public_capture', '_done',
        '_advance', 'select', 'qualify_opening', 'confirm_opening')
} | {
    ('RemoveAuthorIntents', '_done'), ('RemoveAuthorIntents', 'qualify_opening'),
    ('OriginalTerminalOpening', 'qualify_opening'),
    ('AuthorPreApproachRemoveIntents', '_completion_target'),
    ('AuthorPreApproachRemoveIntents', '_advance'), ('AuthorPreApproachRemoveIntents', 'candidates'),
    ('AuthorPreApproachRemoveIntents', 'select'), ('AuthorLiveRoleRemoveIntents', 'select'),
    ('InterceptRemoveIntents', 'candidates'), ('PublicModelFrontend', 'ready'),
}
_SCOPE_LOCK = threading.Lock()


def require(value, message):
    if not value:
        raise ValueError('remove diagnostic v1104: ' + message)


def validate_selection(selection):
    require(type(selection) is dict and set(selection) == {'task', 'method', 'raw_episode_id'},
        'exact single TRAIN selection required')
    require(selection['task'] in FIXED_TRAIN and selection['method'] in ('cv_geometry', 'context_geometry')
        and selection['raw_episode_id'] in FIXED_TRAIN[selection['task']],
        'only original fixed-four remove TRAIN sources and two geometry methods')


def _load_observer(root):
    root = Path(root).resolve()
    path = root / OBSERVER
    require(path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == OBSERVER_SHA,
        'unchanged canonical v892 observer required')
    observer = importlib.import_module('our_mani.multitask.train_remove_capture_ast_v892')
    require(Path(observer.__file__).resolve() == path, 'canonical observer module required')
    return observer


def require_body_plan(root, observer, policy):
    """Validate all source-pinned original descriptors before installing taps."""
    root = Path(root).resolve()
    plan = observer.default_plan(policy)
    actual = []
    for _, owner, name, source, checksum in plan:
        path = Path(source).resolve()
        require(path == root / 'our_mani/multitask' / path.name
            and observer.PINS.get(path.name) == checksum
            and hashlib.sha256(path.read_bytes()).hexdigest() == checksum,
            'canonical unchanged predicate source required')
        actual.append((owner.__name__, name))
    require(len(actual) == 17 and len(set(actual)) == 17 and set(actual) == EXPECTED_METHODS,
        'exact seventeen-method remove/frontend MRO required')
    return [dict(path=str(Path(source).resolve().relative_to(root)), sha256=checksum,
        class_name=owner.__name__, method=name) for _, owner, name, source, checksum in plan]


def _leaf_rows(call):
    rows = []
    for label, expression in call.get('expressions', {}).items():
        for value in call.get('evaluations', {}).get(label, ()):
            rows.append(dict(label=label, expression=expression.get('expression'), kind=expression.get('kind'), **value))
    return rows


def _boolean_leaves(call):
    return {row['expression']: row['value'] for row in _leaf_rows(call)
        if row.get('status') == 'recorded' and type(row.get('value')) is bool}


def _recorded_return(call):
    value = call.get('return_value', {})
    return value.get('value') if value.get('status') == 'recorded' and type(value.get('value')) is bool else None


def _blocker(call, children):
    """Name only a false path proved by its actually evaluated leaf values."""
    leaves = _boolean_leaves(call)
    if leaves.get('target is None') is True:
        return 'completion_target_unavailable'
    if any(value is True and 'timestamp - self.phase_epoch' in expression
           and '<' in expression and 'duration_seconds' in expression for expression, value in leaves.items()):
        return 'minimum_acquire_duration_not_elapsed'
    captures = [child for child in children if child.get('source', {}).get('method') == '_public_capture']
    if captures and _recorded_return(captures[-1]) is False:
        return 'public_capture_returned_false'
    if leaves.get('distance <= 0.02') is False:
        return 'current_position_outside_2cm'
    if leaves.get('angle <= 0.1') is False:
        return 'current_orientation_outside_0_1rad'
    if _recorded_return(call) is True:
        return 'acquire_done_true'
    return 'unknown_unobserved_or_unmaterialized_condition'


def _capture_blocker(call):
    leaves = _boolean_leaves(call)
    early = (
        ('len(ids) != 1', True, 'manipulated_body_count_not_one'),
        ('s.robot.contact', False, 'shared_held_presence_false'),
        ('w is None', True, 'own_close_witness_absent'),
        ("w['phase_index'] != self.phase_index", True, 'close_witness_phase_mismatch'),
        ("tuple(w['manipulated_ids']) != ids", True, 'close_witness_manipulated_identity_mismatch'),
    )
    for expression, value, reason in early:
        if expression in leaves and leaves[expression] is value:
            return reason
    for expression, value in leaves.items():
        if "w['confirmed_after']" in expression and '>' in expression and value is True:
            return 'close_witness_confirmation_in_future'
    required = (
        ('identity not in invalid', 'public_identity_invalid'),
        ('associated == {identity}', 'public_attached_identity_not_unique_or_mismatch'),
        ("bodies.get(identity, {}).get('mode_provenance') == 'caller_controller_touch_action_proprioception'",
            'public_attachment_provenance_mismatch'),
        ("tuple(s.role_bindings.get(goal['template'].manipulated_role, ())) == ids", 'public_role_binding_mismatch'),
        ('identity in f.actor.controller.measured_attachments', 'measured_attachment_transform_absent'),
    )
    for expression, reason in required:
        if leaves.get(expression) is False:
            return reason
    if _recorded_return(call) is True:
        return 'public_capture_true'
    return 'unknown_unobserved_or_unmaterialized_condition'


def summarize_capture_receipt(receipt):
    """Post-process saved allowlisted taps only; never revisit policy objects."""
    calls = receipt.get('calls', [])
    children = {}
    for call in calls:
        children.setdefault(call.get('parent_call_id'), []).append(call)
    done, captures, ready = [], [], []
    for call in calls:
        source = call.get('source', {})
        name = source.get('method')
        if source.get('class_name') == 'SkillGeometryIntents' and name == '_done':
            # RemoveAuthorIntents delegates its acquire branch to this body.
            done.append(dict(call_id=call.get('call_id'), parent_call_id=call.get('parent_call_id'),
                result=_recorded_return(call), first_supported_blocker=_blocker(call, children.get(call.get('call_id'), [])),
                evaluated_leaves=_leaf_rows(call), unknown_not_evaluated=list(call.get('not_evaluated', {})),
                compound_conditions_unknown=list(call.get('unobserved_compound_conditions', {}))))
        elif name == '_public_capture':
            captures.append(dict(call_id=call.get('call_id'), parent_call_id=call.get('parent_call_id'),
                result=_recorded_return(call), first_supported_blocker=_capture_blocker(call), evaluated_leaves=_leaf_rows(call),
                unknown_not_evaluated=list(call.get('not_evaluated', {})),
                compound_conditions_unknown=list(call.get('unobserved_compound_conditions', {}))))
        elif source.get('class_name') == 'PublicModelFrontend' and name == 'ready':
            ready.append(dict(call_id=call.get('call_id'), result=_recorded_return(call),
                evaluated_boolean_leaves=_boolean_leaves(call)))
    return dict(schema=SCHEMA, calls=len(calls), acquire_done_calls=len(done), public_capture_calls=len(captures),
        acquire_done_blocker_counts=dict(Counter(row['first_supported_blocker'] for row in done)),
        public_capture_blocker_counts=dict(Counter(row['first_supported_blocker'] for row in captures)),
        acquire_done=done, public_capture=captures, readiness=ready,
        unknown_not_observed_is_not_false=True, no_native_grasp_identity_or_success_inferred=True,
        observer_errors=receipt.get('errors', []), dropped_calls=receipt.get('dropped_calls', 0))


@contextmanager
def observer_context_factory(root, runtime_module, selection, events, *, permit=None, max_calls=10000):
    """Install the pinned observer in one shared-runner-owned TRAIN process.

    permit is intentionally not interpreted here: the shared runner owns and
    validates the actual resource capability. The v892 observer authenticates
    the original TRAIN component domain and cannot mint resource permission.
    """
    validate_selection(selection)
    require(type(events) is dict and type(max_calls) is int and max_calls > 0,
        'event mapping and bounded positive call cap required')
    require(_SCOPE_LOCK.acquire(blocking=False), 'one isolated same-thread remove observer scope')
    original = runtime_module._geometry
    thread = threading.get_ident()
    instances = []
    try:
        observer = _load_observer(root)
        with ExitStack() as stack:
            def geometry(prepared, env, descriptions):
                require(threading.get_ident() == thread and not instances,
                    'one original geometry construction in the owning thread')
                args, domain = prepared['args'], prepared['component_domain']
                require(args.split == 'train' and args.task == selection['task'] and args.method == selection['method']
                    and args.raw_episode_id == selection['raw_episode_id']
                    and prepared['config'].get('context_checkpoint') == CHECKPOINT,
                    'original prepared fixed TRAIN route and v671 checkpoint required')
                require(domain.require(args.task, args.method) == 'train_qualification'
                    and domain.split == 'train' and domain.raw == selection['raw_episode_id'],
                    'original source-bound TRAIN component capability required')
                result = original(prepared, env, descriptions)
                require(type(result) is tuple and len(result) == 5, 'original geometry tuple required')
                body_plan = require_body_plan(root, observer, result[3])
                value = observer.TrainRemoveCaptureValueObserver(result[3], domain=domain, enabled=True,
                    development_only=True, max_calls=max_calls)
                instances.append(value)
                stack.enter_context(value)
                events.update(schema=SCHEMA, expected_identity=prepared['identity'], component_domain=domain.receipt(),
                    expected_body_plan=body_plan, selection=dict(selection), instrumented=True,
                    formal_score_contribution=False, physical_equivalence_claimed=False,
                    observer_consumes_resource_permit=False)
                return result
            with patch.object(runtime_module, '_geometry', geometry):
                yield
    finally:
        try:
            original_error = sys.exc_info()[1]
            try:
                events['observer_receipts'] = [value.receipt() for value in instances]
                events['restored'] = runtime_module._geometry is original and all(
                    value.get('restored') is True for value in events['observer_receipts'])
                events['geometry_hook_restored'] = runtime_module._geometry is original
                events['summary'] = [summarize_capture_receipt(value) for value in events['observer_receipts']]
            except BaseException as recording_error:
                events['observer_recording_error'] = type(recording_error).__name__
                if original_error is None:
                    raise
        finally:
            _SCOPE_LOCK.release()


# An import or direct invocation never performs a run. v1103 is the only entry.
if __name__ == '__main__':
    raise SystemExit('Use the shared v1103 runner with --diagnostic remove; this module does not launch jobs.')
