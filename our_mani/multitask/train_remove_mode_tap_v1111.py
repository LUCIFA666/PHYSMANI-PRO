"""Passive original PublicModeObserver locals/returns; no extra predicates.

Import is inert. Only the original observe code-object and owned instance are
profiled; no function, native getter, source, condition or return is replaced.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import types

SOURCE = 'our_mani/multitask/mode_observer.py'
SOURCE_SHA = '833b471cb541abebcbc472fb8ab0e6ee3f0b5b1c49d06ac4a97bf6e3c9855bc7'
SCHEMA = 'r10_original_public_remove_mode_locals_passive_profile_v1111'
ROOT = Path(__file__).resolve().parents[2]


def _stored(value):
    descriptor = inspect.getattr_static(type(value), '__dict__', None)
    if type(descriptor) not in (types.GetSetDescriptorType, types.MemberDescriptorType):
        return {}
    result = object.__getattribute__(value, '__dict__')
    return result if type(result) is dict else {}


def _unknown(reason):
    return dict(status='unknown', reason=reason)


def _data(value, depth=0):
    if depth > 5:
        return _unknown('bounded_depth')
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return value if value == value and abs(value) != float('inf') else _unknown('nonfinite')
    if kind in (tuple, list, set, frozenset):
        if len(value) > 32:
            return _unknown('bounded_sequence')
        copied = [_data(v, depth + 1) for v in value]
        return sorted(copied) if kind in (set, frozenset) and all(type(v) is str for v in copied) else copied
    if kind is dict:
        if len(value) > 32 or not all(type(k) is str for k in value):
            return _unknown('bounded_or_nonstring_mapping')
        return {k: _data(v, depth + 1) for k, v in value.items()}
    # Exact NumPy storage only; never arbitrary duck-typed tolist/property calls.
    if kind.__module__ == 'numpy' and kind.__name__ == 'ndarray':
        import numpy as np
        if kind is np.ndarray and value.size <= 16 and value.dtype.kind in 'biuf':
            return _data(np.ndarray.tolist(value), depth + 1)
    return _unknown('value_not_on_public_allowlist')


def _body(value):
    fields = _stored(value)
    pose = _stored(fields.get('pose'))
    return dict(object_id=_data(fields.get('object_id', _unknown('object_id_not_stored'))),
                visible=_data(fields.get('visible', _unknown('visible_not_stored'))),
                timestamp=_data(fields.get('timestamp', _unknown('timestamp_not_stored'))),
                position=_data(pose.get('position', _unknown('position_not_stored'))))


def summarize(receipt):
    counts = {}
    for event in receipt['events']:
        candidate = event.get('original_candidates_local', {})
        count = candidate.get('count') if candidate.get('status') == 'recorded_original_local' else None
        if event.get('original_return_status') != 'recorded_original_tuple':
            branch = 'unknown_nonstandard_or_exception_return'
        elif count is not None:
            branch = ('zero_candidates' if count == 0 else 'unique_candidate' if count == 1 else 'multiple_candidates')
        elif event.get('robot_input', {}).get('contact') is False:
            branch = 'contact_false_no_candidate_filter_evaluated'
        elif event.get('held_before') is not None and type(event.get('held_before')) is str:
            branch = 'existing_held_identity_no_candidate_filter_evaluated'
        else:
            branch = 'unknown_candidate_filter_not_recorded'
        event['source_branch_classification'] = branch
        counts[branch] = counts.get(branch, 0) + 1
    return dict(original_observe_calls=receipt['original_observe_calls'], branch_counts=counts,
                dropped_events=receipt['dropped_events'], observer_errors=receipt['observer_errors'],
                classification_uses_original_locals_and_return=True,
                uncomputed_candidate_list_is_unknown_not_empty=True,
                no_new_filter_or_distance_predicate_evaluated=True)


class PassiveRemoveModeTap:
    def __init__(self, observer, *, max_events=1000):
        if type(max_events) is not int or not 1 <= max_events <= 10000:
            raise ValueError('bounded positive mode event count required')
        self.observer, self.max_events = observer, max_events
        self.events, self.errors, self.frames = [], [], {}
        self.original_observe_calls = self.original_observe_returns = self.dropped_events = 0
        self.observer_errors = 0
        self.entered = self.restored = False
        self.scope_exception_type = None

    def __enter__(self):
        if self.entered or sys.getprofile() is not None:
            raise RuntimeError('one mode profiler scope; existing profiler is preserved')
        if type(self.observer).__module__ != 'our_mani.multitask.mode_observer' or type(self.observer).__name__ != 'PublicModeObserver':
            raise ValueError('actual frozen default PublicModeObserver required')
        method = inspect.getattr_static(type(self.observer), 'observe', None)
        if type(method) is not types.FunctionType or Path(method.__code__.co_filename).resolve() != ROOT / SOURCE:
            raise ValueError('original observe code-object/source required')
        if hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('original frozen mode source changed')
        self.code = method.__code__
        self.entered = True
        sys.setprofile(self._profile)
        return self

    def _profile(self, frame, event, argument):
        if event not in ('call', 'return') or frame.f_code is not self.code:
            return
        try:
            local = frame.f_locals
            if local.get('self') is not self.observer:
                return
            if event == 'call':
                self.original_observe_calls += 1
                if len(self.events) >= self.max_events:
                    self.dropped_events += 1
                    self.frames[frame] = None
                    return
                observer = _stored(self.observer)
                robot = _stored(local.get('robot'))
                pose = _stored(robot.get('pose'))
                previous = local.get('previous_states')
                row = dict(call_index=self.original_observe_calls,
                    held_before=_data(observer.get('held', _unknown('held_not_stored'))),
                    released_before=_data(observer.get('released', _unknown('released_not_stored'))),
                    graspable_ids=_data(observer.get('graspable_ids', _unknown('graspable_ids_not_stored'))),
                    association_maximum_distance_m=_data(observer.get('maximum_distance', _unknown('distance_not_stored'))),
                    robot_input=dict(timestamp=_data(robot.get('timestamp', _unknown('timestamp_not_stored'))),
                                     contact=_data(robot.get('contact', _unknown('contact_not_stored'))),
                                     position=_data(pose.get('position', _unknown('position_not_stored')))),
                    previous_states=[_body(s) for s in previous] if type(previous) in (list, tuple) and len(previous) <= 32
                                    else _unknown('previous_states_not_bounded_plain_sequence'),
                    original_return_status='not_yet_observed',
                    original_candidates_local=dict(status='unknown_not_evaluated'))
                self.events.append(row)
                self.frames[frame] = row
            else:
                self.original_observe_returns += 1
                row = self.frames.pop(frame, None)
                if row is None:
                    return
                observer = _stored(self.observer)
                row['held_after'] = _data(observer.get('held', _unknown('held_not_stored')))
                row['released_after'] = _data(observer.get('released', _unknown('released_not_stored')))
                if 'candidates' in local:
                    candidates = local['candidates']
                    row['original_candidates_local'] = (dict(status='recorded_original_local', count=len(candidates),
                        object_ids=[_data(_stored(s).get('object_id', _unknown('object_id_not_stored'))) for s in candidates])
                        if type(candidates) is list and len(candidates) <= 32 else _unknown('original_candidates_not_bounded_list'))
                if type(argument) is tuple and len(argument) == 2 and type(argument[0]) is dict and type(argument[1]) is set:
                    row.update(original_return_status='recorded_original_tuple', original_modes=_data(argument[0]),
                               original_invalid_ids=_data(argument[1]))
                else:
                    row['original_return_status'] = 'unknown_nonstandard_or_exception_return'
        except BaseException as error:
            # Recording must not change the original method or exception path.
            self.observer_errors += 1
            if len(self.errors) < 20:
                self.errors.append(dict(event=event, exception_type=type(error).__name__))

    def __exit__(self, exc_type, exc, tb):
        if sys.getprofile() == self._profile:
            sys.setprofile(None)
            self.restored = True
        elif sys.getprofile() is None:
            self.restored = True
        if exc_type is not None:
            self.scope_exception_type = exc_type.__name__
        self.frames.clear()
        return False

    def receipt(self):
        result = dict(schema=SCHEMA, source=dict(path=SOURCE, sha256=SOURCE_SHA),
            installed_after_original_geometry_construction=True, owner_thread_only=True,
            restored=self.restored, original_observe_calls=self.original_observe_calls,
            original_observe_returns=self.original_observe_returns, events=list(self.events),
            dropped_events=self.dropped_events, observer_errors=self.observer_errors, errors=list(self.errors),
            scope_exception_type=self.scope_exception_type,
            extra_native_getter_condition_prediction_or_geometry_calls=0,
            original_predicate_calls_added=0, source_or_return_modified=False,
            unqualified=True, native_validated=False, physical_equivalence_claimed=False)
        return result
