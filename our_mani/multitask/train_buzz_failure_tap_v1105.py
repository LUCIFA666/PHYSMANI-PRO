"""Passive TRAIN-only Buzz outcome telemetry; never calls a physical getter.

Observe Python return events from the already executed RLBench Task.success and
Condition.condition_met methods. This module neither wraps those methods nor
changes any task, condition, policy, model, source selection, or return value.
The ordinary public follow-curve trace remains the source of TCP/goal errors.
"""
from contextlib import contextmanager
import sys


TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
METHODS = ('cv_geometry', 'context_geometry')
TASK_CLASSES = {
    TASKS[0]: 'BeatTheRotatingBuzz', TASKS[1]: 'BeatTheRotatingBuzzHighSpeed',
}


def _dict(value):
    """Read only actual instance storage, without descriptors/properties."""
    try:
        result = object.__getattribute__(value, '__dict__')
    except (AttributeError, TypeError):
        return {}
    return result if type(result) is dict else {}


def _pair(value):
    if type(value) not in (tuple, list) or len(value) != 2:
        return None
    result = []
    for item in value:
        if type(item) is bool:
            result.append(item)
        elif type(item).__module__ == 'numpy' and type(item).__name__ == 'bool_':
            result.append(bool(item))
        else:
            return None
    return result


class PassiveBuzzFailureTap:
    def __init__(self, task_name, method, *, max_events=10000):
        if task_name not in TASKS or method not in METHODS:
            raise ValueError('TRAIN Buzz and the two existing geometry methods only')
        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError('bounded positive max_events required')
        self.task_name, self.method = task_name, method
        self.max_events = max_events
        self.events = []
        self.dropped_events = 0
        self.success_calls = self.success_returns = self.condition_calls = self.condition_returns = 0
        self.nonstandard_returns = 0
        self.observer_errors = 0
        self._success_frames = {}
        self._condition_frames = {}
        self._entered = self._restored = False
        self.scope_exception_type = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError('one telemetry scope per original case')
        if sys.getprofile() is not None:
            raise RuntimeError('existing profiler preserved; Buzz tap refuses installation')
        self._entered = True
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Do not erase a profiler unexpectedly installed by another component.
        if sys.getprofile() == self._profile:
            sys.setprofile(None)
            self._restored = True
        elif sys.getprofile() is None:
            self._restored = True
        if exc_type is not None:
            self.scope_exception_type = exc_type.__name__
        self._success_frames.clear()
        self._condition_frames.clear()
        return False

    def _emit(self, row):
        if len(self.events) < self.max_events:
            self.events.append(row)
        else:
            self.dropped_events += 1

    def _is_task(self, value):
        cls = type(value)
        return (cls.__name__ == TASK_CLASSES[self.task_name]
                and cls.__module__ == 'rlbench.tasks.' + self.task_name)

    def _context(self, frame, task):
        result = dict(counted_tick=None, native_tick_phase=None,
                      action_index=None, action_label=None, action_start_tick=None)
        current = frame
        for _ in range(40):
            if current is None:
                break
            if current.f_code.co_name == '_tick' and type(current.f_locals.get('phase')) is str:
                result['native_tick_phase'] = current.f_locals['phase']
            state = _dict(current.f_locals.get('self'))
            if state.get('task') is task:
                ledger = _dict(state.get('ledger'))
                if type(ledger.get('ticks')) is int:
                    result['counted_tick'] = ledger['ticks']
                active = _dict(state.get('_native_action_sampling')).get('active')
                if type(active) is dict:
                    for source, dest in (('action_index', 'action_index'), ('label', 'action_label'),
                                         ('start_tick', 'action_start_tick')):
                        value = active.get(source)
                        if type(value) in (str, int):
                            result[dest] = value
            current = current.f_back
        return result

    def _condition_owner(self, frame):
        current = frame.f_back
        for _ in range(20):
            if current is None:
                break
            token = self._success_frames.get(id(current))
            if token is not None:
                return current.f_locals.get('self'), token
            current = current.f_back
        return None, None

    def _profile(self, frame, event, arg):
        # A telemetry failure is counted, never substituted for a native result.
        # No physical or task API is called here, including when recording fails.
        try:
            self._observe(frame, event, arg)
        except BaseException:
            self.observer_errors += 1

    def _observe(self, frame, event, arg):
        if event not in ('call', 'return'):
            return
        code = frame.f_code
        filename = code.co_filename.replace('\\', '/')
        if code.co_name == 'success' and filename.endswith('/rlbench/backend/task.py'):
            task = frame.f_locals.get('self')
            if not self._is_task(task):
                return
            if event == 'call':
                self.success_calls += 1
                token = dict(success_call=self.success_calls, **self._context(frame, task))
                self._success_frames[id(frame)] = token
            else:
                token = self._success_frames.pop(id(frame), None)
                if token is None:
                    return
                self.success_returns += 1
                pair = _pair(arg)
                self.nonstandard_returns += int(pair is None)
                self._emit(dict(event='original_task_success_return', **token,
                                result=pair, return_type=type(arg).__name__,
                                exception_or_nonstandard_return=pair is None))
            return
        if code.co_name != 'condition_met' or not filename.endswith('/rlbench/backend/conditions.py'):
            return
        if event == 'call':
            condition = frame.f_locals.get('self')
            task, parent = self._condition_owner(frame)
            if parent is None:
                return
            condition_kind = condition_index = None
            storage = _dict(task)
            for key, kind in (('_fail_conditions', 'failure'), ('_success_conditions', 'success')):
                conditions = storage.get(key)
                if type(conditions) is list:
                    for index, item in enumerate(conditions):
                        if item is condition:
                            condition_kind, condition_index = kind, index
            self.condition_calls += 1
            label = {('failure', 0): 'wand_detected_by_middle_sensor',
                     ('success', 0): 'wand_detected_by_right_sensor',
                     ('success', 1): 'nothing_grasped'}.get((condition_kind, condition_index), 'unmapped')
            self._condition_frames[id(frame)] = dict(parent,
                condition_kind=condition_kind, condition_index=condition_index,
                condition_class=type(condition).__name__, source_condition_label=label)
        else:
            token = self._condition_frames.pop(id(frame), None)
            if token is None:
                return
            self.condition_returns += 1
            pair = _pair(arg)
            self.nonstandard_returns += int(pair is None)
            self._emit(dict(event='original_condition_met_return', **token,
                            result=pair, return_type=type(arg).__name__,
                            exception_or_nonstandard_return=pair is None))

    def receipt(self):
        return dict(schema='train_buzz_passive_original_outcome_tap_v1105',
            task=self.task_name, method=self.method, max_events=self.max_events,
            events=list(self.events), dropped_events=self.dropped_events,
            original_success_calls_observed=self.success_calls,
            original_success_returns_observed=self.success_returns,
            original_condition_calls_observed=self.condition_calls,
            original_condition_returns_observed=self.condition_returns,
            exception_or_nonstandard_returns=self.nonstandard_returns,
            observer_errors=self.observer_errors, restored=self._restored,
            scope_exception_type=self.scope_exception_type,
            exception_detail_unknown_for_profile_return_none=True,
            extra_success_calls=0, extra_condition_calls=0, extra_physical_getter_calls=0,
            physical_methods_patched=False, policy_model_or_outcome_modified=False,
            native_truth_passed_to_policy=False, current_thread_only=True,
            existing_profiler_never_overwritten=True)


def summarize(receipt):
    failures = [r for r in receipt['events']
                if r['event'] == 'original_condition_met_return'
                and r['condition_kind'] == 'failure' and r['result'] is not None and r['result'][0]]
    terminals = [r for r in receipt['events']
                 if r['event'] == 'original_task_success_return'
                 and r['result'] is not None and r['result'][1]]
    return dict(task=receipt['task'], method=receipt['method'],
        first_observed_failure_condition=failures[0] if failures else None,
        first_observed_raw_terminal=terminals[0] if terminals else None,
        dropped_events=receipt['dropped_events'], observer_errors=receipt['observer_errors'],
        absence_with_dropped_events_is_unknown=bool(receipt['dropped_events']),
        raw_intermediate_failure_is_not_action_end_failure=True,
        TCP_goal_errors_source='existing geometry_intents.public_trained_follow_curve.events',
        restored=receipt['restored'])


@contextmanager
def observer_context_factory(root, runtime_module, selection, events, *, permit=None):
    """Shared v1103 runner callback; the caller owns its canonical Native lease."""
    if permit is None:
        raise ValueError('shared runner Native lease permit required')
    selected = selection if type(selection) is dict else _dict(selection)
    tap = PassiveBuzzFailureTap(selected.get('task'), selected.get('method'))
    try:
        with tap:
            yield
    finally:
        receipt = tap.receipt()
        events.setdefault('observer_receipts', []).append(receipt)
        events['restored'] = receipt['restored']
        events['summary'] = summarize(receipt)
