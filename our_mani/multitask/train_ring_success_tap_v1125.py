"""Passive evaluation-side ring success tree telemetry for a future TRAIN run.

Only original Task.success/Condition.condition_met call and return events are
observed. Registration ordinals come from plain Python instance dictionaries;
no Native names, identities, poses, predicates or sensor getters are queried.
Nothing in this module is attached to the current matched canaries.
"""
from contextlib import contextmanager
import sys

from .train_buzz_failure_tap_v1105 import _dict, _pair

SCHEMA = 'r10_fixed_TRAIN_ring_passive_original_success_tree_v1125'
TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')
METHODS = ('cv_geometry', 'context_geometry')
TASK_CLASSES = dict(zip(TASKS, ('InsertOntoRotatingPeg', 'InsertOntoRotatingPegHighSpeed')))


class PassiveRingSuccessTap:
    def __init__(self, task_name, method, *, max_events=5000):
        if task_name not in TASKS or method not in METHODS:
            raise ValueError('fixed TRAIN ring and existing geometry methods only')
        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError('bounded positive max_events required')
        self.task_name, self.method, self.max_events = task_name, method, max_events
        self.events = []; self.dropped_events = self.event_sequence = 0
        self.success_calls = self.success_returns = self.condition_calls = self.condition_returns = 0
        self.nonstandard_returns = self.observer_errors = self.registration_truncations = self.inner_sequence_drops = 0
        self._success_frames = {}; self._condition_frames = {}
        self._entered = self._restored = False
        self.scope_exception_type = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError('one ring telemetry scope per original case')
        if sys.getprofile() is not None:
            raise RuntimeError('existing profiler preserved; ring tap refuses installation')
        self._entered = True; sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        if sys.getprofile() == self._profile:
            sys.setprofile(None); self._restored = True
        elif sys.getprofile() is None:
            self._restored = True
        if exc_type is not None:
            self.scope_exception_type = exc_type.__name__
        self._success_frames.clear(); self._condition_frames.clear()
        return False

    def _emit(self, row):
        self.event_sequence += 1
        if len(self.events) < self.max_events:
            self.events.append(dict(event_sequence=self.event_sequence, **row))
        else:
            self.dropped_events += 1

    def _append_inner(self, owner, field, value):
        if len(owner[field]) < 128: owner[field].append(value)
        else: self.inner_sequence_drops += 1

    def _is_task(self, value):
        cls = type(value)
        return cls.__name__ == TASK_CLASSES[self.task_name] and cls.__module__ == 'rlbench.tasks.'+self.task_name

    def _context(self, frame, task):
        result = dict(counted_tick=None, native_tick_phase=None,
                      action_index=None, action_label=None, action_start_tick=None)
        current = frame
        for _ in range(40):
            if current is None: break
            if current.f_code.co_name == '_tick' and type(current.f_locals.get('phase')) is str:
                result['native_tick_phase'] = current.f_locals['phase']
            state = _dict(current.f_locals.get('self'))
            if state.get('task') is task:
                ledger = _dict(state.get('ledger'))
                if type(ledger.get('ticks')) is int: result['counted_tick'] = ledger['ticks']
                active = _dict(state.get('_native_action_sampling')).get('active')
                if type(active) is dict:
                    for source, dest in (('action_index','action_index'),('label','action_label'),('start_tick','action_start_tick')):
                        if type(active.get(source)) in (str, int): result[dest] = active[source]
            current = current.f_back
        return result

    def _registration(self, task):
        """Bounded walk of already stored Python conditions, never Native objects."""
        mapping = {}; nodes = []; seen = set(); truncated = False
        storage = _dict(task)
        def visit(condition, kind, root_index, child_ordinals, depth):
            nonlocal truncated
            if depth > 16 or len(nodes) >= 128 or id(condition) in seen:
                truncated = True; return
            seen.add(id(condition)); data = _dict(condition); cls = type(condition)
            path = [kind, root_index, *child_ordinals]
            children = data.get('_conditions')
            children = children if type(children) is list else None
            node = dict(condition_path=path, condition_kind=kind, root_index=root_index,
                child_ordinals=list(child_ordinals), condition_class=cls.__name__,
                condition_module=cls.__module__, child_count=None if children is None else len(children),
                order_matters=data.get('_order_matters') if type(data.get('_order_matters')) is bool else None)
            mapping[id(condition)] = node; nodes.append(node)
            if children is not None:
                for index, child in enumerate(children): visit(child, kind, root_index, (*child_ordinals,index), depth+1)
        for field, kind in (('_fail_conditions','failure'),('_success_conditions','success')):
            conditions = storage.get(field)
            if type(conditions) is list:
                for index, condition in enumerate(conditions): visit(condition,kind,index,(),0)
        expected_paths = [['success',0], *[['success',0,index] for index in range(4)]]
        exact = (not truncated and [node['condition_path'] for node in nodes] == expected_paths
            and nodes[0]['condition_class'] == 'ConditionSet' and nodes[0]['order_matters'] is False
            and all(node['condition_class'] == 'DetectedCondition' for node in nodes[1:])
            and all(node['condition_module'] == 'rlbench.backend.conditions' for node in nodes))
        self.registration_truncations += int(truncated)
        return mapping, nodes, exact, truncated

    def _owner(self, frame):
        current = frame.f_back; parent_condition_call = None
        for _ in range(40):
            if current is None: break
            condition = self._condition_frames.get(id(current))
            if condition is not None and parent_condition_call is None:
                parent_condition_call = condition['condition_call']
            owner = self._success_frames.get(id(current))
            if owner is not None: return owner, parent_condition_call
            current = current.f_back
        return None, None

    def _profile(self, frame, event, arg):
        try: self._observe(frame,event,arg)
        except BaseException: self.observer_errors += 1

    def _observe(self, frame, event, arg):
        if event not in ('call','return'): return
        code = frame.f_code; filename = code.co_filename.replace('\\','/')
        if code.co_name == 'success' and filename.endswith('/rlbench/backend/task.py'):
            task = frame.f_locals.get('self')
            if not self._is_task(task): return
            if event == 'call':
                self.success_calls += 1
                mapping,nodes,exact,truncated = self._registration(task)
                context = dict(success_call=self.success_calls, **self._context(frame,task))
                self._success_frames[id(frame)] = dict(context=context, mapping=mapping,
                    condition_call_sequence=[], leaf_returns=[], exact_registration=exact)
                self._emit(dict(event='original_task_success_call', **context,
                    registration=nodes, original_four_detector_registration=exact, registration_truncated=truncated))
            else:
                token = self._success_frames.pop(id(frame),None)
                if token is None: return
                self.success_returns += 1; pair = _pair(arg)
                self.nonstandard_returns += int(pair is None)
                self._emit(dict(event='original_task_success_return', **token['context'], result=pair,
                    return_type=type(arg).__name__, exception_or_nonstandard_return=pair is None,
                    condition_call_sequence=list(token['condition_call_sequence']),
                    leaf_returns=list(token['leaf_returns']), original_four_detector_registration=token['exact_registration']))
            return
        if code.co_name != 'condition_met' or not filename.endswith('/rlbench/backend/conditions.py'): return
        if event == 'call':
            owner,parent = self._owner(frame)
            if owner is None: return
            condition = frame.f_locals.get('self'); data = _dict(condition)
            registered = owner['mapping'].get(id(condition), dict(condition_path=None,
                condition_kind=None, root_index=None, child_ordinals=[], condition_class=type(condition).__name__,
                condition_module=type(condition).__module__, child_count=None, order_matters=None))
            self.condition_calls += 1
            path = registered['condition_path']
            ordinal = path[2] if owner['exact_registration'] and type(path) is list and len(path) == 3 else None
            context = dict(owner['context'], **registered, condition_call=self.condition_calls,
                parent_condition_call=parent, registered_leaf_ordinal=ordinal,
                source_condition_label=None if ordinal is None else 'success_detector'+str(ordinal))
            token = dict(context=context, owner=owner, condition_call=self.condition_calls,
                         condition_index_before=data.get('_current_condition_index') if type(data.get('_current_condition_index')) is int else None)
            self._condition_frames[id(frame)] = token
            self._append_inner(owner,'condition_call_sequence',path)
            self._emit(dict(event='original_condition_met_call', **context,
                condition_index_before=token['condition_index_before']))
        else:
            token = self._condition_frames.pop(id(frame),None)
            if token is None: return
            self.condition_returns += 1; pair = _pair(arg)
            self.nonstandard_returns += int(pair is None)
            context = token['context']; data = _dict(frame.f_locals.get('self'))
            locals_result = {}
            for name in ('met','ismet','term'):
                value = frame.f_locals.get(name)
                if type(value) is bool: locals_result[name] = value
                elif type(value).__module__ == 'numpy' and type(value).__name__ == 'bool_': locals_result[name] = bool(value)
            if context['registered_leaf_ordinal'] is not None:
                self._append_inner(token['owner'],'leaf_returns',dict(ordinal=context['registered_leaf_ordinal'],result=pair))
            self._emit(dict(event='original_condition_met_return', **context, result=pair,
                return_type=type(arg).__name__, exception_or_nonstandard_return=pair is None,
                already_computed_locals=locals_result,
                condition_index_before=token['condition_index_before'],
                condition_index_after=data.get('_current_condition_index') if type(data.get('_current_condition_index')) is int else None))

    def receipt(self):
        return dict(schema=SCHEMA, task=self.task_name, method=self.method, max_events=self.max_events,
            events=list(self.events), dropped_events=self.dropped_events, event_sequence=self.event_sequence,
            original_success_calls_observed=self.success_calls, original_success_returns_observed=self.success_returns,
            original_condition_calls_observed=self.condition_calls, original_condition_returns_observed=self.condition_returns,
            exception_or_nonstandard_returns=self.nonstandard_returns, observer_errors=self.observer_errors,
            registration_truncations=self.registration_truncations, inner_sequence_drops=self.inner_sequence_drops,
            restored=self._restored,
            scope_exception_type=self.scope_exception_type, exception_detail_unknown_for_profile_return_none=True,
            extra_success_calls=0, extra_condition_calls=0, extra_physical_getter_calls=0,
            native_object_names_handles_or_poses_read=False, native_truth_passed_to_policy=False,
            source_condition_labels_from_pinned_registration_order_only=True,
            detector_native_names_independently_verified=False,
            physical_methods_patched=False, policy_model_or_outcome_modified=False,
            current_thread_only=True, existing_profiler_never_overwritten=True, formal_VAL_TEST_qualification=False)


def summarize(receipt):
    returns = [row for row in receipt['events'] if row['event'] == 'original_task_success_return']
    exact = [row for row in returns if row['original_four_detector_registration']]
    incomplete = bool(receipt['dropped_events'] or receipt['observer_errors'] or receipt['registration_truncations']
        or receipt['exception_or_nonstandard_returns'] or receipt['inner_sequence_drops'] or not receipt['restored'] or not returns
        or len(exact) != len(returns)
        or receipt['original_success_calls_observed'] != receipt['original_success_returns_observed']
        or receipt['original_condition_calls_observed'] != receipt['original_condition_returns_observed']
        or any(len(row['leaf_returns']) != 4 for row in exact))
    leaf_counts = []
    for ordinal in range(4):
        values = [leaf['result'] for row in exact for leaf in row['leaf_returns'] if leaf['ordinal'] == ordinal]
        leaf_counts.append(dict(registered_leaf_ordinal=ordinal, source_condition_label='success_detector'+str(ordinal),
            observed_returns=len(values), true_returns=sum(v is not None and v[0] for v in values),
            false_returns=sum(v is not None and not v[0] for v in values), unknown_returns=sum(v is None for v in values)))
    return dict(task=receipt['task'], method=receipt['method'], observed_success_returns=len(returns),
        original_four_detector_success_returns=len(exact), leaves=leaf_counts,
        absence_is_unknown=incomplete, dropped_events=receipt['dropped_events'], observer_errors=receipt['observer_errors'],
        restored=receipt['restored'], policy_phase_completion_is_not_native_success=True,
        ordering_source='actual_original_condition_call_and_return_sequence', task_success_called_by_summary=False)


@contextmanager
def observer_context_factory(root, runtime_module, selection, events, *, permit=None):
    """Future independently locked runner callback; caller owns/validates lease."""
    if permit is None: raise ValueError('parent-owned Native lease permit required')
    selected = selection if type(selection) is dict else _dict(selection)
    tap = PassiveRingSuccessTap(selected.get('task'),selected.get('method'))
    try:
        with tap: yield tap
    finally:
        receipt = tap.receipt()
        events.setdefault('observer_receipts',[]).append(receipt)
        events['restored'] = receipt['restored']; events['summary'] = summarize(receipt)
