"""Disabled, unqualified TRAIN taps of collision calls that already occur.

No probe, native getter, forecast, geometry calculation, physics step or resource
acquisition is performed. A future owner must already hold a genuine same-PID
v924 lease. Context is copied from stored public data only. public_scene returns
its original contextmanager object: its enter/exit/yield lifecycle is untouched
and no active scene is inferred. Logging overhead can affect planning wall time;
CPU forwarding tests cannot establish physical equivalence.
"""
from copy import deepcopy
import inspect
import math
import threading
import types

SCHEMA = 'r10_unqualified_TRAIN_original_public_collision_taps_v934'
_MISSING = object()
_RESERVED = set()
_LOCK = threading.Lock()
MAX_EVENTS = 10000
MAX_ITEMS = 64
MAX_ARRAY_ELEMENTS = 256
MAX_DEPTH = 6
MAX_TEXT = 256


def _require(value, message):
    if not value:
        raise ValueError(message)


def _namespace(value):
    """Inspect existing instance storage, never a fallback property/getter."""
    # getattr_static does not execute a custom __getattribute__ or descriptor.
    descriptor = inspect.getattr_static(type(value), '__dict__', None)
    _require(type(descriptor) in (types.GetSetDescriptorType, types.MemberDescriptorType),
             'ordinary instance dictionary storage required')
    result = object.__getattribute__(value, '__dict__')
    _require(type(result) is dict, 'plain stored instance dictionary required')
    return result


def _stored(value, name, default=None):
    return _namespace(value).get(name, default)


def _safe_method(value, name):
    namespace = _namespace(value)
    original = inspect.getattr_static(value, name, _MISSING)
    if name in namespace:
        _require(type(original) in (types.FunctionType, types.MethodType),
                 'stored callable must be a plain Python function/method: '+name)
        return original
    _require(type(original) is types.FunctionType,
             'only plain Python method descriptors may be wrapped: '+name)
    return original.__get__(value, type(value))


def _authorize(domain, development_only):
    from .rml_component_domain_v640 import ComponentDomain
    _require(development_only is True and isinstance(domain, ComponentDomain),
             'explicit authentic TRAIN component domain required')
    _require(domain.require(domain.task, domain.method) == 'train_qualification'
             and domain.split == 'train'
             and domain.method in ('cv_geometry', 'context_geometry'),
             'TRAIN-only CV/Context public geometry domain required')


def _require_lease(permit, root):
    # No injectable production validator, cross-PID token or test-callback permit.
    from .post_TEST_resource_guard_v924 import require_permit
    return require_permit(permit, root, allow_test_callbacks=False)


def _public_types():
    import numpy as np
    from ..contracts import Pose, RobotState, ObjectState, Forecast
    return np, Pose, RobotState, ObjectState, Forecast


class TrainPublicCollisionTap:
    """One isolated, owner-thread scope; disabled mode inspects nothing.

    Entry verifies a real640 TRAIN capability and existing924 lease, snapshots
    registered ghost identities, validates safe Python callable bindings, and
    then replaces only instance attributes. No existing entry installs this tap.
    Calls on another thread pass directly through without context inspection.
    """

    def __init__(self, map_service=None, *, port=None, domain=None, permit=None,
                 root=None, enabled=False, development_only=False,
                 max_events=MAX_EVENTS):
        _require(type(enabled) is bool, 'enabled must be an explicit boolean')
        self.map, self.port, self.domain = map_service, port, domain
        self.permit, self.root = permit, root
        self.enabled, self.development_only = enabled, development_only
        self.max_events = max_events
        self.events, self.errors = [], []
        self.dropped_events = self.error_count = self.dropped_errors = 0
        self.copy_truncations = self.unknown_values = 0
        self.non_owner_thread_calls_unobserved = 0
        self._entered = False
        self._entering = False
        self._patches, self._reserved, self._stack = [], set(), []
        self._ghosts, self._descriptors = [], {}
        self._sequence = 0
        self._types = None
        self._restore_errors = 0

    def _error(self, operation, error):
        self.error_count += 1
        if len(self.errors) < min(self.max_events if type(self.max_events) is int and self.max_events > 0 else 1, 100):
            self.errors.append(dict(operation=operation, exception_type=type(error).__name__))
        else:
            self.dropped_errors += 1

    def _unknown(self, reason):
        self.unknown_values += 1
        return dict(status='unknown', reason=reason)

    def _copy(self, value, depth=0):
        """Concrete public types only; no duck typing, repr, str or coercion."""
        if depth > MAX_DEPTH:
            self.copy_truncations += 1
            return self._unknown('bounded_depth')
        kind = type(value)
        if value is None or kind in (bool, int):
            return value
        if kind is float:
            return value if math.isfinite(value) else self._unknown('nonfinite_float')
        if kind is str:
            if len(value) > MAX_TEXT:
                self.copy_truncations += 1
            return value[:MAX_TEXT]
        if kind is dict:
            result = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= MAX_ITEMS:
                    self.copy_truncations += 1
                    break
                if type(key) is str and len(key) <= MAX_TEXT:
                    result[key] = self._copy(item, depth+1)
                else:
                    self.unknown_values += 1
            return result
        if kind in (list, tuple, set, frozenset):
            result = []
            for index, item in enumerate(value):
                if index >= MAX_ITEMS:
                    self.copy_truncations += 1
                    break
                result.append(self._copy(item, depth+1))
            return result
        if self._types is None:
            return self._unknown('public_types_not_loaded_while_disabled')
        np, Pose, _, _, _ = self._types
        if kind is np.ndarray:
            if value.dtype.kind not in 'biuf' or value.size > MAX_ARRAY_ELEMENTS:
                self.copy_truncations += 1
                return self._unknown('non_numeric_or_bounded_array')
            # Exact ndarray implementation, never arbitrary .tolist dispatch.
            return self._copy(np.ndarray.tolist(value), depth+1)
        if kind in (np.bool_, np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64):
            return self._copy(value.item(), depth+1)
        if kind is Pose:
            fields = _namespace(value)
            return dict(public_type='Pose', position=self._copy_field(fields, 'position', depth+1),
                        rotation=self._copy_field(fields, 'rotation', depth+1))
        if id(value) in self._descriptors:
            return deepcopy(self._descriptors[id(value)])
        return self._unknown('value_not_on_public_allowlist')

    def _attempt(self, operation, function):
        try:
            return function()
        except BaseException as error:
            self._error(operation, error)
            return self._unknown('recording_failed')

    def _copy_field(self, namespace, key, depth=0):
        # A missing stored field is unknown; an explicitly stored None remains
        # None. Never resolve a property or synthesize a replacement value.
        if key not in namespace:
            return self._unknown('public_field_not_stored')
        return self._copy(namespace[key],depth)

    def _registration(self):
        self._ghosts, self._descriptors = [], {}
        namespace = _namespace(self.map)
        static = namespace.get('static_ghosts')
        bodies = namespace.get('estimated_bodies')
        _require(type(static) in (list, tuple) and type(bodies) is dict,
                 'existing public static/estimated registration dictionaries required')
        records = {}
        def add(ghost, role, variant):
            if ghost is None:
                return
            key = id(ghost)
            if key not in records:
                records[key] = ghost
                self._descriptors[key] = dict(
                    kind='registered_public_ghost', registered_roles=[],
                    native_handle='unknown_not_queried', current_native_matrix='unknown_not_queried',
                    collidable_flag='unknown_not_queried', active_scene_membership='unknown_not_observed')
            self._descriptors[key]['registered_roles'].append(dict(role=role, variant=variant))
        for index, ghost in enumerate(static):
            add(ghost, 'static:'+str(index), 'static')
        for role, body in bodies.items():
            _require(type(role) is str and len(role) <= MAX_TEXT, 'public string registration role required')
            stored = _namespace(body)
            add(stored.get('ghost'), role, 'primary')
            add(stored.get('fallback_ghost'), role, 'rotation_uncertainty')
        self._ghosts = list(records.values())

    def _snapshot(self):
        namespace = _namespace(self.port)
        public = dict(candidate_binding=self._copy_field(namespace, 'bound'),
            geometry_modes=self._copy_field(namespace, 'last_geometry_modes'),
            geometry_reasons=self._copy_field(namespace, 'last_geometry_reasons'),
            estimated_tool_to_body=self._copy_field(_namespace(self.map), 'estimated_attachments'),
            active_scene='unknown_public_scene_lifecycle_not_instrumented')
        frame = namespace.get('frame')
        if type(frame) is not tuple or len(frame) != 2:
            public['frame'] = self._unknown('no_stored_public_frame')
            return public
        state, forecast = frame
        stored = _namespace(state)
        robot = stored.get('robot')
        _, _, RobotState, ObjectState, Forecast = self._types
        if type(robot) is RobotState:
            fields = _namespace(robot)
            public['robot'] = {key:self._copy_field(fields,key) for key in
                ('timestamp', 'pose', 'joints', 'joint_velocities', 'gripper_open', 'contact')}
        else:
            public['robot'] = self._unknown('not_exact_public_RobotState')
        public['role_bindings'] = self._copy_field(stored, 'role_bindings')
        bodies = stored.get('bodies')
        if type(bodies) is dict:
            observed = {}
            for index, (role, body) in enumerate(bodies.items()):
                if index >= MAX_ITEMS:
                    self.copy_truncations += 1
                    break
                if type(role) is str and type(body) is ObjectState:
                    fields = _namespace(body)
                    observed[role[:MAX_TEXT]] = {key:self._copy_field(fields,key) for key in ('pose', 'timestamp', 'visible')}
            public['observed_bodies'] = observed
        else:
            public['observed_bodies'] = self._unknown('no_stored_public_bodies')
        if type(forecast) is Forecast:
            fields = _namespace(forecast)
            public['forecast'] = {key:self._copy_field(fields,key) for key in ('object_ids', 'times', 'metadata')}
        else:
            public['forecast'] = self._unknown('not_exact_public_Forecast')
        return public

    def _begin(self, label, args, kwargs, descriptor):
        self._sequence += 1
        index = self._sequence
        parent = self._stack[-1] if self._stack else None
        self._stack.append(index)
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return index, None
        event = dict(call_id=index, parent_call_id=parent, operation=label,
            registered_receiver=deepcopy(descriptor), original_calls_performed=0,
            extra_queries_performed=0, returned=False)
        self.events.append(event)
        event['arguments'] = self._attempt('argument_copy', lambda:dict(args=self._copy(args), kwargs=self._copy(kwargs)))
        event['public_context'] = self._attempt('context_copy', self._snapshot)
        return index, event

    def _wrapper(self, original, label, descriptor):
        def tapped(*args, **kwargs):
            if threading.get_ident() != self._thread:
                self.non_owner_thread_calls_unobserved += 1
                return original(*args, **kwargs)
            index = event = None
            depth = len(self._stack)
            try:
                index, event = self._begin(label, args, kwargs, descriptor)
            except BaseException as error:
                self._error('begin', error)
            try:
                if event is not None:
                    event['original_calls_performed'] = 1
                result = original(*args, **kwargs)
            except BaseException as error:
                if event is not None:
                    event['exception'] = dict(type=type(error).__name__, original_exception_rethrown=True)
                raise
            else:
                if event is not None:
                    event['returned'] = True
                    event['result'] = self._attempt('result_copy', lambda:self._copy(result))
                    if label == 'map.public_scene':
                        event['contextmanager_returned_unchanged'] = True
                        event['enter_exit_yield_not_observed'] = True
                return result
            finally:
                # Never let recording or cleanup replace an original exception.
                try:
                    if len(self._stack) > depth:
                        del self._stack[depth:]
                except BaseException as error:
                    self._error('event_end', error)
        return tapped

    def _patch(self, obj, name, original, label, descriptor):
        namespace = _namespace(obj)
        previous = namespace.get(name, _MISSING)
        wrapper = self._wrapper(original, label, descriptor)
        # All original attributes were validated as plain Python callables.
        object.__setattr__(obj, name, wrapper)
        self._patches.append((obj, name, previous, wrapper))

    def _restore(self):
        errors = 0
        while self._patches:
            obj, name, previous, wrapper = self._patches.pop()
            try:
                if _namespace(obj).get(name, _MISSING) is not wrapper:
                    self._error('intervening_method_mutation', RuntimeError())
                if previous is _MISSING:
                    object.__delattr__(obj, name)
                else:
                    object.__setattr__(obj, name, previous)
            except BaseException as error:
                errors += 1
                self._error('restore:'+name, error)
        with _LOCK:
            _RESERVED.difference_update(self._reserved)
        self._reserved.clear()
        self._restore_errors += errors
        self._entered = False
        return errors

    def __enter__(self):
        with _LOCK:
            _require(not self._entered and not self._entering, 'tap cannot be reentered')
            self._entering = True
        try:
            return self._enter_scope()
        finally:
            with _LOCK:
                self._entering = False

    def _enter_scope(self):
        if self.enabled is False:
            self._entered = True
            return self
        _require(self.enabled is True and type(self.max_events) is int and 1 <= self.max_events <= MAX_EVENTS,
                 'explicit enabled=True and bounded positive max_events required')
        _authorize(self.domain, self.development_only)
        _require_lease(self.permit, self.root)
        self._types = _public_types()
        _require(_stored(self.port, 'map') is self.map, 'actual public port/map pair required')
        model = _stored(self.port, 'model')
        # The owned robot model already stores its arm. Do not invoke an
        # environment property or native robot getter to rediscover it.
        arm = _stored(model, 'arm')
        _require(model is not None and arm is not None, 'stored model and own arm required')
        self._registration()
        specs = [(model,'collision','model.collision',dict(kind='own_robot_model')),
                 (model,'collision_query','model.collision_query',dict(kind='existing_public_collision_query')),
                 (self.map,'check','map.check',dict(kind='public_map')),
                 (self.map,'public_scene','map.public_scene',dict(kind='public_scene_factory')),
                 (arm,'check_arm_collision','arm.check_arm_collision',dict(kind='own_arm_collision_collection', robot_member='unknown_not_queried'))]
        specs.extend((ghost,'check_collision','registered_ghost.check_collision',self._descriptors[id(ghost)]) for ghost in self._ghosts)
        # Validate the complete set without executing descriptors before patching.
        prepared = [(obj,name,_safe_method(obj,name),label,descriptor) for obj,name,label,descriptor in specs]
        _require(_stored(model, 'collision_query') is _stored(self.port, 'query_callable'),
                 'original public port callable binding required')
        reserved = {id(obj) for obj,_,_,_,_ in prepared}
        with _LOCK:
            _require(not (_RESERVED & reserved), 'overlapping collision tap scopes prohibited')
            _RESERVED.update(reserved)
            self._reserved = reserved
        self._thread = threading.get_ident()
        try:
            for obj,name,original,label,descriptor in prepared:
                self._patch(obj,name,original,label,descriptor)
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type, exc, traceback):
        errors = self._restore()
        if errors and exc is None:
            raise RuntimeError('collision tap restoration failed; see receipt')
        return False

    def receipt(self):
        return dict(schema=SCHEMA, enabled=self.enabled, active=self._entered,
            restored=not self._patches and not self._reserved and self._restore_errors == 0,
            native_validated=False, unqualified=True, qualification=False,
            participates_in_VAL_TEST=False, formal_score_contribution=False,
            physical_equivalence_claimed=False, planning_wall_clock_invariance_claimed=False,
            extra_collision_or_probe_calls=0, extra_forecast_or_step_calls=0,
            observer_native_getter_calls=0, extra_native_getter_calls=0,
            original_native_getter_call_count='unknown_not_instrumented', active_scene_observed=False,
            public_scene_lifecycle_not_instrumented=True,
            registered_ghost_snapshot_only=True, robot_collision_member_known=False,
            registered_ghost_count=len(self._ghosts), original_calls_seen=self._sequence,
            non_owner_thread_calls_unobserved=self.non_owner_thread_calls_unobserved,
            events=deepcopy(self.events), dropped_events=self.dropped_events,
            errors=deepcopy(self.errors), error_count=self.error_count,
            dropped_errors=self.dropped_errors, copy_truncations=self.copy_truncations,
            unknown_values=self.unknown_values, unknown_values_not_filled=True)
