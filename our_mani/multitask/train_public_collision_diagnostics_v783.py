"""Unenabled, opt-in diagnostics for a future independent TRAIN process.

This module is not imported by any executor.  Use only *after* all public map
ghosts have been registered, in a single-threaded standalone TRAIN diagnostic:

    with TrainPublicCollisionDiagnostics(map_service, port=collision_port,
            domain=train_domain, enabled=True, development_only=True) as trace:
        # The existing TRAIN caller may now make its normal queries.
        ...
    sidecar = trace.receipt()  # The caller decides whether/where to save it.

The original map, arm and held/world calls run once with identical arguments;
their exact return objects and exceptions are forwarded.  On a positive arm
collection check, additional read-only collection-to-public-ghost queries run
*inside the original public_scene and contact exemptions*.  They never query
native task objects, step physics, move bodies or alter collision flags.  A
positive ghost target is exact; the member of the robot collision collection
is explicitly unresolved.  No positive public target means unresolved, not an
assertion of self-collision.  Held/world pairs are observed original calls.

Only instance attributes are patched, and all shadows are restored/deleted in
finally.  Concurrent or overlapping scopes on these instances are prohibited.
This is development evidence, excluded from qualification and VAL/TEST scores.
Native execution of this candidate has NOT been validated.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import threading


SCHEMA = 'r10_development_TRAIN_public_collision_diagnostics_v783'
_MISSING = object()
_RESERVED = set()
_RESERVATION_LOCK = threading.Lock()


def _plain(value):
    """Copy only known public values; never enumerate an environment/object."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    # Public Pose, not a native object: no get_pose/get_matrix invocation here.
    if hasattr(value, 'position') and hasattr(value, 'rotation'):
        return dict(position=_plain(value.position), rotation=_plain(value.rotation))
    return dict(unrecorded_type=type(value).__name__)


def _error(error):
    return dict(type=type(error).__name__, message=str(error))


class TrainPublicCollisionDiagnostics:
    """Scoped observer; disabled construction/entry touches no map or domain.

    ``domain`` must be an authentic v640 TRAIN capability.  This additionally
    requires the explicit development_only opt-in and never issues or changes
    any qualification/admission receipt.  The scope owns no files or processes.
    """

    def __init__(self, map_service=None, *, port=None, domain=None, enabled=False,
                 development_only=False, max_events=10000):
        self.map, self.port, self.domain = map_service, port, domain
        self.enabled, self.development_only = enabled, development_only
        self.max_events = max_events
        self.events, self.errors = [], []
        self.dropped_events = 0
        self._patches, self._reserved = [], set()
        self._entered = False
        self._query, self._active, self._future_offset = None, None, None
        self._ghosts, self._descriptors = [], {}
        self._source_sha = None

    def _diagnostic(self, operation, fn, default=None):
        try:
            return fn()
        except BaseException as error:
            self.errors.append(dict(operation=operation, error=_error(error)))
            return default

    def _patch(self, obj, name, wrapper):
        # PyRep Object instances support normal Python instance attributes.
        # Save the raw shadow, not a bound method that would outlive the scope.
        namespace = vars(obj)
        previous = namespace.get(name, _MISSING)
        setattr(obj, name, wrapper)
        self._patches.append((obj, name, previous))

    def _restore(self):
        errors = []
        while self._patches:
            obj, name, previous = self._patches.pop()
            try:
                if previous is _MISSING:
                    delattr(obj, name)
                else:
                    setattr(obj, name, previous)
            except Exception as error:
                errors.append(dict(attribute=name, error=_error(error)))
        with _RESERVATION_LOCK:
            _RESERVED.difference_update(self._reserved)
        self._reserved.clear()
        self._entered = False
        return errors

    def _inventory(self):
        ghosts, descriptors = [], {}
        static_records = getattr(self.map, 'map', ())
        for index, ghost in enumerate(self.map.static_ghosts):
            record = static_records[index] if index < len(static_records) else {}
            descriptors[id(ghost)] = dict(kind='public_static_snapshot_ghost',
                role='static:'+str(index), geometry_mode='static',
                shape_handle=int(ghost.get_handle()), snapshot_name=record.get('name'),
                mesh_sha256=record.get('mesh_sha256'), snapshot_pose=_plain(record.get('pose')))
            ghosts.append(ghost)
        for role, body in self.map.estimated_bodies.items():
            for mode, ghost, checksum in (
                ('primary', body.ghost, body.geometry_sha256),
                ('rotation_uncertainty', body.fallback_ghost, body.fallback_geometry_sha256)):
                if ghost is None:
                    continue
                if id(ghost) in descriptors:
                    raise ValueError('public ghost belongs to multiple roles/variants')
                descriptors[id(ghost)] = dict(kind='public_estimated_ghost', role=str(role),
                    geometry_mode=mode, shape_handle=int(ghost.get_handle()),
                    mesh_sha256=checksum, TRAIN_source_sha256=body.source_sha256,
                    provenance=body.provenance)
                ghosts.append(ghost)
        if len({id(ghost) for ghost in ghosts}) != len(ghosts):
            raise ValueError('duplicate public static/estimated ghost')
        self._ghosts, self._descriptors = ghosts, descriptors

    def __enter__(self):
        if self._entered:
            raise RuntimeError('diagnostic scope cannot be reentered')
        if not self.enabled:
            self._entered = True
            return self
        from .rml_component_domain_v640 import ComponentDomain
        if self.development_only is not True:
            raise ValueError('explicit development_only=True is required')
        if (not isinstance(self.domain, ComponentDomain)
                or self.domain.require(self.domain.task) != 'train_qualification'
                or self.domain.split != 'train'):
            raise ValueError('authentic TRAIN-only component domain required; no VAL/TEST')
        if not isinstance(self.max_events, int) or self.max_events < 1:
            raise ValueError('positive max_events required')
        if self.port is None or self.port.map is not self.map:
            raise ValueError('the actual public PhaseCollisionPort/map pair is required')
        self._inventory()
        arm = self.map.env.arm
        instances = [self.map, arm, self.port.model, *self._ghosts]
        reserved = {id(obj) for obj in instances}
        with _RESERVATION_LOCK:
            if _RESERVED & reserved:
                raise RuntimeError('overlapping public collision diagnostic scopes prohibited')
            _RESERVED.update(reserved)
            self._reserved = reserved
        self._thread = threading.get_ident()
        try:
            if self.port.model.collision_query is not self.port.query_callable:
                raise ValueError('collision model must still use this public port callable')
            self._source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            self._install(arm)
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type, exc, traceback):
        errors = self._restore()
        if errors:
            self.errors.extend(dict(operation='hook_restore', **item) for item in errors)
            if exc is None:
                raise RuntimeError('diagnostic hook restoration failed: '+repr(errors))
            if hasattr(exc, 'add_note'):
                exc.add_note('diagnostic hook restoration failed: '+repr(errors))
        return False

    def _mine(self):
        return threading.get_ident() == self._thread and self._query is not None

    def _descriptor(self, ghost):
        return deepcopy(self._descriptors.get(id(ghost), dict(kind='unregistered_not_inspected')))

    def _snapshot(self, args, kwargs):
        poses = args[0] if args else kwargs.get('poses')
        state, forecast = self.port.frame if self.port.frame is not None else (None, None)
        public = dict(future_offset=_plain(self._future_offset),
            candidate_binding=_plain(self.port.bound),
            requested_geometry_modes=_plain(kwargs.get('geometry_modes')),
            geometry_mode_evidence=_plain(self.port.last_geometry_reasons),
            supplied_estimated_poses=_plain(poses),
            ignored_object_ids=_plain(kwargs.get('ignored_object_ids', ())),
            allowed_robot_contact_ids=_plain(kwargs.get('allowed_robot_contact_ids', ())),
            estimated_tool_to_body=_plain(self.map.estimated_attachments))
        if state is not None:
            robot = state.robot
            public['current_observation'] = dict(timestamp=_plain(robot.timestamp),
                robot_pose=_plain(robot.pose), joints=_plain(robot.joints),
                gripper_open=_plain(robot.gripper_open), contact=_plain(robot.contact),
                role_bindings=_plain(state.role_bindings),
                body_poses={key: dict(pose=_plain(body.pose), timestamp=_plain(body.timestamp))
                            for key, body in state.bodies.items()},
                perception_metadata=_plain(state.perception_metadata))
        if forecast is not None:
            # These and the exact supplied poses identify the public forecast
            # sample.  Do not call forecast.at again or a native role resolver.
            public['forecast'] = dict(object_ids=_plain(forecast.object_ids),
                times=_plain(forecast.times), metadata=_plain(forecast.metadata),
                sampled_timestamp=(None if state is None or self._future_offset is None
                    else float(state.robot.timestamp)+max(0., float(self._future_offset))))
        arm, gripper = self.map.env.arm, self.map.env.gripper
        public['query_robot_proprioception'] = dict(
            arm_joints=_plain(arm.get_joint_positions()),
            gripper_joints=_plain(gripper.get_joint_positions()),
            gripper_open_amount=_plain(gripper.get_open_amount()),
            tool_pose_xyzw=_plain(arm.get_tip().get_pose()))
        return public

    def _install(self, arm):
        original_check = self.map.check
        original_scene = self.map.public_scene
        original_arm = arm.check_arm_collision
        original_future = self.port.model.collision_query

        def check(*args, **kwargs):
            if threading.get_ident() != self._thread:
                return original_check(*args, **kwargs)
            previous = self._query
            query = dict(sequence=len(self.events)+self.dropped_events+1,
                public_context=self._diagnostic('public_snapshot', lambda:self._snapshot(args, kwargs)),
                arm_world=[], held_world_calls=[], scenes=[])
            self._query = query
            try:
                result = original_check(*args, **kwargs)
                self._diagnostic('map_result', lambda:query.update(result=_plain(result)))
                return result
            except BaseException as error:
                self._diagnostic('map_exception', lambda:query.update(exception=_error(error)))
                raise
            finally:
                self._query = previous
                if len(self.events) < self.max_events:
                    self.events.append(query)
                else:
                    self.dropped_events += 1

        @contextmanager
        def scene(*args, **kwargs):
            with original_scene(*args, **kwargs) as active:
                previous = self._active
                self._active = active
                try:
                    if self._mine():
                        def snapshot():
                            self._query['scenes'].append(dict(
                                inventory_complete=all(id(g) in self._descriptors for g in active.values()),
                                active_roles={str(role):dict(shape=self._descriptor(ghost),
                                    world_matrix=_plain(ghost.get_matrix())) for role,ghost in active.items()}))
                        self._diagnostic('public_scene_snapshot', snapshot)
                    yield active  # Preserve the original yielded object.
                finally:
                    self._active = previous

        def arm_check(*args, **kwargs):
            result = original_arm(*args, **kwargs)
            if self._mine() and self._active is not None:
                self._diagnostic('arm_world_diagnostics', lambda:self._arm_evidence(
                    arm, original_arm, args, kwargs, result))
            return result

        def future(*args, **kwargs):
            if threading.get_ident() != self._thread:
                return original_future(*args, **kwargs)
            previous = self._future_offset
            self._future_offset = args[0] if args else kwargs.get('future_offset')
            try:
                return original_future(*args, **kwargs)
            finally:
                self._future_offset = previous

        self._patch(self.map, 'check', check)
        self._patch(self.map, 'public_scene', scene)
        self._patch(arm, 'check_arm_collision', arm_check)
        for ghost in self._ghosts:
            self._patch(ghost, 'check_collision', self._held_wrapper(ghost, ghost.check_collision))
        self._patch(self.port.model, 'collision_query', future)

    def _held_wrapper(self, ghost, original):
        def collision(*args, **kwargs):
            other = args[0] if args else kwargs.get('obj')
            # Observe only original held-vs-known-public-world pair calls.
            observed = (self._mine() and self._active is not None
                and any(self._active.get(role) is ghost for role in self.map.estimated_attachments)
                and id(other) in self._descriptors)
            pair = dict(branch='held-world', observation='original_pair_call',
                first=self._descriptor(ghost), second=self._descriptor(other)) if observed else None
            try:
                result = original(*args, **kwargs)
                if pair is not None:
                    self._diagnostic('held_world_result', lambda:pair.update(result=_plain(result)))
                return result
            except BaseException as error:
                if pair is not None:
                    self._diagnostic('held_world_exception', lambda:pair.update(exception=_error(error)))
                raise
            finally:
                if pair is not None:
                    self._query['held_world_calls'].append(pair)
        return collision

    def _arm_evidence(self, arm, original, args, kwargs, result):
        # maps.check calls check_arm_collision() with no target.  Never
        # reinterpret an explicit different target or probe a native object.
        aggregate = not args and not kwargs
        event = dict(branch='arm-world', original_result=_plain(result),
            original_query_is_full_collection=aggregate,
            robot=dict(kind='own_robot_arm_collision_collection',
                collection_handle=getattr(arm, '_collision_collection', None),
                member_shape='unresolved'), public_target_probes=[],
            collidable_public_targets=[], noncollidable_public_targets=[],
            contact_exempt_public_targets=[])
        self._query['arm_world'].append(event)
        eligible = []
        for ghost in self._ghosts:
            flag = bool(ghost.is_collidable())
            label = self._descriptor(ghost)
            if flag:
                eligible.append(ghost)
                event['collidable_public_targets'].append(label)
            else:
                event['noncollidable_public_targets'].append(label)
                active_role = next((role for role, value in self._active.items() if value is ghost), None)
                allowed = self._query['public_context'] or {}
                if active_role in self.map.estimated_attachments or active_role in allowed.get('allowed_robot_contact_ids', ()):
                    event['contact_exempt_public_targets'].append(label)
        # PyRep check_arm_collision is documented/implemented to return bool.
        # Do not coerce arbitrary caller return objects merely for logging.
        if aggregate and result is True:
            for ghost in eligible:
                probe = dict(target=self._descriptor(ghost),
                    observation='additional_read_only_collection_to_public_ghost')
                event['public_target_probes'].append(probe)
                try:
                    probe['result'] = _plain(original(ghost))
                except BaseException as error:
                    probe['exception'] = _error(error)
            event['resolved_public_collision_targets'] = [item['target']
                for item in event['public_target_probes'] if item.get('result') is True]
            event['resolution'] = ('exact_public_target_robot_member_unresolved'
                if event['resolved_public_collision_targets']
                else 'unresolved_no_positive_public_target_no_self_collision_claim')
        else:
            event['resolution'] = 'original_negative_or_nonaggregate_no_extra_queries'

    def receipt(self):
        """Return an in-memory sidecar; never write to a case/admission record."""
        return deepcopy(dict(schema=SCHEMA, enabled=bool(self.enabled),
            source_sha256=self._source_sha, development_only=True, TRAIN_only=True,
            participates_in_formal_qualification=False, participates_in_VAL_TEST=False,
            native_validated=False, native_task_pose_queries=False,
            extra_queries='read-only own arm collection versus already-created public ghosts',
            original_results_and_exceptions_forwarded=True,
            domain=(self.domain.receipt() if self.enabled else None),
            events=self.events, diagnostic_errors=self.errors, dropped_events=self.dropped_events,
            hook_scope_active=self._entered, restored=not self._patches))
