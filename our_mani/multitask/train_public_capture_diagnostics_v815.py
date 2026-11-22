"""UNENABLED partial observer for a future isolated TRAIN remove-cup process.

Feasibility limit: existing capture/ready implementations use short-circuit
expressions. Their individual boolean evaluations and _done's distance/angle
locals are not returned. Instance wrappers CANNOT recover that complete vector.
This module records ONLY original calls, existing public operands, original
returned geometry/proofs, and nested-call identities. It never calls a predicate
again, computes a distance/angle, predicts, resolves roles, or accesses native
environment/task state. Unknown subpredicate evaluations stay explicitly unknown.

A future separately reviewed TRAIN runner could install this scope after actor
construction, then supply its already-known counted tick/action metadata with
bind_context before the original calls. No runner/import/control-chain hook is
provided here. To obtain the remaining exact local/short-circuit vector, a
future isolated version of those predicate bodies would need explicit receipts,
source fingerprints, and parity checks. Global monkeypatches, frame tracing and
re-executing capture/readiness are deliberately not used.

That future runner must separately honor STOP, wait for formal775/772 to exit,
and hold the existing global resource locks in a fresh TRAIN-only output. This
observer implements no resource admission, runner, queue, or automatic launch.

Default construction/entry is inert. Explicit development_only opt-in and an
authentic TRAIN capability are mandatory. Native operation is NOT validated;
this is neither a fix nor qualification/VAL/TEST evidence. No files are written.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import threading


SCHEMA = 'r10_partial_public_capture_observer_TRAIN_only_v815'
TASKS = ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')
_MISSING = object()
_RESERVED = set()
_LOCK = threading.Lock()

LIMITATIONS = (
    'Short-circuit subpredicate evaluation order/results are not exposed by existing returns.',
    '_done distance/angle and elapsed-duration comparison locals are unknown; no geometry is recomputed.',
    'Readiness proof is observed only when the original frontend actually calls collision_port.readiness.',
    'Only instance-dispatched entry calls are hooked; explicit class/super calls do not produce separate intermediate-method events.',
    'Missing child calls are not automatically classified as a false subpredicate or failed physical capture.',
    'Exact tick/action metadata requires a future standalone TRAIN runner bind_context call; absent/stale context is unknown.',
    'No native attachment, target truth, success/failure outcome or physical capture sensor is queried.',
)

SUBPREDICATES = {
    '_public_capture': (
        'exactly_one_manipulated_id', 'public_contact', 'close_witness_present',
        'witness_phase_matches', 'witness_manipulated_ids_match', 'witness_confirmation_not_future',
        'identity_not_invalid', 'unique_associated_attached_identity', 'own_action_mode_provenance',
        'public_role_binding_matches', 'measured_attachment_present'),
    '_done': ('completion_target_available', 'elapsed_duration_sufficient',
        'public_capture_passed', 'current_position_within_2cm', 'current_angle_within_0_1rad'),
    'ready': ('state_available', 'forecast_available', 'role_ready',
        'no_invalid_modes', 'all_modes_known', 'original_collision_readiness'),
}


def _plain(value):
    if value is None or type(value) in (str, bool, int): return value
    if isinstance(value, float): return value if math.isfinite(value) else str(value)
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if hasattr(value, 'tolist'): return _plain(value.tolist())
    return dict(unrecorded_type=type(value).__name__)


def _pose(value):
    if value is None: return None
    # Already computed public contract Pose operands only; no native getter.
    return dict(position=_plain(value.position), rotation=_plain(value.rotation))


def _error(problem): return dict(type=type(problem).__name__, message=str(problem))


class TrainPublicCaptureDiagnostics:
    def __init__(self, policy=None, *, domain=None, enabled=False,
                 development_only=False, max_events=10000):
        self.policy, self.domain = policy, domain
        self.enabled, self.development_only = enabled, development_only
        self.max_events = max_events
        self.events, self.diagnostic_errors = [], []
        self.dropped_events = 0
        self._patches, self._reserved, self._calls = [], set(), []
        self._entered, self._sequence, self._context = False, 0, None
        self._source_sha = None

    def _diagnostic(self, operation, fn, default=None):
        try: return fn()
        except BaseException as problem:
            self.diagnostic_errors.append(dict(operation=operation, error=_error(problem)))
            return default

    def _patch(self, obj, name, wrapper):
        previous = vars(obj).get(name, _MISSING)
        setattr(obj, name, wrapper)
        self._patches.append((obj, name, previous))

    def _restore(self):
        errors = []
        while self._patches:
            obj, name, previous = self._patches.pop()
            try:
                if previous is _MISSING: delattr(obj, name)
                else: setattr(obj, name, previous)
            except BaseException as problem: errors.append(dict(attribute=name, error=_error(problem)))
        with _LOCK: _RESERVED.difference_update(self._reserved)
        self._reserved.clear()
        self._entered, self._context = False, None
        return errors

    def __enter__(self):
        if self._entered: raise RuntimeError('capture diagnostic scope cannot be reentered')
        if not self.enabled:
            self._entered = True
            return self
        from .rml_component_domain_v640 import ComponentDomain
        if (self.development_only is not True or not isinstance(self.domain, ComponentDomain)
                or self.domain.require(self.policy.task) != 'train_qualification'
                or self.domain.split != 'train' or self.domain.task not in TASKS
                or self.domain.method not in ('cv_geometry', 'context_geometry')):
            raise ValueError('explicit development-only authentic remove-cup TRAIN capability required')
        if type(self.max_events) is not int or self.max_events < 1:
            raise ValueError('positive max_events required')
        self.frontend = self.policy.frontend
        self.port = self.frontend.actor.collision_port
        hooks = [(self.policy, name) for name in
            ('select', '_done', '_public_capture', '_completion_target', 'qualify_opening', 'confirm_opening')]
        hooks += [(self.frontend, 'ready'), (self.port, 'readiness')]
        originals = [(obj, name, getattr(obj, name)) for obj,name in hooks]
        if any(not callable(fn) for _,_,fn in originals):
            raise ValueError('actual existing public predicate callables required')
        reserved = {id(obj) for obj,_ in hooks}
        with _LOCK:
            if reserved & _RESERVED: raise RuntimeError('overlapping capture diagnostic scopes prohibited')
            _RESERVED.update(reserved); self._reserved = reserved
        self._thread = threading.get_ident()
        try:
            self._source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            for obj,name,original in originals: self._patch(obj,name,self._wrapper(name,original))
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, kind, problem, traceback):
        errors = self._restore()
        if errors:
            self.diagnostic_errors.extend(dict(operation='restore', **item) for item in errors)
            if problem is None: raise RuntimeError('capture diagnostic hook restore failed: '+repr(errors))
            if hasattr(problem,'add_note'): problem.add_note('capture diagnostic hook restore failed: '+repr(errors))
        return False

    def bind_context(self, *, tick, action_id, action_kind, action_tick=None, dt=.05):
        """Only diagnostic metadata, supplied by a future runner; no native reads."""
        if not self.enabled: return False
        if not self._entered or threading.get_ident()!=self._thread:
            raise RuntimeError('active same-thread TRAIN observer required')
        if (type(tick) is not int or tick<0 or type(action_id) not in (str,int)
                or action_kind not in ('initial_observation','geometry_execute','public_hold','outside_action')
                or action_tick is not None and (type(action_tick) is not int or action_tick<0)
                or dt!=.05):
            raise ValueError('explicit counted50ms tick/action metadata required')
        self._context = dict(tick=tick,action_id=action_id,action_kind=action_kind,action_tick=action_tick,dt=dt)
        return True

    def _snapshot(self, name, args, kwargs):
        p, f = self.policy, self.frontend
        state = f.state
        goal = (args[0] if args else kwargs.get('goal')) if name in ('_done','_public_capture','_completion_target') else p.selected
        result = dict(phase_index=p.phase_index,phase_epoch=p.phase_epoch,
            close_witness=_plain(p.close_witness),state_available=state is not None,
            forecast_available=f.forecast is not None,role_ready=f.role_ready,
            caller_context=_plain(self._context),
            action_context_provenance='caller-supplied standalone TRAIN metadata; no native action probe')
        if goal is not None:
            template=goal['template']
            result['goal'] = dict(phase=goal.get('phase'),reference_id=goal.get('reference_id'),
                manipulated_ids=_plain(goal.get('manipulated_ids')),selected_at=goal.get('selected_at'),
                predicted_reference_time=goal.get('predicted_reference_time'),
                author_substage=goal.get('author_substage'),
                template=dict(manipulated_role=template.manipulated_role,duration_seconds=template.duration_seconds,
                    reference_kind=template.reference_kind,reference_time_basis=template.reference_time_basis))
        if state is not None:
            robot=state.robot
            now=float(robot.timestamp)
            result['public_frame'] = dict(timestamp=now,contact=_plain(robot.contact),
                gripper_open=_plain(robot.gripper_open),robot_pose=_pose(robot.pose),joints=_plain(robot.joints),
                role_bindings=_plain(state.role_bindings),
                invalid_mode_ids=_plain(state.perception_metadata.get('invalid_mode_ids',())),
                body_mode_operands={key:{field:_plain(value.get(field)) for field in
                    ('motion_mode','mode_provenance','registration_valid','directly_visible','pose_evidence_until')}
                    for key,value in state.perception_metadata.get('bodies',{}).items()},
                measured_attachment_ids=list(f.actor.controller.measured_attachments),
                measured_attachment_public_records=_plain(f.actor.controller.measured_attachments))
            current = self._context is not None and abs(self._context['tick']*.05-now)<=1e-7
            result['exact_tick'] = self._context['tick'] if current else None
            result['exact_action_id'] = self._context['action_id'] if current else None
            result['tick_context_status'] = 'matches_current_public_frame' if current else 'missing_or_stale_unknown'
        else:
            result.update(exact_tick=None,exact_action_id=None,tick_context_status='public_frame_unavailable_unknown')
        result['frontend_mode_operands'] = _plain({key:f.actor.last_diagnostics.get(key)
            for key in ('invalid_mode_ids','mode_estimates')})
        if f.forecast is not None:
            result['existing_public_forecast'] = dict(object_ids=_plain(f.forecast.object_ids),
                times=_plain(f.forecast.times),metadata=_plain(f.forecast.metadata))
        if name=='qualify_opening':
            result['original_opening_request'] = _plain(dict(requested=args[0] if args else kwargs.get('requested'),
                current_requested=args[1] if len(args)>1 else kwargs.get('current_requested')))
        if name=='confirm_opening':
            result['original_confirmation_operands'] = _plain(dict(opening=args[0] if args else kwargs.get('opening'),
                proof=args[1] if len(args)>1 else kwargs.get('proof'),
                proposed=args[2] if len(args)>2 else kwargs.get('proposed'),confirmed_after=kwargs.get('confirmed_after')))
        if name in ('ready','readiness'):
            result['requested_phase'] = args[0] if name=='ready' and args else args[2] if name=='readiness' and len(args)>2 else kwargs.get('phase')
        return result

    def _wrapper(self, name, original):
        def observed(*args, **kwargs):
            if threading.get_ident()!=self._thread: return original(*args,**kwargs)
            self._sequence += 1
            conditions = SUBPREDICATES.get(name,())
            if name=='_done':
                goal=args[0] if args else kwargs.get('goal')
                if goal is None or goal.get('phase')!='acquire': conditions=()
            event = dict(call_id=self._sequence,parent_call_id=self._calls[-1] if self._calls else None,
                method=name,public_operands=self._diagnostic('snapshot:'+name,lambda:self._snapshot(name,args,kwargs)),
                original_subpredicate_evaluations={key:dict(value=None,status='not_exposed_by_original_call')
                    for key in conditions},extra_predicate_or_geometry_calls=0)
            self._calls.append(event['call_id'])
            try:
                result = original(*args,**kwargs)  # Exactly one original invocation.
                if name=='_completion_target':
                    self._diagnostic('target_return',lambda:event.update(original_returned_public_target=_pose(result)))
                else:
                    self._diagnostic('return:'+name,lambda:event.update(original_return=_plain(result)))
                if name in ('select','confirm_opening'):
                    self._diagnostic('post_snapshot:'+name,lambda:event.update(
                        public_operands_after=self._snapshot(name,args,kwargs)))
                return result
            except BaseException as problem:
                self._diagnostic('exception:'+name,lambda:event.update(original_exception=_error(problem)))
                raise
            finally:
                self._calls.pop()
                if len(self.events)<self.max_events: self.events.append(event)
                else: self.dropped_events += 1
        return observed

    def receipt(self):
        return deepcopy(dict(schema=SCHEMA,enabled=bool(self.enabled),source_sha256=self._source_sha,
            development_only=True,TRAIN_only=True,native_validated=False,fix_claimed=False,
            participates_in_formal_qualification=False,participates_in_VAL_TEST=False,
            full_subpredicate_vector_available=False,geometry_recomputed=False,native_state_queried=False,
            predicates_called_for_diagnostics=0,class_or_module_hooks_installed=False,
            resource_gate_implemented_here=False,standalone_runner_provided=False,
            original_results_and_exceptions_forwarded=True,limitations=list(LIMITATIONS),
            domain=self.domain.receipt() if self.enabled else None,
            events=self.events,diagnostic_errors=self.diagnostic_errors,dropped_events=self.dropped_events,
            hook_scope_active=self._entered,restored=not self._patches))
