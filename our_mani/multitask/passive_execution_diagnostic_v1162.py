"""Bounded, evaluation-only copies of calls that already occur.

This prototype installs one current-thread Python profiler, not a worker or a
policy hook.  It never invokes a registered function, a descriptor, a physical
getter, a forecast, or a predicate.  In particular, profile ``return(None)`` is
ambiguous between a real None return and Python exception unwinding.  C return
events do not contain a C result and are deliberately ignored.
"""
import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType, GetSetDescriptorType, MemberDescriptorType, MethodType

import numpy as np

from ..contracts import Forecast, ObjectState, Pose, RobotState
from .interaction import PolicyState

SCHEMA = 'r10_passive_existing_execution_diagnostic_v1162'


def _storage(value):
    """Only an actual built-in instance dictionary, never a user property."""
    cls = type(value)
    for base in type.__getattribute__(cls, '__mro__'):
        descriptor = type.__getattribute__(base, '__dict__').get('__dict__')
        if descriptor is not None:
            if type(descriptor) not in (GetSetDescriptorType, MemberDescriptorType):
                return {}
            result = descriptor.__get__(value, cls)
            return result if type(result) is dict else {}
    return {}


def unavailable(reason):
    return dict(status='unavailable', reason=reason)


@dataclass(frozen=True)
class ProfileSite:
    label: str
    filename: str
    qualname: str
    firstlineno: int
    kind: str
    source_sha256: str | None = None
    owner: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_callable(cls, label, function, *, kind):
        owner = None
        if type(function) is MethodType:
            owner, function = function.__self__, function.__func__
        if type(function) is not FunctionType:
            raise TypeError('ordinary Python function or bound method required')
        code = function.__code__
        return cls(label, code.co_filename, function.__qualname__,
                   code.co_firstlineno, kind, owner=owner)

    def key(self):
        # Python3.10 code objects have no co_qualname. Filename+def line+name
        # identify the AST declaration, including a nested same-name function.
        return self.filename, self.qualname.rsplit('.',1)[-1], self.firstlineno


# Exact source declarations, resolved with AST rather than importing PyRep.
# 548 imports the *529* completion helper; 593 is also declared explicitly.
_PROJECT_SITES = (
    ('release.done', 'physmani_ring_release_v590.py', 'AuthorRingReleaseIntents._done', 'release_done'),
    ('ring.done', 'physmani_ring_compensation_v580.py', 'CorrectedRingIntents._done', 'done'),
    ('skill.completion_target', 'physmani_skill_policy_v548.py', 'SkillGeometryIntents._completion_target', 'completion'),
    ('completion.current529', 'physmani_geometry_goals_v529.py', 'current_completion_pose', 'completion'),
    ('completion.current593', 'physmani_geometry_goals_v593.py', 'current_completion_pose', 'completion'),
    ('geometry.errors', 'physmani_skill_policy_v548.py', 'errors', 'errors'),
    ('frontend.ready', 'physmani_geometry_policy_v518.py', 'PublicModelFrontend.ready', 'ready'),
    ('frontend.observe', 'physmani_geometry_policy_v518.py', 'PublicModelFrontend.observe', 'observe'),
    ('collision.readiness', 'pipeline.py', 'PhaseCollisionPort.readiness', 'port'),
    ('reference.observed', 'interaction.py', 'CandidateGenerator.observed_reference', 'reference'),
)
_EXECUTION_SITES = (
    ('terminal.execute', 'our_mani/multitask/physmani_executor_v7.py', 'TerminalArmAdapter.execute', 'terminal'),
    ('terminal.command', 'our_mani/multitask/physmani_executor_v7.py', '_command_with_clipping_receipt', 'command'),
    ('actuator.apply', 'our_mani/sensor_alignment/physmani_executor_v6.py', '_apply_direct_target', 'apply'),
    ('actuator.command', 'our_mani/sensor_alignment/physmani_executor_v6.py',
     'PhysManiAdapter.execute.<locals>.direct_command_receipt', 'apply_outer'),
    ('path.extract', 'our_mani/ablations/physmani_executor_v4.py', 'path_poses_without_physics', 'extract'),
    ('path.filter', 'our_mani/ablations/physmani_executor_v4.py', 'modify_released_path', 'filter'),
    ('forecast.at', 'our_mani/contracts.py', 'Forecast.at', 'forecast_at'),
)


def _function_lines(tree):
    result = {}
    def walk(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                result[name] = min([node.lineno] + [d.lineno for d in node.decorator_list])
                walk(node.body, name + '.<locals>.')
    walk(tree.body)
    return result


def frozen_source_sites(root, *, pyrep_path_file, owners=None):
    """Read only explicit source files; do not import or construct a Native app.

    ``owners`` optionally maps a site label to its already owned Python object.
    The caller must install this prototype in an exclusive evaluation scope.
    Source hashes are checked again before installation.  This is not a policy
    graph, admission, source-overlay writer, or authorization to run anything.
    """
    root = Path(root).resolve(); owners = {} if owners is None else owners
    if type(owners) is not dict:
        raise TypeError('plain owner mapping required')
    declarations = [(label, root/'our_mani/multitask'/name, qual, kind)
                    for label, name, qual, kind in _PROJECT_SITES]
    declarations += [(label, root/name, qual, kind)
                     for label, name, qual, kind in _EXECUTION_SITES]
    declarations += [('path.step', Path(pyrep_path_file).resolve(), 'ArmConfigurationPath.step', 'path_step'),
                     ('path.motion', Path(pyrep_path_file).resolve(), 'ArmConfigurationPath._step_motion', 'path_motion')]
    cache = {}; sites = []
    for label, path, qual, kind in declarations:
        path = path.resolve()
        if path not in cache:
            data = path.read_bytes()
            cache[path] = (hashlib.sha256(data).hexdigest(), _function_lines(ast.parse(data)))
        sha, lines = cache[path]
        if qual not in lines:
            raise ValueError('declared original Python function missing: '+label)
        sites.append(ProfileSite(label, str(path), qual, lines[qual], kind, sha, owners.get(label)))
    return tuple(sites)


class PassiveExecutionDiagnostic:
    def __init__(self, sites, *, max_events=10000, max_serialized_bytes=8_000_000,
                 max_active_frames=128, max_values_per_event=4096):
        for value, upper in ((max_events, 50000), (max_serialized_bytes, 64_000_000),
                             (max_active_frames, 1024), (max_values_per_event, 20000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError('bounded positive diagnostic limits required')
        self.sites = tuple(sites)
        if not self.sites or len(self.sites) > 64 or any(type(s) is not ProfileSite for s in self.sites):
            raise ValueError('1..64 exact Python source sites required')
        if len({s.label for s in self.sites}) != len(self.sites):
            raise ValueError('unique site labels required')
        self._sites = {}
        for s in self.sites:
            self._sites.setdefault(s.key(), []).append(s)
        self.max_events, self.max_serialized_bytes = max_events, max_serialized_bytes
        self.max_active_frames, self.max_values_per_event = max_active_frames, max_values_per_event
        self.events = []; self._active = {}; self._entered = False; self._disabled = False
        self.call_sequence = self.event_sequence = self.serialized_bytes = 0
        self.dropped_events = self.recording_errors = self.value_truncations = 0
        self.active_frame_overflows = self.ambiguous_none_returns = 0
        self.restored = False; self.restoration_error = self.scope_exception_type = None
        self._previous_profile = None
        self.counts = {s.label:dict(calls=0, return_events=0) for s in self.sites}

    def __enter__(self):
        if self._entered:
            raise RuntimeError('one use per passive diagnostic scope')
        self._previous_profile = sys.getprofile()
        if self._previous_profile is not None:
            raise RuntimeError('existing profiler preserved; use a single explicit owner')
        for site in self.sites:
            if site.source_sha256 is not None:
                if hashlib.sha256(Path(site.filename).read_bytes()).hexdigest() != site.source_sha256:
                    raise ValueError('diagnostic declared source bytes changed: '+site.label)
        self._entered = True
        sys.setprofile(self.observe_profile_event)
        return self

    def __exit__(self, exc_type, exc, tb):
        current = sys.getprofile()
        if current == self.observe_profile_event or current is self._previous_profile:
            sys.setprofile(self._previous_profile)
            self.restored = sys.getprofile() is self._previous_profile
        else:
            # Never erase another component's subsequently installed profiler.
            self.restoration_error = 'profile_owner_changed; foreign_profile_preserved'
        self._active.clear()
        if exc_type is not None:
            self.scope_exception_type = exc_type.__name__
        return False

    def _limited(self, reason):
        self.value_truncations += 1
        return unavailable(reason)

    def _copy(self, value, budget, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 10:
            return self._limited('snapshot_value_or_depth_budget')
        cls = type(value)
        if value is None or cls in (bool, int): return value
        if cls is float: return value if math.isfinite(value) else unavailable('nonfinite_original_number')
        if cls is str: return value if len(value) <= 512 else self._limited('original_string_limit')
        if cls in (np.bool_, np.float64, np.float32, np.int64, np.int32):
            return self._copy(value.item(), budget, depth+1)
        if cls is np.ndarray:
            if value.dtype.kind not in 'biuf' or value.size > 256:
                return self._limited('original_array_type_or_element_limit')
            if not np.isfinite(value).all(): return unavailable('nonfinite_original_array')
            return self._copy(value.tolist(), budget, depth+1)
        if cls in (tuple, list):
            if len(value) > 256: return self._limited('original_sequence_limit')
            return [self._copy(v, budget, depth+1) for v in value]
        if cls is dict:
            if len(value) > 64 or any(type(k) is not str or len(k)>128 for k in value):
                return self._limited('original_dict_limit_or_nonstring_key')
            return {k:self._copy(v,budget,depth+1) for k,v in value.items()}
        data = _storage(value)
        if cls in (Pose, RobotState, ObjectState, PolicyState):
            fields = {
                Pose:('position','rotation'),
                RobotState:('timestamp','pose','joints','joint_velocities','linear_velocity','angular_velocity','gripper_open','contact'),
                ObjectState:('object_id','timestamp','pose','linear_velocity','angular_velocity','visible','confidence'),
                PolicyState:('robot','bodies','role_bindings'),
            }[cls]
            return dict(dto=cls.__name__, **{k:self._copy(data[k],budget,depth+1)
                        if k in data else unavailable('DTO_field_not_stored:'+k) for k in fields})
        if cls is Forecast:
            # Existing Forecast.at calls are observed separately. Never call at
            # ourselves, nor duplicate its potentially large prediction tensor.
            return dict(dto='Forecast', object_ids=self._copy(data.get('object_ids'),budget,depth+1),
                times=self._copy(data.get('times'),budget,depth+1),
                tensor_shapes={k:list(data[k].shape) if type(data.get(k)) is np.ndarray
                               else unavailable('not_stored_ndarray') for k in ('positions','rotations','linear_velocities','angular_velocities')},
                full_tensors=unavailable('not_copied_by_design; use_original_forecast.at_return'))
        return unavailable('not_whitelisted_plain_data_or_public_DTO')

    def _fields(self, data, names, budget):
        return {name:self._copy(data[name],budget) if name in data
                else unavailable('not_in_original_locals_or_storage:'+name) for name in names}

    def _path(self, value, budget):
        if value is None: return None
        data = _storage(value)
        return dict(python_object_token=id(value), **self._fields(data,
                    ('_path_done','_joint_position_action','_rml_handle','_num_joints'),budget))

    def _goal(self, value, budget):
        if type(value) is not dict: return unavailable('original_goal_not_stored_dict')
        return self._fields(value, ('phase','reference_id','manipulated_ids','selected_at','opening',
                                   'predicted_reference_time','pose','present_pose'), budget)

    def _path_rows(self, value, budget, selected=None):
        """Existing FK/joint output endpoints only; never recompute FK."""
        if type(value) is not np.ndarray or value.ndim!=2 or value.shape[1]!=7 or not len(value):
            return unavailable('original_nonempty_N_by7_array_not_returned')
        result=dict(shape=list(value.shape),first=self._copy(value[0],budget),
                    last=self._copy(value[-1],budget),full_rows=unavailable('not_copied_by_design'))
        if type(selected) is list and selected and type(selected[-1]) is int and 0<=selected[-1]<len(value):
            result['last_selected_index']=selected[-1]
            result['last_selected_row']=self._copy(value[selected[-1]],budget)
        else:result['last_selected_row']=unavailable('no_original_selected_index')
        return result

    def _snapshot(self, frame, site, budget):
        local = frame.f_locals
        names = ('target','current','canonical','robot','state','forecast','phase','invalid','modes',
                 'done','tick','ticks','executed','last_target','action','error','arm_error',
                 'dt','lengths','posVelAccel','pos','i','t','p1','p2','dx','qs',
                 'joints','requested','bounded','opening','before','after','native','applied',
                 'poses','selected','object_id','timestamp')
        # Only selected ordinary locals, not an arbitrary dump of self/env/task.
        result = {k:self._copy(local[k],budget) for k in names if k in local
                  and not (site.kind in ('extract','filter') and k in ('poses','joints'))}
        if site.kind in ('extract','filter'):
            for key in ('poses','joints','joint_waypoints'):
                if key in local:
                    result[key+'_existing_endpoints']=self._path_rows(local[key],budget,local.get('selected'))
        for required in {'release_done':('target','state'), 'path_motion':('state','qs'),
                         'command':('last_target',), 'apply':('current','bounded'),
                         'apply_outer':('after','native')}.get(site.kind, ()):
            result.setdefault(required, unavailable('not_yet_computed_or_not_present:'+required))
        if 'goal' in local: result['goal'] = self._goal(local['goal'],budget)
        if 'path' in local: result['path'] = self._path(local['path'],budget)
        own = _storage(local.get('self'))
        if site.kind in ('path_step','path_motion'):
            result['path'] = self._path(local.get('self'),budget)
        frontend = local.get('frontend', own.get('frontend'))
        if site.kind in ('ready','observe'): frontend = local.get('self')
        fdata = _storage(frontend)
        if fdata:
            result['frontend'] = self._fields(fdata,('state','forecast','role_ready'),budget)
        if 'selected' in own:
            result['selected_goal'] = self._goal(own['selected'],budget)
        if 'phase_index' in own:
            result['source_phase'] = self._fields(own,('phase_index','phase_order','finished'),budget)
        if site.kind == 'command':
            args, kwargs = local.get('args'),local.get('kwargs')
            result['forwarded_joints'] = (self._copy(args[1],budget) if type(args) is tuple and len(args)>1
                                          else unavailable('original_forwarded_joints_absent'))
            result['forwarded_kwargs'] = self._fields(kwargs if type(kwargs) is dict else {},('opening','phase'),budget)
            caller = frame.f_back
            if caller is not None and self._site(caller) is not None and self._site(caller).kind == 'terminal':
                result['terminal_after_step_assignment_before_dispatch'] = self._fields(caller.f_locals,
                    ('done','tick','ticks','executed','last_target','action','error','arm_error'),budget)
                result['terminal_path'] = self._path(caller.f_locals.get('path'),budget)
            else: result['terminal_after_step_assignment_before_dispatch'] = unavailable('registered_terminal_caller_absent')
        return result

    def _site(self, frame):
        code = frame.f_code
        for site in self._sites.get((code.co_filename,code.co_name,code.co_firstlineno), ()):
            if site.owner is None or frame.f_locals.get('self') is site.owner: return site
        return None

    def _parent(self, frame):
        parent = frame.f_back
        for _ in range(40):
            if parent is None: break
            token = self._active.get(id(parent))
            if token is not None: return token
            parent = parent.f_back
        return None

    def _context(self, frame, budget):
        result = {}; current = frame
        for _ in range(24):
            if current is None: break
            local = current.f_locals
            env = local.get('env',local.get('current_env'))
            env_data = _storage(env); ledger = _storage(env_data.get('ledger'))
            for key in ('ticks','dt'):
                if key in ledger: result.setdefault('ledger_'+key,self._copy(ledger[key],budget))
            if '_requested_opening' in env_data:
                result.setdefault('requested_opening',self._copy(env_data['_requested_opening'],budget))
            sampling=env_data.get('_native_action_sampling')
            sampling_data=sampling if type(sampling) is dict else _storage(sampling)
            active=sampling_data.get('active')
            if type(active) is dict:
                for key,dest in (('action_index','action_index'),('label','action_label'),('start_tick','action_start_tick')):
                    if key in active:result.setdefault(dest,self._copy(active[key],budget))
            robot = local.get('robot')
            state = local.get('state')
            if type(state) is PolicyState: robot = _storage(state).get('robot')
            if type(robot) is RobotState:
                result.setdefault('public_timestamp',self._copy(_storage(robot).get('timestamp'),budget))
            goal = local.get('goal')
            if type(goal) is dict and type(goal.get('phase')) is str:
                result.setdefault('phase',goal['phase'])
            current = current.f_back
        for key in ('public_timestamp','ledger_ticks','ledger_dt','requested_opening',
                    'action_index','action_label','action_start_tick','phase'):
            result.setdefault(key,unavailable('existing_'+key+'_not_in_bounded_stack'))
        return result

    def _emit(self, row):
        self.event_sequence += 1
        row['event_sequence'] = self.event_sequence
        size = len(json.dumps(row,ensure_ascii=True,allow_nan=False).encode('utf-8'))
        if len(self.events)>=self.max_events or self.serialized_bytes+size>self.max_serialized_bytes:
            self.dropped_events += 1; self._disabled = True; self._active.clear()
            return
        self.events.append(row); self.serialized_bytes += size

    def observe_profile_event(self, frame, event, arg):
        """Single-owner composable callback; ignores C events and never raises."""
        try:
            if event not in ('call','return'): return
            site = self._site(frame)
            if site is None: return
            if self._disabled:
                self.dropped_events += 1; return
            budget = [self.max_values_per_event]
            if event == 'call':
                if len(self._active)>=self.max_active_frames:
                    self.active_frame_overflows += 1; self.dropped_events += 1
                    self._disabled=True; self._active.clear(); return
                parent = self._parent(frame); self.call_sequence += 1
                token = dict(call_id=self.call_sequence,label=site.label,kind=site.kind,
                             parent_call_id=None if parent is None else parent['call_id'], errors_return=None,
                             port_return=None)
                self._active[id(frame)] = token; self.counts[site.label]['calls'] += 1
                self._emit(dict(event='original_python_call',call_id=token['call_id'],
                    parent_call_id=token['parent_call_id'],site=site.label,
                    context=self._context(frame,budget),snapshot=self._snapshot(frame,site,budget)))
            else:
                token = self._active.pop(id(frame),None)
                if token is None: return
                self.counts[site.label]['return_events'] += 1
                parent = self._parent(frame)
                if site.kind=='errors' and parent is not None and parent['kind'] in ('release_done','done'):
                    if type(arg) is tuple and len(arg)==2 and all(type(v) is float for v in arg):
                        parent['errors_return']=arg
                if site.kind=='port' and parent is not None and parent['kind']=='ready':
                    parent['port_return']=self._copy(arg,budget)
                if arg is None: self.ambiguous_none_returns += 1
                returned=(dict(poses_existing_endpoints=self._path_rows(arg[0],budget),
                               joint_existing_endpoints=self._path_rows(arg[1],budget))
                          if site.kind=='extract' and type(arg) is tuple and len(arg)==2 else
                          self._copy(arg,budget) if arg is not None else unavailable('None_return_or_Python_exception_unwind'))
                row = dict(event='original_python_return_event',call_id=token['call_id'],
                    parent_call_id=token['parent_call_id'],site=site.label,
                    context=self._context(frame,budget),snapshot=self._snapshot(frame,site,budget),
                    return_status='none_return_or_python_exception_unwind_unknown' if arg is None else 'non_none_original_return',
                    original_return=returned,
                    direct_done_errors_return=self._copy(token['errors_return'],budget) if token['errors_return'] is not None
                        else unavailable('no_original_errors_return_observed_for_this_done_call'),
                    direct_ready_port_return=token['port_return'] if token['port_return'] is not None
                        else unavailable('no_original_port_return_observed_for_this_ready_call'))
                if site.kind=='release_done' and type(arg) is bool:
                    row['release_leaf_evidence']=self._release_leaves(frame,token,budget)
                self._emit(row)
        except BaseException:
            self.recording_errors += 1

    def _release_leaves(self, frame, token, budget):
        local=frame.f_locals; state=local.get('state'); robot=_storage(state).get('robot') if type(state) is PolicyState else None
        data=_storage(robot) if type(robot) is RobotState else {}
        if 'target' not in local:
            return {key:unavailable('original_target_local_absent; shortcircuit_unknown') for key in
                    ('target_present','gripper_open','not_contact','distance_m','distance_le_3cm')}
        present=local['target'] is not None
        result={'target_present':present}
        open_value=data.get('gripper_open'); contact=data.get('contact')
        result['gripper_open']=(open_value if present and type(open_value) is bool else
            dict(status='not_evaluated',reason='original_target_shortcircuit') if not present else unavailable('original_robot_open_not_stored_bool'))
        evaluate_contact=present and open_value is True
        result['not_contact']=not contact if evaluate_contact and type(contact) is bool else (
            dict(status='not_evaluated',reason='earlier_original_leaf_shortcircuit') if not present or open_value is False
            else unavailable('original_open_or_contact_not_stored_bool; shortcircuit_unknown'))
        errors=token['errors_return']
        if errors is not None:
            result['distance_m']=self._copy(errors[0],budget); result['distance_le_3cm']=errors[0]<=.03
        elif not present or open_value is False or evaluate_contact and contact is True:
            result['distance_m']=result['distance_le_3cm']=dict(status='not_evaluated',reason='earlier_original_leaf_shortcircuit')
        else:
            result['distance_m']=result['distance_le_3cm']=unavailable('original_errors_return_not_observed; never_recomputed')
        return result

    def receipt(self):
        return dict(schema=SCHEMA,events=list(self.events),method_call_counts={k:dict(v) for k,v in self.counts.items()},
            max_events=self.max_events,max_serialized_bytes=self.max_serialized_bytes,serialized_bytes=self.serialized_bytes,
            dropped_events=self.dropped_events,value_truncations=self.value_truncations,recording_errors=self.recording_errors,
            active_frame_overflows=self.active_frame_overflows,ambiguous_none_returns=self.ambiguous_none_returns,
            restored=self.restored,restoration_error=self.restoration_error,scope_exception_type=self.scope_exception_type,
            diagnostic_available=bool(self.restored and self.call_sequence and not any((self.dropped_events,self.value_truncations,
                self.recording_errors,self.active_frame_overflows,self.restoration_error))),
            sites=[dict(label=s.label,filename=s.filename,qualname=s.qualname,firstlineno=s.firstlineno,
                        kind=s.kind,source_sha256=s.source_sha256,owned_instance_filter=s.owner is not None) for s in self.sites],
            current_thread_only=True,extra_native_getter_observe_predict_predicate_calls=0,
            native_grasp_ID_read=False,functions_or_frame_locals_modified=False,
            previous_profile_never_overwritten=True,Python_exception_details_unavailable_from_profile=True,
            full_forecast_tensors_copied=False,forecast_query_returns_copied_only_if_originally_called=True,
            path_step_return_caller_assignment_not_yet_completed=True,
            post_step_issued_target_source='original_terminal_command_call_after_assignment',
            apply_current_is_pre_tick_q_not_post_tick=True,
            post_tick_q_source='original_outer_direct_command_receipt_after_local',
            context_public_timestamp_is_existing_DTO_and_may_precede_current_ledger_tick=True,
            prototype_only_no_worker_owner_source_lock_graph_or_Native_launch=True)
