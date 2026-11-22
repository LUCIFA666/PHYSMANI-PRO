"""Evaluation-only copies of readiness/goal calls the policy already makes.

Enter after1132.runtime_scope and1126.public_execution_trace_scope. This probe
does not request readiness, references, forecasts, poses, sensors or predicates.
Four owned methods are wrapped; each original call/return/exception is preserved.
The source _done method is not patched. Its reference-call context is identified
from bounded Python code frames, without reading frame locals or native objects.
"""
from contextlib import contextmanager, ExitStack
import math
import sys
from types import FunctionType, GetSetDescriptorType, MemberDescriptorType, MethodType
from unittest.mock import patch

import numpy as np

from ..contracts import Forecast, ObjectState, Pose, RobotState
from .interaction import PhaseTemplate, PolicyState

SCHEMA = 'r10_evaluation_existing_public_readiness_probe_v1134'
METHODS = ('frontend.ready', 'collision_port.readiness', 'source_policy.candidates',
           'generator.observed_reference')
_DONE_FILES = frozenset(('physmani_skill_policy_v548.py', 'physmani_ring_compensation_v580.py',
    'physmani_ring_release_v590.py', 'physmani_geometry_goals_v529.py',
    'public_ring_seating_policy_v1129.py'))


def _storage(value):
    """Read a built-in instance dictionary descriptor; never a user property."""
    cls = type(value)
    for base in type.__getattribute__(cls, '__mro__'):
        descriptor = type.__getattribute__(base, '__dict__').get('__dict__')
        if descriptor is not None:
            if type(descriptor) not in (GetSetDescriptorType, MemberDescriptorType):
                return {}
            result = descriptor.__get__(value, cls)
            return result if type(result) is dict else {}
    return {}


def _owned_callable(owned, attribute):
    data = _storage(owned)
    if attribute in data:
        value = data[attribute]
    else:
        value = None
        for base in type.__getattribute__(type(owned), '__mro__'):
            raw = type.__getattribute__(base, '__dict__').get(attribute)
            if raw is not None:
                # Only ordinary Python method binding; no arbitrary descriptor.
                if type(raw) is not FunctionType:
                    return None
                value = MethodType(raw, owned)
                break
    return value if callable(value) else None


class ReadinessProbe:
    def __init__(self, *, max_events=1000):
        if type(max_events) is not int or not 1 <= max_events <= 20000:
            raise ValueError('bounded positive readiness event limit required')
        self.max_events = max_events
        self.events = []; self.errors = []; self.unknown_reasons = {}
        self.method_call_counts = {name:dict(calls=0, returns=0, exceptions=0) for name in METHODS}
        self.geometry_calls = self.geometry_returns = self.geometry_exceptions = 0
        self.dropped_events = self.recording_errors = self.call_sequence = self.finish_sequence = 0
        self.hooks = []; self._installed = set(); self._active = []
        self.scope_exception_type = self.restoration_error = None

    def unknown(self, reason):
        self.unknown_reasons[reason] = self.unknown_reasons.get(reason, 0) + 1
        return dict(status='unknown', reason=reason)

    def error(self, stage, error):
        self.recording_errors += 1
        if len(self.errors) < 32:
            self.errors.append(dict(stage=stage, exception_type=type(error).__name__))

    def plain(self, value, *, depth=0, budget=None):
        if budget is None: budget = [1024]
        budget[0] -= 1
        if budget[0] < 0 or depth > 6:
            return self.unknown('plain_copy_value_or_depth_limit')
        cls = type(value)
        if value is None or cls in (bool, int): return value
        if cls is str:
            return value if len(value) <= 1024 else self.unknown('plain_string_limit')
        if cls is float:
            return value if math.isfinite(value) else self.unknown('nonfinite_plain_number')
        if cls is np.ndarray:
            if value.dtype.kind not in 'biuf' or value.size > 128:
                return self.unknown('plain_array_type_or_size_limit')
            return value.tolist() if np.isfinite(value).all() else self.unknown('nonfinite_plain_array')
        if cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'float64', 'float32', 'int64', 'int32'):
            return self.plain(value.item(), depth=depth+1, budget=budget)
        if cls in (tuple, list) and len(value) <= 128:
            return [self.plain(item, depth=depth+1, budget=budget) for item in value]
        if cls is dict and len(value) <= 64 and all(type(key) is str and len(key) <= 256 for key in value):
            return {key:self.plain(item, depth=depth+1, budget=budget) for key,item in value.items()}
        return self.unknown('not_bounded_plain_data_no_getter_read')

    def pose(self, value):
        if type(value) is not Pose: return self.unknown('not_original_Pose_DTO')
        data = _storage(value)
        return dict(position=self.plain(data.get('position')), rotation=self.plain(data.get('rotation')))

    def template(self, value):
        cls = type(value)
        corrected = (cls.__name__ == 'CorrectedTemplate' and cls.__module__ in (
            'our_mani.multitask.grasp_frame_compensation',
            'our_mani.multitask.grasp_frame_compensation_native_compat') and PhaseTemplate in cls.__mro__)
        if cls is not PhaseTemplate and not corrected:
            return self.unknown('not_original_PhaseTemplate_DTO')
        data = _storage(value)
        return {key:self.plain(data[key]) if key in data else self.unknown('template_field_not_stored:'+key)
                for key in ('phase','anchor_id','reference_role','reference_kind','reference_mode',
                            'manipulated_role','lag_seconds','reference_time_basis')}

    def selected(self, value):
        if type(value) is not dict: return self.unknown('no_stored_selected_context')
        result = {key:self.plain(value[key]) for key in ('phase','reference_id','manipulated_ids',
                  'opening','selected_at','predicted_reference_time','lookahead_seconds') if key in value}
        result['template'] = self.template(value.get('template'))
        return result

    def forecast(self, value):
        if type(value) is not Forecast: return self.unknown('not_original_Forecast_DTO')
        data = _storage(value); times = data.get('times'); metadata = data.get('metadata')
        interval = (dict(count=int(times.size), first=self.plain(times[0]), last=self.plain(times[-1]))
                    if type(times) is np.ndarray and times.ndim == 1 and 2 <= times.size <= 10000
                    else self.unknown('forecast_time_range_not_stored'))
        return dict(object_ids=self.plain(data.get('object_ids')), time_range=interval,
            observed_until=self.plain(metadata.get('observed_until')) if type(metadata) is dict
                           else self.unknown('forecast_metadata_not_plain_dict'))

    def snapshot(self, frontend, source):
        fdata = _storage(frontend); actor = _storage(fdata.get('actor'))
        controller = _storage(actor.get('controller')); diagnostics = actor.get('last_diagnostics')
        sdata = _storage(source); state = fdata.get('state')
        if type(state) is not PolicyState:
            public = self.unknown('not_original_PolicyState_DTO')
        else:
            state_data = _storage(state); robot = state_data.get('robot')
            timestamp = (_storage(robot).get('timestamp') if type(robot) is RobotState
                         else self.unknown('not_original_RobotState_DTO'))
            metadata = state_data.get('perception_metadata'); body_map = state_data.get('bodies')
            metadata_bodies = metadata.get('bodies') if type(metadata) is dict else None
            body_summary = {}
            if type(body_map) is dict and len(body_map) <= 64:
                for identity, body in body_map.items():
                    if type(identity) is not str: continue
                    if type(body) is not ObjectState:
                        body_summary[identity] = self.unknown('not_original_ObjectState_DTO'); continue
                    bdata = _storage(body); info = metadata_bodies.get(identity) if type(metadata_bodies) is dict else None
                    fields = {key:(self.plain(info[key]) if type(info) is dict and key in info
                                   else self.unknown('body_metadata_field_not_stored:'+key))
                        for key in ('motion_mode','mode_provenance','directly_visible','registration_valid',
                                    'orientation_observable','pose_evidence_until','rigid_root','pose_source','points_source')}
                    fields.update(timestamp=self.plain(bdata.get('timestamp')), visible=self.plain(bdata.get('visible')),
                                  confidence=self.plain(bdata.get('confidence')))
                    body_summary[identity] = fields
            else: body_summary = self.unknown('public_body_map_outside_bound')
            public = dict(timestamp=self.plain(timestamp), role_bindings=self.plain(state_data.get('role_bindings')),
                invalid_mode_ids=self.plain(metadata.get('invalid_mode_ids')) if type(metadata) is dict else
                                 self.unknown('metadata_invalid_modes_not_stored'), bodies=body_summary,
                rigid_groups=self.plain(metadata['rigid_groups']) if type(metadata) is dict and 'rigid_groups' in metadata
                             else self.unknown('public_rigid_group_provenance_not_stored'))
        return dict(public_state=public, role_ready=self.plain(fdata.get('role_ready')),
            diagnostic_modes=self.plain(diagnostics.get('mode_estimates')) if type(diagnostics) is dict
                             else self.unknown('last_diagnostics_not_stored'),
            diagnostic_invalid_ids=self.plain(diagnostics.get('invalid_mode_ids')) if type(diagnostics) is dict
                                   else self.unknown('last_diagnostics_not_stored'),
            committed_roles=self.plain(controller.get('committed_roles')),
            phase_index=self.plain(sdata.get('phase_index')), phase_order=self.plain(sdata.get('phase_order')),
            selected=self.selected(sdata.get('selected')), committed=self.selected(sdata.get('committed')),
            forecast=self.forecast(fdata.get('forecast')))

    def context(self):
        if any(row[1] == 'source_policy.candidates' for row in self._active):
            return dict(kind='candidate', evidence='owned_original_candidates_call_active')
        frame = None; completion_frame = None
        try:
            frame = sys._getframe(2)
            for _ in range(20):
                if frame is None: break
                code = frame.f_code; filename = code.co_filename.replace('\\','/')
                if filename.rsplit('/',1)[-1] in _DONE_FILES and '/our_mani/multitask/' in filename:
                    if code.co_name == '_done' or (code.co_name == 'done' and filename.endswith('/public_ring_seating_policy_v1129.py')):
                        return dict(kind='done', evidence='existing_python_done_code_frame',
                                    function=code.co_name, source_file=filename.rsplit('/',1)[-1])
                    if code.co_name in ('_completion_target','current_completion_pose'):
                        completion_frame = dict(function=code.co_name,source_file=filename.rsplit('/',1)[-1])
                frame = frame.f_back
        finally:
            del frame
        if completion_frame is not None:
            return dict(kind='other_or_unknown', evidence='completion_target_outside_known_done_frame',**completion_frame)
        return dict(kind='other_or_unknown', evidence='no_owned_candidate_or_known_done_code_frame')

    def request(self, method, args, kwargs):
        def arg(index, name): return args[index] if len(args)>index else kwargs.get(name)
        if method in ('frontend.ready','source_policy.candidates'):
            return dict(phase=self.plain(arg(0,'phase')))
        if method == 'collision_port.readiness':
            return dict(phase=self.plain(arg(2,'phase')))
        return dict(reference_id=self.plain(arg(1,'body_id')), template=self.template(arg(2,'template')))

    def returned(self, method, result):
        if method in ('frontend.ready','collision_port.readiness'):
            return self.plain(result)
        if method == 'source_policy.candidates':
            if type(result) not in (list,tuple): return self.unknown('candidate_return_not_original_sequence')
            return dict(candidate_count=len(result), first_candidates=[self.selected(row) for row in result[:8]],
                        candidate_details_omitted=max(0,len(result)-8))
        if type(result) is not ObjectState: return self.unknown('reference_return_not_original_ObjectState_DTO')
        data = _storage(result)
        return dict(object_id=self.plain(data.get('object_id')), timestamp=self.plain(data.get('timestamp')),
                    visible=self.plain(data.get('visible')), pose=self.pose(data.get('pose')))

    def wrap(self, original, method, frontend, source, geometry_index):
        def wrapped(*args, **kwargs):
            counts = self.method_call_counts[method]; counts['calls'] += 1
            self.call_sequence += 1; sequence = self.call_sequence
            row = None
            if len(self.events) < self.max_events:
                try:
                    row = dict(method=method,call_sequence=sequence,geometry_index=geometry_index,
                        parent_call_sequence=self._active[-1][0] if self._active else None,
                        context=self.context(), request=self.request(method,args,kwargs),
                        snapshot=self.snapshot(frontend,source), evaluation_only=True)
                except BaseException as error: self.error('before_original_copy',error)
            self._active.append((sequence,method))
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                counts['exceptions'] += 1
                self.finish(row,method,None,error)
                raise
            else:
                counts['returns'] += 1
                self.finish(row,method,result,None)
                return result
            finally:
                self._active.pop()
        return wrapped

    def finish(self, row, method, result, error):
        self.finish_sequence += 1
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        if row is None:
            row = dict(method=method, snapshot=self.unknown('original_call_snapshot_copy_failed'))
        try:
            row.update(finish_sequence=self.finish_sequence,original_returned=error is None)
            if error is None: row['original_return'] = self.returned(method,result)
            else:
                arguments = BaseException.args.__get__(error,type(error))
                row['original_exception'] = dict(type=type(error).__name__,args=self.plain(arguments),
                    reason=self.plain(arguments[0]) if len(arguments)==1 and type(arguments[0]) is str
                           else self.unknown('original_exception_has_no_single_plain_string_reason'))
            self.events.append(row)
        except BaseException as recording_error:
            self.error('after_original_copy',recording_error)

    def hook(self, stack, owned, attribute, replacement, label):
        data = _storage(owned); prior = (attribute in data,data.get(attribute))
        stack.enter_context(patch.object(owned,attribute,replacement))
        self.hooks.append((owned,attribute,prior,label))

    def install(self, stack, result, geometry_index):
        if type(result) not in (tuple,list) or len(result)!=5:
            self.unknown('geometry_factory_shape_unknown'); return
        frontend, policy = result[2:4]
        source = policy; seen = set()
        for _ in range(8):
            data = _storage(source)
            if 'selected' in data and 'phase_index' in data: break
            next_source = data.get('source_policy')
            if next_source is None or id(next_source) in seen:
                self.unknown('owned_source_policy_not_in_storage'); return
            seen.add(id(source)); source = next_source
        else:
            self.unknown('owned_source_policy_chain_limit'); return
        key = (id(frontend),id(source))
        if key in self._installed:
            self.unknown('same_owned_components_already_instrumented'); return
        self._installed.add(key)
        actor = _storage(_storage(frontend).get('actor')); controller = _storage(actor.get('controller'))
        owned_methods = ((frontend,'ready',METHODS[0]),(actor.get('collision_port'),'readiness',METHODS[1]),
                         (source,'candidates',METHODS[2]),(controller.get('generator'),'observed_reference',METHODS[3]))
        for owned, attribute, label in owned_methods:
            original = _owned_callable(owned,attribute)
            if original is None:
                self.unknown('owned_original_method_unavailable:'+label); continue
            self.hook(stack,owned,attribute,self.wrap(original,label,frontend,source,geometry_index),label)

    def receipt(self):
        checks = []
        for owned, attribute, prior, label in self.hooks:
            data = _storage(owned)
            restored = (attribute in data)==prior[0] and (not prior[0] or data.get(attribute) is prior[1])
            checks.append(dict(attribute=label,restored=bool(restored)))
        restored = self.restoration_error is None and bool(checks) and all(row['restored'] for row in checks)
        return dict(schema=SCHEMA,events=list(self.events),max_events=self.max_events,
            method_call_counts={name:dict(value) for name,value in self.method_call_counts.items()},
            geometry_calls=self.geometry_calls,geometry_returns=self.geometry_returns,geometry_exceptions=self.geometry_exceptions,
            dropped_events=self.dropped_events,recording_errors=self.recording_errors,errors=list(self.errors),
            unknown_reasons=dict(self.unknown_reasons),restored=restored,restoration_checks=checks,
            scope_exception_type=self.scope_exception_type,restoration_error=self.restoration_error,
            sysprofile_installed=False,extra_readiness_reference_forecast_or_native_calls=0,
            original_return_or_exception_rewritten=False,data_is_evaluation_only_and_not_policy_input=True,
            original_exceptions_are_not_observer_errors=True,absent_or_uncaptured_reason_is_unknown=True,
            goal_generation_after_reference_calls_not_reexecuted_or_assumed=True)


@contextmanager
def readiness_probe_scope(runtime_module, events, *, max_events=1000):
    """After parent runtime/trace scopes, publish events['readiness_probe']."""
    if type(events) is not dict or 'readiness_probe' in events:
        raise ValueError('fresh evaluation-side readiness event dictionary required')
    original = _owned_callable(runtime_module,'_geometry')
    if original is None: raise ValueError('owned original geometry factory required')
    probe = ReadinessProbe(max_events=max_events); stack = ExitStack(); original_error = None
    def geometry(*args, **kwargs):
        probe.geometry_calls += 1
        try: result = original(*args, **kwargs)
        except BaseException:
            probe.geometry_exceptions += 1
            raise
        probe.geometry_returns += 1
        try: probe.install(stack,result,probe.geometry_calls)
        except BaseException as error: probe.error('owned_component_installation',error)
        return result
    probe.hook(stack,runtime_module,'_geometry',geometry,'runtime._geometry')
    try:
        yield probe
    except BaseException as error:
        original_error = error; probe.scope_exception_type = type(error).__name__
        raise
    finally:
        try: stack.close()
        except BaseException as error:
            probe.restoration_error = type(error).__name__; probe.error('restoration',error)
            if original_error is None: raise
        finally:
            receipt = probe.receipt(); events['readiness_probe'] = receipt; events['restored'] = receipt['restored']
            events['summary'] = {name:receipt[name] for name in ('geometry_calls','method_call_counts',
                'dropped_events','recording_errors','restored','unknown_reasons')}
