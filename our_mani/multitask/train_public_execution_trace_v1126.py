"""Bounded evaluation-side copies of existing public frames and planned paths.

Install AFTER the new worker's owned runtime_scope. No observation, prediction,
condition, physical getter, path step or FK is requested by this observer. The
original observer/planner return objects and exceptions are left unchanged.
This module does not install sys.setprofile and can accompany1105/1125.
"""
from contextlib import contextmanager, ExitStack
from enum import Enum
import math
from types import MethodType
from unittest.mock import patch
import numpy as np

from ..contracts import Pose, RobotState, ObjectState
from .interaction import PhaseTemplate

SCHEMA = 'r10_evaluation_existing_public_execution_trace_v1126'
MAX_PLAN_VALUES = 50000


def _storage(value):
    try:
        result = object.__getattribute__(value, '__dict__')
    except (AttributeError, TypeError):
        return {}
    return result if type(result) is dict else {}


def _callable_key(value):
    if type(value) is MethodType:
        return id(value.__func__), id(value.__self__)
    return (id(value),)


class PublicExecutionTrace:
    def __init__(self, *, max_frames=201, max_plans=24):
        if type(max_frames) is not int or not 1 <= max_frames <= 10000:
            raise ValueError('bounded positive public frame limit required')
        if type(max_plans) is not int or not 1 <= max_plans <= 1000:
            raise ValueError('bounded positive original plan limit required')
        self.max_frames, self.max_plans = max_frames, max_plans
        self.frames = []; self.plans = []
        self.frame_calls = self.frame_returns = self.frame_exceptions = self.frame_dropped = 0
        self.plan_calls = self.plan_returns = self.plan_exceptions = self.plan_dropped = 0
        self.geometry_calls = self.recording_errors = self.plan_value_truncations = 0
        self.errors = []; self.unknown_reasons = {}; self.hooks = []
        self.scope_exception_type = None
        self.restoration_error = None
        self._installed = set()

    def unknown(self, reason):
        self.unknown_reasons[reason] = self.unknown_reasons.get(reason, 0) + 1
        return dict(status='unknown', reason=reason)

    def error(self, stage, error):
        self.recording_errors += 1
        if len(self.errors) < 32:
            self.errors.append(dict(stage=stage, exception_type=type(error).__name__))

    def plain(self, value, *, depth=0, maximum_values=128):
        kind = type(value)
        if depth > 5:
            return self.unknown('plain_data_depth_limit')
        if value is None or kind in (bool, int, str):
            return value
        if kind is float:
            return value if math.isfinite(value) else self.unknown('nonfinite_plain_number')
        if kind is np.ndarray:
            if value.dtype.kind not in 'biuf' or value.size > maximum_values:
                return self.unknown('array_type_or_size_not_admitted')
            return value.tolist() if np.isfinite(value).all() else self.unknown('nonfinite_array')
        if kind.__module__ == 'numpy' and kind.__name__ in ('bool_', 'float64', 'float32', 'int64', 'int32'):
            return self.plain(value.item(), depth=depth+1, maximum_values=maximum_values)
        if kind in (list, tuple):
            if len(value) > maximum_values:
                return self.unknown('plain_sequence_limit')
            return [self.plain(item, depth=depth+1, maximum_values=maximum_values) for item in value]
        if kind is dict and len(value) <= 64 and all(type(key) is str for key in value):
            return {key:self.plain(item, depth=depth+1, maximum_values=maximum_values) for key,item in value.items()}
        if isinstance(value, Enum):
            data = _storage(value)
            return dict(enum_class=kind.__name__, enum_module=kind.__module__,
                        name=self.plain(data.get('_name_')), value=self.plain(data.get('_value_')))
        return self.unknown('not_plain_data_no_descriptor_read')

    def pose(self, value):
        if type(value) is not Pose:
            return self.unknown('not_original_Pose_DTO')
        data = _storage(value)
        return dict(position=self.plain(data['position']), rotation=self.plain(data['rotation']))

    def robot(self, value):
        if type(value) is not RobotState:
            return self.unknown('not_original_RobotState_DTO')
        data = _storage(value)
        return dict(timestamp=self.plain(data['timestamp']), pose=self.pose(data['pose']),
            joints=self.plain(data['joints']), joint_velocities=self.plain(data['joint_velocities']),
            gripper_open=self.plain(data['gripper_open']), contact=self.plain(data['contact']))

    def body(self, value):
        if type(value) is not ObjectState:
            return self.unknown('not_original_ObjectState_DTO')
        data = _storage(value)
        return dict(object_id=self.plain(data['object_id']), timestamp=self.plain(data['timestamp']),
            pose=self.pose(data['pose']), visible=self.plain(data['visible']), confidence=self.plain(data['confidence']))

    def selected(self, policy):
        data = _storage(policy)
        seen = set()
        for _ in range(5):
            if 'selected' in data:
                break
            source = data.get('source_policy')
            if source is None or id(source) in seen:
                return self.unknown('selected_public_intent_not_in_instance_storage')
            seen.add(id(source)); data = _storage(source)
        selected = data.get('selected')
        if type(selected) is not dict:
            return self.unknown('no_current_selected_public_intent')
        template = selected.get('template')
        cls = type(template)
        corrected = (cls.__name__ == 'CorrectedTemplate' and cls.__module__ in (
            'our_mani.multitask.grasp_frame_compensation',
            'our_mani.multitask.grasp_frame_compensation_native_compat') and PhaseTemplate in cls.__mro__)
        # The two original correction DTOs inherit and store the same anchor;
        # never invoke their at(), source-template delegation or descriptors.
        anchor = (_storage(template).get('anchor_id') if cls is PhaseTemplate or corrected else
                  self.unknown('not_original_PhaseTemplate_anchor_unknown'))
        if type(anchor) is not str and not (type(anchor) is dict and anchor.get('status') == 'unknown'):
            anchor = self.unknown('original_template_anchor_not_stored_string')
        result = {name:(self.plain(selected[name]) if name in selected else self.unknown('selected_field_not_stored:'+name)) for name in (
            'phase', 'reference_id', 'manipulated_ids', 'selected_at', 'opening', 'predicted_reference_time',
            'curve_index', 'curve_progress', 'author_substage')}
        result.update(pose=self.pose(selected.get('pose')), anchor_id=self.plain(anchor))
        result['present_pose'] = self.pose(selected['present_pose']) if 'present_pose' in selected else self.unknown('present_pose_not_stored')
        return result

    def frame(self, frontend, policy, robot, geometry_index):
        state = _storage(_storage(frontend).get('state'))
        bodies = state.get('bodies')
        if type(bodies) is dict and len(bodies) <= 64 and all(type(key) is str for key in bodies):
            observed = {key:self.body(value) for key,value in bodies.items()}
        else:
            observed = self.unknown('public_body_map_missing_or_outside_bound')
        return dict(frame_sequence=self.frame_calls, geometry_index=geometry_index,
            original_observe_returned=True, robot=self.robot(robot), bodies=observed,
            selected=self.selected(policy), evaluation_only=True)

    def observe(self, original, frontend, policy, geometry_index):
        def observe(observation, robot):
            self.frame_calls += 1
            try:
                result = original(observation, robot)
            except BaseException:
                self.frame_exceptions += 1
                raise
            self.frame_returns += 1
            if len(self.frames) >= self.max_frames:
                self.frame_dropped += 1
            else:
                try:
                    self.frames.append(self.frame(frontend, policy, robot, geometry_index))
                except BaseException as error:
                    self.error('public_frame_copy', error)
            return result
        return observe

    def plan(self, original, frontend, policy, geometry_index):
        def plan(*args, **kwargs):
            self.plan_calls += 1
            sequence = self.plan_calls
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self.plan_exceptions += 1
                try:
                    self._plan_record(None, args, kwargs, frontend, policy, geometry_index,
                                      sequence, exception_type=type(error).__name__)
                except BaseException as recording_error:
                    self.error('original_plan_exception_copy', recording_error)
                raise
            self.plan_returns += 1
            try:
                self._plan_record(result, args, kwargs, frontend, policy, geometry_index, sequence)
            except BaseException as error:
                self.error('original_plan_return_copy', error)
            return result
        return plan

    def _plan_record(self, result, args, kwargs, frontend, policy, geometry_index, sequence, exception_type=None):
        if len(self.plans) >= self.max_plans:
            self.plan_dropped += 1
            return
        state = _storage(_storage(frontend).get('state'))
        robot = state.get('robot')
        timestamp = _storage(robot).get('timestamp') if type(robot) is RobotState else self.unknown('plan_public_timestamp_unavailable')
        row = dict(plan_sequence=sequence, geometry_index=geometry_index,
            public_timestamp=self.plain(timestamp), original_args=self.plain(args), original_kwargs=self.plain(kwargs),
            exception_type=exception_type, original_returned=exception_type is None,
            selected=self.selected(policy), evaluation_only=True,
            capture_stage='original_plan_return_before_existing_waypoint_modification')
        if exception_type is not None:
            row['path_points'] = self.unknown('original_plan_raised_no_returned_path')
        else:
            storage = _storage(result)
            points = storage.get('_path_points')
            if type(points) is np.ndarray and points.size > MAX_PLAN_VALUES:
                self.plan_value_truncations += 1
                row['path_points'] = self.unknown('original_path_points_value_limit')
            elif type(points) is np.ndarray and points.dtype.kind in 'fiu':
                row['path_points'] = self.plain(points, maximum_values=MAX_PLAN_VALUES)
                row['path_points_shape'] = list(points.shape)
            else:
                row['path_points'] = self.unknown('original_path_points_not_stored_numeric_array')
            row['stored_num_joints'] = self.plain(storage['_num_joints']) if '_num_joints' in storage else self.unknown('num_joints_not_stored')
            row['stored_path_done'] = self.plain(storage['_path_done']) if '_path_done' in storage else self.unknown('path_done_not_stored')
        self.plans.append(row)

    def hook(self, stack, owned, attribute, replacement, label):
        data = _storage(owned)
        prior = (attribute in data, data.get(attribute))
        stack.enter_context(patch.object(owned, attribute, replacement))
        self.hooks.append((owned, attribute, prior, label))

    def install_components(self, stack, result, env, geometry_index):
        if type(result) not in (tuple, list) or len(result) != 5:
            self.unknown('geometry_factory_shape_unknown'); return
        frontend, policy, executor = result[2:]
        key = (id(frontend), id(executor))
        if key in self._installed:
            self.unknown('same_components_already_instrumented'); return
        self._installed.add(key)
        original = frontend.observe
        callback = executor.observe_public_frame
        if _callable_key(original) == _callable_key(callback):
            wrapped = self.observe(original, frontend, policy, geometry_index)
            self.hook(stack, frontend, 'observe', wrapped, 'frontend.observe')
            self.hook(stack, executor, 'observe_public_frame', wrapped, 'executor.observe_public_frame')
        else:
            self.unknown('different_observe_callback_not_rebound')
        service = _storage(env).get('map_service')
        if service is None:
            self.unknown('owned_map_service_not_in_instance_storage'); return
        self.hook(stack, service, 'plan_path', self.plan(service.plan_path, frontend, policy, geometry_index),
                  'owned_map_service.plan_path')

    def receipt(self):
        checks = []
        for owned, attribute, prior, label in self.hooks:
            data = _storage(owned)
            restored = (attribute in data) == prior[0] and (not prior[0] or data.get(attribute) is prior[1])
            checks.append(dict(attribute=label, restored=bool(restored)))
        restored = self.restoration_error is None and bool(checks) and all(item['restored'] for item in checks)
        return dict(schema=SCHEMA, frames=list(self.frames), plans=list(self.plans),
            max_frames=self.max_frames, max_plans=self.max_plans, max_plan_values=MAX_PLAN_VALUES,
            geometry_calls=self.geometry_calls, frame_calls=self.frame_calls, frame_returns=self.frame_returns,
            frame_exceptions=self.frame_exceptions, frame_dropped=self.frame_dropped,
            plan_calls=self.plan_calls, plan_returns=self.plan_returns, plan_exceptions=self.plan_exceptions,
            plan_dropped=self.plan_dropped, plan_value_truncations=self.plan_value_truncations,
            recording_errors=self.recording_errors, errors=list(self.errors), unknown_reasons=dict(self.unknown_reasons),
            restored=restored, restoration_checks=checks, restoration_error=self.restoration_error,
            scope_exception_type=self.scope_exception_type, sysprofile_installed=False,
            extra_env_observe_calls=0, extra_native_getter_or_condition_calls=0, extra_prediction_calls=0,
            extra_path_step_or_FK_calls=0, original_return_or_exception_rewritten=False,
            path_points_are_original_planner_output_not_executed_TCP_or_collision_certificate=True,
            data_is_evaluation_only_and_not_added_to_policy_inputs=True,
            absence_or_unsupported_values_are_unknown=True)


@contextmanager
def public_execution_trace_scope(runtime_module, events, *, max_frames=201, max_plans=24):
    """Enter after1122.runtime_scope; publish events['trace_receipt'] on exit."""
    if type(events) is not dict or 'trace_receipt' in events:
        raise ValueError('fresh separate evaluation-side trace event dictionary required')
    trace = PublicExecutionTrace(max_frames=max_frames, max_plans=max_plans)
    stack = ExitStack(); original = runtime_module._geometry
    original_error = None
    def geometry(*args, **kwargs):
        trace.geometry_calls += 1
        result = original(*args, **kwargs)
        try:
            env = args[1] if len(args) >= 2 else kwargs.get('env')
            trace.install_components(stack, result, env, trace.geometry_calls)
        except BaseException as error:
            trace.error('owned_component_installation', error)
        return result
    trace.hook(stack, runtime_module, '_geometry', geometry, 'runtime._geometry')
    try:
        yield trace
    except BaseException as error:
        original_error = error; trace.scope_exception_type = type(error).__name__
        raise
    finally:
        try:
            stack.close()
        except BaseException as error:
            trace.restoration_error = type(error).__name__; trace.error('restoration', error)
            if original_error is None:
                raise
        finally:
            receipt = trace.receipt()
            events['trace_receipt'] = receipt
            events['restored'] = receipt['restored']
            events['summary'] = {name:receipt[name] for name in (
                'geometry_calls','frame_calls','frame_dropped','plan_calls','plan_dropped',
                'plan_value_truncations','recording_errors','restored','unknown_reasons')}
