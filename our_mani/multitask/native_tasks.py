"""Native reset plans and privileged metrics for the prepared R10 tasks.

No policy builder imports this module. Functions never call expert waypoints,
setters, reset, step, rendering, or training. A ResetPlan is a description for
later integration, not an executor. MetricOnlySnapshot is deliberately not a
Mapping: it cannot be merged directly into an observation dictionary.
"""
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .specs import (NATIVE_COMMON_SOURCE_SHA256, NativeRole, ResetDispatch,
                    TaskSpec, get_task_spec)


class NativeAuditError(ValueError):
    pass


@dataclass(frozen=True)
class ResetStep:
    action: str
    args: tuple = ()
    kwargs: tuple = ()


@dataclass(frozen=True)
class ResetPlan:
    task_name: str
    variation: int
    dispatch: ResetDispatch
    steps: tuple[ResetStep, ...]
    metric_only: bool = True
    policy_use_allowed: bool = False


@dataclass(frozen=True)
class NativeResult:
    success: bool
    terminate: bool
    terminal_failure: bool


@dataclass
class NativeOutcomeLatch:
    """First terminal outcome wins, including buzz failure before later success."""
    outcome: NativeResult | None = None

    def update(self, result):
        if not isinstance(result, NativeResult):
            raise TypeError('NativeOutcomeLatch requires a NativeResult, not policy diagnostics')
        if self.outcome is None or not self.outcome.terminate:
            self.outcome = result
        return self.outcome


def native_result(task):
    value = task.success()
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise NativeAuditError('native task.success must return (success, terminate)')
    if not all(isinstance(x, (bool, np.bool_)) for x in value):
        raise NativeAuditError('native outcome fields must be booleans')
    success, terminate = map(bool, value)
    return NativeResult(success, terminate, terminate and not success)


def _spec(value):
    return get_task_spec(value) if isinstance(value, str) else value


def _variation(spec, variation):
    if type(variation) is not int or variation not in spec.supported_variations:
        raise NativeAuditError('unsupported variation for ' + spec.task_name + ': ' + str(variation))
    return variation


def official_reset_plan(task_spec, variation):
    """Author rlbench_env.py:372–425 task-specific ordering, without executing it."""
    spec = _spec(task_spec)
    _variation(spec, variation)
    # Author evaluation selects the task variation outside its per-case block.
    # A standalone R10 case must stage that same variation explicitly.
    steps = [ResetStep('set_variation', (variation,))]
    if spec.preload_before_reset_to_demo:
        steps.append(ResetStep('load_archived_motion_state_before_reset_to_demo'))
    steps += [ResetStep('reset_to_demo'), ResetStep('load_archived_motion_state')]
    if spec.reset_dispatch in (ResetDispatch.ROTATING, ResetDispatch.PEG_NO_RANDOM):
        steps.append(ResetStep('initialize_rotation_driver_from_archived_variation'))
    steps += [ResetStep('init_episode', (variation,), spec.init_episode_kwargs),
              ResetStep('pyrep_step_once'), ResetStep('get_observation'),
              ResetStep('disable_expert_plan')]
    return ResetPlan(spec.task_name, variation, spec.reset_dispatch, tuple(steps))


def _finite_vector(value, length, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise NativeAuditError(label + ' must be a finite ' + str(length) + '-vector')
    return array.tolist()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json_value(x) for x in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise NativeAuditError('nonfinite or nonserializable native metric: ' + type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _archived_variation(states, variation):
    try:
        if isinstance(states, Mapping):
            return states[variation] if variation in states else states[str(variation)]
        return states[variation]
    except (KeyError, IndexError, TypeError):
        raise NativeAuditError('archive lacks requested native variation') from None


def validate_archived_motion_state(task_spec, variation, states):
    """Validate reset-only driver schema; return no policy feature vectors."""
    spec = _spec(task_spec)
    _variation(spec, variation)
    selected = _archived_variation(states, variation)
    if spec.reset_dispatch in (ResetDispatch.ROTATING, ResetDispatch.PEG_NO_RANDOM):
        if not isinstance(selected, Mapping) or 'yaw_speed' not in selected:
            raise NativeAuditError('rotation archive requires yaw_speed')
        yaw = np.asarray(selected['yaw_speed'], dtype=float)
        if yaw.size != 1 or not np.isfinite(yaw).all():
            raise NativeAuditError('native yaw_speed must be a finite scalar or one-element list')
        drivers = 1
    else:
        if spec.family == 'moving_basketball_in_hoop':
            if not isinstance(selected, (list, tuple)) or len(selected) != 2:
                raise NativeAuditError('basketball archive must contain separate ball and hoop drivers')
            motion = selected
        elif spec.family == 'push_moving_button':
            if not isinstance(selected, (list, tuple)) or len(selected) != 1:
                raise NativeAuditError('button archive must contain its one-driver list')
            motion = selected
        else:
            motion = (selected,)
        for driver in motion:
            if not isinstance(driver, Mapping):
                raise NativeAuditError('translation driver must be a mapping')
            for field in ('x', 'v', 'a'):
                if field not in driver:
                    raise NativeAuditError('translation archive lacks ' + field)
                _finite_vector(driver[field], 3, 'archive ' + field)
        drivers = len(motion)
    serializable = _json_value(selected)
    return dict(task_name=spec.task_name, variation=variation, driver_count=drivers,
                archived_variation_sha256=_digest(serializable),
                metric_only=True, policy_use_allowed=False)


def validate_reset_metadata(task_spec, episode, *, archived_variation=None,
                            archived_states=None, official=True):
    spec = _spec(task_spec)
    if episode.get('task') != spec.task_name:
        raise NativeAuditError('episode task does not match TaskSpec')
    if not isinstance(episode.get('raw_episode_id'), str) or not episode['raw_episode_id']:
        raise NativeAuditError('missing raw episode identity')
    variation = _variation(spec, episode.get('variation'))
    if archived_variation is not None:
        if type(archived_variation) is not int or archived_variation != variation:
            raise NativeAuditError('variation_number.pkl does not match manifest')
    if official and (type(episode.get('official_episode_index')) is not int
                     or episode['official_episode_index'] < 0):
        raise NativeAuditError('missing official episode index')
    result = dict(task_name=spec.task_name, raw_episode_id=episode['raw_episode_id'],
                  variation=variation, metric_only=True, policy_use_allowed=False)
    if archived_states is not None:
        result['archive'] = validate_archived_motion_state(spec, variation, archived_states)
    return result


def verify_native_sources(native_repo, task_spec):
    spec = _spec(task_spec)
    expected = {spec.native_source_relative_path: spec.native_source_sha256,
                **NATIVE_COMMON_SOURCE_SHA256}
    for name, digest in expected.items():
        path = Path(native_repo) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise NativeAuditError('native source changed: ' + name)
    return expected


def resolve_native_roles(task, task_spec, *, scene_object_resolver=None):
    """Privileged handles for reset/metrics/map partitioning, never perception."""
    spec = _spec(task_spec)
    result = {}
    for role in spec.roles:
        current = task
        try:
            if not role.attribute_path:
                raise AttributeError('role has no task attribute')
            for component in role.attribute_path:
                current = current[component] if isinstance(component, int) else getattr(current, component)
        except (AttributeError, IndexError, KeyError, TypeError):
            if scene_object_resolver is None or role.scene_name is None:
                if role.required:
                    raise NativeAuditError('native role unavailable: ' + role.name) from None
                continue
            current = scene_object_resolver(role.scene_name)
        if current is None:
            raise NativeAuditError('native role resolved to None: ' + role.name)
        result[role.name] = current
    return result


class MetricOnlySnapshot:
    """Explicit report export only: no Mapping/iteration/observation adapter."""
    __slots__ = ('__report',)

    def __init__(self, report):
        self.__report = deepcopy(report)

    def to_report(self):
        return deepcopy(self.__report)


def _read_body(obj, role, *, strict):
    pose = _finite_vector(obj.get_pose(), 7, role.name + ' pose xyzw')
    if np.linalg.norm(pose[3:]) < 1e-8:
        raise NativeAuditError('native pose has a zero quaternion: ' + role.name)
    record = dict(pose_xyzw=pose,
                  native_name=obj.get_name() if hasattr(obj, 'get_name') else None,
                  kind=role.kind)
    if hasattr(obj, 'get_handle'):
        record['native_handle_metric_only'] = int(obj.get_handle())
    if hasattr(obj, 'get_velocity'):
        linear, angular = obj.get_velocity()
        record['linear_velocity_m_s'] = _finite_vector(linear, 3, role.name + ' linear velocity')
        record['angular_velocity_rad_s'] = _finite_vector(angular, 3, role.name + ' angular velocity')
    else:
        record['linear_velocity_m_s'] = record['angular_velocity_rad_s'] = None
        if strict and role.kind == 'body':
            raise NativeAuditError('body velocity unavailable: ' + role.name)
    if role.kind == 'joint':
        position = float(obj.get_joint_position())
        if not np.isfinite(position):
            raise NativeAuditError('nonfinite native joint position')
        record['joint_position_native_units'] = position
        record['joint_velocity_native_units_s'] = (
            float(obj.get_joint_velocity()) if hasattr(obj, 'get_joint_velocity') else None)
        if record['joint_velocity_native_units_s'] is not None and not np.isfinite(record['joint_velocity_native_units_s']):
            raise NativeAuditError('nonfinite native joint velocity')
    return record


def metric_initial_snapshot(task, task_spec, *, scene_object_resolver=None,
                            physics_time=None, include_dynamic_descendants=True,
                            strict=True):
    """Read complete requested audit fields after reset; never put them in obs.

    physics_time may be a read-only callable (clock invariance is then checked)
    or a caller-provided scalar. No PyRep or GPU imports are required here.
    Engine contact caches/internal numerical solver state are not claimed equal.
    """
    spec = _spec(task_spec)
    before_t, before_step = float(task.t), int(task.step_id)
    before_physics = physics_time() if callable(physics_time) else physics_time
    objects = resolve_native_roles(task, spec, scene_object_resolver=scene_object_resolver)
    bodies = {role.name: _read_body(objects[role.name], role, strict=strict)
              for role in spec.roles if role.name in objects}
    descendants, skipped_expert_dummies = {}, []
    seen = {record.get('native_handle_metric_only') for record in bodies.values()
            if record.get('native_handle_metric_only') is not None}
    if include_dynamic_descendants:
        for root in spec.dynamic_roots:
            obj = objects[root]
            if not hasattr(obj, 'get_objects_in_tree'):
                if strict:
                    raise NativeAuditError('dynamic root subtree API unavailable: ' + root)
                continue
            for child in obj.get_objects_in_tree():
                name = child.get_name()
                if 'waypoint' in name.lower() or name.lower().startswith('expertwp'):
                    skipped_expert_dummies.append(name)
                    continue
                handle = int(child.get_handle())
                if handle in seen:
                    continue
                seen.add(handle)
                # Some native trees contain markers/sensors as well as shapes;
                # absence of their velocity is recorded, never replaced by zero.
                role = NativeRole(root + '/' + name, kind='subtree_object')
                descendants[root + '/' + name] = _read_body(child, role, strict=False)
    if not task.target_state_list:
        raise NativeAuditError('snapshot requires an initialized active native motion schedule')
    schedules = _json_value(task.target_state_list)
    fields = {name: _json_value(getattr(task, name)) for name in
              ('stage', 'var_index', *spec.metric_state_fields) if hasattr(task, name)}
    result = native_result(task)
    report = dict(
        version='r10_metric_only_native_initial_snapshot_v1', task_name=spec.task_name,
        metric_only=True, policy_use_allowed=False,
        native_task_source_sha256=spec.native_source_sha256,
        bodies=bodies, dynamic_descendants=descendants,
        dynamic_roots=list(spec.dynamic_roots),
        skipped_expert_waypoint_dummies=sorted(set(skipped_expert_dummies)),
        task_time_seconds=before_t, task_step_id=before_step,
        physics_time_seconds=None if before_physics is None else float(before_physics),
        active_motion_schedule=schedules[-1], motion_schedule_history=schedules,
        motion_schedule_sha256=_digest(schedules), task_internal_fields=fields,
        native_result=dict(success=result.success, terminate=result.terminate,
                           terminal_failure=result.terminal_failure),
        full_native_physics_state_claimed=False,
        unrecorded_state='native contact/solver caches; opaque engine state; no bit-identical full-state claim',
    )
    variation = next((getattr(task, name) for name in ('var_index', '_index', 'variation_index', '_variation_index')
                      if hasattr(task, name)), None)
    if variation is not None and hasattr(task, 'var2target_state_list'):
        report['restored_current_variation_state'] = _json_value(
            _archived_variation(task.var2target_state_list, int(variation)))
    native_robot = getattr(task, 'robot', None)
    if native_robot is not None:
        arm, gripper = native_robot.arm, native_robot.gripper
        report['robot'] = dict(
            tip_pose_xyzw=_finite_vector(arm.get_tip().get_pose(), 7, 'robot tip'),
            joints=_json_value(arm.get_joint_positions()),
            joint_velocities=_json_value(arm.get_joint_velocities()),
            gripper_open_amount=_json_value(gripper.get_open_amount()),
            grasped_native_names_metric_only=[obj.get_name() for obj in gripper.get_grasped_objects()],
        )
    after_physics = physics_time() if callable(physics_time) else physics_time
    if float(task.t) != before_t or int(task.step_id) != before_step:
        raise NativeAuditError('native audit read advanced the task clock')
    if before_physics is not None and (not np.isfinite([before_physics, after_physics]).all()
                                       or abs(float(after_physics) - float(before_physics)) > 1e-9):
        raise NativeAuditError('native audit read advanced the physics clock')
    return MetricOnlySnapshot(_json_value(report))
