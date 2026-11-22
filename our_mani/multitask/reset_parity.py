"""Parent-only comparison of complete *recorded* R10 reset conditions.

This module consumes numeric audit receipts, never policy observations or
native handles. It does not assert equality of opaque contact/solver state.
New tolerances remain pending until TRAIN repeat-reset evidence is frozen;
no TEST observations or outcomes may fit or relax them.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .specs import TASK_SPECS, NATIVE_COMMON_SOURCE_SHA256, get_task_spec

VERSION = 'r10_recorded_initial_conditions_v1'
PROFILE_VERSION = 'r10_recorded_initial_condition_tolerances_v1'
METHODS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')
INHERITED = dict(robot_position_m=.001, robot_orientation_rad=.002, robot_joints_rad=.0001,
    body_position_component_m=.0001, fixture_pose_component=.0001,
    camera_calibration_component=.00001, clock_seconds=.00002)
NEW_FIELDS = ('body_orientation_rad', 'body_linear_velocity_m_s', 'body_angular_velocity_rad_s',
    'robot_joint_velocity_rad_s', 'gripper_open_amount', 'native_joint_position_units',
    'native_joint_velocity_units_s', 'schedule_numeric_absolute', 'internal_numeric_absolute')
PERMISSIONS = ('current_instance_masks', 'information_setting', 'map_access',
    'shared_semantic_instance_binding', 'semantic_binding_available_to_baseline', 'shared_TRAIN_CAD_geometry')


class RecordedConditionError(Exception):
    """Malformed metric evidence is infrastructure, not a policy failure."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pending_tolerance_profile(legacy):
    mapping = dict(robot_position_m='robot_position_m', robot_orientation_rad='robot_orientation_rad',
        robot_joints_rad='robot_joints_rad', body_position_component_m='target_position_m',
        fixture_pose_component='fixture_pose_component', camera_calibration_component='camera_calibration_component')
    if any(legacy.get(old) != INHERITED[new] for new, old in mapping.items()):
        raise RecordedConditionError('inherited R09 tolerances must not be changed automatically')
    return dict(version=PROFILE_VERSION, status='TRAIN_repeat_validation_pending',
        calibration_split='train', TEST_used=False, thresholds={**INHERITED, **dict.fromkeys(NEW_FIELDS)},
        inherited_thresholds=dict(INHERITED), new_thresholds_require_TRAIN_fit_and_independent_repeat_validation=True,
        threshold_relaxation_automatic=False, TRAIN_evidence=None)


def profile_problems(profile, *, evidence_root=None):
    problems = []
    if (not isinstance(profile, Mapping) or profile.get('version') != PROFILE_VERSION
            or profile.get('calibration_split') != 'train' or profile.get('TEST_used') is not False
            or profile.get('threshold_relaxation_automatic') is not False):
        return ['invalid_tolerance_profile']
    thresholds = profile.get('thresholds', {})
    if not isinstance(thresholds, Mapping):
        return ['tolerance_thresholds_not_mapping']
    if profile.get('inherited_thresholds') != INHERITED or any(thresholds.get(k) != v for k, v in INHERITED.items()):
        problems.append('inherited_tolerances_changed')
    if set(thresholds) != set(INHERITED) | set(NEW_FIELDS):
        problems.append('tolerance_fields_incomplete_or_unknown')
    for key, value in thresholds.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value < 0):
            problems.append('invalid_tolerance:' + key)
    if profile.get('status') != 'frozen_after_TRAIN_repeat_validation' or any(thresholds.get(k) is None for k in NEW_FIELDS):
        problems.append('new_tolerances_not_TRAIN_validated_and_frozen')
    evidence = profile.get('TRAIN_evidence')
    if evidence_root is None or not isinstance(evidence, Mapping):
        problems.append('TRAIN_repeat_evidence_unverified')
    else:
        try:
            root = Path(evidence_root).resolve();path = (root / evidence['path']).resolve()
            if Path(evidence['path']).is_absolute() or not path.is_relative_to(root):
                raise RecordedConditionError('TRAIN repeat evidence must stay under its artifact root')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != evidence['sha256']:
                raise RecordedConditionError('TRAIN repeat evidence SHA differs')
            report = json.loads(data)
            if (report.get('split') != 'train' or report.get('TEST_used') is not False
                    or report.get('thresholds_sha256') != digest(thresholds)
                    or report.get('fit_repeat_data_passed') is not True
                    or report.get('independent_repeat_validation_passed') is not True
                    or report.get('native_repeat_execution_complete') is not True
                    or report.get('synthetic_mock_data_used') is not False
                    or report.get('threshold_relaxation_automatic') is not False
                    or set(report.get('per_task', {})) != set(TASK_SPECS)):
                raise RecordedConditionError('TRAIN fit/validation coverage or selected thresholds differ')
            for task, row in report['per_task'].items():
                if (row.get('fit_pairs', 0) < 1 or row.get('independent_validation_pairs', 0) < 1
                        or row.get('passed') is not True):
                    raise RecordedConditionError('TRAIN repeat acceptance missing: ' + task)
        except (OSError, KeyError, TypeError, ValueError, RecordedConditionError) as exc:
            problems.append('TRAIN_repeat_evidence:' + str(exc))
    return problems


def capture_recorded_initial_conditions(env, observation=None, robot=None):
    """Capture once at reset tick0 for the parent evaluator/collector only."""
    from .simulation import public_initial_conditions
    if not env._aligned or env.ledger.ticks != 0 or env.time != 0:
        raise RecordedConditionError('capture initial metric conditions before the first counted action')
    if (observation is None) != (robot is None):
        raise RecordedConditionError('provide both current observation and robot or neither')
    before = env.physics_time(), float(env.task.t), int(env.task.step_id)
    initial = env.reset_receipt['native_initial_snapshot']
    if (abs(float(initial['physics_time_seconds'])-before[0]) > INHERITED['clock_seconds']
            or abs(float(initial['task_time_seconds'])-before[1]) > INHERITED['clock_seconds']
            or initial['task_step_id'] != before[2]):
        raise RecordedConditionError('reset receipt is stale or physics/task advanced before initial capture')
    if observation is None:
        observation, robot = env.observe()
    if robot.timestamp != 0 or observation['timestamp'] != 0:
        raise RecordedConditionError('initial observation must have reset timestamp zero')
    cyclic, intervals = env.arm.get_joint_intervals()
    capability = dict(joint_cyclic=list(map(bool, cyclic)), joint_intervals_min_range=np.asarray(intervals).tolist(),
        upper_velocity_limits_rad_s=np.asarray(env.arm.get_joint_upper_velocity_limits()).tolist(),
        simulation_timestep_seconds=float(env.pyrep.get_simulation_timestep()))
    common = {key: deepcopy(env.protocol[key]) for key in PERMISSIONS if key in env.protocol}
    common.update(dt=env.protocol['dt'], simulation_seconds=env.protocol['simulation_seconds'],
        gripper_velocity=env.protocol['gripper_velocity'],
        gripper_semantics='single_grasp_attempt_per_close_transition_incremental_actuation')
    value = dict(version=VERSION, metric_only=True, policy_use_allowed=False,
        reset_receipt=deepcopy(env.reset_receipt),
        public_initial_conditions=deepcopy(public_initial_conditions(env, observation, robot)),
        native_robot_capabilities=capability, common_external_protocol=common,
        full_native_physics_state_claimed=False,
        unrecorded_state='opaque native contact/solver caches and engine internals; no bit-identical full-state claim')
    if (env.physics_time(), float(env.task.t), int(env.task.step_id)) != before:
        raise RecordedConditionError('initial metric capture advanced physics/task clock')
    return value


def _record(value):
    if not isinstance(value, Mapping):
        raise RecordedConditionError('initial condition record must be a mapping')
    return value.get('recorded_initial_conditions', value)


class _Compare:
    def __init__(self, profile):
        self.thresholds = profile.get('thresholds', {}) if isinstance(profile, Mapping) else {}
        if not isinstance(self.thresholds, Mapping): self.thresholds = {}
        self.problems, self.deltas, self.pending = [], {}, set()

    def need(self, value, key, path):
        if not isinstance(value, Mapping) or key not in value:
            raise RecordedConditionError('missing_field:' + path + '.' + key)
        return value[key]

    def exact(self, a, b, path):
        if a != b:
            self.problems.append('mismatch:' + path)

    def numeric(self, a, b, path, threshold, *, shape=None, norm=False):
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if aa.shape != bb.shape or (shape is not None and aa.shape != shape) or not np.isfinite(aa).all() or not np.isfinite(bb).all():
            raise RecordedConditionError('invalid_numeric:' + path)
        delta = float(np.linalg.norm(aa-bb) if norm else np.max(np.abs(aa-bb), initial=0))
        self.deltas[path] = delta
        bound = self.thresholds.get(threshold)
        if bound is None:
            self.pending.add(threshold);bound = 0.
        if delta > bound + 1e-15:
            self.problems.append('outside_tolerance:' + path + ':' + threshold)

    def pose(self, a, b, path, position_key, *, angle_key=None, quaternion_component_key=None, norm=False):
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if aa.shape != (7,) or bb.shape != (7,) or not np.isfinite(aa).all() or not np.isfinite(bb).all():
            raise RecordedConditionError('invalid_pose:' + path)
        na, nb = np.linalg.norm(aa[3:]), np.linalg.norm(bb[3:])
        if abs(na-1) > 1e-4 or abs(nb-1) > 1e-4:
            raise RecordedConditionError('nonunit_quaternion:' + path)
        self.numeric(aa[:3], bb[:3], path + '.position', position_key, norm=norm)
        qa, qb = aa[3:]/na, bb[3:]/nb
        if np.dot(qa, qb) < 0: qb = -qb
        angle = float(4*np.arctan2(np.linalg.norm(qa-qb), np.linalg.norm(qa+qb)))
        if angle_key:
            self.numeric(angle, 0., path + '.orientation_rad', angle_key)
        else:
            self.deltas[path + '.orientation_rad'] = angle
        if quaternion_component_key:
            self.numeric(qa, qb, path + '.quaternion_component', quaternion_component_key, shape=(4,))

    def tree(self, a, b, path, numeric_key):
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                raise RecordedConditionError('field_set_mismatch:' + path)
            for key in sorted(a): self.tree(a[key], b[key], path + '.' + key,
                'clock_seconds' if key == 't0' else numeric_key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b): raise RecordedConditionError('sequence_length_mismatch:' + path)
            for index, (x, y) in enumerate(zip(a, b)): self.tree(x, y, path + '[' + str(index) + ']', numeric_key)
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            if isinstance(a, int) and isinstance(b, int): self.exact(a, b, path)
            else: self.numeric(a, b, path, numeric_key)
        else:
            self.exact(a, b, path)


def _compare_record(a, b, c):
    for value, label in ((a, 'a'), (b, 'b')):
        if (c.need(value, 'version', label) != VERSION or c.need(value, 'metric_only', label) is not True
                or c.need(value, 'policy_use_allowed', label) is not False
                or c.need(value, 'full_native_physics_state_claimed', label) is not False):
            raise RecordedConditionError('invalid_parent_metric_boundary:' + label)
    ra, rb = [c.need(value, 'reset_receipt', label) for value, label in ((a, 'a'), (b, 'b'))]
    for key in ('task', 'raw_episode_id', 'variation', 'reset_kind', 'metric_only', 'policy_use_allowed',
                'native_source_sha256', 'expert_actions_used', 'future_observations_used', 'original_source_replayed'):
        c.exact(c.need(ra, key, 'reset'), c.need(rb, key, 'reset'), 'reset.' + key)
    spec = get_task_spec(ra['task'])
    expected_sources = {spec.native_source_relative_path: spec.native_source_sha256, **NATIVE_COMMON_SOURCE_SHA256}
    for reset in (ra, rb):
        if (reset['metric_only'] is not True or reset['policy_use_allowed'] is not False
                or reset['expert_actions_used'] is not False or reset['future_observations_used'] is not False
                or type(reset['original_source_replayed']) is not bool
                or reset['native_source_sha256'] != expected_sources):
            raise RecordedConditionError('reset_source_or_metric_boundary_invalid')
    if ra['variation'] not in spec.supported_variations:
        raise RecordedConditionError('unsupported_reset_variation')
    if ra['original_source_replayed']:
        for key in ('official_episode_index', 'target_state_sha256', 'verified_source_inputs', 'reset_sequence'):
            c.exact(c.need(ra, key, 'reset'), c.need(rb, key, 'reset'), 'reset.' + key)
        verified = ra['verified_source_inputs']
        from .simulation import OFFICIAL_SOURCE_FILES
        required_sources = c.need(verified, 'source_required_sha256', 'source')
        if (set(required_sources) != set(OFFICIAL_SOURCE_FILES)
                or c.need(verified, 'task_source_sha256', 'source') != spec.native_source_sha256
                or c.need(verified, 'task_scene_sha256', 'source') is None
                or required_sources['target_state.pkl'] != ra['target_state_sha256']
                or any(not isinstance(value, str) or len(value) != 64 or
                       any(char not in '0123456789abcdef' for char in value)
                       for value in [*required_sources.values(), verified['task_scene_sha256']])):
            raise RecordedConditionError('official_source_hash_evidence_incomplete')
    else:
        for key in ('source_split', 'source_hdf5_sha256', 'seed', 'seed_identity', 'replica',
                    'source_motion_or_expert_trajectory_replayed'):
            c.exact(c.need(ra, key, 'reset'), c.need(rb, key, 'reset'), 'reset.' + key)
        if ra['source_split'] not in ('train', 'validation'):
            raise RecordedConditionError('synthetic_reset_split_must_be_development')
    na, nb = [c.need(reset, 'native_initial_snapshot', 'reset') for reset in (ra, rb)]
    for value in (na, nb):
        if (c.need(value, 'version', 'native') != 'r10_metric_only_native_initial_snapshot_v1'
                or c.need(value, 'metric_only', 'native') is not True
                or c.need(value, 'policy_use_allowed', 'native') is not False
                or c.need(value, 'task_name', 'native') != spec.task_name
                or c.need(value, 'native_task_source_sha256', 'native') != spec.native_source_sha256
                or c.need(value, 'full_native_physics_state_claimed', 'native') is not False):
            raise RecordedConditionError('native_snapshot_boundary_or_source_differs')
        schedules = c.need(value, 'motion_schedule_history', 'native')
        if (not isinstance(schedules, list) or not schedules
                or c.need(value, 'motion_schedule_sha256', 'native') != digest(schedules)
                or c.need(value, 'active_motion_schedule', 'native') != schedules[-1]):
            raise RecordedConditionError('motion_schedule_hash_or_active_entry_inconsistent')
        def driver_clocks(node):
            if isinstance(node, Mapping):
                if any(key in node for key in ('x', 'v', 'a', 'yaw_speed')):
                    t0 = c.need(node, 't0', 'motion_driver')
                    if not isinstance(t0, (int, float)) or not np.isfinite(t0):
                        raise RecordedConditionError('native_driver_t0_unavailable')
                for child in node.values(): driver_clocks(child)
            elif isinstance(node, list):
                for child in node: driver_clocks(child)
        driver_clocks(schedules)
        if type(c.need(value, 'task_step_id', 'native')) is not int or value['task_step_id'] < 0:
            raise RecordedConditionError('native_step_id_unavailable_or_invalid')
        if not isinstance(c.need(c.need(value, 'task_internal_fields', 'native'), 'stage', 'native.internal'), str):
            raise RecordedConditionError('native_stage_unavailable')
    for key in ('task_step_id', 'native_result', 'dynamic_roots'):
        c.exact(c.need(na, key, 'native'), c.need(nb, key, 'native'), 'native.' + key)
    if set(na['dynamic_roots']) != set(spec.dynamic_roots):
        raise RecordedConditionError('native_dynamic_root_coverage_differs')
    for key in ('task_time_seconds', 'physics_time_seconds'):
        c.numeric(c.need(na, key, 'native'), c.need(nb, key, 'native'), 'native.' + key, 'clock_seconds', shape=())
    for key in ('motion_schedule_history', 'active_motion_schedule', 'restored_current_variation_state'):
        c.tree(c.need(na, key, 'native'), c.need(nb, key, 'native'), 'native.' + key, 'schedule_numeric_absolute')
    c.tree(c.need(na, 'task_internal_fields', 'native'), c.need(nb, 'task_internal_fields', 'native'),
           'native.task_internal_fields', 'internal_numeric_absolute')
    for collection in ('bodies', 'dynamic_descendants'):
        ba, bb = c.need(na, collection, 'native'), c.need(nb, collection, 'native')
        if not isinstance(ba, Mapping) or not isinstance(bb, Mapping) or set(ba) != set(bb):
            raise RecordedConditionError('native_body_coverage_mismatch:' + collection)
        if collection == 'bodies' and set(ba) != {role.name for role in spec.roles if role.required}:
            raise RecordedConditionError('declared_native_role_coverage_incomplete')
        for identity in sorted(ba):
            x, y = ba[identity], bb[identity];path = 'native.' + collection + '.' + identity
            for key in ('kind', 'native_name'):
                c.exact(c.need(x, key, path), c.need(y, key, path), path + '.' + key)
            expected_kind = ({role.name: role.kind for role in spec.roles}[identity]
                             if collection == 'bodies' else 'subtree_object')
            if x['kind'] != expected_kind or y['kind'] != expected_kind:
                raise RecordedConditionError('native_role_kind_differs:' + path)
            c.pose(c.need(x, 'pose_xyzw', path), c.need(y, 'pose_xyzw', path), path,
                   'body_position_component_m', angle_key='body_orientation_rad')
            for key, threshold in (('linear_velocity_m_s', 'body_linear_velocity_m_s'),
                                   ('angular_velocity_rad_s', 'body_angular_velocity_rad_s')):
                vx, vy = c.need(x, key, path), c.need(y, key, path)
                if vx is None or vy is None:
                    if x['kind'] == 'body' or vx is not None or vy is not None:
                        raise RecordedConditionError('declared_body_velocity_unavailable:' + path + '.' + key)
                else: c.numeric(vx, vy, path + '.' + key, threshold, shape=(3,))
            if x['kind'] == 'joint':
                for key, threshold in (('joint_position_native_units', 'native_joint_position_units'),
                                       ('joint_velocity_native_units_s', 'native_joint_velocity_units_s')):
                    c.numeric(c.need(x, key, path), c.need(y, key, path), path + '.' + key, threshold, shape=())
    xa, xb = c.need(na, 'robot', 'native'), c.need(nb, 'robot', 'native')
    c.pose(c.need(xa, 'tip_pose_xyzw', 'robot'), c.need(xb, 'tip_pose_xyzw', 'robot'), 'native.robot',
           'robot_position_m', angle_key='robot_orientation_rad', norm=True)
    for key, threshold, shape in (('joints', 'robot_joints_rad', (7,)),
            ('joint_velocities', 'robot_joint_velocity_rad_s', (7,)), ('gripper_open_amount', 'gripper_open_amount', (2,))):
        c.numeric(c.need(xa, key, 'robot'), c.need(xb, key, 'robot'), 'native.robot.' + key, threshold, shape=shape)
    c.exact(sorted(c.need(xa, 'grasped_native_names_metric_only', 'robot')),
            sorted(c.need(xb, 'grasped_native_names_metric_only', 'robot')), 'native.robot.grasped_object_names')
    pa, pb = [c.need(value, 'public_initial_conditions', label) for value, label in ((a, 'a'), (b, 'b'))]
    for key in ('information_setting', 'sensor_protocol'):
        c.exact(c.need(pa, key, 'public'), c.need(pb, key, 'public'), 'public.' + key)
    from ..sensor_alignment.author_rgbd import PROTOCOL as SENSOR_PROTOCOL, RGB_SPEC
    for public in (pa, pb):
        sensor = c.need(public, 'sensor_protocol', 'public')
        if (not isinstance(sensor, Mapping) or c.need(sensor, 'version', 'sensor') != SENSOR_PROTOCOL
                or c.need(sensor, 'camera_render_modes', 'sensor') != {name: 'OPENGL' for name in ('front', '8', '16', '36')}
                or c.need(sensor, 'rgb_processing', 'sensor') != RGB_SPEC):
            raise RecordedConditionError('sensor_protocol_incomplete_or_not_author_RGBD')
    c.numeric(c.need(pa, 'robot_joints', 'public'), c.need(pb, 'robot_joints', 'public'),
              'public.robot_joints', 'robot_joints_rad', shape=(7,))
    c.pose(c.need(pa, 'robot_pose', 'public'), c.need(pb, 'robot_pose', 'public'), 'public.robot',
           'robot_position_m', angle_key='robot_orientation_rad', norm=True)
    cameras_a, cameras_b = c.need(pa, 'cameras', 'public'), c.need(pb, 'cameras', 'public')
    if set(cameras_a) != {'front', '8', '16', '36'} or set(cameras_a) != set(cameras_b):
        raise RecordedConditionError('camera_coverage_incomplete')
    for name, x in cameras_a.items():
        y = cameras_b[name];path = 'public.camera.' + name
        for key in ('render_mode', 'raw_rgb_255_processing'):
            c.exact(c.need(x, key, path), c.need(y, key, path), path + '.' + key)
        if (x['render_mode'] != 'OPENGL' or y['render_mode'] != 'OPENGL'
                or x['raw_rgb_255_processing'] != RGB_SPEC['observation/raw_rgb_255'][name]
                or y['raw_rgb_255_processing'] != RGB_SPEC['observation/raw_rgb_255'][name]):
            raise RecordedConditionError('RGBD_render_mode_or_processing_is_not_author_protocol')
        c.numeric(c.need(x, 'intrinsics', path), c.need(y, 'intrinsics', path), path + '.K',
                  'camera_calibration_component', shape=(3, 3))
        c.pose(c.need(x, 'camera_to_world', path), c.need(y, 'camera_to_world', path), path,
               'camera_calibration_component', quaternion_component_key='camera_calibration_component')
    maps = []
    for public in (pa, pb):
        values = c.need(public, 'static_map', 'public')
        if not isinstance(values, list) or len({v['name'] for v in values}) != len(values):
            raise RecordedConditionError('static_map_schema_or_duplicate_names')
        maps.append({value['name']: value for value in values})
    if maps[0].keys() != maps[1].keys(): raise RecordedConditionError('static_map_shape_set_differs')
    for name, x in maps[0].items():
        y = maps[1][name];path = 'public.static.' + name
        c.exact(c.need(x, 'mesh_sha256', path), c.need(y, 'mesh_sha256', path), path + '.mesh_sha256')
        c.numeric(c.need(x, 'bounding_box', path), c.need(y, 'bounding_box', path), path + '.bounding_box',
                  'fixture_pose_component', shape=(6,))
        c.pose(c.need(x, 'pose', path), c.need(y, 'pose', path), path,
               'fixture_pose_component', quaternion_component_key='fixture_pose_component')
    for key in ('native_robot_capabilities', 'common_external_protocol'):
        c.exact(c.need(a, key, 'record'), c.need(b, key, 'record'), key)
    for value in (a, b):
        cap = value['native_robot_capabilities']
        cyclic = c.need(cap, 'joint_cyclic', 'capability')
        if not isinstance(cyclic, list) or len(cyclic) != 7 or any(type(x) is not bool for x in cyclic):
            raise RecordedConditionError('native_joint_cyclic_flags_unavailable')
        for key, shape in (('joint_intervals_min_range', (7, 2)), ('upper_velocity_limits_rad_s', (7,))):
            values = np.asarray(c.need(cap, key, 'capability'), dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise RecordedConditionError('native_robot_capability_unavailable:' + key)
        if np.any(np.asarray(cap['upper_velocity_limits_rad_s']) <= 0):
            raise RecordedConditionError('native_velocity_limits_nonpositive')
        dt = c.need(cap, 'simulation_timestep_seconds', 'capability')
        if not isinstance(dt, (int, float)) or not np.isfinite(dt) or abs(dt-.05) > 1e-8:
            raise RecordedConditionError('native_timestep_differs_from_50ms')
        common = value['common_external_protocol']
        for key, expected in (('dt', .05), ('simulation_seconds', 10.), ('gripper_velocity', .2),
                              ('current_instance_masks', True),
                              ('gripper_semantics', 'single_grasp_attempt_per_close_transition_incremental_actuation')):
            if c.need(common, key, 'external') != expected:
                raise RecordedConditionError('common_external_protocol_invalid:' + key)
        if c.need(common, 'information_setting', 'external') != value['public_initial_conditions']['information_setting']:
            raise RecordedConditionError('information_permission_receipts_conflict')


def compare_initial_conditions(a, b, profile, *, evidence_root=None, require_frozen=True):
    """Fail closed on missing fields; compare only initial parent metric data."""
    c = _Compare(profile);profile_errors = profile_problems(profile, evidence_root=evidence_root)
    try:
        _compare_record(_record(a), _record(b), c)
    except (KeyError, TypeError, ValueError, RecordedConditionError) as exc:
        c.problems.append('malformed_record:' + str(exc))
    matched = not c.problems
    eligible = not profile_errors
    return dict(version='r10_recorded_condition_comparison_v1', accepted=matched and (eligible or not require_frozen),
        recorded_conditions_match=matched, formal_acceptance_eligible=eligible,
        problems=c.problems + (profile_errors if require_frozen else []), tolerance_profile_problems=profile_errors,
        measured_deltas=c.deltas, provisional_exact_only_fields=sorted(c.pending),
        ignored_nonphysical_fields=['native_handle_metric_only', 'process_id', 'evaluation_process_pid',
                                    'compute_seconds', 'wall_seconds', 'created_utc'],
        recorded_native_state_scope='declared bodies/subtrees/robot/velocities/schedules/task clocks; parent metric only',
        full_native_physics_state_identical_claimed=False,
        unrecorded_state='opaque contact/solver caches and engine internals remain an attribution limit')


def compare_five_way(records_by_method, profile, *, evidence_root=None, require_frozen=True):
    """Compare all ten method pairs for one common official/development case."""
    if set(records_by_method) != set(METHODS):
        return dict(accepted=False, complete=False, problems=['five_method_coverage_incomplete'], checks=[])
    checks = [dict(a=a, b=b, **compare_initial_conditions(records_by_method[a], records_by_method[b], profile,
                evidence_root=evidence_root, require_frozen=require_frozen))
              for a, b in itertools.combinations(METHODS, 2)]
    return dict(accepted=all(row['accepted'] for row in checks), complete=True, checks=checks,
        metric_only=True, policy_use_allowed=False,
        problems=[dict(a=row['a'], b=row['b'], problems=row['problems']) for row in checks if not row['accepted']])
