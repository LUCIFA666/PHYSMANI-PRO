"""Explicit shared TRAIN/VAL physical reset, separate from demo reachability.

The archived TEST reset and frozen v11 acquisition do not import this module.
This profile changes development initialization: only after the author's full
forty-attempt waypoint-0 validation failure may the SAME sampled scene proceed
through independent physical guards and the original eleven settling steps.
It never certifies the expert waypoint reachable. The teacher's existing pure
planner and censored/no-resample path failure remain unchanged.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import MultitaskInfrastructureError, MultitaskProtocolError
from .simulation import AlignedDevelopmentSimulation, file_sha256

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'r10_shared_development_physical_init_separate_expert_waypoint_v4'
JOINT_INTERVAL_NUMERICAL_TOLERANCE = {'arm': 1e-6, 'gripper': 1e-5}
TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
DIAGNOSTIC = dict(
    path='artifacts/framework/multitask_r10/reset_failure_diagnostic_v1/report.json',
    sha256='75ecd11e00d7cf7b2e5c75008a6350a30be3d24c1608f278cceab15469f15a82')
NATIVE_FILES = (
    'third_party/RLBench/rlbench/task_environment.py',
    'third_party/RLBench/rlbench/backend/scene.py',
    'third_party/RLBench/rlbench/backend/task.py',
    'third_party/RLBench/rlbench/tasks/beat_the_rotating_buzz.py',
    'third_party/RLBench/rlbench/tasks/beat_the_rotating_buzz_high_speed.py',
    'third_party/PyRep/pyrep/objects/object.py',
)
BODY_FIELDS = ('_frame_base', 'wand', 'w0', 'w1', 'w2', 'w3', 'w3_')
POLICY = dict(version=VERSION, allowed_tasks=list(TASKS), allowed_splits=['train', 'validation'],
    shared_methods='ours_and_PhysMani_same_development_profile', official_TEST_allowed=False,
    original_native_reset_attempts=40, settling_pyrep_steps=10, post_reset_pyrep_steps=1,
    strict_physical_guards=True, resample_source_seed_variation_or_scene=False,
    native_object_matrix_shape=[4, 4], homogeneous_matrix_bottom_row=[0., 0., 0., 1.],
    joint_interval_numerical_tolerance=JOINT_INTERVAL_NUMERICAL_TOLERANCE,
    joint_interval_native_units={'arm': 'radian', 'gripper': 'metre'},
    raw_joint_positions_clipped=False,
    expert_waypoint_reachability_certified=False,
    teacher_path_failure='existing_TeacherPathPlanningFailure_censored_no_resample',
    diagnostic_native_success_claimed=False)


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def rng_fingerprint():
    state = np.random.get_state()
    return dict(numpy=hashlib.sha256(state[0].encode()+state[1].tobytes()
        +repr(state[2:]).encode()).hexdigest(),
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())


def exception_chain(error):
    rows, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        rows.append(dict(type=type(error).__module__+'.'+type(error).__name__, message=str(error)))
        error = error.__cause__
    return rows


def _require(condition, message):
    if not condition: raise MultitaskInfrastructureError('development physical reset: '+message)


def _pointer(pointer, root):
    if not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}:
        raise MultitaskProtocolError('development reset permission requires an exact path/SHA pointer')
    path = (Path(root)/pointer['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or file_sha256(path) != pointer['sha256']:
        raise MultitaskProtocolError('development reset permission artifact changed')
    return path


def load_permission(pointer, *, root=ROOT, native_repo=None):
    """Verify the permission, limited diagnostic, and exact native reset sources."""
    path = _pointer(pointer, root)
    permission = json.loads(path.read_text())
    expected_keys = set(POLICY) | {'diagnostic_report', 'native_inputs_sha256', 'base_simulation_sha256'}
    if set(permission) != expected_keys or any(json.dumps(permission.get(k), sort_keys=True)
            != json.dumps(v, sort_keys=True) for k, v in POLICY.items()):
        raise MultitaskProtocolError('development reset permission scope or guards changed')
    if permission['diagnostic_report'] != DIAGNOSTIC:
        raise MultitaskProtocolError('development reset requires the disclosed single failed-source diagnostic')
    report = json.loads(_pointer(permission['diagnostic_report'], root).read_text())
    if (report.get('complete') is not True or report.get('native_cleanup_complete') is not True
            or report.get('original_reset_succeeded') is not False
            or report.get('Point_flags', {}).get('linear_only') is not False
            or report.get('outputs_may_enter_fitting') is not False
            or report.get('physical_scene_legality_certified') is not False
            or len(report.get('queries', [])) != 2
            or any(row.get('query_success') is not False for row in report['queries'])
            or any(row.get(key) is not True for row in report['queries'] for key in
                   ('clock_unchanged', 'robot_restored', 'Python_numpy_RNG_unchanged', 'task_body_poses_unchanged'))):
        raise MultitaskProtocolError('development reset diagnostic scope/result changed')
    from . import simulation
    if file_sha256(simulation.__file__) != permission['base_simulation_sha256']:
        raise MultitaskProtocolError('frozen base development simulation changed')
    if set(permission['native_inputs_sha256']) != set(NATIVE_FILES) or native_repo is None:
        raise MultitaskProtocolError('development reset needs all exact native reset/task sources')
    for relative, expected in permission['native_inputs_sha256'].items():
        if file_sha256(Path(native_repo)/relative) != expected:
            raise MultitaskProtocolError('native reset source changed: '+relative)
    return permission


def failure_evidence(error, env, variation):
    """Only the exact original forty-attempt waypoint-0 failure is eligible.

    Traceback locals bind the attempt budget and actual scene/task identities;
    a matching text message or a fabricated outer exception is insufficient.
    No extra reset attempts, validate bypass, or per-attempt instrumentation.
    """
    chain = exception_chain(error)
    expected = [dict(type='rlbench.backend.exceptions.TaskEnvironmentError', message=
        'Could not place the task %s in the scene. This should not happen, please raise an issues on this task.'
        % env.spec.task_name), dict(type='rlbench.backend.exceptions.WaypointError', message=
        "Error in task %s. Infeasible episode. Can't reach waypoint 0." % env.spec.task_name)]
    _require(chain == expected, 'exception is not the exact original waypoint-0 prevalidation failure')
    frames, seen_frames = [], set()
    for current in (error, error.__cause__):
        tb = current.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if id(frame) not in seen_frames:
                seen_frames.add(id(frame))
                frames.append((Path(frame.f_code.co_filename).resolve(), frame.f_code.co_name, frame.f_locals))
            tb = tb.tb_next
    native = Path(env.native_repo).resolve()/'third_party/RLBench/rlbench'
    original_reset = [locals_ for path, name, locals_ in frames
        if path == native/'task_environment.py' and name == 'reset' and locals_.get('self') is env.task_environment]
    attempts = [locals_ for path, name, locals_ in frames
        if path == native/'backend/scene.py' and name == 'init_episode' and locals_.get('self') is env.scene]
    waypoint = [locals_ for path, name, locals_ in frames
        if path == native/'backend/task.py' and name == '_get_waypoints' and locals_.get('self') is env.task]
    _require(len(original_reset) == len(attempts) == len(waypoint) == 1,
             'missing exact native reset/scene/waypoint traceback frames')
    local = attempts[0]
    _require(type(local.get('attempts')) is int and local['attempts'] == 40
        and type(local.get('max_attempts')) is int and local['max_attempts'] == 40
        and local.get('index') == variation and waypoint[0].get('way_i') == 0
        and waypoint[0].get('feasible') is False, 'original native forty-attempt failure not established')
    return dict(exception_chain=chain, traceback= ''.join(traceback.format_exception(error)),
        original_native_attempts_observed=40, original_max_attempts_observed=40,
        waypoint_index=0, per_attempt_log_available=False,
        attempt_evidence='unmodified Scene.init_episode traceback locals after exhaustion')


def _finite(value, shape, label):
    value = np.asarray(value, dtype=float)
    _require(value.shape == shape and bool(np.isfinite(value).all()), label+' missing/nonfinite/wrong shape')
    return value


def _pose(value, label):
    value = _finite(value, (7,), label)
    _require(abs(float(np.linalg.norm(value[3:]))-1.) <= 1e-5, label+' quaternion is not a unit SO(3) pose')
    return value


def _pose_equal(a, b):
    a, b = _pose(a, 'before pose'), _pose(b, 'after pose')
    return bool(np.allclose(a[:3], b[:3], rtol=0, atol=1e-6)
        and (Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude() <= 1e-6)


def _joints(component, label):
    q = np.asarray(component.get_joint_positions(), dtype=float)
    _require(q.ndim == 1 and len(q) > 0 and bool(np.isfinite(q).all()), label+' joints not finite')
    n = len(q)
    targets = _finite(component.get_joint_target_positions(), (n,), label+' targets')
    velocities = _finite(component.get_joint_target_velocities(), (n,), label+' target velocities')
    actual_velocities = _finite(component.get_joint_velocities(), (n,), label+' actual velocities')
    cyclic, intervals = component.get_joint_intervals()
    intervals = _finite(intervals, (n, 2), label+' native intervals')
    _require(len(cyclic) == n and all(type(c) in (bool, np.bool_) for c in cyclic)
        and bool((intervals[:, 1] > 0).all()), label+' invalid native joint limits')
    cyclic = np.asarray(cyclic, bool)
    violation = np.maximum(np.maximum(intervals[:, 0]-q, q-intervals.sum(axis=1)), 0.)
    violation[cyclic] = 0.
    maximum_violation = float(np.max(violation))
    tolerance = JOINT_INTERVAL_NUMERICAL_TOLERANCE[label]
    _require(maximum_violation <= tolerance,
             label+' actual joints outside native limits beyond fixed numerical tolerance: '
             +json.dumps(dict(raw_q=q.tolist(), native_intervals=intervals.tolist(),
                 maximum_violation=maximum_violation, tolerance=tolerance)))
    return dict(q=q.tolist(), targets=targets.tolist(), target_velocities=velocities.tolist(),
        actual_velocities=actual_velocities.tolist(), cyclic=cyclic.tolist(), intervals=intervals.tolist(),
        native_interval_violation=violation.tolist(), maximum_native_interval_violation=maximum_violation,
        native_interval_numerical_tolerance=tolerance, raw_joint_positions_clipped=False)


def snapshot(env):
    """Raw metric-only state; no restoration or physics occurs in this function."""
    _require(type(env.seed) is int and type(env.task.var_index) is int, 'missing actual seed/variation')
    bodies = {}
    for name in BODY_FIELDS:
        obj = getattr(env.task, name)
        pose = _pose(obj.get_pose(), name+' pose')
        matrix = _finite(obj.get_matrix(), (4, 4), name+' matrix')
        _require(np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-7, rtol=0),
                 name+' matrix is not homogeneous SE(3)')
        rotation = matrix[:3, :3]
        _require(np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5, rtol=0)
            and abs(float(np.linalg.det(rotation))-1.) <= 1e-5
            and np.allclose(matrix[:3, 3], pose[:3], atol=1e-6, rtol=0)
            and np.allclose(rotation, Rotation.from_quat(pose[3:]).as_matrix(), atol=1e-5, rtol=0),
            name+' matrix/pose is not consistent finite SE(3)')
        bodies[name] = dict(pose_xyzw=pose.tolist(), matrix=matrix.tolist())
    arm, gripper = _joints(env.arm, 'arm'), _joints(env.gripper, 'gripper')
    _require(len(arm['q']) == 7, 'the shared reset supports the original seven-joint Panda only')
    opening = _finite(env.gripper.get_open_amount(), (len(gripper['q']),), 'gripper opening')
    _require(bool(np.all((opening >= -1e-6) & (opening <= 1+1e-6))), 'invalid gripper opening')
    attached = env.gripper.get_grasped_objects()
    _require(len(attached) == 0, 'initial gripper has attached objects')
    driver = clean(env.task.var2target_state_list)
    active = clean(env.task.target_state_list)
    # allow_nan=False validates every stored driver field without normalization.
    driver_sha, active_sha = digest(driver), digest(active)
    state = dict(physics_time=float(env.physics_time()), task_time=float(env.task.t),
        step_id=env.task.step_id, seed=env.seed, variation=env.task.var_index,
        scene_variation=env.scene._variation_index, arm=arm, gripper=gripper,
        tip_pose_xyzw=_pose(env.arm.get_tip().get_pose(), 'tip').tolist(),
        IK_target_pose_xyzw=_pose(env.arm._ik_target.get_pose(), 'IK target').tolist(),
        gripper_opening=opening.tolist(), attachment_count=len(attached), bodies=bodies,
        driver=driver, driver_sha256=driver_sha, active_driver=active, active_driver_sha256=active_sha,
        RNG=rng_fingerprint(), scene_has_init_task=env.scene._has_init_task,
        scene_has_init_episode=env.scene._has_init_episode,
        task_environment_reset_called=env.task_environment._reset_called)
    _require(np.isfinite(state['physics_time']) and np.isfinite(state['task_time'])
        and type(state['step_id']) is int and state['step_id'] == 0 and state['task_time'] == 0,
        'development reset is not at the original task clock origin')
    return clean(state)


def _same_reconstruction(before, after):
    fields = ('physics_time', 'task_time', 'step_id', 'seed', 'variation', 'scene_variation',
        'driver_sha256', 'RNG', 'scene_has_init_task', 'scene_has_init_episode', 'task_environment_reset_called')
    _require(all(before[k] == after[k] for k in fields), 'same-variation reconstruction changed clock/RNG/seed/driver/state flags')
    for component in ('arm', 'gripper'):
        _require(all(np.allclose(before[component][k], after[component][k], atol=1e-6, rtol=0)
            for k in ('q', 'targets', 'target_velocities', 'actual_velocities', 'intervals'))
            and before[component]['cyclic'] == after[component]['cyclic'],
            'same-variation reconstruction changed '+component)
    _require(all(_pose_equal(before[k], after[k]) for k in ('tip_pose_xyzw', 'IK_target_pose_xyzw')),
             'same-variation reconstruction changed robot tip/IK target')
    _require(all((Rotation.from_quat(before['bodies'][k]['pose_xyzw'][3:]).inv()
        * Rotation.from_quat(after['bodies'][k]['pose_xyzw'][3:])).magnitude() <= 1e-6
        for k in BODY_FIELDS), 'same-variation reconstruction changed original native body orientations')


def guarded_collision(env, before):
    collision = env.arm.check_arm_collision()
    after = snapshot(env)
    _require(after == before, 'read-only collision query mutated raw state')
    _require(type(collision) in (bool, np.bool_) and not collision, 'arm initial collision')
    return dict(arm_collision=False, query_advanced_clock=False, query_changed_raw_state=False)


def guarded_native_observation(env, label, receipt):
    before = snapshot(env)
    env.task_environment.get_observation()
    after = snapshot(env)
    _require(after == before, label+' native observation changed raw clock/RNG/robot/driver state')
    receipt.setdefault('native_observation_queries', []).append(dict(label=label,
        before=before, after=after, raw_state_unchanged=True))


def restore_same_physical_episode(env, variation, receipt):
    """Reconstruct only the existing native var2; fail closed before all steps."""
    before = snapshot(env); receipt['post_failure_restored_state'] = before
    _require(env.task.is_static_workspace() is True and env.scene._has_init_task is True
        and env.scene._has_init_episode is False and env.task_environment._reset_called is False,
        'not a fresh initialized static task with an exhausted reset')
    _require(before['variation'] == before['scene_variation'] == variation,
             'failed native reset variation differs from the requested source')
    descriptions = env.task.init_episode(variation)
    after = snapshot(env); receipt['same_variation_reconstructed_state'] = after
    _same_reconstruction(before, after)
    driver = after['driver'].get(str(variation))
    _require(isinstance(driver, dict) and set(driver) == {'yaw_speed', 'frame_position'}, 'unexpected native buzz driver schema')
    target = _finite(driver['frame_position'], (3,), 'frozen var2 frame position')
    _require(np.isfinite(driver['yaw_speed']) and driver['yaw_speed'] != 0
        and np.allclose(after['bodies']['_frame_base']['pose_xyzw'][:3], target, atol=1e-6, rtol=0),
        'reconstructed scene does not match the original frozen var2 geometry')
    _require(after['active_driver'] == [dict(yaw_speed=driver['yaw_speed'], t0=0)]
        and env.task.yaw_speed == driver['yaw_speed'] and env.task.stage == 'wp0',
        'native driver activation/stage differs from same-source init_episode')
    receipt['physical_guards_before_settling'] = guarded_collision(env, after)
    receipt['reconstruction_guards_passed'] = True
    receipt['settling_steps'] = []
    for i in range(11):
        if i == 10:
            # Mirror Scene.init_episode completion and TaskEnvironment.reset
            # completion BEFORE the owned development reset's extra step.
            env.scene._has_init_episode = True
            env.task_environment._reset_called = True
            receipt['state_flags_explicitly_completed_after_native10'] = True
            guarded_native_observation(env, 'original_TaskEnvironment_reset_observation', receipt)
        previous = snapshot(env)
        env.pyrep.step()
        current = snapshot(env)
        _require(abs(current['physics_time']-previous['physics_time']-env.protocol['dt']) <= 1e-6,
                 'settling step did not advance exactly one declared physics dt')
        _require(all(current[k] == previous[k] for k in
            ('task_time', 'step_id', 'seed', 'variation', 'scene_variation', 'driver_sha256',
             'active_driver_sha256', 'RNG', 'scene_has_init_task', 'scene_has_init_episode', 'task_environment_reset_called')),
            'settling changed task clock/RNG/source/driver/state flags')
        receipt['settling_steps'].append(dict(index=i+1,
            phase='original_native10' if i < 10 else 'original_owned_development1',
            before=previous, after=current))
    final = snapshot(env)
    receipt['physical_guards_after_settling'] = guarded_collision(env, final)
    receipt['final_settled_state'] = final
    receipt['original_native10_and_owned1_physics_steps_measured'] = True
    receipt['physics_delta_seconds'] = final['physics_time']-after['physics_time']
    receipt['task_delta_seconds'] = final['task_time']-after['task_time']
    return descriptions


class SharedPhysicalDevelopmentSimulation(AlignedDevelopmentSimulation):
    """Recovery-only opt-in; original successful resets use the base verbatim."""
    def __init__(self, *args, **kwargs):
        protocol = kwargs['protocol']
        pointer = protocol.get('development_reset_profile')
        self._reset_profile = load_permission(pointer, native_repo=protocol['baseline_repo'])
        self._reset_profile_pointer = deepcopy(pointer)
        self.reset_recovery_receipt = dict(version=VERSION, permission=deepcopy(pointer),
            diagnostic_report=dict(DIAGNOSTIC), recovery_applied=False, complete=False,
            official_TEST_allowed=False, source_resampled=False,
            expert_waypoint_reachability_certified=False, dataset_or_closed_loop_success_claimed=False)
        self.reset_recovery_receipt.update(native_Cpp_planner_RNG_not_observable=True,
            physical_guards_are_initial_state_checks_not_global_collision_or_reachability_proof=True,
            original_v11_completed_cases_not_relabelled=True)
        super().__init__(*args, **kwargs)

    def reset_development(self, *, variation=None):
        receipt = self.reset_recovery_receipt
        _require(self.development_split in ('train', 'validation'), 'shared reset is TRAIN/VAL only')
        if hasattr(self, 'ledger') or receipt['complete']:
            raise MultitaskProtocolError('shared development reset also needs a fresh process')
        actual_variation = (self.development_source.get('variation',
            self.spec.supported_variations[self.seed % self.spec.variation_count]) if variation is None else variation)
        receipt.update(task=self.spec.task_name, split=self.development_split, seed=self.seed,
            seed_identity=deepcopy(self.development_seed_identity), replica=self.replica,
            source=deepcopy(self.development_source), variation=actual_variation)
        try:
            result = super().reset_development(variation=variation)
        except Exception as error:
            receipt.update(original_reset_succeeded=False, original_reset_exception_chain=exception_chain(error),
                original_reset_traceback=''.join(traceback.format_exception(error)))
            if self.spec.task_name not in TASKS:
                raise
            try:
                receipt['failure_eligibility'] = failure_evidence(error, self, actual_variation)
                descriptions = restore_same_physical_episode(self, actual_variation, receipt)
                guarded_native_observation(self, 'original_owned_development_post_step_observation', receipt)
                self.task.disable_expert_plan()
                self._start_alignment()
                self._reset_metric_receipt(dict(task=self.spec.task_name,
                    raw_episode_id=self.development_source['raw_episode_id'], variation=actual_variation,
                    source_split=self.development_split, source_path=str(self.calibration_source),
                    source_hdf5_sha256=file_sha256(self.calibration_source), seed=self.seed,
                    seed_identity=self.development_seed_identity, replica=self.replica,
                    reset_kind='native_generated_development_shared_physical_init_after_waypoint0_failure',
                    original_source_replayed=False, official_TEST=False,
                    source_fields_used='source identity for seed and frame-zero camera calibration only',
                    source_motion_or_expert_trajectory_replayed=False,
                    development_reset_profile=deepcopy(self._reset_profile_pointer)))
                result, _robot = self.observe()
                receipt.update(recovery_applied=True, complete=True, original_reset_succeeded=False,
                    expert_waypoint_validation_skipped_after_original40=True,
                    independent_physical_guards_passed=True,
                    teacher_planner_unchanged=True, teacher_path_failure_censored_no_resample=True)
                self.reset_receipt['reset_recovery_receipt'] = deepcopy(receipt)
                return descriptions, result
            except Exception as recovery_error:
                receipt.update(complete=False, recovery_error=exception_chain(recovery_error),
                    recovery_traceback=''.join(traceback.format_exception(recovery_error)),
                    physical_guard_failure_is_infrastructure=True)
                raise MultitaskInfrastructureError('same-source development physical reset failed; no resample') from recovery_error
        receipt.update(complete=True, original_reset_succeeded=True, recovery_applied=False)
        self.reset_receipt['development_reset_profile'] = deepcopy(self._reset_profile_pointer)
        self.reset_receipt['reset_recovery_receipt'] = deepcopy(receipt)
        return result
