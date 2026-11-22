"""CPU-only guards for the opt-in shared physical development reset."""
from copy import deepcopy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.multitask import development_reset_recovery as r
from our_mani.multitask.simulation import AlignedDevelopmentSimulation
from our_mani.multitask.protocol import MultitaskInfrastructureError, MultitaskProtocolError


class Body:
    def __init__(self, position=(0., 0., .9)):
        self.pose = list(position)+[0., 0., 0., 1.]
        self.matrix_override = None
    def get_pose(self): return list(self.pose)
    def get_matrix(self):
        if self.matrix_override is not None: return self.matrix_override
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(self.pose[3:]).as_matrix()
        matrix[:3, 3] = self.pose[:3]
        return matrix


class Component:
    def __init__(self, n, *, gripper=False):
        self.q = [.02 if gripper else 0.] * n
        self.targets, self.velocities, self.actual_velocities = list(self.q), [0.] * n, [0.] * n
        self.intervals = [[0., .04] if gripper else [-3., 6.] for _ in self.q]
        self.cyclic = [False] * n
        self.tip, self._ik_target = Body(), Body()
        self.attached, self.collision = [], False
        self.collision_effect = None
    def get_joint_positions(self): return list(self.q)
    def get_joint_target_positions(self): return list(self.targets)
    def get_joint_target_velocities(self): return list(self.velocities)
    def get_joint_velocities(self): return list(self.actual_velocities)
    def get_joint_intervals(self): return list(self.cyclic), deepcopy(self.intervals)
    def get_tip(self): return self.tip
    def get_open_amount(self): return [.5] * len(self.q)
    def get_grasped_objects(self): return list(self.attached)
    def check_arm_collision(self):
        if self.collision_effect: self.collision_effect()
        return self.collision


def fixture(tmp):
    env = object.__new__(r.SharedPhysicalDevelopmentSimulation)
    env.seed, env.replica = 123, 0
    env.native_repo = Path(tmp)/'native'
    env.spec = SimpleNamespace(task_name=r.TASKS[0], supported_variations=(0,), variation_count=1)
    env.development_split = 'train'
    env.calibration_source = Path(tmp)/'source.hdf5'; env.calibration_source.write_bytes(b'fixed-source')
    env.development_source = dict(task=env.spec.task_name, raw_episode_id='episode104', variation=0,
                                  path=str(env.calibration_source))
    env.development_seed_identity = dict(raw_episode_id='episode104', seed=123)
    env.protocol = dict(dt=.05)
    env._reset_profile_pointer = dict(path='permission.json', sha256='f'*64)
    env._reset_profile = deepcopy(r.POLICY)
    env.reset_recovery_receipt = dict(version=r.VERSION, complete=False, recovery_applied=False)
    env.arm, env.gripper = Component(7), Component(2, gripper=True)
    env.scene = SimpleNamespace(_has_init_task=True, _has_init_episode=False, _variation_index=0)
    env.task_environment = SimpleNamespace(_reset_called=False)
    env.task = SimpleNamespace(var_index=0, step_id=0, t=0., stage='wp0', yaw_speed=3.,
        var2target_state_list={0: dict(yaw_speed=3., frame_position=[.3, -.2, .9])},
        target_state_list=[], is_static_workspace=lambda: True, init_calls=0)
    for name in r.BODY_FIELDS: setattr(env.task, name, Body())
    env.seconds, env.step_calls, env.obs_calls = .2, 0, 0
    env.step_effect, env.observation_effect, env.reconstruction_effect = None, None, None
    def step():
        env.seconds += .05; env.step_calls += 1
        if env.step_effect: env.step_effect()
    env.pyrep = SimpleNamespace(step=step)
    env.physics_time = lambda: env.seconds
    def observation():
        env.obs_calls += 1
        if env.observation_effect: env.observation_effect()
        return 'native-observation'
    env.task_environment.get_observation = observation
    def init_episode(index):
        env.task.init_calls += 1
        driver = env.task.var2target_state_list[index]
        env.task._frame_base.pose[:3] = driver['frame_position']
        env.task.target_state_list = [dict(yaw_speed=driver['yaw_speed'], t0=0)]
        if env.reconstruction_effect: env.reconstruction_effect()
        return ['same-native-description']
    env.task.init_episode = init_episode
    env.task.disable_expert_plan = lambda: setattr(env.task, 'expert_disabled', True)
    def align(): env.ledger = SimpleNamespace(ticks=0); env._aligned = True
    env._start_alignment = align
    env._reset_metric_receipt = lambda metadata: setattr(env, 'reset_receipt', metadata)
    env.observe = lambda: ({'timestamp': 0.}, None)
    return env


def exhausted_reset(env, *, attempts=40, maximum=40, way_i=0, variation=0, unrelated=False):
    """Synthetic tracebacks at exact native paths; no native imports or writes."""
    root = env.native_repo/'third_party/RLBench/rlbench'
    TaskEnvironmentError = type('TaskEnvironmentError', (Exception,), {'__module__': 'rlbench.backend.exceptions'})
    WaypointError = type('WaypointError', (Exception,), {'__module__': 'rlbench.backend.exceptions'})
    scope = dict(TaskEnvironmentError=TaskEnvironmentError, WaypointError=WaypointError,
                 task_name=env.spec.task_name, fail_way=way_i, fail_attempts=attempts,
                 fail_max=maximum, unrelated=unrelated)
    exec(compile('def _get_waypoints(self):\n'
        '    feasible = False\n    way_i = fail_way\n'
        '    if unrelated: raise ValueError("not a native reset error")\n'
        '    raise WaypointError("Error in task %s. Infeasible episode. Can\\\'t reach waypoint %d." % (task_name, way_i))\n',
        str(root/'backend/task.py'), 'exec'), scope)
    env.task._get_waypoints = scope['_get_waypoints'].__get__(env.task)
    exec(compile('def init_episode(self, index, max_attempts):\n'
        '    attempts = fail_attempts\n    max_attempts = fail_max\n'
        '    self.task._get_waypoints()\n', str(root/'backend/scene.py'), 'exec'), scope)
    env.scene.task = env.task
    env.scene.init_episode = scope['init_episode'].__get__(env.scene)
    exec(compile('def reset(self):\n'
        '    try: self._scene.init_episode(self.variation, max_attempts=40)\n'
        '    except WaypointError as e:\n'
        '        raise TaskEnvironmentError("Could not place the task %s in the scene. This should not happen, please raise an issues on this task." % task_name) from e\n',
        str(root/'task_environment.py'), 'exec'), scope)
    env.task_environment._scene, env.task_environment.variation = env.scene, variation
    return scope['reset'](env.task_environment)


class RecoveryTests(unittest.TestCase):
    def setUp(self): self.tmp = TemporaryDirectory(); self.env = fixture(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def reset(self, **kw):
        def original(_self, **_kwargs): return exhausted_reset(_self, **kw)
        with patch.object(AlignedDevelopmentSimulation, 'reset_development', original):
            return self.env.reset_development(variation=0)

    def test_actual_native_get_matrix_contract_is_four_by_four(self):
        source = (r.ROOT/'../phsimani/PhysMani/third_party/PyRep/pyrep/objects/object.py').resolve()
        if not source.exists():
            self.skipTest('native repository is unavailable in this fixture environment')
        import ast
        definition = next(node for node in ast.walk(ast.parse(source.read_text()))
                          if isinstance(node, ast.FunctionDef) and node.name == 'get_matrix')
        namespace = {'np': np, 'sim': SimpleNamespace(
            simGetObjectMatrix=lambda _handle, _relative: [1., 0., 0., .3, 0., 1., 0., -.2, 0., 0., 1., .9])}
        module = ast.Module(body=[definition], type_ignores=[])
        exec(compile(module, str(source), 'exec'), namespace)
        native = namespace['get_matrix'](SimpleNamespace(_handle=1))
        self.assertEqual(native.shape, (4, 4))
        self.env.task.wand.matrix_override = native
        self.env.task.wand.pose[:3] = native[:3, 3].tolist()
        state = r.snapshot(self.env)
        self.assertEqual(np.asarray(state['bodies']['wand']['matrix']).shape, (4, 4))

    def test_invalid_homogeneous_bottom_row_is_rejected_before_physics(self):
        matrix = np.eye(4)
        matrix[:3, 3] = self.env.task.wand.pose[:3]
        matrix[3, 0] = .001
        self.env.task.wand.matrix_override = matrix
        with self.assertRaises(MultitaskInfrastructureError) as failure:
            self.reset()
        self.assertIn('homogeneous SE', str(failure.exception.__cause__))
        self.assertEqual(self.env.step_calls, 0)

    def test_native_float_interval_residual_is_recorded_without_clipping(self):
        self.env.gripper.q[0] = .04000002
        state = r.snapshot(self.env)
        self.assertEqual(state['gripper']['q'][0], .04000002)
        self.assertGreater(state['gripper']['maximum_native_interval_violation'], 0.)
        self.assertLess(state['gripper']['maximum_native_interval_violation'], 1e-6)
        self.assertFalse(state['gripper']['raw_joint_positions_clipped'])

    def test_joint_violation_over_fixed_tolerance_retains_raw_diagnostic(self):
        self.env.gripper.q[0] = .04002
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'maximum_violation.*tolerance'):
            r.snapshot(self.env)
        self.assertEqual(self.env.gripper.q[0], .04002)
        self.assertEqual(self.env.step_calls, 0)

    def test_measured_native_settling_residual_preserves_distinct_component_units(self):
        self.env.gripper.q = [0.040001556277275085, 0.039998602122068405]
        self.env.gripper.intervals = [[0., 0.03999999910593033]]*2
        state = r.snapshot(self.env)
        self.assertAlmostEqual(state['gripper']['maximum_native_interval_violation'], 1.55717134475708e-6)
        self.assertEqual(state['gripper']['native_interval_numerical_tolerance'], 1e-5)
        self.assertEqual(state['arm']['native_interval_numerical_tolerance'], 1e-6)
        self.assertEqual(state['gripper']['q'], self.env.gripper.q)
        self.env.arm.q[0] = 3.000002
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'arm actual joints outside'):
            r.snapshot(self.env)

    def test_exact_failure_recovers_same_scene_and_measures_only_eleven_reset_steps(self):
        rng = r.rng_fingerprint(); driver = deepcopy(self.env.task.var2target_state_list)
        descriptions, obs = self.reset()
        receipt = self.env.reset_recovery_receipt
        self.assertEqual(descriptions, ['same-native-description'])
        self.assertEqual(obs, {'timestamp': 0.})
        self.assertEqual(self.env.task.init_calls, 1)
        self.assertEqual(self.env.step_calls, 11)
        self.assertEqual(self.env.obs_calls, 2)
        self.assertEqual(self.env.task.var2target_state_list, driver)
        self.assertEqual(r.rng_fingerprint(), rng)
        self.assertAlmostEqual(receipt['physics_delta_seconds'], .55)
        self.assertEqual(receipt['task_delta_seconds'], 0)
        self.assertEqual([x['phase'] for x in receipt['settling_steps']], ['original_native10']*10+['original_owned_development1'])
        self.assertFalse(receipt['settling_steps'][9]['after']['scene_has_init_episode'])
        self.assertTrue(receipt['settling_steps'][10]['before']['scene_has_init_episode'])
        self.assertTrue(self.env.task_environment._reset_called)
        self.assertTrue(self.env.task.expert_disabled)
        self.assertEqual(self.env.ledger.ticks, 0)
        self.assertTrue(receipt['teacher_planner_unchanged'])
        self.assertTrue(receipt['teacher_path_failure_censored_no_resample'])
        self.assertEqual(self.env.reset_receipt['reset_recovery_receipt'], receipt)

    def test_high_speed_validation_has_identical_recovery_scope(self):
        self.env.spec.task_name = r.TASKS[1]; self.env.development_split = 'validation'
        self.reset()
        self.assertTrue(self.env.reset_recovery_receipt['complete'])

    def test_original_success_is_returned_without_reconstruction_or_extra_steps(self):
        def success(env, **kw):
            env.reset_receipt = {'original': True}; return ('native-description', 'native-observation')
        with patch.object(AlignedDevelopmentSimulation, 'reset_development', success):
            self.assertEqual(self.env.reset_development(), ('native-description', 'native-observation'))
        self.assertEqual(self.env.step_calls, 0); self.assertEqual(self.env.task.init_calls, 0)
        self.assertFalse(self.env.reset_recovery_receipt['recovery_applied'])

    def test_other_failures_and_other_tasks_never_reconstruct(self):
        for kw in ({'way_i': 1}, {'attempts': 39}, {'maximum': 50}, {'variation': 1}, {'unrelated': True}):
            with self.subTest(kw=kw), self.assertRaises(Exception): self.reset(**kw)
            self.assertEqual(self.env.step_calls, 0); self.assertEqual(self.env.task.init_calls, 0)
        self.env.spec.task_name = 'insert_onto_rotating_peg'
        with self.assertRaises(Exception): self.reset()
        self.assertEqual(self.env.task.init_calls, 0)

    def test_formal_test_is_rejected_before_original_reset(self):
        self.env.development_split = 'test'
        with patch.object(AlignedDevelopmentSimulation, 'reset_development') as base:
            with self.assertRaises(MultitaskInfrastructureError): self.env.reset_development()
            base.assert_not_called()

    def test_reconstruction_cannot_change_rng_robot_clock_seed_or_driver(self):
        mutations = [lambda e: np.random.random(), lambda e: random.random(),
            lambda e: e.arm.targets.__setitem__(0, .1), lambda e: e.arm._ik_target.pose.__setitem__(0, .1),
            lambda e: setattr(e, 'seconds', e.seconds+.05), lambda e: setattr(e, 'seed', 999),
            lambda e: e.task.var2target_state_list[0].__setitem__('yaw_speed', 6.),
            lambda e: e.task.wand.pose.__setitem__(3, .1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.env = fixture(self.tmp.name); self.env.reconstruction_effect = lambda: mutate(self.env)
                with self.assertRaises(MultitaskInfrastructureError): self.reset()
                self.assertEqual(self.env.step_calls, 0)

    def test_out_of_limits_nonfinite_attachment_collision_and_invalid_se3_fail_closed(self):
        mutations = [lambda e: e.arm.q.__setitem__(0, 3.01), lambda e: e.arm.q.__setitem__(0, np.nan),
            lambda e: e.gripper.attached.append(object()), lambda e: setattr(e.arm, 'collision', True),
            lambda e: e.task.wand.pose.__setitem__(6, 2.),
            lambda e: setattr(e.task.wand, 'matrix_override', np.zeros((3,4))),
            lambda e: e.gripper.q.__setitem__(0, .05)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.env = fixture(self.tmp.name); mutate(self.env)
                with self.assertRaises(MultitaskInfrastructureError): self.reset()
                self.assertEqual(self.env.step_calls, 0)
                self.assertFalse(self.env.reset_recovery_receipt['complete'])

    def test_read_only_collision_query_cannot_mutate_robot_or_clock(self):
        self.env.arm.collision_effect = lambda: self.env.arm.targets.__setitem__(0, .1)
        with self.assertRaises(MultitaskInfrastructureError): self.reset()
        self.assertEqual(self.env.step_calls, 0)

    def test_settling_dt_taskclock_driver_rng_and_observation_are_guarded(self):
        effects = [lambda e: setattr(e, 'seconds', e.seconds+.01),
            lambda e: setattr(e.task, 't', .05), lambda e: np.random.random(),
            lambda e: e.task.target_state_list[0].__setitem__('yaw_speed', 4.)]
        for mutate in effects:
            with self.subTest(mutate=mutate):
                self.env = fixture(self.tmp.name); self.env.step_effect = lambda: mutate(self.env)
                with self.assertRaises(MultitaskInfrastructureError): self.reset()
                self.assertEqual(self.env.step_calls, 1)
        self.env = fixture(self.tmp.name)
        self.env.observation_effect = lambda: setattr(self.env, 'seconds', self.env.seconds+.01)
        with self.assertRaises(MultitaskInfrastructureError): self.reset()
        self.assertEqual(self.env.step_calls, 10)

    def test_fixed_var2_translation_and_active_native_driver_required(self):
        self.env.reconstruction_effect = lambda: self.env.task._frame_base.pose.__setitem__(0, .35)
        with self.assertRaises(MultitaskInfrastructureError): self.reset()
        self.assertEqual(self.env.step_calls, 0)

    def test_fresh_process_flags_required_and_repeated_reset_rejected(self):
        self.env.scene._has_init_episode = True
        with self.assertRaises(MultitaskInfrastructureError): self.reset()
        self.assertEqual(self.env.step_calls, 0)
        self.env = fixture(self.tmp.name); self.reset()
        with self.assertRaises(MultitaskProtocolError): self.env.reset_development()
        self.assertEqual(self.env.step_calls, 11)

    def test_permission_scope_and_pointer_reject_changed_artifact_before_native(self):
        p = Path(self.tmp.name)/'permission.json'
        policy = dict(r.POLICY, diagnostic_report=dict(r.DIAGNOSTIC), native_inputs_sha256={},
                      base_simulation_sha256='f'*64)
        policy['official_TEST_allowed'] = True
        p.write_text(json.dumps(policy))
        with self.assertRaises(MultitaskProtocolError):
            r.load_permission(dict(path=p.name, sha256=r.file_sha256(p)), root=self.tmp.name, native_repo=self.tmp.name)
        with self.assertRaises(MultitaskProtocolError):
            r.load_permission(dict(path=p.name, sha256='0'*64), root=self.tmp.name, native_repo=self.tmp.name)
        with self.assertRaises(MultitaskProtocolError): r.load_permission(None, root=self.tmp.name)


if __name__ == '__main__': unittest.main()
