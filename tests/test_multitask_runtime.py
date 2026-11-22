"""R10 reset/clock/terminal/map contracts without native simulation or GPU."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.aligned.protocol import TickLedger
from our_mani.contracts import Forecast, Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask import physmani_executor_v7 as executor
from our_mani.ablations import physmani_executor_v4 as frozen_arm
from our_mani.multitask.maps import SharedMultiObjectMap, KnownMapForecastCollision, AlignedRobotModel
from our_mani.multitask.native_tasks import NativeOutcomeLatch
from our_mani.multitask.protocol import (EpisodeTerminatedError, MultitaskProtocolError,
                                         MultitaskInfrastructureError)
from our_mani.multitask.simulation import (MultitaskSimulation, AlignedOfficialSimulation,
    AlignedDevelopmentSimulation, development_seed, OFFICIAL_SOURCE_FILES,
    verify_official_source_files, file_sha256)
from our_mani.multitask.specs import TASK_SPECS, get_task_spec


class Shape:
    serial = 0
    __hash__ = None  # Match native PyRep Object.__eq__ / unhashable instances.

    def __init__(self, name, *, children=(), forbid_pose=False):
        Shape.serial += 1
        self.name, self.handle = name, Shape.serial
        self.children, self.forbid_pose = list(children), forbid_pose
        self.collidable, self.matrix, self.removed = True, np.eye(4), False

    def get_handle(self): return self.handle
    def __eq__(self, other): return isinstance(other, Shape) and self.handle == other.handle
    def get_name(self): return self.name
    def get_objects_in_tree(self, **kwargs):
        return [self, *[part for child in self.children for part in child.get_objects_in_tree(**kwargs)]]
    def get_mesh_data(self): return [0., 0., 0., .1, 0., 0., 0., .1, 0.], [0, 1, 2], []
    def get_bounding_box(self): return [-.1, .1] * 3
    def get_pose(self):
        if self.forbid_pose: raise AssertionError('live dynamic pose was read')
        return [0., 0., 0., 0., 0., 0., 1.]
    def get_matrix(self):
        if self.forbid_pose: raise AssertionError('live dynamic matrix was read')
        return self.matrix.copy()
    def set_matrix(self, value): self.matrix = np.asarray(value).copy()
    def is_collidable(self): return self.collidable
    def is_respondable(self): return getattr(self, 'respondable', False)
    def set_collidable(self, value): self.collidable = value
    def set_dynamic(self, value): assert value is False
    def set_respondable(self, value): assert value is False
    def set_renderable(self, value): assert value is False
    def remove(self): self.removed = True


class Arm(Shape):
    def __init__(self):
        super().__init__('robot')
        self.q, self.targets, self.velocity = np.zeros(7), np.full(7, .4), np.zeros(7)
        self._ik_pose = [0., 0., 0., 0., 0., 0., 1.]
        self._ik_target = SimpleNamespace(get_pose=lambda: self._ik_pose.copy(),
            set_pose=lambda value: setattr(self, '_ik_pose', list(value)))
    def get_joint_positions(self): return self.q.tolist()
    def set_joint_positions(self, value): self.q = np.asarray(value).copy()
    def get_joint_target_positions(self): return self.targets.tolist()
    def set_joint_target_positions(self, value): self.targets = np.asarray(value).copy()
    def get_joint_velocities(self): return self.velocity.tolist()
    def get_tip(self): return SimpleNamespace(get_pose=lambda: [*self.q[:3], 0., 0., 0., 1.])


def runtime(fatal_tick=2, success_tick=None):
    env = object.__new__(MultitaskSimulation)
    env.arm = Arm()
    env.gripper = SimpleNamespace(get_open_amount=lambda: [1., 1.], get_grasped_objects=lambda: [],
                                  release=Mock(), grasp=Mock(), actuate=Mock())
    env.ledger = TickLedger(dt=.05, budget=10.)
    env.time = env._clock = env._physics_origin = env._task_origin = 0.
    env.previous_robot = None
    env._aligned = True
    env._requested_opening = 1.
    env._outcome, env._terminal_tick = NativeOutcomeLatch(), None
    env._native_success_latched = env._native_failure_latched = env._native_terminal_latched = False
    env.physics_time = lambda: env._clock
    env.protocol = dict(dt=.05, simulation_seconds=10., gripper_velocity=.2)
    env.robot_model = SimpleNamespace(velocity_limits=np.full(7, 2.), lower=np.full(7, -3.), upper=np.full(7, 3.))
    env.task = SimpleNamespace(t=0., get_graspable_objects=lambda: [])
    def result():
        if fatal_tick is not None and env.ledger.ticks >= fatal_tick: return False, True
        if success_tick is not None and env.ledger.ticks >= success_tick: return True, True
        return False, False
    env.task.success = result
    def step():
        delta = np.clip(env.arm.targets - env.arm.q, -.1, .1)
        env.arm.q += delta
        env.arm.velocity = delta / .05
        env._clock += .05
        env.task.t += .05
    env.scene = SimpleNamespace(step=step, check_target_in_workspace=lambda point: True)
    env.map_service = SimpleNamespace(plan_path=Mock(side_effect=RuntimeError('native planner rejected')))
    env.observe = lambda: ({}, SimpleNamespace(timestamp=env.time))
    return env


class RuntimeTests(unittest.TestCase):
    def test_first_native_fatal_latches_failure_and_rejects_future_action(self):
        env = runtime(fatal_tick=1)
        env.hold_tick()
        assert env.ledger.ticks == 1 and env.native_outcome.terminal_failure
        assert not env._native_success_latched
        env.task.success = lambda: (True, True)
        env._latch_native_result()
        assert env.native_outcome.terminal_failure and not env._native_success_latched
        with self.assertRaises(EpisodeTerminatedError): env.hold_tick()
        assert env._clock == .05

    def test_clock_drift_is_infrastructure_before_setting_references(self):
        env = runtime()
        old = env.arm.targets.copy()
        env._clock = .025
        with self.assertRaises(MultitaskProtocolError): env.command_joints(np.ones(7), opening=0.)
        np.testing.assert_array_equal(env.arm.targets, old)
        env.gripper.actuate.assert_not_called()

    def test_seed_varies_by_source_split_and_replica_without_consuming_rng(self):
        row = dict(task='push_moving_button', raw_episode_id='episode5')
        before = np.random.get_state()
        a, _ = development_seed(row['task'], row, 'train')
        assert a == development_seed(row['task'], row, 'train')[0]
        assert a != development_seed(row['task'], row, 'validation')[0]
        assert a != development_seed(row['task'], row, 'train', replica=1)[0]
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        with self.assertRaises(MultitaskProtocolError): development_seed(row['task'], row, 'test')

    def test_official_dispatch_all_14_tasks_restores_archives_and_peg_no_random(self):
        driver = dict(x=[0., 0., .8], v=[0., .2, 0.], a=[0., 0., 0.])
        with tempfile.TemporaryDirectory() as directory:
            for spec in TASK_SPECS.values():
                var = spec.supported_variations[-1]
                path = Path(directory) / spec.task_name / 'all_variations/episodes/episode99'
                path.mkdir(parents=True)
                alias = path.parents[2] / ('variation' + str(var)) / 'episodes/episode0'
                alias.parent.mkdir(parents=True)
                alias.symlink_to(path)
                if spec.family == 'push_moving_button': selected = [driver]
                elif spec.family == 'moving_basketball_in_hoop': selected = [driver, driver]
                elif spec.reset_dispatch.value != 'author_translation': selected = {'yaw_speed': [.5]}
                else: selected = driver
                archive = {var: selected}
                for name, value in [('low_dim_obs.pkl', SimpleNamespace()), ('variation_number.pkl', var),
                                    ('target_state.pkl', archive)]:
                    (path / name).write_bytes(pickle.dumps(value))
                required = {}
                for name in OFFICIAL_SOURCE_FILES:
                    native_file = path / name
                    native_file.parent.mkdir(parents=True, exist_ok=True)
                    if not native_file.exists(): native_file.write_bytes(b'fixture')
                    required[name] = dict(path=str(native_file), sha256=file_sha256(native_file),
                                          bytes=native_file.stat().st_size)
                repo = Path(directory) / 'native_repo'
                source = repo / spec.native_source_relative_path
                scene = repo / 'third_party/RLBench/rlbench/task_ttms' / (spec.task_name + '.ttm')
                for item in (source, scene):
                    item.parent.mkdir(parents=True, exist_ok=True);item.write_bytes(b'fixture source')
                spec = replace(spec, native_source_sha256=file_sha256(source))
                env = object.__new__(AlignedOfficialSimulation)
                env.spec, env.calibration_source = spec, path
                env.native_repo = repo
                env.native_source_hashes = {}
                env.task = SimpleNamespace(var2target_state_list=None)
                log = []
                def to_demo(demo):
                    if spec.preload_before_reset_to_demo: assert env.task.var2target_state_list == archive
                    log.append('reset_to_demo')
                    env.task.var2target_state_list = {'randomized': True}
                    return ['native description'], None
                def init(*args, **kwargs):
                    assert env.task.var2target_state_list == archive
                    assert args == (var,) and kwargs == dict(spec.init_episode_kwargs)
                    log.append('init_episode')
                env.task.init_episode = init
                env.task.disable_expert_plan = lambda: log.append('disable')
                env.task_environment = SimpleNamespace(set_variation=lambda v: log.append(('variation', v)),
                    reset_to_demo=to_demo, get_observation=lambda: log.append('observation'))
                env.pyrep = SimpleNamespace(step=lambda: log.append('physics_staging'))
                env._start_alignment = lambda: log.append('alignment')
                env._reset_metric_receipt = lambda value: setattr(env, 'reset_receipt', value)
                env.observe = lambda: ({'current': True}, None)
                row = dict(task=spec.task_name, raw_episode_id='episode99', path=str(path),
                    variation=var, official_episode_index=0, official_alias_path=str(alias), mapping_status='verified',
                    source_required_files=required, target_state_sha256=required['target_state.pkl']['sha256'],
                    task_source_path=str(source), task_source_sha256=file_sha256(source),
                    task_scene_path=str(scene), task_scene_sha256=file_sha256(scene))
                descriptions, observation = env.reset_official(row)
                assert descriptions == ['native description'] and observation == {'current': True}
                assert log[-4:] == ['physics_staging', 'observation', 'disable', 'alignment']
                assert env.reset_receipt['original_source_replayed'] is True
                assert len(env.reset_receipt['verified_source_inputs']['source_required_sha256']) == 19
                broken = dict(row);broken.pop('source_required_files')
                with self.assertRaises(MultitaskProtocolError): verify_official_source_files(broken, spec, repo)
                broken = deepcopy(row);broken['source_required_files'].pop('target_state.pkl')
                with self.assertRaises(MultitaskProtocolError): verify_official_source_files(broken, spec, repo)
                scene.write_bytes(b'changed scene')
                with self.assertRaises(MultitaskProtocolError): verify_official_source_files(row, spec, repo)

    def test_generic_observation_uses_four_current_cameras_and_no_native_outcomes(self):
        env = runtime(fatal_tick=None)
        del env.observe
        env._native_success_latched = True
        rgb = np.full((256, 256, 3), .501, dtype=np.float32)
        env.cameras = {name: SimpleNamespace(handle_explicitly=Mock(), capture_rgb=lambda: rgb.copy(),
            capture_depth=lambda **kw: np.full((256, 256), .8),
            get_render_mode=lambda: SimpleNamespace(name='OPENGL')) for name in ('front', '8', '16', '36')}
        env.mask_cameras = {name: SimpleNamespace(handle_explicitly=Mock(), capture_rgb=lambda: rgb.copy())
                            for name in env.cameras}
        env.calibrations = {name: CameraCalibration(np.eye(3), Pose(np.zeros(3))) for name in env.cameras}
        observation, robot = env.observe()
        assert robot.task_success is False and set(observation['observation/images']) == set(env.cameras)
        assert not any('outcome' in key or 'goal' in key or 'target' in key for key in observation)
        assert observation['observation/images']['front'][0, 0, 0] == 128
        assert observation['observation/raw_rgb_255']['front'][0, 0, 0] == 127
        np.testing.assert_allclose(observation['observation/raw_rgb_255']['8'][0, 0], .501 * 255)
        env.hold_tick();env.arm.q[:3] = [.01, .02, .03]
        _, first = env.observe();_, duplicate = env.observe()
        np.testing.assert_allclose(first.linear_velocity, [.2, .4, .6])
        np.testing.assert_array_equal(duplicate.linear_velocity, first.linear_velocity)
        assert env.ledger.ticks == 1

    def test_development_reset_explicitly_declares_synthetic_not_source_motion_replay(self):
        env = object.__new__(AlignedDevelopmentSimulation)
        env.spec = get_task_spec('push_moving_button')
        env.seed, env.replica, env.development_seed_identity = 12, 0, {'version': 'fixture'}
        env.development_source = dict(raw_episode_id='episode1', variation=1)
        env.development_split = 'validation'
        with tempfile.TemporaryDirectory() as directory:
            env.calibration_source = Path(directory) / 'source.h5';env.calibration_source.write_bytes(b'fixture')
            env.task_environment = SimpleNamespace(set_variation=Mock(), reset=lambda: (['generated'], None),
                                                   get_observation=Mock())
            env.task = SimpleNamespace(disable_expert_plan=Mock())
            env.pyrep = SimpleNamespace(step=Mock())
            env._start_alignment = Mock()
            env._reset_metric_receipt = lambda value: setattr(env, 'reset_receipt', value)
            env.observe = lambda: ({}, None)
            old_state = np.random.get_state()
            try: descriptions, _ = env.reset_development()
            finally: np.random.set_state(old_state)
        assert descriptions == ['generated']
        assert env.reset_receipt['original_source_replayed'] is False
        assert env.reset_receipt['source_motion_or_expert_trajectory_replayed'] is False
        assert env.reset_receipt['official_TEST'] is False
        assert env.reset_receipt['reset_kind'] == 'native_generated_synthetic_development_augmentation'
        env.task_environment.set_variation.assert_called_once_with(1)


class MapTests(unittest.TestCase):
    def make_map(self):
        parts = {name: Shape(name, forbid_pose=True) for name in ('rubbish', 'bin', 'tomato1', 'tomato2', 'bin-child')}
        parts['bin'].children = [parts['bin-child']]
        table, arm, gripper = Shape('table'), Arm(), Shape('gripper')
        env = SimpleNamespace(spec=get_task_spec('put_rubbish_in_moving_bin'), physics_time=lambda: 0.,
            task=SimpleNamespace(t=0., rubbish=parts['rubbish'], bin=parts['bin'], get_graspable_objects=lambda: [parts['rubbish']]),
            arm=arm, gripper=gripper, resolve_scene_object=lambda name: parts.get(name, Shape('sensor')))
        env.pyrep = SimpleNamespace(get_objects_in_tree=lambda **kw: [table, arm, gripper, *parts.values()])
        ghosts = []
        def factory(*args):
            ghost = Shape('ghost' + str(len(ghosts)));ghosts.append(ghost);return ghost
        service = SharedMultiObjectMap(env, shape_type='shape', ghost_factory=factory)
        return env, service, parts, table, ghosts

    def test_all_dynamic_roles_and_descendants_excluded_without_live_pose_reads(self):
        env, service, parts, table, ghosts = self.make_map()
        assert [entry['name'] for entry in service.map] == ['table']
        assert service.receipt()['excluded_native_dynamic_shapes'] == 5
        env.arm.check_arm_collision = lambda: any(ghost.collidable for ghost in ghosts)
        assert service.check()
        assert all(body.collidable for body in parts.values()) and table.collidable
        assert all(not ghost.collidable for ghost in ghosts)

    def test_predicted_ghost_transform_and_flags_restored_after_query_error(self):
        env, service, parts, table, ghosts = self.make_map()
        vertices, indices = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], [[0, 1, 2]]
        service.register_estimated_geometry('observed-bin', vertices, indices, provenance='current_rgbd')
        def failure():
            assert all(not body.collidable for body in parts.values()) and not table.collidable
            np.testing.assert_array_equal(ghosts[-1].matrix[:3, 3], [.2, .3, .4])
            raise RuntimeError('collision backend error')
        env.arm.check_arm_collision = failure
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'collision backend failed'):
            service.check({'observed-bin': Pose(np.array([.2, .3, .4]))})
        assert all(body.collidable for body in parts.values()) and table.collidable
        assert all(not ghost.collidable for ghost in ghosts)
        with self.assertRaises(MultitaskProtocolError):
            service.register_estimated_geometry('oracle', vertices, indices, provenance='native_live_mesh')
        with self.assertRaises(MultitaskProtocolError):
            service.register_estimated_geometry('cad', vertices, indices, provenance='shared_CAD_template')

    def test_unknown_geometry_forecast_fails_closed(self):
        env, service, *_ = self.make_map()
        forecast = Forecast(('unknown',), np.array([0., .05]), np.zeros((2, 1, 3)),
            np.broadcast_to(np.eye(3), (2, 1, 3, 3)), np.zeros((2, 1, 3)), np.zeros((2, 1, 3)))
        collision = KnownMapForecastCollision(service)
        collision.bind(forecast, 0.)
        assert collision.check(.025) is True
        # Omitting an unobserved physical body from the supplied forecast must
        # not silently remove that body from collision screening.
        service.register_estimated_geometry('bin', [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]],
            [[0, 1, 2]], provenance='current_rgbd')
        only_bin = Forecast(('bin',), forecast.times, forecast.positions, forecast.rotations,
                            forecast.linear_velocities, forecast.angular_velocities)
        collision.bind(only_bin, 0., body_ids=('bin',))
        assert collision.check(.025) is True

    def test_noncolliding_train_role_needs_no_invented_ghost_or_forecast(self):
        env, service, *_ = self.make_map()
        vertices, indices = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], [[0, 1, 2]]
        with self.assertRaises(MultitaskProtocolError):
            service.declare_noncolliding_geometry('bin', source_sha256='bad')
        service.declare_noncolliding_geometry('bin', source_sha256='a' * 64)
        service.declare_noncolliding_geometry('rubbish', source_sha256='b' * 64)
        for identity in set(service.required_dynamic_role_ids) - {'bin', 'rubbish'}:
            service.declare_noncolliding_geometry(identity, source_sha256='b' * 64)
        assert not service.estimated_bodies
        with self.assertRaises(MultitaskProtocolError):
            service.register_estimated_geometry('bin', vertices, indices, provenance='current_rgbd')
        with self.assertRaises(MultitaskProtocolError):
            service.declare_noncolliding_geometry('bin', source_sha256='c' * 64)
        forecast = Forecast((), np.array([0., .05]), np.zeros((2, 0, 3)),
            np.zeros((2, 0, 3, 3)), np.zeros((2, 0, 3)), np.zeros((2, 0, 3)))
        env.arm.check_arm_collision = lambda: False
        collision = KnownMapForecastCollision(service)
        collision.bind(forecast, 0.)
        assert not collision.body_ids and not collision.check(.025)
        assert service.receipt()['noncolliding_geometry']['bin']['source_sha256'] == 'a' * 64

    def test_respondable_only_static_geometry_is_preserved(self):
        env, old, parts, table, _ = self.make_map()
        old.close()
        table.collidable, table.respondable = False, True
        ghosts = []
        def factory(*args):
            ghost = Shape('ghost'); ghosts.append(ghost); return ghost
        service = SharedMultiObjectMap(env, shape_type='shape', ghost_factory=factory)
        assert [row['name'] for row in service.map] == ['table']
        assert service.map[0]['physical_component_flags'] == dict(collidable=False, respondable=True)
        env.arm.check_arm_collision = lambda: any(ghost.collidable for ghost in ghosts)
        assert service.check() and not table.collidable and table.respondable

    def test_uncertainty_receipt_binds_mesh_source_and_is_not_mutable_by_caller(self):
        import hashlib
        env, service, *_ = self.make_map()
        vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]],np.float32)
        triangles=np.array([[0,1,2]],np.int32)
        checksum=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
        receipt=dict(TRAIN_asset_sha256='a'*64,registered_mesh_sha256=checksum,mode='uncertain')
        service.register_estimated_geometry('x',vertices,triangles,provenance='TRAIN_geometry_model',
            source_sha256='a'*64,collision_geometry_receipt=receipt)
        receipt['mode']='tampered'
        assert service.receipt()['estimated_geometry']['x']['collision_geometry_receipt']['mode']=='uncertain'
        with self.assertRaisesRegex(MultitaskProtocolError,'declaration changed'):
            service.register_estimated_geometry('x',vertices,triangles,provenance='TRAIN_geometry_model',
                source_sha256='a'*64,collision_geometry_receipt=receipt)
        receipt['registered_mesh_sha256']='b'*64
        with self.assertRaisesRegex(MultitaskProtocolError,'registered mesh'):
            service.register_estimated_geometry('y',vertices,triangles,provenance='TRAIN_geometry_model',
                source_sha256='a'*64,collision_geometry_receipt=receipt)

    def test_held_ghost_uses_only_estimated_attachment_and_checks_world_sweeps(self):
        env, service, parts, table, ghosts = self.make_map()
        vertices, triangles = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], [[0, 1, 2]]
        service.register_estimated_geometry('held', vertices, triangles, provenance='current_rgbd')
        service.bind_estimated_attachment('held', Pose(np.array([.2, 0., 0.])),
                                         provenance='current_rgbd_robot_proprioception')
        held = service.estimated_bodies['held'].ghost
        env.arm.q[:3] = [.3, .1, .8]
        def arm_collision():
            assert held.collidable is False
            return False
        def body_collision(other):
            np.testing.assert_allclose(held.matrix[:3, 3], [.5, .1, .8])
            return other is ghosts[0]
        env.arm.check_arm_collision, held.check_collision = arm_collision, body_collision
        assert service.check(ignored_object_ids=('held',)) is True
        assert all(body.collidable for body in parts.values()) and table.collidable
        assert all(not ghost.collidable for ghost in ghosts)
        with self.assertRaises(MultitaskProtocolError):
            service.bind_estimated_attachment('held', Pose(np.zeros(3)), provenance='native_grasp_parent')
        service.bind_estimated_attachment('held', None, provenance='current_rgbd_robot_proprioception')
        assert not service.estimated_attachments

    def test_allowed_robot_contact_target_remains_solid_to_held_body(self):
        env, service, parts, table, ghosts = self.make_map()
        vertices, triangles = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], [[0, 1, 2]]
        for identity in ('held-ring', 'selected-peg'):
            service.register_estimated_geometry(identity, vertices, triangles, provenance='current_rgbd')
        service.bind_estimated_attachment('held-ring', Pose(np.zeros(3)),
                                         provenance='current_rgbd_robot_proprioception')
        held, peg = (service.estimated_bodies[key].ghost for key in ('held-ring', 'selected-peg'))
        def arm_collision():
            assert held.collidable is False and peg.collidable is False
            return False
        def held_collision(other):
            assert held.collidable is True and peg.collidable is True
            return other is peg
        env.arm.check_arm_collision, held.check_collision = arm_collision, held_collision
        assert service.check({'selected-peg': Pose(np.array([.2, 0., .8]))},
                             allowed_robot_contact_ids=('selected-peg',)) is True
        assert all(not ghost.collidable for ghost in ghosts)
        assert all(body.collidable for body in parts.values()) and table.collidable
        assert service.receipt()['allowed_robot_contact_ids_seen'] == ['selected-peg']
        with self.assertRaises(MultitaskProtocolError):
            service.check({'selected-peg': Pose(np.zeros(3))}, ignored_object_ids=('selected-peg',),
                          allowed_robot_contact_ids=('selected-peg',))

    def test_kinematic_probe_restores_joints_targets_and_ik_even_on_error(self):
        env, service, *_ = self.make_map()
        model = object.__new__(AlignedRobotModel);model.arm, model.service = env.arm, service
        model.lower,model.upper=np.full(7,-3.),np.full(7,3.)
        q, targets = env.arm.q.copy(), env.arm.targets.copy()
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'kinematic backend probe failed'):
            model._probe(np.ones(7), lambda: (_ for _ in ()).throw(RuntimeError('IK failure')))
        np.testing.assert_array_equal(env.arm.q, q)
        np.testing.assert_array_equal(env.arm.targets, targets)

    def test_kinematic_probe_accepts_quaternion_sign_but_rejects_rotation_change(self):
        env, service, *_ = self.make_map()
        model = object.__new__(AlignedRobotModel);model.arm, model.service = env.arm, service
        model.lower,model.upper=np.full(7,-3.),np.full(7,3.)
        original_set = env.arm._ik_target.set_pose
        def equivalent(value):
            value = list(value);value[3:] = [-x for x in value[3:]];original_set(value)
        env.arm._ik_target.set_pose = equivalent
        self.assertEqual(model._probe(np.ones(7), lambda: 42), 42)
        def changed(value):
            value = list(value);value[3:] = [.001, 0., 0., (1.-1e-6)**.5];original_set(value)
        env.arm._ik_target.set_pose = changed
        with self.assertRaisesRegex(MultitaskProtocolError, 'failed to restore robot state'):
            model._probe(np.ones(7), lambda: 42)

    def test_out_of_interval_actual_joint_is_not_mutated_by_probe(self):
        env,service,*_=self.make_map()
        model=object.__new__(AlignedRobotModel);model.arm,model.service=env.arm,service
        model.lower,model.upper=np.full(7,-3.),np.full(7,3.)
        env.arm.q[4]=-3.0002
        before=env.arm.q.copy();targets=env.arm.targets.copy()
        callback=Mock()
        with self.assertRaisesRegex(MultitaskProtocolError,'probe unavailable'):
            model._probe(before,callback)
        self.assertFalse(callback.called)
        self.assertIsNone(model.inverse(Pose(np.zeros(3)),before))
        self.assertTrue(model.collision(np.zeros(7),0.))
        self.assertEqual(model.forward(before).position.tolist(),before[:3].tolist())
        np.testing.assert_array_equal(env.arm.q,before)
        np.testing.assert_array_equal(env.arm.targets,targets)


class TerminalExecutorTests(unittest.TestCase):
    def test_v7_nonterminal_decision_keeps_all_ten_counted_ticks_and_helper_cleanup(self):
        env = runtime(fatal_tick=None)
        constants, errors = ModuleType('pyrep.const'), ModuleType('pyrep.errors')
        constants.ConfigurationPathAlgorithms = SimpleNamespace(RRTConnect=1)
        errors.ConfigurationPathError = type('ConfigurationPathError', (Exception,), {})
        errors.IKError = type('IKError', (Exception,), {})
        adapter = object.__new__(executor.PhysManiAdapter)
        adapter.config = {'baseline_decision_ticks': 10}
        adapter.observe, observer = Mock(), Mock()
        previous = frozen_arm._command_with_clipping_receipt
        with patch.dict(sys.modules, {'pyrep.const': constants, 'pyrep.errors': errors}), \
                patch.object(executor, 'physics_time', side_effect=env.physics_time), \
                patch.object(frozen_arm, 'physics_time', side_effect=env.physics_time):
            result = adapter.execute(env, [0., 0., 1., 0., 0., 0., 1., 0.], observer=observer)
            assert result['ticks'] == env.ledger.ticks == 10
            assert result['partial_decision_due_to_native_terminal'] is False
            assert observer.call_count == adapter.observe.call_count == 10
            assert env.ledger.phases == {'physmani_path': 9, 'physmani_gripper': 1}
            assert env.ledger.grasp_requests == 1 and env.ledger.time == .5
            observer.side_effect = MultitaskInfrastructureError('observer state corruption')
            with self.assertRaises(MultitaskInfrastructureError):
                adapter.execute(env, [0., 0., 1., 0., 0., 0., 1., 0.], observer=observer)
        assert frozen_arm._command_with_clipping_receipt is previous

    def test_v7_mro_keeps_v6_scope_and_stops_at_first_fatal_with_partial_receipts(self):
        env = runtime(fatal_tick=2)
        constants, errors = ModuleType('pyrep.const'), ModuleType('pyrep.errors')
        constants.ConfigurationPathAlgorithms = SimpleNamespace(RRTConnect=1)
        errors.ConfigurationPathError = type('ConfigurationPathError', (Exception,), {})
        errors.IKError = type('IKError', (Exception,), {})
        adapter = object.__new__(executor.PhysManiAdapter)
        adapter.config = {'baseline_decision_ticks': 10}
        adapter.observe, observer = Mock(), Mock()
        previous = frozen_arm._command_with_clipping_receipt
        with patch.dict(sys.modules, {'pyrep.const': constants, 'pyrep.errors': errors}), \
                patch.object(executor, 'physics_time', side_effect=env.physics_time), \
                patch.object(frozen_arm, 'physics_time', side_effect=env.physics_time):
            result = adapter.execute(env, [0., 0., 1., 0., 0., 0., 1., 0.], observer=observer)
        assert result['executor'] == executor.EXECUTOR
        assert result['ticks'] == result['actual_speed_receipt']['command_ticks'] == 2
        assert result['native_terminal_failure_latched'] is True and result['native_success_latched'] is False
        assert result['partial_decision_due_to_native_terminal'] is True
        assert result['common_joint_command_clipping']['velocity_clip_ticks'] == 0
        assert result['counterfactual_ours_reference_clip']['velocity_clip_ticks'] == 2
        assert env.ledger.ticks == 2 and env._clock == .1
        assert observer.call_count == 2 and adapter.observe.call_count == 1
        assert frozen_arm._command_with_clipping_receipt is previous
        with self.assertRaises(EpisodeTerminatedError):
            adapter.execute(env, [0., 0., 1., 0., 0., 0., 1., 0.])


if __name__ == '__main__':
    unittest.main()
