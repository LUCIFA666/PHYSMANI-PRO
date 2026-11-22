"""Fixed TRAIN collision variants: conservative defaults and world contacts.

The fake native backend records which actual mesh is active. These checks do
not infer correctness from a receipt alone or require a physics/GPU process.
"""
from copy import deepcopy
import warnings
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask.collision_geometry import (
    collision_geometry_for_orientation, mesh_sha256,
)
from our_mani.multitask.protocol import (
    MultitaskInfrastructureError, MultitaskProtocolError,
)
import test_multitask_runtime as native_fixtures


class MapVariantTests(unittest.TestCase):
    def setUp(self):
        self.env, self.service, self.parts, self.table, self.ghosts = (
            native_fixtures.MapTests().make_map())
        self.clock = [7.25]
        self.env.physics_time = lambda: self.clock[0]
        self.env.task.t = 3.5
        # Restoration must preserve false flags as well as true ones.
        self.parts['bin-child'].collidable = False
        self.native_flags = {obj.get_handle(): obj.is_collidable()
                             for obj in [self.table, *self.parts.values()]}
        self.source = 'a' * 64
        self.vertices = np.asarray([[0., 0., 0.], [.1, 0., 0.], [0., .08, .02]], np.float32)
        self.triangles = np.asarray([[0, 1, 2]], np.int32)

    def tearDown(self):
        self.service.close()

    def geometry(self):
        primary = collision_geometry_for_orientation(self.vertices, self.triangles,
                                                     orientation_observability='full')
        fallback = collision_geometry_for_orientation(self.vertices, self.triangles,
                                                      orientation_observability='none')
        primary_receipt = dict(primary.receipt, TRAIN_asset_sha256=self.source)
        fallback_mesh = dict(vertices=fallback.vertices, triangles=fallback.triangles,
                             collision_geometry_receipt=dict(fallback.receipt,
                                                             TRAIN_asset_sha256=self.source))
        return primary_receipt, fallback_mesh

    def register(self, identity='body'):
        receipt, fallback = self.geometry()
        self.service.register_estimated_geometry(identity, self.vertices, self.triangles,
            provenance='TRAIN_geometry_model', source_sha256=self.source,
            collision_geometry_receipt=receipt, fallback_mesh=fallback)
        return self.service.estimated_bodies[identity]

    def assert_restored(self):
        for obj in [self.table, *self.parts.values()]:
            self.assertEqual(obj.is_collidable(), self.native_flags[obj.get_handle()])
        self.assertTrue(all(not obj.is_collidable() for obj in self.ghosts))
        self.assertEqual((self.env.physics_time(), self.env.task.t), (7.25, 3.5))

    def test_default_activates_only_fallback_and_primary_requires_explicit_selection(self):
        body = self.register()
        pose = Pose(np.asarray([.2, -.3, .8]), Rotation.from_euler('z', .4).as_matrix())
        for modes, selected, inactive in ((None, body.fallback_ghost, body.ghost),
                ({'body': 'primary'}, body.ghost, body.fallback_ghost)):
            with self.service.public_scene({'body': pose}, geometry_modes=modes) as active:
                self.assertIs(active['body'], selected)
                self.assertTrue(selected.is_collidable())
                self.assertFalse(inactive.is_collidable())
                np.testing.assert_allclose(selected.matrix[:3, 3], pose.position)
                np.testing.assert_allclose(selected.matrix[:3, :3], pose.rotation)
                self.assertTrue(all(not x.is_collidable() for x in self.parts.values()))
            self.assert_restored()
        self.assertEqual(self.service.receipt()['geometry_mode_queries'],
                         {'rotation_uncertainty': 1, 'primary': 1})

    def test_legacy_body_without_fallback_keeps_primary_and_rejects_missing_variant(self):
        self.service.register_estimated_geometry('legacy', self.vertices, self.triangles,
                                                provenance='current_rgbd')
        body = self.service.estimated_bodies['legacy']
        self.assertIsNone(body.fallback_ghost)
        with self.service.public_scene({'legacy': Pose(np.zeros(3))}) as active:
            self.assertIs(active['legacy'], body.ghost)
        with self.assertRaisesRegex(MultitaskProtocolError, 'no registered rotation uncertainty fallback'):
            with self.service.public_scene({'legacy': Pose(np.zeros(3))},
                    geometry_modes={'legacy': 'rotation_uncertainty'}):
                self.fail('nonexistent mesh was accepted')
        self.assert_restored()

    def test_invalid_mode_or_unknown_body_does_not_query_backend(self):
        self.register()
        calls = []
        self.env.arm.check_arm_collision = lambda: calls.append(True)
        for modes in ({'body': 'native'}, {'missing': 'primary'}, {'body': None}):
            with self.subTest(modes=modes), self.assertRaises(MultitaskProtocolError):
                self.service.check({'body': Pose(np.zeros(3))}, geometry_modes=modes)
            self.assert_restored()
        with self.assertRaisesRegex(MultitaskProtocolError, 'geometry missing'):
            self.service.check({'unknown': Pose(np.zeros(3))})
        self.assertTrue(self.service.check(allowed_robot_contact_ids=('unknown',)))
        self.assertEqual(calls, [])
        self.assert_restored()

    def test_bad_fallback_receipts_and_provenance_are_rejected_before_mesh_creation(self):
        receipt, fallback = self.geometry()
        bad = []
        for key, value in (('TRAIN_asset_sha256', 'b' * 64),
                ('registered_mesh_sha256', 'b' * 64),
                ('collision_geometry_kind', 'exact_TRAIN_triangles'),
                ('native_pose_or_future_state_used', True),
                ('native_pose_or_future_state_used', None),
                ('extra_nonfinite', float('nan'))):
            item = deepcopy(fallback); item['collision_geometry_receipt'][key] = value
            bad.append(item)
        item = deepcopy(fallback); item['collision_geometry_receipt'] = None; bad.append(item)
        item = deepcopy(fallback); item['unexpected'] = True; bad.append(item)
        initial_ghosts = len(self.ghosts)
        for i, item in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(MultitaskProtocolError):
                self.service.register_estimated_geometry('body', self.vertices, self.triangles,
                    provenance='TRAIN_geometry_model', source_sha256=self.source,
                    collision_geometry_receipt=receipt, fallback_mesh=item)
            self.assertEqual(len(self.ghosts), initial_ghosts)
            self.assertFalse(self.service.estimated_bodies)
        with self.assertRaises(MultitaskProtocolError):
            self.service.register_estimated_geometry('body', self.vertices, self.triangles,
                provenance='current_rgbd', fallback_mesh=fallback)
        self.assert_restored()

    def test_finite_float64_fallback_that_overflows_native_float32_is_rejected(self):
        receipt, fallback = self.geometry()
        overflow = np.asarray(fallback['vertices'], dtype=np.float64)
        overflow[0, 0] = 1e100
        fallback['vertices'] = overflow
        # A caller-supplied checksum cannot make an infinite native mesh valid.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fallback['collision_geometry_receipt']['registered_mesh_sha256'] = mesh_sha256(
                overflow, fallback['triangles'])
            with self.assertRaises(MultitaskProtocolError):
                self.service.register_estimated_geometry('body', self.vertices, self.triangles,
                    provenance='TRAIN_geometry_model', source_sha256=self.source,
                    collision_geometry_receipt=receipt, fallback_mesh=fallback)
        self.assertFalse(self.service.estimated_bodies)
        self.assert_restored()

    def test_receipts_are_copied_and_registered_fallback_cannot_be_redeclared(self):
        receipt, fallback = self.geometry()
        saved = deepcopy(fallback['collision_geometry_receipt'])
        self.service.register_estimated_geometry('body', self.vertices, self.triangles,
            provenance='TRAIN_geometry_model', source_sha256=self.source,
            collision_geometry_receipt=receipt, fallback_mesh=fallback)
        receipt['construction_rule'] = 'changed caller receipt'
        fallback['collision_geometry_receipt']['construction_rule'] = 'changed caller receipt'
        self.assertEqual(self.service.estimated_bodies['body'].fallback_geometry_receipt, saved)
        original_receipt, _ = self.geometry()
        with self.assertRaisesRegex(MultitaskProtocolError, 'fallback collision geometry changed'):
            self.service.register_estimated_geometry('body', self.vertices, self.triangles,
                provenance='TRAIN_geometry_model', source_sha256=self.source,
                collision_geometry_receipt=original_receipt, fallback_mesh=fallback)
        self.assert_restored()

    def test_held_fallback_and_allowed_target_still_collide_in_world(self):
        held, target = self.register('held'), self.register('target')
        self.service.bind_estimated_attachment('held', Pose(np.asarray([.2, 0., 0.])),
            provenance='current_rgbd_robot_proprioception')
        self.env.arm.q[:3] = [.3, .1, .8]
        expected = [held.fallback_ghost, target.fallback_ghost]
        calls = []
        def arm_collision():
            self.assertTrue(all(not ghost.collidable for ghost in expected))
            self.assertFalse(held.ghost.collidable)
            return False
        def held_collision(other):
            self.assertTrue(all(ghost.collidable for ghost in expected))
            np.testing.assert_allclose(held.fallback_ghost.matrix[:3, 3], [.5, .1, .8])
            calls.append(other.get_handle())
            return other is expected[1]
        self.env.arm.check_arm_collision = arm_collision
        held.fallback_ghost.check_collision = held_collision
        # An ignored held object must be restored by attachment handling.
        self.assertTrue(self.service.check({'target': Pose(np.asarray([.5, .1, .8]))},
            ignored_object_ids=('held',), allowed_robot_contact_ids=('target',)))
        self.assertIn(target.fallback_ghost.get_handle(), calls)
        self.assert_restored()
        expected[1] = target.ghost
        self.assertTrue(self.service.check({'target': Pose(np.asarray([.5, .1, .8]))},
            allowed_robot_contact_ids=('target',), geometry_modes={'target': 'primary'}))
        self.assert_restored()
        with self.assertRaisesRegex(MultitaskProtocolError, 'cannot remove that obstacle'):
            self.service.check({'target': Pose(np.zeros(3))}, ignored_object_ids=('target',),
                               allowed_robot_contact_ids=('target',))
        self.assert_restored()

    def test_arm_and_held_backend_exceptions_restore_every_native_and_variant_flag(self):
        held, target = self.register('held'), self.register('target')
        self.service.bind_estimated_attachment('held', Pose(np.zeros(3)),
            provenance='current_rgbd_robot_proprioception')
        def failure(): raise RuntimeError('native arm failure')
        self.env.arm.check_arm_collision = failure
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'collision backend failed'):
            self.service.check({'target': Pose(np.zeros(3))},
                               allowed_robot_contact_ids=('target',))
        self.assert_restored()
        self.env.arm.check_arm_collision = lambda: False
        def body_failure(other): raise RuntimeError('native held-body failure')
        held.fallback_ghost.check_collision = body_failure
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'collision backend failed'):
            self.service.check({'target': Pose(np.zeros(3))},
                               allowed_robot_contact_ids=('target',))
        self.assert_restored()

    def test_variant_pose_setter_exception_restores_all_flags(self):
        body = self.register()
        def failure(matrix): raise RuntimeError('native pose setter failure')
        body.fallback_ghost.set_matrix = failure
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'collision backend failed'):
            self.service.check({'body': Pose(np.zeros(3))})
        self.assert_restored()

    def test_close_removes_static_primary_fallback_and_continues_after_backend_error(self):
        body = self.register()
        all_ghosts = list(self.ghosts)
        def failure(): raise RuntimeError('first native remove failure')
        all_ghosts[0].remove = failure
        with self.assertRaisesRegex(MultitaskInfrastructureError, 'ghost cleanup failed'):
            self.service.close()
        self.assertTrue(body.ghost.removed and body.fallback_ghost.removed)
        self.assertTrue(all(x.removed for x in all_ghosts[1:]))
        self.assertFalse(self.service.static_ghosts or self.service.estimated_bodies)
        self.assertEqual((self.env.physics_time(), self.env.task.t), (7.25, 3.5))


if __name__ == '__main__':
    unittest.main()
