"""Synthetic buffer/visibility transaction tests, not Native performance tests."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from our_mani.local_mpc_20261103 import depth_owner_capture_sensors as d
from our_mani.local_mpc_20261103 import perception_capture_sensors as old
from tests.test_near_goal_perception_capture_sensors_20261103 import FakeCamera, FakeShape, FakeSim


class Robot:
    def __init__(self, size):
        self.q = np.zeros(size)
        self.v = np.zeros(size)
        self.target = np.zeros(size)
        self.target_v = np.zeros(size)
    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.v.copy()
    def get_joint_target_positions(self): return self.target.copy()
    def get_joint_target_velocities(self): return self.target_v.copy()


class VisibilitySim(FakeSim):
    def __init__(self):
        super().__init__()
        self.flags = {88: 513, 83: 530}
        self.writes = []
        self.on_set = None
    def simGetObjectSpecialProperty(self, identity): return self.flags[identity]
    def simSetObjectSpecialProperty(self, identity, value):
        self.writes.append((identity, value))
        self.flags[identity] = value
        for shape in FakeShape.registry.values():
            if shape.identity == identity: shape.renderable = bool(value & 512)
        if self.on_set: self.on_set(identity, value)


class DepthOwnerTests(unittest.TestCase):
    def setUp(self):
        FakeCamera.created, FakeCamera.world, FakeCamera.fail_create = [], 0, None
        FakeCamera.mask_shift, FakeCamera.gl3_depth_offset = False, 0.
        FakeShape.registry = {}
        self.sim = VisibilitySim()
        self.cube, self.goal = FakeShape('pick_and_lift_target', 88), FakeShape('success_visual', 83)
        self.world = dict(cube_present=True, goal_depth=True, negative=False, conflict=False, tint=0)
        self.clock = [1.75, 0., 0, 0]
        self.backend = SimpleNamespace(task=SimpleNamespace(target_block=self.cube), arm=Robot(7), gripper=Robot(2),
            _clock_tuple=lambda: tuple(self.clock), clock=lambda: 0.)
        world, sim = self.world, self.sim
        def render(camera):
            rgb = np.full((256, 256, 3), 30+world['tint'], dtype=np.float32)
            depth = np.full((256, 256), 5., dtype=np.float32)
            cube = bool(sim.flags[88] & 512) and world['cube_present']
            goal = bool(sim.flags[83] & 512)
            if cube:
                rgb[80:160, 80:160] = np.array([160, 0, 0])+world['tint']
                depth[80:160, 80:160] = 1.
            if goal:
                rgb[20:60, 20:60] = np.array([150, 10, 10])+world['tint']
                if world['goal_depth']: depth[20:60, 20:60] = 2.
            if not cube and world['negative']: depth[180:184, 180:184] = 4.
            if not goal and world['conflict'] and cube: depth[80:160, 80:160] = 2.
            camera.rgb, camera.depth = rgb/255., depth
        self.render_patch = patch.object(FakeCamera, 'refresh', render)
        self.render_patch.start(); self.addCleanup(self.render_patch.stop)
        originals = {n: FakeCamera(mode=3 if n == 'front' else 1, explicit=i % 2) for i, n in enumerate(d.CAMERAS)}
        self.backend.raw = SimpleNamespace(cameras=originals)
        api = (self.sim, SimpleNamespace(OPENGL=1, OPENGL_COLOR_CODED=2, OPENGL3=3), FakeShape, FakeCamera)
        self.api = patch.object(old, '_load_native_api', return_value=api)
        self.api.start(); self.addCleanup(self.api.stop)

    def bundle(self):
        b = d.create_depth_owner_cameras(self.backend)
        self.addCleanup(b.close)
        return b

    def test_depth_ownership_public_label_isolation_and_fixed_no_step_warmups(self):
        b = self.bundle()
        original_flags = [c.explicit for c in self.backend.raw.cameras.values()]
        public, labels, receipt = b.capture()
        self.assertEqual(set(public), d.PUBLIC_ARRAY_KEYS)
        self.assertEqual(labels['semantic_role_ids'].dtype, np.uint8)
        np.testing.assert_array_equal(labels['cube_depth_owner'], labels['semantic_role_ids'] == 1)
        self.assertTrue(labels['cube_depth_owner'][:, 100, 100].all())
        self.assertTrue(labels['goal_depth_owner'][:, 40, 40].all())
        self.assertTrue((labels['semantic_role_ids'][:, 80, 100] == 255).all())
        self.assertTrue((labels['semantic_role_ids'][:, 79, 100] == 255).all())
        self.assertTrue((labels['semantic_role_ids'][:, 0, 0] == 0).all())
        self.assertTrue(all(receipt[k] is True for k in ('original_clone_equal', 'all_counterfactual_flags_restored',
            'all_restore_RGBD_equal', 'clock_unchanged', 'handling_restored', 'global_vision_sensors_enabled', 'created_nonpassive')))
        self.assertEqual(receipt['label_method'], d.LABEL_METHOD)
        self.assertEqual(receipt['semantic_kind'], 'current_measured_depth_owner')
        self.assertEqual([r['rounds'] for r in receipt['warmups']], [12, 12, 12])
        self.assertEqual(receipt['render_call_count'], 320)
        self.assertEqual(receipt['completed_render_call_count'], 320)
        self.assertEqual(original_flags, [c.explicit for c in self.backend.raw.cameras.values()])
        self.assertEqual(self.sim.flags, {88: 513, 83: 530})
        self.assertEqual(self.sim.writes, [(88, 1), (88, 513), (83, 18), (83, 530)])
        self.assertEqual(len(FakeCamera.created), 4)
        self.assertFalse(any(c.mode == 2 for c in FakeCamera.created))
        self.assertFalse(any('instance' in key for key in labels))
        self.assertEqual(self.clock, [1.75, 0., 0, 0])
        self.assertGreaterEqual(receipt['wall_seconds'], 0.)

    def test_cube_absent_is_not_hallucinated_and_transparent_rgb_only_goal_is_unknown(self):
        self.world.update(cube_present=False, goal_depth=False)
        _, labels, receipt = self.bundle().capture()
        self.assertFalse(labels['cube_depth_owner'].any())
        self.assertFalse(labels['goal_depth_owner'].any())
        self.assertTrue(labels['goal_rgb_only_unknown'][:, 30, 30].all())
        self.assertTrue((labels['semantic_role_ids'][:, 30, 30] == 255).all())
        self.assertTrue(receipt['all_restore_RGBD_equal'])

    def test_negative_counterfactual_and_conflicting_owners_never_become_positives(self):
        self.world.update(negative=True, conflict=True)
        _, labels, _ = self.bundle().capture()
        self.assertTrue(labels['cube_negative_delta'][:, 181, 181].all())
        self.assertTrue((labels['semantic_role_ids'][:, 181, 181] == 255).all())
        self.assertTrue(labels['conflicting_depth_owner'][:, 100, 100].all())
        self.assertFalse(labels['cube_depth_owner'][:, 100, 100].any())
        self.assertTrue((labels['semantic_role_ids'][:, 100, 100] == 255).all())

    def test_visibility_bypass_camera_is_rejected(self):
        camera = self.backend.raw.cameras['front']
        self.sim.params[(camera.identity, 1007)] = 1048576
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'bypasses'):
            d.create_depth_owner_cameras(self.backend)
        self.assertTrue(all(c.removed for c in FakeCamera.created))
        self.assertFalse(self.sim.writes)

    def test_collection_entity_and_arbitrary_warmup_count_rejected(self):
        self.backend.raw.cameras['front'].entity = 900
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'bypasses'):
            d.create_depth_owner_cameras(self.backend)
        with self.assertRaisesRegex(ValueError, 'exactly12'):
            d.create_depth_owner_cameras(self.backend, warmup_rounds=1)

    def test_counterfactual_sensor_failure_restores_full_flags_and_handling(self):
        b = self.bundle()
        original = [c.explicit for c in self.backend.raw.cameras.values()]
        def fail():
            if not self.sim.flags[88] & 512: raise RuntimeError('synthetic hidden render failure')
        b.rgb_cameras['front'].on_handle = fail
        with self.assertRaisesRegex(RuntimeError, 'synthetic hidden'):
            b.capture()
        self.assertEqual(self.sim.flags, {88: 513, 83: 530})
        self.assertEqual(original, [c.explicit for c in self.backend.raw.cameras.values()])
        self.assertTrue(b.last_receipt['all_counterfactual_flags_restored'])
        self.assertTrue(b.last_receipt['clock_unchanged'])
        self.assertFalse(b.last_receipt['all_restore_RGBD_equal'])
        self.assertEqual(set(b.last_public_arrays), d.PUBLIC_ARRAY_KEYS)

    def test_extra_flag_corruption_is_not_silently_accepted(self):
        b = self.bundle()
        def corrupt(identity, value):
            if not value & 512: self.sim.flags[identity] |= 4096
        self.sim.on_set = corrupt
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'unexpected special'):
            b.capture()
        self.assertEqual(self.sim.flags, {88: 513, 83: 530})

    def test_clock_or_actual_q_movement_cannot_be_repaired_by_fabricating_state(self):
        b = self.bundle()
        def move(identity, value):
            if not value & 512: self.backend.arm.q[0] = .001
        self.sim.on_set = move
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'actual q/dq/pose'):
            b.capture()
        self.assertEqual(self.sim.flags, {88: 513, 83: 530})
        self.assertEqual(self.backend.arm.q[0], .001)
        self.assertTrue(b.last_receipt['restoration_errors'])

    def test_nonrestored_shadow_or_rgb_cache_causes_hard_rejection_without_adaptive_retry(self):
        b = self.bundle()
        def tint(identity, value):
            if identity == 88 and value & 512: self.world['tint'] = 1
        self.sim.on_set = tint
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'not byte-identical'):
            b.capture()
        self.assertEqual([r['rounds'] for r in b.last_receipt['warmups']], [12, 12])
        self.assertFalse(b.last_receipt['all_restore_RGBD_equal'])
        self.assertEqual(self.sim.flags, {88: 513, 83: 530})
        self.assertEqual(self.clock, [1.75, 0., 0, 0])

    def test_unchanged_camera_mode_and_closed_bundle_ownership(self):
        b = self.bundle()
        b.capture()
        self.assertEqual(b.rgb_cameras['front'].mode, 3)
        self.assertTrue(all(b.rgb_cameras[n].mode == 1 for n in ('8', '16', '36')))
        b.close(); b.close()
        self.assertTrue(all(c.removed for c in FakeCamera.created))
        self.assertFalse(any(c.removed for c in self.backend.raw.cameras.values()))
        with self.assertRaisesRegex(d.DepthOwnerCaptureError, 'closed'): b.capture()


if __name__ == '__main__': unittest.main()
