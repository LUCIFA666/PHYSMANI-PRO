"""CPU contracts; fake sensor buffers change only when explicitly handled."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_20261103 import perception_capture_sensors as sensors


class FakeCamera:
    created = []
    world = 0
    fail_create = None
    mask_shift = False
    gl3_depth_offset = 0.
    counter = 10

    def __init__(self, mode=1, explicit=0):
        type(self).counter += 1
        self.identity = type(self).counter
        self.mode, self.explicit = mode, explicit
        self.matrix = np.eye(4)
        self.resolution = [256, 256]
        self.perspective, self.angle, self.ortho = 1, 60., 1.
        self.near, self.far, self.entity = .01, 10., -1
        self.removed = False
        self.handles = 0
        self.rgb_delta = 0.
        self.on_handle = None
        self.refresh()

    @classmethod
    def create(cls, resolution, **kwargs):
        if cls.fail_create is not None and len(cls.created) == cls.fail_create:
            raise RuntimeError('construction failure')
        assert kwargs['explicit_handling'] is True and kwargs['passive'] is False
        obj = cls(kwargs['render_mode'], 1)
        obj.creation_kwargs = kwargs
        obj.resolution = list(resolution)
        cls.created.append(obj)
        return obj

    def refresh(self):
        if self.mode == 2:
            self.rgb = np.broadcast_to(np.array([0x56, 0x34, 0x12], np.float32) / 255., (256, 256, 3)).copy()
        else:
            base = 140 if self.mode == 3 else 40
            self.rgb = np.full((256, 256, 3), (base + self.world) / 255. + self.rgb_delta, dtype=np.float32)
        self.depth = np.full((256, 256), 1 + self.world * .01 + (self.gl3_depth_offset if self.mode == 3 else 0.), np.float32)

    def handle_explicitly(self):
        if not self.explicit:
            raise RuntimeError('not explicit')
        self.handles += 1
        if self.on_handle:
            self.on_handle()
        self.refresh()

    def capture_rgb(self): return self.rgb
    def capture_depth(self, *, in_meters):
        assert in_meters is True
        return self.depth
    def get_explicit_handling(self): return self.explicit
    def set_explicit_handling(self, value): self.explicit = value
    def get_matrix(self): return self.matrix.copy()
    def set_matrix(self, value):
        self.matrix = np.asarray(value).copy()
        if self.mode == 2 and self.mask_shift:
            self.matrix[0, 3] += .001
    def get_intrinsic_matrix(self): return np.array([[-100., 0, 128], [0, -100., 128], [0, 0, 1.]])
    def get_resolution(self): return self.resolution.copy()
    def get_perspective_mode(self): return self.perspective
    def set_perspective_mode(self, value): self.perspective = value
    def get_perspective_angle(self): return self.angle
    def set_perspective_angle(self, value): self.angle = value
    def get_orthographic_size(self): return self.ortho
    def set_orthographic_size(self, value): self.ortho = value
    def get_near_clipping_plane(self): return self.near
    def set_near_clipping_plane(self, value): self.near = value
    def get_far_clipping_plane(self): return self.far
    def set_far_clipping_plane(self, value): self.far = value
    def get_render_mode(self): return self.mode
    def get_entity_to_render(self): return self.entity
    def set_entity_to_render(self, value): self.entity = value
    def get_handle(self): return self.identity
    def remove(self): self.removed = True


class FakeShape:
    registry = {}
    def __new__(cls, name, identity=None):
        if name in cls.registry:
            return cls.registry[name]
        return super().__new__(cls)
    def __init__(self, name, identity=None):
        if hasattr(self, 'name'):
            return
        self.name, self.identity = name, identity
        self.renderable = True
        self.pose = np.array([.2, .1, .8, 0, 0, 0, 1.])
        self.pose_reads = 0
        self.registry[name] = self
    def get_name(self): return self.name
    def get_handle(self): return self.identity
    def is_renderable(self): return self.renderable
    def get_parent(self): return None
    def get_pose(self):
        self.pose_reads += 1
        return self.pose.copy()
    def get_bounding_box(self): return [-.025, .025, -.025, .025, -.025, .025]


class FakeSim:
    sim_boolparam_vision_sensor_handling_enabled = 10
    sim_visionintparam_disabled_light_components = 1006
    sim_visionintparam_rendering_attributes = 1007
    enabled = 1
    def __init__(self): self.params = {}
    def simGetBoolParameter(self, parameter):
        assert parameter == 10
        return self.enabled
    def simGetObjectInt32Parameter(self, identity, parameter): return self.params.get((identity, parameter), 7)
    def simSetObjectInt32Parameter(self, identity, parameter, value): self.params[identity, parameter] = value


class CaptureSensorTests(unittest.TestCase):
    def setUp(self):
        FakeCamera.created, FakeCamera.world, FakeCamera.fail_create = [], 0, None
        FakeCamera.mask_shift = False
        FakeCamera.gl3_depth_offset = 0.
        FakeShape.registry = {}
        self.sim = FakeSim()
        self.original = {name: FakeCamera(explicit=i % 2) for i, name in enumerate(sensors.CAMERAS)}
        self.values = [0., 0., 0, 0]
        self.cube = FakeShape('pick_and_lift_target', 0xABCDEF)
        self.goal = FakeShape('success_visual', 0x123456)
        self.backend = SimpleNamespace(raw=SimpleNamespace(cameras=self.original),
                                       task=SimpleNamespace(target_block=self.cube),
                                       _clock_tuple=lambda: tuple(self.values), clock=lambda: 0.)
        self.api = patch.object(sensors, '_load_native_api', return_value=(self.sim, SimpleNamespace(OPENGL=1, OPENGL_COLOR_CODED=2, OPENGL3=3), FakeShape, FakeCamera))
        self.api.start()
        self.addCleanup(self.api.stop)

    def bundle(self):
        bundle = sensors.create_capture_cameras(self.backend)
        self.addCleanup(bundle.close)
        return bundle

    def test_current_buffers_refresh_without_time_and_high_rgb24_ids_decode(self):
        bundle = self.bundle()
        FakeCamera.world = 3
        arrays, ids, receipt = bundle.capture()
        self.assertEqual(set(arrays), sensors.PUBLIC_ARRAY_KEYS)
        np.testing.assert_array_equal(arrays['rgb'], np.full((4, 256, 256, 3), 43, np.uint8))
        np.testing.assert_array_equal(arrays['depth_m'], np.full((4, 256, 256), 1.03, np.float32))
        self.assertTrue(np.all(ids == 0x123456))
        self.assertEqual(ids.dtype, np.uint32)
        self.assertTrue(receipt['original_clone_equal'])
        self.assertTrue(receipt['handling_restored'])
        self.assertTrue(receipt['mask_rgb_projection_equal'])
        self.assertTrue(receipt['mask_rgb_depth_equal'])
        self.assertEqual(set(bundle.last_annotation_arrays), {'mask_depth_m'})
        np.testing.assert_array_equal(bundle.last_annotation_arrays['mask_depth_m'], arrays['depth_m'])
        self.assertEqual(receipt['clock_before'], receipt['clock_after'])
        self.assertEqual(self.cube.pose_reads, 0)
        self.assertEqual(self.goal.pose_reads, 0)
        for camera in self.original.values(): self.assertEqual(camera.handles, 1)
        for camera in FakeCamera.created: self.assertEqual(camera.handles, 1)
        self.assertLess(bundle.calibrations['front'].intrinsics[0, 0], 0)

    def test_original_flags_and_render_mode_restore_success_and_failure(self):
        bundle = self.bundle()
        before = [x.explicit for x in self.original.values()]
        bundle.capture()
        self.assertEqual(before, [x.explicit for x in self.original.values()])
        def fail(): raise RuntimeError('read failure')
        bundle.rgb_cameras['8'].on_handle = fail
        with self.assertRaisesRegex(RuntimeError, 'read failure'):
            bundle.capture()
        self.assertEqual(before, [x.explicit for x in self.original.values()])
        self.assertTrue(bundle.last_receipt['handling_restored'])
        self.assertTrue(all(x.mode == 1 for x in self.original.values()))

    def test_original_clone_difference_is_reported_not_hidden(self):
        bundle = self.bundle()
        bundle.rgb_cameras['16'].rgb_delta = 1 / 255.
        _, _, receipt = bundle.capture()
        self.assertFalse(receipt['original_clone_equal'])
        self.assertEqual(receipt['per_camera']['16']['rgb_max_abs_diff_uint8'], 1)
        self.assertTrue(receipt['per_camera']['16']['depth_equal'])

    def test_heterogeneous_original_render_modes_are_preserved_per_camera(self):
        self.original['front'].mode = 3
        self.original['front'].refresh()
        bundle = self.bundle()
        arrays, _, receipt = bundle.capture()
        self.assertEqual(bundle.rgb_cameras['front'].mode, 3)
        self.assertTrue(all(bundle.rgb_cameras[name].mode == 1 for name in ('8', '16', '36')))
        self.assertTrue(all(x.mode == 2 for x in bundle.mask_cameras.values()))
        self.assertTrue(receipt['original_clone_equal'])
        self.assertTrue(np.all(arrays['rgb'][0] == 140))
        self.assertTrue(receipt['mask_rgb_depth_equal'])

    def test_same_projection_does_not_hide_colorcoded_occlusion_difference(self):
        self.original['front'].mode = 3
        FakeCamera.gl3_depth_offset = .2
        self.original['front'].refresh()
        bundle = self.bundle()
        arrays, _, receipt = bundle.capture()
        self.assertTrue(receipt['original_clone_equal'])
        self.assertTrue(receipt['mask_rgb_projection_equal'])
        self.assertFalse(receipt['mask_rgb_depth_equal'])
        front = receipt['per_camera']['front']
        self.assertEqual(front['mask_rgb_depth_mismatched_pixel_count'], 256 * 256)
        self.assertAlmostEqual(front['mask_rgb_depth_max_abs_diff_m'], .2, places=6)
        self.assertFalse(np.array_equal(bundle.last_annotation_arrays['mask_depth_m'][0], arrays['depth_m'][0]))
        self.assertNotIn('mask_depth_m', arrays)

    def test_global_disabled_rejects_stale_buffers_before_read(self):
        bundle = self.bundle()
        self.sim.enabled = 0
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'global vision'):
            bundle.capture()
        self.assertTrue(all(x.handles == 0 for x in self.original.values()))

    def test_clock_advance_fails_and_restores_flags(self):
        bundle = self.bundle()
        before = [x.explicit for x in self.original.values()]
        bundle.mask_cameras['front'].on_handle = lambda: self.values.__setitem__(0, .05)
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'changed physics'):
            bundle.capture()
        self.assertEqual(before, [x.explicit for x in self.original.values()])
        self.assertTrue(bundle.last_receipt['restoration_errors'])

    def test_camera_change_cannot_redefine_the_initial_calibration(self):
        bundle = self.bundle()
        self.original['front'].matrix[0, 3] += .01
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'frozen capture'):
            bundle.capture()

    def test_cleanup_removes_only_owned_sensors_and_is_idempotent(self):
        bundle = self.bundle()
        bundle.close(); bundle.close()
        self.assertTrue(all(x.removed for x in FakeCamera.created))
        self.assertFalse(any(x.removed for x in self.original.values()))
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'closed'):
            bundle.capture()

    def test_partial_construction_failure_cleans_up_without_touching_original(self):
        FakeCamera.fail_create = 3
        before = [x.explicit for x in self.original.values()]
        with self.assertRaisesRegex(RuntimeError, 'construction failure'):
            sensors.create_capture_cameras(self.backend)
        self.assertEqual(before, [x.explicit for x in self.original.values()])
        self.assertTrue(all(x.removed for x in FakeCamera.created))
        self.assertFalse(any(x.removed for x in self.original.values()))

    def test_shifted_mask_camera_cannot_generate_wrong_same_clock_labels(self):
        FakeCamera.mask_shift = True
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'not exactly aligned'):
            sensors.create_capture_cameras(self.backend)
        self.assertTrue(all(x.removed for x in FakeCamera.created))
        self.assertFalse(any(x.removed for x in self.original.values()))

    def test_roles_current_shape_identity_and_label_only_poses(self):
        nodes, bindings, receipt = sensors.resolve_annotation_roles(self.backend)
        self.assertEqual(bindings, {'cube': [0xABCDEF], 'goal': [0x123456]})
        self.assertTrue(receipt['annotation_only'])
        self.assertEqual(self.cube.pose_reads + self.goal.pose_reads, 0)
        labels = sensors.annotation_pose_labels(nodes)
        self.assertEqual(set(labels), {'cube_pose_world', 'goal_mesh_pose_world', 'cube_local_bbox', 'goal_local_bbox'})
        np.testing.assert_array_equal(labels['goal_mesh_pose_world'], self.goal.pose)
        self.goal.renderable = False
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'renderable'):
            sensors.resolve_annotation_roles(self.backend)

    def test_nonshape_sensor_and_duplicate_ids_rejected(self):
        self.backend.task.target_block = SimpleNamespace(get_name=lambda: 'pick_and_lift_target')
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'Shape'):
            sensors.resolve_annotation_roles(self.backend)
        self.backend.task.target_block = self.cube
        self.goal.identity = self.cube.identity
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'cannot own'):
            sensors.resolve_annotation_roles(self.backend)

    def test_label_role_swapping_cannot_silently_change_oracle_center(self):
        with self.assertRaisesRegex(sensors.CaptureSensorError, 'verified cube'):
            sensors.annotation_pose_labels({'cube': self.goal, 'goal': self.cube})


if __name__ == '__main__': unittest.main()
