"""Explicitly refreshed, zero-physics RGBD/instance annotation sensors.

Native imports are lazy.  Public arrays never contain annotation poses, handles,
task state, contact or success.  The separate label helpers are privileged and
must only be called by the perception-only collector.  CPU tests exercise call
ordering and failure restoration; actual rendering still needs a Native canary.
"""
from __future__ import annotations

import numpy as np

from ..contracts import Pose
from ..perception import CameraCalibration


CAMERAS = ('front', '8', '16', '36')
RESOLUTION = (256, 256)
PUBLIC_ARRAY_KEYS = frozenset(('rgb', 'depth_m', 'camera_intrinsics', 'camera_to_world', 'camera_names'))


class CaptureSensorError(RuntimeError):
    pass


def _load_native_api():
    from pyrep.backend import sim
    from pyrep.const import RenderMode
    from pyrep.objects.shape import Shape
    from pyrep.objects.vision_sensor import VisionSensor
    return sim, RenderMode, Shape, VisionSensor


def _clock(backend):
    values = tuple(backend._clock_tuple())
    control = backend.clock()
    if (len(values) != 4 or any(isinstance(x, (bool, np.bool_)) for x in (*values, control))
            or not np.isfinite(np.asarray((*values, control), dtype=float)).all()):
        raise CaptureSensorError('finite physics/task/step/ledger and public clock required')
    return values, float(control)


def _same_clock(backend, before, operation):
    after = _clock(backend)
    if after != before:
        raise CaptureSensorError(operation + ' changed physics/task/step/ledger or public clock')
    return after


def _matrix(camera):
    value = np.asarray(camera.get_matrix(), dtype=np.float64)
    if value.shape == (3, 4):
        value = np.vstack((value, [0., 0., 0., 1.]))
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise CaptureSensorError('finite camera-to-world matrix required')
    Pose(value[:3, 3], value[:3, :3])
    return value.copy()


def _signature(camera):
    return dict(matrix=_matrix(camera), K=np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64).copy(),
                resolution=tuple(camera.get_resolution()), perspective=camera.get_perspective_mode(),
                angle=float(camera.get_perspective_angle()), ortho=float(camera.get_orthographic_size()),
                near=float(camera.get_near_clipping_plane()), far=float(camera.get_far_clipping_plane()),
                render_mode=camera.get_render_mode(), entity=int(camera.get_entity_to_render()))


def _signature_equal(a, b):
    return all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


def _same_projection(a, b):
    # Render modes intentionally differ; camera/raster geometry and scene
    # entity selection must not differ between RGBD and visible-ID labels.
    keys = ('matrix', 'K', 'resolution', 'perspective', 'angle', 'ortho', 'near', 'far', 'entity')
    return all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in keys)


def _global_enabled(sim):
    # PyRep's installed v4.1 constants expose this parameter as 10.
    flag = sim.simGetBoolParameter(sim.sim_boolparam_vision_sensor_handling_enabled)
    if flag != 1:
        raise CaptureSensorError('global vision sensor handling is disabled; buffers may be stale')
    return True


def _rgb(camera):
    values = np.asarray(camera.capture_rgb(), dtype=np.float32)
    if (values.shape != (*RESOLUTION, 3) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise CaptureSensorError('finite normalized original-resolution RGB required')
    return np.clip(np.rint(values * 255.), 0, 255).astype(np.uint8), values.copy()


def _depth(camera):
    value = np.asarray(camera.capture_depth(in_meters=True), dtype=np.float32)
    if value.shape != RESOLUTION or not np.isfinite(value).all() or np.any(value < 0):
        raise CaptureSensorError('finite nonnegative metric-camera-Z depth required')
    return value.copy()


def _instances(camera):
    value = np.asarray(camera.capture_rgb(), dtype=np.float32)
    if (value.shape != (*RESOLUTION, 3) or not np.isfinite(value).all()
            or np.any(value < 0) or np.any(value > 1)):
        raise CaptureSensorError('normalized current RGB24 instance image required')
    channels = np.rint(value * 255.).astype(np.uint32)
    return channels[..., 0] + 256 * channels[..., 1] + 65536 * channels[..., 2]


class CaptureCameraBundle:
    def __init__(self, backend):
        self.backend = backend
        self.sim, self.render_modes, self.shape_type, self.sensor_type = _load_native_api()
        self.original = getattr(backend.raw, 'cameras', None)
        if not isinstance(self.original, dict) or set(self.original) != set(CAMERAS):
            raise CaptureSensorError('exact four public cameras required')
        self.rgb_cameras, self.mask_cameras, self.calibrations = {}, {}, {}
        self._owned = []
        self._closed = False
        self.last_receipt = None
        self.last_annotation_arrays = {}
        before = _clock(backend)
        original_signatures = {name: _signature(self.original[name]) for name in CAMERAS}
        original_flags = {name: self.original[name].get_explicit_handling() for name in CAMERAS}
        _global_enabled(self.sim)
        try:
            for name in CAMERAS:
                source = self.original[name]
                if tuple(source.get_resolution()) != RESOLUTION:
                    raise CaptureSensorError('public cameras must retain original256 resolution')
                # The original front camera can use OPENGL3 while the other
                # views use classic OPENGL. Preserve the actual per-view mode.
                self.rgb_cameras[name] = self._clone(source, source.get_render_mode())
                self.mask_cameras[name] = self._clone(source, self.render_modes.OPENGL_COLOR_CODED)
                if not _same_projection(_signature(self.rgb_cameras[name]), _signature(self.mask_cameras[name])):
                    raise CaptureSensorError('RGBD and instance-label clone camera projections are not exactly aligned')
                camera = self.rgb_cameras[name]
                K, T = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64), _matrix(camera)
                if K.shape != (3, 3) or not np.isfinite(K).all():
                    raise CaptureSensorError('finite signed calibration required')
                self.calibrations[name] = CameraCalibration(K, Pose(T[:3, 3], T[:3, :3]))
            if any(not _signature_equal(original_signatures[n], _signature(self.original[n]))
                   or original_flags[n] != self.original[n].get_explicit_handling() for n in CAMERAS):
                raise CaptureSensorError('camera creation changed an original RGB sensor')
            self._frozen_signatures = [(camera, _signature(camera)) for camera in
                [self.original[n] for n in CAMERAS] + [self.rgb_cameras[n] for n in CAMERAS]
                + [self.mask_cameras[n] for n in CAMERAS]]
            _same_clock(backend, before, 'camera creation')
        except BaseException:
            for camera in reversed(self._owned):
                try:
                    camera.remove()
                except Exception:
                    pass
            self._closed = True
            _same_clock(backend, before, 'failed camera creation/cleanup')
            raise

    def _clone(self, source, render_mode):
        camera = self.sensor_type.create(list(RESOLUTION), explicit_handling=True, passive=False,
                                         show_volume_not_detecting=False, show_volume_detecting=False,
                                         render_mode=render_mode)
        self._owned.append(camera)
        camera.set_perspective_mode(source.get_perspective_mode())
        camera.set_perspective_angle(source.get_perspective_angle())
        camera.set_orthographic_size(source.get_orthographic_size())
        camera.set_near_clipping_plane(source.get_near_clipping_plane())
        camera.set_far_clipping_plane(source.get_far_clipping_plane())
        camera.set_matrix(source.get_matrix())
        camera.set_entity_to_render(source.get_entity_to_render())
        # Preserve rendering attributes/disabled-light components when the
        # installed sim API exposes them. Different defaults must not be hidden.
        for constant in ('sim_visionintparam_disabled_light_components', 'sim_visionintparam_rendering_attributes'):
            parameter = getattr(self.sim, constant, None)
            if parameter is not None:
                value = self.sim.simGetObjectInt32Parameter(source.get_handle(), parameter)
                self.sim.simSetObjectInt32Parameter(camera.get_handle(), parameter, value)
        return camera

    def capture(self):
        if self._closed:
            raise CaptureSensorError('capture camera bundle is closed')
        self.last_annotation_arrays = {}
        before = _clock(self.backend)
        _global_enabled(self.sim)
        if any(not _signature_equal(signature, _signature(camera)) for camera, signature in self._frozen_signatures):
            raise CaptureSensorError('camera calibration/render properties differ from the frozen capture bundle')
        sensors = [self.original[n] for n in CAMERAS]
        sensors += [self.rgb_cameras[n] for n in CAMERAS] + [self.mask_cameras[n] for n in CAMERAS]
        flags = [(camera, camera.get_explicit_handling()) for camera in sensors]
        signatures = [(camera, _signature(camera)) for camera in sensors]
        rows, rgbs, depths, mask_depths, ids, Ks, Ts = {}, [], [], [], [], [], []
        error = None
        restoration_errors = []
        receipt = dict(global_vision_sensors_enabled=True, created_nonpassive=True,
                       mask_rgb_projection_equal=True,
                       mask_rgb_depth_equal=False,
                       original_clone_equal=False, handling_restored=False,
                       clock_before=before[0], control_clock_before=before[1], per_camera=rows,
                       RGB_render_modes_changed=False, Native_pose_or_state_in_public_arrays=False,
                       Native_started_by_helper=False)
        try:
            for camera, flag in flags:
                if not flag:
                    camera.set_explicit_handling(1)
            for name in CAMERAS:
                original, rgb, mask = self.original[name], self.rgb_cameras[name], self.mask_cameras[name]
                # The getters alone do NOT render. Each explicit handle reads
                # the current world and must immediately precede its buffers.
                original.handle_explicitly()
                original_rgb, original_float = _rgb(original)
                original_depth = _depth(original)
                rgb.handle_explicitly()
                image, image_float = _rgb(rgb)
                depth = _depth(rgb)
                mask.handle_explicitly()
                instances = _instances(mask)
                mask_depth = _depth(mask)
                same_rgb = np.array_equal(original_rgb, image)
                same_depth = np.array_equal(original_depth, depth)
                same_mask_depth = np.array_equal(mask_depth, depth)
                original_T, clone_T = _matrix(original), _matrix(rgb)
                original_K = np.asarray(original.get_intrinsic_matrix(), dtype=np.float64)
                clone_K = np.asarray(rgb.get_intrinsic_matrix(), dtype=np.float64)
                rows[name] = dict(rgb_equal=bool(same_rgb), depth_equal=bool(same_depth),
                                  rgb_max_abs_diff_uint8=int(np.max(np.abs(original_rgb.astype(np.int16) - image.astype(np.int16)))),
                                  rgb_float_max_abs_diff=float(np.max(np.abs(original_float - image_float))),
                                  depth_max_abs_diff_m=float(np.max(np.abs(original_depth - depth))),
                                  mask_rgb_depth_equal=bool(same_mask_depth),
                                  mask_rgb_depth_max_abs_diff_m=float(np.max(np.abs(mask_depth - depth))),
                                  mask_rgb_depth_mismatched_pixel_count=int(np.count_nonzero(mask_depth != depth)),
                                  K_max_abs_diff=float(np.max(np.abs(original_K - clone_K))),
                                  camera_to_world_max_abs_diff=float(np.max(np.abs(original_T - clone_T))),
                                  original_render_mode=str(original.get_render_mode()),
                                  clone_render_mode=str(rgb.get_render_mode()),
                                  mask_render_mode=str(mask.get_render_mode()))
                rgbs.append(image); depths.append(depth); mask_depths.append(mask_depth); ids.append(instances)
                Ks.append(clone_K.copy()); Ts.append(clone_T)
                _same_clock(self.backend, before, 'current sensor rendering/buffer read')
            receipt['original_clone_equal'] = all(v['rgb_equal'] and v['depth_equal'] for v in rows.values())
            receipt['mask_rgb_depth_equal'] = all(v['mask_rgb_depth_equal'] for v in rows.values())
            # Color-coded and OPENGL3 renderers need not share occlusion even
            # with exactly equal K/T. This array belongs in labels/diagnostics,
            # never in public model inputs. A False gate must remain a rejection.
            self.last_annotation_arrays = {'mask_depth_m': np.stack(mask_depths).astype(np.float32)}
        except BaseException as exc:
            error = exc
        finally:
            for camera, flag in flags:
                try:
                    if camera.get_explicit_handling() != flag:
                        camera.set_explicit_handling(flag)
                except Exception as exc:
                    restoration_errors.append('explicit handling restoration: ' + str(exc))
            for camera, expected in signatures:
                try:
                    if not _signature_equal(expected, _signature(camera)):
                        restoration_errors.append('camera geometry/render mode changed')
                except Exception as exc:
                    restoration_errors.append('camera signature readback: ' + str(exc))
            try:
                _global_enabled(self.sim)
                after = _same_clock(self.backend, before, 'capture/finally restoration')
                receipt.update(clock_after=after[0], control_clock_after=after[1])
            except Exception as exc:
                restoration_errors.append(str(exc))
            try:
                flags_equal = all(camera.get_explicit_handling() == flag for camera, flag in flags)
            except Exception as exc:
                flags_equal = False
                restoration_errors.append('explicit handling final readback: ' + str(exc))
            receipt['handling_restored'] = not restoration_errors and flags_equal
            receipt['error'] = None if error is None else type(error).__name__ + ': ' + str(error)
            receipt['restoration_errors'] = restoration_errors
            self.last_receipt = receipt
        if restoration_errors:
            raise CaptureSensorError('; '.join(restoration_errors)) from error
        if error is not None:
            raise error
        public = dict(rgb=np.stack(rgbs).astype(np.uint8), depth_m=np.stack(depths).astype(np.float32),
                      camera_intrinsics=np.stack(Ks).astype(np.float64),
                      camera_to_world=np.stack(Ts).astype(np.float64), camera_names=np.asarray(CAMERAS))
        if set(public) != PUBLIC_ARRAY_KEYS:
            raise CaptureSensorError('public sensor output whitelist changed')
        return public, np.stack(ids).astype(np.uint32), receipt

    def close(self):
        if self._closed:
            return
        before = _clock(self.backend)
        errors = []
        for camera in reversed(self._owned):
            try:
                camera.remove()
            except Exception as exc:
                errors.append(str(exc))
        self._closed = True
        _same_clock(self.backend, before, 'owned sensor removal')
        if errors:
            raise CaptureSensorError('owned sensor cleanup failed: ' + '; '.join(errors))


def create_capture_cameras(backend):
    return CaptureCameraBundle(backend)


def resolve_annotation_roles(backend):
    """Return actual current renderable shape IDs for LABELS, never public input."""
    _, _, Shape, _ = _load_native_api()
    before = _clock(backend)
    cube, goal = backend.task.target_block, Shape('success_visual')
    nodes = dict(cube=cube, goal=goal)
    names = dict(cube='pick_and_lift_target', goal='success_visual')
    bindings, inventory = {}, {}
    for role, node in nodes.items():
        if type(node) is not Shape or node.get_name() != names[role] or not node.is_renderable():
            raise CaptureSensorError('exact current renderable ' + names[role] + ' Shape required')
        identity = node.get_handle()
        if isinstance(identity, (bool, np.bool_)) or not isinstance(identity, (int, np.integer)) or identity < 0:
            raise CaptureSensorError('current native integer shape ID required')
        bindings[role] = [int(identity)]
        parent = node.get_parent()
        inventory[role] = dict(name=node.get_name(), type='Shape', handle=int(identity), renderable=True,
                               parent_name=None if parent is None else parent.get_name())
    if bindings['cube'][0] == bindings['goal'][0]:
        raise CaptureSensorError('one rendered instance cannot own cube and goal labels')
    after = _same_clock(backend, before, 'annotation role inventory')
    return nodes, bindings, dict(clock_before=before[0], clock_after=after[0],
                                  control_clock_before=before[1], control_clock_after=after[1],
                                  annotation_only=True, public_inputs=False, inventory=inventory,
                                  goal_is_visible_shape_not_success_proximity_sensor=True)


def annotation_pose_labels(role_nodes):
    """Privileged same-shot label payload. Call under the collector clock guard."""
    _, _, Shape, _ = _load_native_api()
    if not isinstance(role_nodes, dict) or set(role_nodes) != {'cube', 'goal'} or any(type(x) is not Shape for x in role_nodes.values()):
        raise CaptureSensorError('exact annotation role nodes required')
    if (role_nodes['cube'].get_name() != 'pick_and_lift_target'
            or role_nodes['goal'].get_name() != 'success_visual'
            or role_nodes['cube'].get_handle() == role_nodes['goal'].get_handle()):
        raise CaptureSensorError('annotation label roles must retain the verified cube/visual-goal identities')
    result = {}
    for role, key in (('cube', 'cube_pose_world'), ('goal', 'goal_mesh_pose_world')):
        pose = np.asarray(role_nodes[role].get_pose(), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5, rtol=0):
            raise CaptureSensorError('finite same-frame Native label SE3 required')
        result[key] = pose.copy()
        bbox = np.asarray(role_nodes[role].get_bounding_box(), dtype=np.float64)
        if bbox.shape != (6,) or not np.isfinite(bbox).all() or np.any(bbox[1::2] < bbox[::2]):
            raise CaptureSensorError('finite label-only shape bounding box required')
        result[role + '_local_bbox'] = bbox.copy()
    return result
