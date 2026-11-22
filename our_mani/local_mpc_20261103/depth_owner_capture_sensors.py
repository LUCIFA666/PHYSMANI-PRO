"""Label-only visibility counterfactuals in the unchanged RGBD renderer.

The returned public arrays are the normal base image. Counterfactual images,
depth differences, role handles and state checks are annotation evidence only.
No material, alpha, pose, joint or physics setter is used. This module does not
claim RGB instance-ID semantics: classes denote current measured-depth owners.
"""
from __future__ import annotations

import time
import numpy as np

from ..contracts import Pose
from ..perception import CameraCalibration
from . import perception_capture_sensors as old


CAMERAS = old.CAMERAS
PUBLIC_ARRAY_KEYS = old.PUBLIC_ARRAY_KEYS
LABEL_METHOD = 'counterfactual_current_depth_owner'
SEMANTIC_KIND = 'current_measured_depth_owner'
RENDERABLE = 512
IGNORE_RENDERABLE = 1048576
WARMUP_ROUNDS = 12


class DepthOwnerCaptureError(old.CaptureSensorError):
    pass


def _equal_arrays(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def _boundary(mask):
    """Fixed one-pixel inner and outer ring, including the raster border."""
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    eroded, dilated = np.ones_like(mask), np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            part = padded[:, dy:dy+mask.shape[1], dx:dx+mask.shape[2]]
            eroded &= part
            dilated |= part
    return dilated & ~eroded


def _labels(base, counterfactuals):
    depth = base['depth_m']
    annotations = {}
    positives, negatives, invalids, boundaries, responses = {}, {}, {}, {}, {}
    for role in ('cube', 'goal'):
        changed = counterfactuals[role]
        without = changed['depth_m']
        valid = np.isfinite(depth) & np.isfinite(without) & (depth > 0) & (without > 0)
        delta = without.astype(np.float64) - depth.astype(np.float64)
        positives[role] = valid & (delta > 0)
        negatives[role] = valid & (delta < 0)
        invalids[role] = ~valid
        boundaries[role] = _boundary(positives[role])
        responses[role] = np.any(changed['rgb'] != base['rgb'], axis=-1)
        annotations.update({f'without_{role}_depth_m': without.copy(),
                            f'without_{role}_rgb': changed['rgb'].copy(),
                            f'{role}_signed_depth_delta_m': delta,
                            f'{role}_raw_positive': positives[role],
                            f'{role}_negative_delta': negatives[role],
                            f'{role}_invalid_depth': invalids[role],
                            f'{role}_one_pixel_boundary': boundaries[role],
                            f'{role}_rgb_response': responses[role]})
    conflict = positives['cube'] & positives['goal']
    unknown = (negatives['cube'] | negatives['goal'] | invalids['cube'] | invalids['goal']
               | boundaries['cube'] | boundaries['goal'] | conflict)
    # A transparent goal can contribute RGB without writing the measured depth.
    # RGB/shadow response alone never creates a positive semantic goal label.
    goal_rgb_only = responses['goal'] & ~positives['goal'] & ~positives['cube']
    unknown |= goal_rgb_only
    cube = positives['cube'] & ~unknown
    goal = positives['goal'] & ~positives['cube'] & ~unknown
    semantic = np.zeros(depth.shape, dtype=np.uint8)
    semantic[cube], semantic[goal], semantic[unknown] = 1, 2, 255
    annotations.update(semantic_role_ids=semantic, cube_depth_owner=cube,
                       goal_depth_owner=goal, cube_unknown=unknown.copy(),
                       semantic_unknown=unknown, conflicting_depth_owner=conflict,
                       goal_rgb_only_unknown=goal_rgb_only)
    return annotations


class DepthOwnerCameraBundle(old.CaptureCameraBundle):
    """Owned normal-render clones; no ColorCoded sensor is created or used."""
    def __init__(self, backend, *, warmup_rounds=WARMUP_ROUNDS):
        if type(warmup_rounds) is not int or warmup_rounds != WARMUP_ROUNDS:
            raise ValueError('this version fixes exactly12 finite normal-render warmup rounds')
        self.backend, self.warmup_rounds = backend, warmup_rounds
        self.sim, self.render_modes, self.shape_type, self.sensor_type = old._load_native_api()
        self.original = dict(backend.raw.cameras)
        if set(self.original) != set(CAMERAS):
            raise DepthOwnerCaptureError('exact four public cameras required')
        self.rgb_cameras, self.mask_cameras, self.calibrations, self._owned = {}, {}, {}, []
        self._closed = False
        self.last_receipt, self.last_public_arrays, self.last_annotation_arrays = None, {}, {}
        self.role_nodes, self.role_bindings, self.role_receipt = old.resolve_annotation_roles(backend)
        before = old._clock(backend)
        original = {n: old._signature(self.original[n]) for n in CAMERAS}
        old._global_enabled(self.sim)
        try:
            for name in CAMERAS:
                source = self.original[name]
                if tuple(source.get_resolution()) != old.RESOLUTION:
                    raise DepthOwnerCaptureError('original256 resolution required')
                self.rgb_cameras[name] = self._clone(source, source.get_render_mode())
                camera = self.rgb_cameras[name]
                K, T = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64), old._matrix(camera)
                self.calibrations[name] = CameraCalibration(K, Pose(T[:3, 3], T[:3, :3]))
            if any(not old._signature_equal(original[n], old._signature(self.original[n])) for n in CAMERAS):
                raise DepthOwnerCaptureError('normal clone construction changed a source camera')
            self._sensors = [(n, 'original', self.original[n]) for n in CAMERAS]
            self._sensors += [(n, 'clone', self.rgb_cameras[n]) for n in CAMERAS]
            self._frozen_cameras = [(c, old._signature(c), self._attributes(c)) for _, _, c in self._sensors]
            self._guard_cameras()
            old._same_clock(backend, before, 'depth-owner camera creation')
        except BaseException:
            for camera in reversed(self._owned):
                try: camera.remove()
                except Exception: pass
            self._closed = True
            old._same_clock(backend, before, 'failed depth-owner camera cleanup')
            raise

    def _attributes(self, camera):
        return int(self.sim.simGetObjectInt32Parameter(camera.get_handle(), self.sim.sim_visionintparam_rendering_attributes))

    def _guard_cameras(self):
        old._global_enabled(self.sim)
        for camera, expected, attributes in self._frozen_cameras:
            actual_attributes = self._attributes(camera)
            if camera.get_entity_to_render() != -1 or actual_attributes & IGNORE_RENDERABLE:
                raise DepthOwnerCaptureError('camera bypasses renderable flags/uses an entity or collection override')
            if not old._signature_equal(expected, old._signature(camera)) or actual_attributes != attributes:
                raise DepthOwnerCaptureError('camera properties changed from frozen normal renderer')

    def _flags(self):
        result = {}
        for role, node in self.role_nodes.items():
            value = self.sim.simGetObjectSpecialProperty(node.get_handle())
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise DepthOwnerCaptureError('integer local object special properties required')
            result[role] = int(value)
        return result

    def _snapshot(self):
        result = dict(clock=old._clock(self.backend), flags=self._flags())
        for role, node in self.role_nodes.items():
            value = np.asarray(node.get_pose(), dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]), 1., atol=1e-5, rtol=0):
                raise DepthOwnerCaptureError('finite unchanged native annotation pose required')
            result[role+'_pose'] = value.copy()
        for role, node, size in (('arm', self.backend.arm, 7), ('finger', self.backend.gripper, 2)):
            for name in ('get_joint_positions', 'get_joint_velocities', 'get_joint_target_positions'):
                value = np.asarray(getattr(node, name)(), dtype=float)
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise DepthOwnerCaptureError('finite actual joint/target state required')
                result[role+'_'+name] = value.copy()
        result['finger_target_velocity'] = np.asarray(self.backend.gripper.get_joint_target_velocities(), dtype=float).copy()
        if result['finger_target_velocity'].shape != (2,) or not np.isfinite(result['finger_target_velocity']).all():
            raise DepthOwnerCaptureError('finite actual finger target velocity required')
        return result

    def _check_state(self, expected, *, flags=None):
        actual = self._snapshot()
        for key, value in expected.items():
            if key == 'flags':
                if actual[key] != (value if flags is None else flags):
                    raise DepthOwnerCaptureError('visibility transaction changed unexpected special-property bits')
            elif isinstance(value, np.ndarray):
                if not np.isfinite(actual[key]).all() or not np.array_equal(value, actual[key]):
                    raise DepthOwnerCaptureError('visibility-only transaction changed actual q/dq/pose/servo state: '+key)
            elif actual[key] != value:
                raise DepthOwnerCaptureError('visibility-only transaction changed physics/task/control clock')
        self._guard_cameras()

    def _handle(self, camera, *, phase, name, kind, round_index=None):
        row = dict(phase=phase, camera=name, sensor=kind, round=round_index, completed=False)
        self._calls.append(row)
        camera.handle_explicitly()
        row['completed'] = True

    def _warmup(self, phase, expected):
        started = time.perf_counter()
        for index in range(self.warmup_rounds):
            for name in CAMERAS:
                for kind, camera in (('original', self.original[name]), ('clone', self.rgb_cameras[name])):
                    self._handle(camera, phase=phase, name=name, kind=kind, round_index=index+1)
            self._check_state(expected)
        self._warmups.append(dict(phase=phase, rounds=self.warmup_rounds,
                                  render_calls=2*len(CAMERAS)*self.warmup_rounds,
                                  wall_seconds=time.perf_counter()-started, adaptive=False))

    def _read_normal(self, phase):
        arrays, comparisons = self._read_clones(phase, include_original=True)
        return arrays, all(row['rgb_equal'] and row['depth_equal'] for row in comparisons.values()), comparisons

    def _read_clones(self, phase, *, include_original=False):
        images, depths, Ks, Ts, comparisons = [], [], [], [], {}
        for name in CAMERAS:
            if include_original:
                original = self.original[name]
                self._handle(original, phase=phase, name=name, kind='original')
                original_rgb, _ = old._rgb(original)
                original_depth = old._depth(original)
            clone = self.rgb_cameras[name]
            self._handle(clone, phase=phase, name=name, kind='clone')
            image, _ = old._rgb(clone)
            depth = old._depth(clone)
            images.append(image); depths.append(depth)
            Ks.append(np.asarray(clone.get_intrinsic_matrix(), dtype=np.float64).copy())
            Ts.append(old._matrix(clone))
            if include_original:
                comparisons[name] = dict(rgb_equal=bool(np.array_equal(original_rgb, image)),
                    depth_equal=bool(np.array_equal(original_depth, depth)),
                    rgb_max_abs_diff=int(np.max(np.abs(original_rgb.astype(np.int16)-image.astype(np.int16)))),
                    depth_max_abs_diff_m=float(np.max(np.abs(original_depth-depth))))
        arrays = dict(rgb=np.stack(images), depth_m=np.stack(depths), camera_intrinsics=np.stack(Ks),
                      camera_to_world=np.stack(Ts), camera_names=np.asarray(CAMERAS))
        if set(arrays) != PUBLIC_ARRAY_KEYS:
            raise DepthOwnerCaptureError('normal public array whitelist changed')
        return arrays, comparisons

    def capture(self):
        if self._closed:
            raise DepthOwnerCaptureError('depth-owner bundle is closed')
        began = time.perf_counter()
        self.last_receipt = None
        self.last_public_arrays, self.last_annotation_arrays = {}, {}
        self._calls, self._warmups = [], []
        self._guard_cameras()
        state = self._snapshot()
        if any(not flag & RENDERABLE for flag in state['flags'].values()):
            raise DepthOwnerCaptureError('both original role shapes must start locally renderable')
        saved_handling = [(camera, camera.get_explicit_handling()) for _, _, camera in self._sensors]
        receipt = dict(label_method=LABEL_METHOD, semantic_kind=SEMANTIC_KIND,
            original_clone_equal=False, global_vision_sensors_enabled=True, created_nonpassive=True,
            handling_restored=False, all_counterfactual_flags_restored=False,
            all_restore_RGBD_equal=False, clock_unchanged=False,
            warmup_rounds_fixed=self.warmup_rounds, adaptive_rendering=False,
            clock_before=state['clock'][0], control_clock_before=state['clock'][1],
            per_role={}, render_calls=self._calls, warmups=self._warmups,
            camera_render_modes={n: str(self.original[n].get_render_mode()) for n in CAMERAS},
            Native_pose_or_state_in_public_arrays=False, semantic_pixels_are_Native_instance_IDs=False,
            alpha_or_material_modified=False, poses_or_joints_set=False)
        counterfactuals, restored_results, error, restoration_errors = {}, [], None, []
        try:
            for camera, flag in saved_handling:
                if not flag: camera.set_explicit_handling(1)
            self._warmup('base_normal_warmup', state)
            base, equal, comparison = self._read_normal('base_normal_capture')
            self.last_public_arrays = {k: v.copy() for k, v in base.items()}
            receipt.update(original_clone_equal=bool(equal), base_camera_comparison=comparison)
            self._check_state(state)
            if not equal:
                raise DepthOwnerCaptureError('normal renderer original/clone RGBD differs')
            for role in ('cube', 'goal'):
                expected_flags = dict(state['flags'])
                expected_flags[role] &= ~RENDERABLE
                role_receipt = dict(flags_before=dict(state['flags']), flags_hidden_expected=expected_flags,
                                    flags_restored=False, restored_RGBD_equal=False)
                receipt['per_role'][role] = role_receipt
                try:
                    self.sim.simSetObjectSpecialProperty(self.role_nodes[role].get_handle(), expected_flags[role])
                    self._check_state(state, flags=expected_flags)
                    arrays, _ = self._read_clones('without_'+role)
                    counterfactuals[role] = arrays
                    self.last_annotation_arrays['without_'+role+'_depth_m'] = arrays['depth_m'].copy()
                    self.last_annotation_arrays['without_'+role+'_rgb'] = arrays['rgb'].copy()
                    self._check_state(state, flags=expected_flags)
                finally:
                    self.sim.simSetObjectSpecialProperty(self.role_nodes[role].get_handle(), state['flags'][role])
                    self._check_state(state)
                    role_receipt['flags_restored'] = True
                    self._warmup('restore_'+role+'_normal_warmup', state)
                    normal, normal_equal, details = self._read_normal('restore_'+role+'_normal_capture')
                    same = bool(normal_equal and _equal_arrays(base, normal))
                    restored_results.append(same)
                    role_receipt.update(restored_RGBD_equal=same, restored_camera_comparison=details)
                    self._check_state(state)
                if not same:
                    raise DepthOwnerCaptureError('restored normal RGBD is not byte-identical to base')
            self.last_annotation_arrays = _labels(base, counterfactuals)
            receipt['per_view_counts'] = {name: {key: int(self.last_annotation_arrays[key][j].sum()) for key in
                ('cube_raw_positive', 'cube_negative_delta', 'cube_one_pixel_boundary', 'cube_depth_owner',
                 'goal_raw_positive', 'goal_depth_owner', 'goal_rgb_only_unknown', 'semantic_unknown')}
                for j, name in enumerate(CAMERAS)}
        except BaseException as exc:
            error = exc
        finally:
            for role, value in state['flags'].items():
                try:
                    if self._flags()[role] != value:
                        self.sim.simSetObjectSpecialProperty(self.role_nodes[role].get_handle(), value)
                except Exception as exc: restoration_errors.append('visibility restore: '+str(exc))
            for camera, flag in saved_handling:
                try:
                    if camera.get_explicit_handling() != flag: camera.set_explicit_handling(flag)
                except Exception as exc: restoration_errors.append('explicit handling restore: '+str(exc))
            try:
                self._check_state(state)
                receipt['all_counterfactual_flags_restored'] = self._flags() == state['flags']
                receipt['clock_unchanged'] = old._clock(self.backend) == state['clock']
                receipt['clock_after'], receipt['control_clock_after'] = old._clock(self.backend)
                receipt['handling_restored'] = all(c.get_explicit_handling() == f for c, f in saved_handling)
            except Exception as exc: restoration_errors.append(str(exc))
            receipt['all_restore_RGBD_equal'] = len(restored_results) == 2 and all(restored_results)
            receipt.update(wall_seconds=time.perf_counter()-began, render_call_count=len(self._calls),
                           completed_render_call_count=sum(r['completed'] for r in self._calls),
                           error=None if error is None else type(error).__name__+': '+str(error),
                           restoration_errors=restoration_errors,
                           label_contract={'cube_and_goal':'current measured-depth attribution; not all RGB contribution or amodal geometry',
                            'positive':'valid without-role depth strictly greater than normal base depth',
                            'unknown':'negative/invalid/conflicting responses, fixed1pixel inner+outer boundary, goal RGB-only response',
                            'goal_alpha':'original translucent state retained, no forced class2 coverage',
                            'Native_handles':'offline role bookkeeping only, never pixel IDs/model inputs'})
            self.last_receipt = receipt
        if restoration_errors:
            raise DepthOwnerCaptureError('; '.join(restoration_errors)) from error
        if error is not None:
            raise error
        return self.last_public_arrays, self.last_annotation_arrays, receipt


def create_depth_owner_cameras(backend, *, warmup_rounds=WARMUP_ROUNDS):
    return DepthOwnerCameraBundle(backend, warmup_rounds=warmup_rounds)
