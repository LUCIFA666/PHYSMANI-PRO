"""Current-depth free-space evidence for a candidate CAD root-frame pose.

This is a one-sided visibility cost, not a pose estimator or observability
certificate. A surface predicted behind observed depth may be truly occluded
and is never penalized. Only a surface in observed free space is contradictory.
No masks, native poses, future frames, task handles, or outcomes are inputs.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..contracts import Pose
from ..perception import CameraCalibration


SCHEMA = 'r10_current_depth_CAD_free_space_cost_v1'


@dataclass(frozen=True)
class FreeSpaceConfig:
    tolerance_m: float = .004
    maximum_gap_m: float = .03
    weight: float = 2.
    maximum_camera_age_seconds: float = .035

    def __post_init__(self):
        values = (self.tolerance_m, self.maximum_gap_m, self.weight,
                  self.maximum_camera_age_seconds)
        if (any(type(value) not in (int, float) or not np.isfinite(value) for value in values)
                or min(self.tolerance_m, self.weight, self.maximum_camera_age_seconds) < 0
                or self.maximum_gap_m <= 0):
            raise ValueError('finite nonnegative tolerance/weight/age and positive clipping gap required')


def current_depth_pose_cost(surface_pieces, candidate_pose: Pose, views: Mapping,
                            *, config=FreeSpaceConfig(), reference_time=None,
                            view_timestamps=None):
    """Return a bounded m² cost from validated current public depth views.

    ``surface_pieces`` is a mapping or sequence of Nx3 surface arrays expressed
    in the candidate root CAD frame. ``views[camera]`` is
    ``(CameraCalibration, current_depth)``; a third current mask element is
    accepted for direct use with the tracker's already synchronized views and
    ignored. Every projected pixel contributes only its nearest CAD surface,
    avoiding duplicate-surface or point-density weighting within a ray.

    Cost = weight * mean(clip(measured_z - model_z - tolerance, 0, maxgap)²)
    over all in-frustum model pixels with finite positive current depth.
    The denominator includes consistent and truly occluded valid rays. Missing
    depth and unobserved rays supply no evidence and no penalty. A zero cost
    with zero evaluated rays therefore is not a successful pose test.

    Normally ``views`` comes directly from the caller's current synchronized
    observation. For raw caches, pass ``reference_time`` and all camera
    ``view_timestamps`` together: future or stale views are then rejected,
    after applying the public calibration's timestamp offset.
    """
    if not isinstance(candidate_pose, Pose) or not isinstance(config, FreeSpaceConfig):
        raise ValueError('pure candidate Pose and explicit FreeSpaceConfig required')
    if not isinstance(views, Mapping) or not views:
        raise ValueError('at least one current public calibration/depth view required')
    if (reference_time is None) != (view_timestamps is None):
        raise ValueError('supply both current reference_time and per-view timestamps')
    if reference_time is not None:
        if (not np.isfinite(reference_time) or not isinstance(view_timestamps, Mapping)
                or set(view_timestamps) != set(views)):
            raise ValueError('finite current time and exact public view timestamp coverage required')
    chunks = list(surface_pieces.values()) if isinstance(surface_pieces, Mapping) else list(surface_pieces)
    pieces = []
    for points in chunks:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('finite Nx3 local CAD surface pieces required')
        if len(points):
            pieces.append(points)
    local = np.concatenate(pieces) if pieces else np.empty((0, 3))
    world = candidate_pose.apply(local)
    totals = dict(model_samples=len(local), projected_samples=0, behind_camera_samples=0,
                  outside_frustum_samples=0, deduplicated_model_pixels=0,
                  invalid_depth_pixels=0, evaluated_pixels=0, contradictory_pixels=0,
                  occluded_pixels=0, consistent_pixels=0, clipped_contradictory_pixels=0)
    per_view, squared_sum = {}, 0.
    for camera, view in views.items():
        if not isinstance(view, (tuple, list)) or len(view) not in (2, 3):
            raise ValueError('views must contain only calibration, current depth, optional ignored current mask')
        calibration, depth = view[:2]
        if not isinstance(calibration, CameraCalibration):
            raise ValueError('public CameraCalibration required, no environment or native handle')
        if reference_time is not None:
            observed = float(view_timestamps[camera]) + calibration.timestamp_offset
            age = float(reference_time) - observed
            if not np.isfinite(age) or age < -1e-9 or age > config.maximum_camera_age_seconds + 1e-9:
                raise ValueError('future or stale camera depth cannot contribute current-pose evidence')
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2 or min(depth.shape, default=0) <= 0:
            raise ValueError('current depth must be a nonempty HxW image')
        points = calibration.camera_to_world.inverse_apply(world)
        positive = points[:, 2] > 1e-8
        z = points[positive, 2]
        projected = points[positive] @ calibration.intrinsics.T
        uv = projected[:, :2] / projected[:, 2:]
        rounded = np.rint(uv)
        h, w = depth.shape
        inside = (np.isfinite(rounded).all(1) & (rounded[:, 0] >= 0) & (rounded[:, 0] < w)
                  & (rounded[:, 1] >= 0) & (rounded[:, 1] < h))
        uv, z = rounded[inside].astype(np.int64), z[inside]
        pixels = uv[:, 1] * w + uv[:, 0]
        # Sorting pixel first, depth second produces deterministic frontmost
        # CAD samples without introducing a hidden rasterizer or live geometry.
        order = np.lexsort((z, pixels))
        unique = np.r_[True, pixels[order][1:] != pixels[order][:-1]] if len(order) else np.zeros(0, bool)
        selected = order[unique]
        uv, z = uv[selected], z[selected]
        measured = depth[uv[:, 1], uv[:, 0]] * calibration.depth_scale
        valid = np.isfinite(measured) & (measured > 0)
        difference = measured[valid] - z[valid]
        contradictory = difference > config.tolerance_m
        occluded = difference < -config.tolerance_m
        gaps = np.clip(difference - config.tolerance_m, 0., config.maximum_gap_m)
        subtotal = float(gaps @ gaps)
        count = dict(projected_samples=int(inside.sum()), behind_camera_samples=int((~positive).sum()),
            outside_frustum_samples=int((~inside).sum()), deduplicated_model_pixels=len(uv),
            invalid_depth_pixels=int((~valid).sum()), evaluated_pixels=int(valid.sum()),
            contradictory_pixels=int(contradictory.sum()), occluded_pixels=int(occluded.sum()),
            consistent_pixels=int((~contradictory & ~occluded).sum()),
            clipped_contradictory_pixels=int((difference - config.tolerance_m >= config.maximum_gap_m).sum()))
        per_view[str(camera)] = dict(count, unweighted_squared_gap_sum_m2=subtotal,
            cost_m2=config.weight * subtotal / max(1, count['evaluated_pixels']))
        for name, value in count.items():
            totals[name] += value
        squared_sum += subtotal
    unweighted = squared_sum / max(1, totals['evaluated_pixels'])
    return dict(schema=SCHEMA, cost_m2=float(config.weight * unweighted),
        unweighted_cost_m2=float(unweighted), maximum_cost_m2=float(config.weight * config.maximum_gap_m ** 2),
        counts=totals, per_view=per_view, evidence_available=totals['evaluated_pixels'] > 0,
        tolerance_m=config.tolerance_m, maximum_gap_m=config.maximum_gap_m, weight=config.weight,
        timestamp_validation='explicit_current_time_and_public_view_timestamps' if reference_time is not None else
            'caller_supplied_already_synchronized_current_views',
        valid_occlusion_penalized=False, invalid_depth_or_frustum_penalized=False,
        GT_native_or_future_input_used=False, full_pose_observability_proven=False)
