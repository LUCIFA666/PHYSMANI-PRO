"""Numerical bridge from current public camera data to the frozen seven keys.

This does not approve a checkpoint or the provenance of an initial sphere fit.
A future live binding must separately certify that fit, calibration identity,
frame order, and a qualified model. No controller uses this module yet.
"""
import numpy as np

from .goal_aware_perception import VisualGoalSphere, _validate_public
from ..near_goal_segmentation_20261103.data import CAMERAS, validate_public


def pack_current_public_refiner_inputs(observation, initial_sphere):
    """Copy the unmodified four current views, with signed K and world T.

    Depth must already be metric float32 at the original 256 resolution.
    Robot state, predictions, counterfactual data and labels are rejected by
    the observation whitelist. The caller owns the causal initial-fit binding.
    """
    _validate_public(observation)
    timestamp = np.asarray(observation['timestamp'])
    if timestamp.shape != () or timestamp.dtype.kind not in 'fi':
        raise ValueError('One numeric actual observation timestamp required')
    now = float(timestamp)
    if type(initial_sphere) is not VisualGoalSphere:
        raise TypeError('Explicit frozen public initial-sphere representation required')
    if initial_sphere.observed_at > now:
        raise ValueError('A future sphere fit cannot enter the current observation')
    intrinsics, transforms, colors, depths = [], [], [], []
    for name in CAMERAS:
        calibration = observation['observation/calibrations'][name]
        if calibration.depth_scale != 1. or calibration.timestamp_offset != 0.:
            raise ValueError('Original metric and synchronized camera calibration required')
        rgb = np.asarray(observation['observation/images'][name])
        depth = np.asarray(observation['observation/depths'][name])
        if rgb.shape != (256, 256, 3) or depth.shape != (256, 256) or depth.dtype != np.float32:
            raise ValueError('Original 256 RGB and metric float32 depth only; no resize or conversion')
        transform = np.eye(4)
        transform[:3, :3] = calibration.camera_to_world.rotation
        transform[:3, 3] = calibration.camera_to_world.position
        intrinsics.append(calibration.intrinsics)
        transforms.append(transform)
        colors.append(rgb)
        depths.append(depth)
    result = dict(rgb=np.stack(colors), depth_m=np.stack(depths),
                  K=np.stack(intrinsics), T=np.stack(transforms),
                  public_sphere_center=np.asarray(initial_sphere.center).copy(),
                  camera_names=np.asarray(CAMERAS), timestamp=np.asarray(now))
    validate_public(result)
    # Every result array is owned by this call; callers cannot mutate the raw
    # observation through it or quietly alter the packed model input later.
    for value in result.values():
        value.setflags(write=False)
    return result
