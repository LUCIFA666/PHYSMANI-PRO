"""RGB-D geometry and a causal, timestamped rigid point tracker.

Segmentation is supplied by a perception backend; simulator labels must be
declared oracle inputs in experiments. Correspondence fitting never sees future frames.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contracts import ObjectState, Pose, array


@dataclass(frozen=True)
class CameraCalibration:
    intrinsics: np.ndarray
    camera_to_world: Pose
    depth_scale: float = 1.0
    timestamp_offset: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "intrinsics", array(self.intrinsics, (3, 3)))
        # CoppeliaSim stores signed focal lengths according to its camera axes.
        # Preserve their signs together with the supplied extrinsic frame.
        if self.depth_scale <= 0 or abs(np.linalg.det(self.intrinsics)) < 1e-12 or abs(self.intrinsics[0, 0]) < 1e-8 or abs(self.intrinsics[1, 1]) < 1e-8:
            raise ValueError("invalid camera/depth scale")

    def backproject(self, depth, mask):
        z = np.asarray(depth, dtype=float)*self.depth_scale
        if z.ndim != 2 or np.asarray(mask).shape != z.shape:
            raise ValueError("depth and segmentation must be aligned H x W images")
        y, x = np.nonzero(np.asarray(mask, dtype=bool) & np.isfinite(z) & (z > 0))
        uv = np.c_[x, y, np.ones(len(x))]
        rays = uv @ np.linalg.inv(self.intrinsics).T
        return self.camera_to_world.apply(rays*z[y, x, None])


def fuse_views(views, reference_time, max_skew=0.035):
    """Each view is (calibration, timestamp, depth_metres_or_raw, target_mask)."""
    points = []
    for calibration, timestamp, depth, mask in views:
        aligned_time = timestamp + calibration.timestamp_offset
        if abs(aligned_time-reference_time) > max_skew:
            raise ValueError("camera frame outside synchronization window")
        points.append(calibration.backproject(depth, mask))
    result = np.concatenate(points) if points else np.empty((0, 3))
    if len(result) < 3:
        raise ValueError("insufficient visible target geometry")
    return result


def rigid_fit(source, target):
    """Kabsch on corresponding points, with reflections corrected."""
    source, target = np.asarray(source), np.asarray(target)
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    return Pose(b-r@a, r)


class CausalTracker:
    def __init__(self, object_id, window=5, min_dt=0.005, max_gap=0.2, max_match=0.08):
        self.object_id = object_id
        self.history = deque(maxlen=window)
        self.min_dt, self.max_gap, self.max_match = min_dt, max_gap, max_match

    def update(self, points_world, timestamp, *, corresponded=False, confidence=1.0):
        points = array(points_world)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("invalid point cloud")
        rotation = np.eye(3)
        angular = np.zeros(3)
        if self.history:
            previous = self.history[-1]
            dt = timestamp-previous.timestamp
            if dt < self.min_dt:
                raise ValueError("non-monotonic or duplicate observation timestamp")
            if dt > self.max_gap:
                self.history.clear()  # reacquisition starts a new causal track
            else:
                if corresponded:
                    if points.shape != previous.points_world.shape:
                        raise ValueError("correspondence shape changed")
                    transform = rigid_fit(previous.points_world, points)
                else:
                    # Translation-initialized nearest-neighbour association.
                    old = previous.points_world
                    shifted = old + points.mean(0)-old.mean(0)
                    distances, indices = cKDTree(points).query(shifted)
                    valid = distances < self.max_match
                    if valid.sum() < 3:
                        raise ValueError("target association lost")
                    transform = rigid_fit(old[valid], points[indices[valid]])
                    confidence *= float(valid.mean())
                rotation = transform.rotation @ previous.pose.rotation
                angular = Rotation.from_matrix(transform.rotation).as_rotvec()/dt
        pose = Pose(points.mean(0), rotation)
        # Least-squares slope over only the available past centroid observations.
        samples = [(s.timestamp, s.pose.position) for s in self.history] + [(timestamp, pose.position)]
        velocity = np.zeros(3)
        if len(samples) > 1:
            times = np.array([s[0]-timestamp for s in samples])
            positions = np.array([s[1] for s in samples])
            centered = times-times.mean()
            velocity = (centered[:, None]*(positions-positions.mean(0))).sum(0)/(centered@centered)
        state = ObjectState(self.object_id, timestamp, pose, points, velocity, angular, confidence)
        self.history.append(state)
        return state
