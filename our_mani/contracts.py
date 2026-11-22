"""Shared contracts: metres, seconds, radians; quaternion convention is xyzw."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def array(value, shape=None):
    result = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("non-finite geometry")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        object.__setattr__(self, "position", array(self.position, (3,)))
        object.__setattr__(self, "rotation", array(self.rotation, (3, 3)))
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-5):
            raise ValueError("rotation must be right handed")

    @classmethod
    def from_xyzw(cls, values):
        v = array(values, (7,))
        if np.linalg.norm(v[3:]) < 1e-8:
            raise ValueError("zero quaternion")
        return cls(v[:3], Rotation.from_quat(v[3:]).as_matrix())

    def xyzw(self):
        # Older SciPy builds used by the simulator require a writable buffer.
        return np.r_[self.position, Rotation.from_matrix(self.rotation.copy()).as_quat()]

    def apply(self, points):
        return np.asarray(points) @ self.rotation.T + self.position

    def inverse_apply(self, points):
        return (np.asarray(points) - self.position) @ self.rotation


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    timestamp: float
    pose: Pose
    points_world: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    confidence: float = 1.0
    visible: bool = True

    def __post_init__(self):
        p = array(self.points_world)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
            raise ValueError("object needs at least three world points")
        if not np.isfinite(self.timestamp) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid observation time/confidence")
        object.__setattr__(self, "points_world", p)
        object.__setattr__(self, "linear_velocity", array(self.linear_velocity, (3,)))
        object.__setattr__(self, "angular_velocity", array(self.angular_velocity, (3,)))


@dataclass(frozen=True)
class RobotState:
    timestamp: float
    pose: Pose
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joints: np.ndarray
    joint_velocities: np.ndarray
    gripper_open: bool = True
    contact: bool = False
    task_success: bool = False

    def __post_init__(self):
        for key in ("linear_velocity", "angular_velocity"):
            object.__setattr__(self, key, array(getattr(self, key), (3,)))
        object.__setattr__(self, "joints", array(self.joints))
        object.__setattr__(self, "joint_velocities", array(self.joint_velocities, self.joints.shape))
        if self.joints.ndim != 1 or not np.isfinite(self.timestamp):
            raise ValueError("invalid robot state")


@dataclass(frozen=True)
class Forecast:
    object_ids: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray  # [time, object, xyz], object frame origins
    rotations: np.ndarray  # [time, object, 3, 3]
    linear_velocities: np.ndarray
    angular_velocities: np.ndarray
    points_world: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        t = array(self.times)
        n = len(self.object_ids)
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("forecast times must increase")
        if len(set(self.object_ids)) != n:
            raise ValueError("duplicate object IDs")
        object.__setattr__(self, "times", t)
        for key in ("positions", "linear_velocities", "angular_velocities"):
            object.__setattr__(self, key, array(getattr(self, key), (len(t), n, 3)))
        object.__setattr__(self, "rotations", array(self.rotations, (len(t), n, 3, 3)))
        for r in self.rotations.reshape(-1, 3, 3):
            if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or np.linalg.det(r) < 0:
                raise ValueError("invalid forecast rotation")

    def at(self, object_id, timestamp):
        """Interpolate only inside the forecast; never silently clamp old goals."""
        from scipy.spatial.transform import Slerp
        if timestamp < self.times[0] - 1e-8 or timestamp > self.times[-1] + 1e-8:
            raise ValueError("time outside forecast")
        timestamp = float(np.clip(timestamp, self.times[0], self.times[-1]))
        o = self.object_ids.index(object_id)
        hi = min(max(int(np.searchsorted(self.times, timestamp, side="right")), 1), len(self.times)-1)
        lo = hi - 1
        weight = (timestamp - self.times[lo]) / (self.times[hi] - self.times[lo])
        interp = lambda v: (1-weight)*v[lo, o] + weight*v[hi, o]
        r = Slerp(self.times[lo:hi+1], Rotation.from_matrix(self.rotations[lo:hi+1, o].copy()))([timestamp]).as_matrix()[0]
        return Pose(interp(self.positions), r), interp(self.linear_velocities), interp(self.angular_velocities)


@dataclass(frozen=True)
class HandoverRegion:
    region_id: str
    object_id: str
    local_pose: Pose
    radii: np.ndarray
    max_relative_speed: float = 0.12
    max_orientation_error: float = 0.25
    value: float = 0.5
    source: str = "configured_untrained"

    def __post_init__(self):
        object.__setattr__(self, "radii", array(self.radii, (3,)))
        if np.any(self.radii <= 0) or self.max_relative_speed <= 0 or not 0 < self.max_orientation_error <= np.pi:
            raise ValueError("invalid handover tolerances")
        if not 0 <= self.value <= 1:
            raise ValueError("value must be a probability")

    def in_world(self, object_pose, object_velocity, angular_velocity):
        offset = object_pose.rotation @ self.local_pose.position
        pose = Pose(object_pose.position + offset, object_pose.rotation @ self.local_pose.rotation)
        velocity = object_velocity + np.cross(angular_velocity, offset)
        return pose, velocity


@dataclass(frozen=True)
class Reachability:
    feasible: bool
    reason: str
    minimum_duration: float = 0.0
    joints: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterceptGoal:
    goal_id: int
    observation_time: float
    arrival_time: float
    region: HandoverRegion
    pose: Pose
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    forecast: Forecast
    reachability: Reachability
    score: float
