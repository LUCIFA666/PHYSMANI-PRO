"""Commands distinguish a smoothed destination from already sampled references.

All joint positions are absolute radians in the backend's declared joint order.
Trajectory points are references for t0+50ms, t0+100ms, ... . They are sent to
the native POSITION servo unchanged, not assigned to actual joint state. Neither
command promises an actual arrival time or motor-velocity feedforward.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence


def vector(value, name, size=None):
    if isinstance(value, (str, bytes)):
        raise ValueError(name + " must be a finite numeric vector")
    try:
        result = tuple(float(v) for v in value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(name + " must be a finite numeric vector") from error
    if not result or (size is not None and len(result) != size) or not all(map(isfinite, result)):
        raise ValueError(name + " has invalid length or nonfinite values")
    return result


def opening(value):
    if value is None:
        return None
    # Partial apertures used to be silently thresholded. This API is binary.
    if isinstance(value, (str, bytes)) or value not in (0, 1):
        raise ValueError("opening must be 0=CLOSE, 1=OPEN, or None=retain")
    return float(value)


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("a finite nonnegative simulator timestamp is required")
    value = float(value)
    if not isfinite(value) or value < 0:
        raise ValueError("a finite nonnegative simulator timestamp is required")
    return value


def names(value):
    if value is None:
        return None
    value = tuple(value)
    if not value or any(not isinstance(n, str) or not n for n in value) or len(set(value)) != len(value):
        raise ValueError("distinct nonempty joint names are required")
    return value


@dataclass(frozen=True)
class JointGoal:
    positions: Sequence[float]
    opening: Optional[float] = None
    ticks: int = 2
    expected_time: Optional[float] = None
    joint_names: Optional[Sequence[str]] = None

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions, "joint goal"))
        object.__setattr__(self, "opening", opening(self.opening))
        object.__setattr__(self, "expected_time", timestamp(self.expected_time))
        object.__setattr__(self, "joint_names", names(self.joint_names))
        if type(self.ticks) is not int or self.ticks not in (1, 2):
            raise ValueError("a receding command executes one or two native ticks")


@dataclass(frozen=True)
class JointTrajectory:
    start_time: float
    positions: Sequence[Sequence[float]]
    times_from_start: Sequence[float]
    opening: Optional[float] = None
    joint_names: Optional[Sequence[str]] = None

    def __post_init__(self):
        at = timestamp(self.start_time)
        if at is None:
            raise ValueError("trajectory start_time is required; stale points cannot be retimed")
        points = tuple(vector(row, "trajectory reference") for row in self.positions)
        times = vector(self.times_from_start, "trajectory times")
        if len(points) not in (1, 2) or len(times) != len(points):
            raise ValueError("supply one or two time-aligned reference points")
        if any(len(row) != len(points[0]) for row in points):
            raise ValueError("all trajectory references must have the same joint dimension")
        if any(abs(t - (i + 1) * .05) > 1e-9 for i, t in enumerate(times)):
            raise ValueError("trajectory samples must be at +50ms and optionally +100ms")
        object.__setattr__(self, "start_time", at)
        object.__setattr__(self, "positions", points)
        object.__setattr__(self, "times_from_start", times)
        object.__setattr__(self, "opening", opening(self.opening))
        object.__setattr__(self, "joint_names", names(self.joint_names))
