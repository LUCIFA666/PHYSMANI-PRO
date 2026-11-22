"""Experimental SO3 history fit using public past/current observations only.

No task, simulator, native target, label or future-state inputs are accepted.
The fit is a constant-angular-velocity diagnostic, not a calibrated probability.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RotationHistoryFit:
    cutoff: float
    start: float
    frames: int
    angular_velocity_world: np.ndarray
    smoothed_current_rotation: np.ndarray
    residual_rms_rad: float


def fit_rotation_history(*, times, rotations, valid, orientation_observable,
                         modes, confidence, cutoff, window_seconds=.95):
    times = np.asarray(times, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    observable = np.asarray(orientation_observable, dtype=bool)
    modes = np.asarray(modes)
    confidence = np.asarray(confidence, dtype=float)
    n = len(times)
    if (times.ndim != 1 or n < 5 or rotations.shape != (n, 3, 3)
            or any(x.shape != (n,) for x in (valid, observable, modes, confidence))
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isin(modes, (0, 1, 2)).all()
            or not np.isfinite(cutoff) or abs(times[-1] - cutoff) > 1e-8
            or not np.isfinite(window_seconds) or not .2 <= window_seconds <= .95):
        raise ValueError('finite causal ordered history ending at exact cutoff required')
    if not valid[-1] or not observable[-1] or modes[-1] == 1:
        raise ValueError('a visible orientation-observable external/released current body is required')
    begin = n - 1
    while (begin > 0 and n - begin < 20 and times[-1] - times[begin - 1] <= window_seconds + 1e-8
           and times[begin] - times[begin - 1] <= .075
           and valid[begin - 1] and observable[begin - 1] and modes[begin - 1] == modes[-1]):
        begin -= 1
    if n - begin < 5 or times[-1] - times[begin] < .2 - 1e-8:
        raise ValueError('five contiguous same-mode public rotation frames over at least .2s required')
    R = rotations[begin:].copy()
    q = confidence[begin:]
    if (not np.isfinite(R).all() or not np.isfinite(q).all() or np.any((q <= 0) | (q > 1))
            or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4)
            or np.any(np.linalg.det(R) < .9999)):
        raise ValueError('valid public SO3 matrices and positive bounded confidence required')
    tangent = Rotation.from_matrix(R @ R[-1].T).as_rotvec()
    if np.linalg.norm(tangent, axis=1).max() > 2.5:
        raise ValueError('history tangent exceeds the declared local-chart bound')
    design = np.c_[np.ones(len(R)), times[begin:] - cutoff]
    weighted = np.sqrt(q)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(design * weighted, tangent * weighted, rcond=None)
    if rank != 2 or not np.isfinite(coefficients).all():
        raise ValueError('rotation history fit is not identifiable')
    fitted_tangent = design @ coefficients
    rms = float(np.sqrt(np.average(np.sum((tangent - fitted_tangent) ** 2, axis=1), weights=q)))
    smoothed = Rotation.from_rotvec(coefficients[0]).as_matrix() @ R[-1]
    return RotationHistoryFit(float(cutoff), float(times[begin]), len(R), coefficients[1].copy(), smoothed, rms)


def forecast_rotation(fit, horizon_seconds, *, current_rotation=None):
    if not isinstance(fit, RotationHistoryFit) or not np.isfinite(horizon_seconds) or not 0 <= horizon_seconds <= 1.1:
        raise ValueError('history fit and declared prediction horizon0..1.1s required')
    R0 = fit.smoothed_current_rotation if current_rotation is None else np.asarray(current_rotation, dtype=float)
    if (R0.shape != (3, 3) or not np.isfinite(R0).all()
            or not np.allclose(R0.T @ R0, np.eye(3), atol=1e-4) or np.linalg.det(R0) < .9999):
        raise ValueError('SO3 current rotation required')
    return Rotation.from_rotvec(fit.angular_velocity_world * horizon_seconds).as_matrix() @ R0
