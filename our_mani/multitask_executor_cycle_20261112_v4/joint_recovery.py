"""Measured joint-bound recovery using legal motor references, never setters."""
import numpy as np


def mechanical_recovery_reference(actual, previous_reference, lower, upper, *, margin=.002):
    actual, reference, lower, upper = [np.asarray(x, float) for x in (actual, previous_reference, lower, upper)]
    if (any(x.shape != (7,) or not np.isfinite(x).all() for x in (actual, reference, lower, upper))
            or np.any(lower+2*margin >= upper)):
        raise ValueError('Finite seven-joint measurements/references and valid native limits required')
    unsafe = (actual < lower+1e-6) | (actual > upper-1e-6)
    if not unsafe.any():
        return None
    goal = reference.copy()
    goal[unsafe] = np.clip(reference[unsafe], lower[unsafe]+margin, upper[unsafe]-margin)
    # Measured overshoot is retained. Only a legal commanded reference may be
    # sent; never clip and relabel the actual sensor value as recovered state.
    return goal, dict(unsafe_joint_indices=np.flatnonzero(unsafe).tolist(),
                      actual_q=actual.tolist(), original_reference=reference.tolist(),
                      recovery_reference=goal.tolist(), inward_reference_margin_rad=margin,
                      recovery_requires_real_physics=True, measured_q_clipped=False,
                      native_limits_relaxed=False)
