"""Offline V60 geometry experiment; no producer or live binding approval.

The feasible centre box is conditional on a correctly associated, world-axis
aligned 5 cm Cube and a declared 6 mm per-axis point error allowance.  The
allowance is a development hypothesis, not a calibrated error distribution.
This function proves neither semantic ownership nor physical separation.
"""
import numpy as np

HALF_SIDE_M = .025
POINT_AXIS_ALLOWANCE_M = .006
NEAR_CENTRE_SCOPE_M = .08


def component_far_consistency(component_world, sphere_center):
    """Return FAR_CONSISTENT only if all compatible centres exclude the scope.

    Only the actual currently selected component belongs in ``component_world``.
    Completed geometry, forecasts and simulator labels are not admissible
    component inputs.  This pure numeric function cannot authenticate that
    provenance; the offline caller and any future binding must do so.

    Invalid/empty/inconsistent components and touching bounds return UNKNOWN.
    No pose, velocity, visible flag or NN availability is manufactured.
    """
    p = np.asarray(component_world)
    s = np.asarray(sphere_center)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.dtype.kind not in 'fiu':
        raise ValueError('numeric current component [N,3] required')
    if s.shape != (3,) or s.dtype.kind not in 'fiu':
        raise ValueError('numeric initial public sphere centre [3] required')
    p, s = p.astype(np.float64, copy=False), s.astype(np.float64, copy=False)
    result = dict(status='UNKNOWN', far_consistent=False, reason=None,
        point_count=len(p), centre_lower_m=None, centre_upper_m=None,
        minimum_compatible_centre_distance_m=None, numerical_boundary_guard_m=None,
        half_side_m=HALF_SIDE_M, point_axis_allowance_m=POINT_AXIS_ALLOWANCE_M,
        near_centre_scope_m=NEAR_CENTRE_SCOPE_M,
        statistical_or_physical_guarantee=False, semantic_ownership_verified=False,
        NN_measurement_available=False, known_axis_aligned_Cube_assumed=True)
    if not np.isfinite(s).all() or not np.isfinite(p).all():
        result['reason'] = 'nonfinite_component_or_initial_sphere'
        return result
    if not len(p):
        result['reason'] = 'empty_current_component'
        return result
    with np.errstate(over='ignore', invalid='ignore'):
        lo = p.max(0)-HALF_SIDE_M-POINT_AXIS_ALLOWANCE_M
        hi = p.min(0)+HALF_SIDE_M+POINT_AXIS_ALLOWANCE_M
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        result['reason'] = 'derived_geometry_nonfinite_or_overflow'
        return result
    result.update(centre_lower_m=lo.tolist(), centre_upper_m=hi.tolist())
    if np.any(lo > hi):
        result['reason'] = 'empty_compatible_centre_box'
        return result
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.maximum(np.maximum(lo-s, s-hi), 0.)
        distance = float(np.linalg.norm(delta))
    if not np.isfinite(delta).all() or not np.isfinite(distance):
        result['reason'] = 'derived_geometry_nonfinite_or_overflow'
        return result
    # Conservative floating arithmetic only; this is not another physical
    # uncertainty/tuning parameter.  A numerically touching sphere stays unknown.
    scale = max(1., float(np.abs(s).max()), float(np.abs(lo).max()), float(np.abs(hi).max()))
    guard = 16*np.finfo(np.float64).eps*scale
    if not np.isfinite(guard):
        result['reason'] = 'derived_geometry_nonfinite_or_overflow'
        return result
    result.update(minimum_compatible_centre_distance_m=distance,
                  numerical_boundary_guard_m=guard)
    if distance-NEAR_CENTRE_SCOPE_M <= guard:
        result['reason'] = 'compatible_centres_intersect_or_touch_near_scope'
        return result
    result.update(status='FAR_CONSISTENT', far_consistent=True,
                  reason='all_compatible_centres_outside_declared_near_scope')
    return result
