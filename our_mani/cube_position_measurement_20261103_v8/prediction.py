"""Two fixed current public proposals; no label, history or route oracle.

This module does not grant position or raw-model qualification. The second
proposal is considered only after the first F geometry is unavailable.
"""
import numpy as np

from ..near_goal_cube_geometry_20261103_v1 import measure_current_cube, VERSION as GEOMETRY_VERSION
from ..near_goal_depth_refiner_20261103.data import validate_public

VERSION='current_Cube_position_adaptive_public_proposals_fixed995_then90_20261103_v8'
PROPOSAL_THRESHOLDS=(.995,.9)


def _check_probabilities(probabilities):
    p=np.asarray(probabilities)
    if (p.shape!=(4,2,64,64) or p.dtype!=np.float32 or not np.isfinite(p).all()
        or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=0,atol=1e-5)):
        raise ValueError('Unchanged normalized current4view float32 probabilities required')
    return p


def route_current_public_proposals(crops,probabilities,proposals):
    """Replay already frozen current public F proposals without labels.

    Prefix1 is valid only when .995 is available. Otherwise exact prefix2 is
    required. The first available proposal wins; no scored quality enters.
    """
    p=_check_probabilities(probabilities)
    domain=np.asarray(crops['valid_crop'])
    if domain.shape!=(4,64,64) or domain.dtype!=np.bool_:
        raise ValueError('Exact current observed crop domain required')
    if not isinstance(proposals,(tuple,list)) or not 1<=len(proposals)<=2:
        raise ValueError('Only the fixed ordered public proposal prefix allowed')
    trace=[]
    for index,(threshold,measurement) in enumerate(proposals):
        if (type(threshold) is not float or threshold!=PROPOSAL_THRESHOLDS[index]
            or not isinstance(measurement,dict)
            or type(measurement.get('refiner_available')) is not bool
            or measurement.get('geometry_profile')!=GEOMETRY_VERSION):
            raise ValueError('Exact .995 then .9 F current proposals required')
        available=measurement['refiner_available']
        trace.append(dict(threshold=threshold,geometry_available=available,
            reasons=list(measurement['reasons']),measured_points=measurement['measured_points'],
            visible_views=measurement['visible_views'],geometry_confidence=measurement['geometry_confidence']))
        if available or index==1:
            if available and (np.asarray(measurement.get('centre')).shape!=(3,)
                              or not np.isfinite(measurement['centre']).all()):
                raise ValueError('Only a finite available current centre can be selected')
            return dict(measurement=measurement,threshold=threshold,
                selected_threshold=threshold if available else None,
                proposal_thresholds=PROPOSAL_THRESHOLDS,thresholds_tried=[x['threshold'] for x in trace],
                proposal_trace=trace,
                proposal_route='first_0p995' if available and index==0 else 'fallback_0p9' if available else 'UNKNOWN',
                position_measurement_available=available,centre=measurement['centre'] if available else None,
                raw_Cube_mask=(p[:,1]>=threshold)&domain,raw_mask_qualified=False,
                geometry_version=GEOMETRY_VERSION,completion=False,
                adaptive_public_proposal_version=VERSION,route_uses_GT_or_future=False)
    raise ValueError('Unavailable .995 proposal requires the second .9 current proposal')


def predict_current_position(inputs,crops,probabilities):
    """Only current public7 RGBD and frozen probabilities; no supervision args."""
    validate_public(inputs)
    p=_check_probabilities(probabilities)
    proposals=[]
    for threshold in PROPOSAL_THRESHOLDS:
        raw=(p[:,1]>=threshold)&crops['valid_crop']
        measurement=measure_current_cube(inputs,crops,raw)
        proposals.append((threshold,measurement))
        if measurement['refiner_available']:
            break
    return route_current_public_proposals(crops,p,proposals)
