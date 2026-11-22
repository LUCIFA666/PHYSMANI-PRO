"""Old full V8 exactpass, then same masks with original EF at scope12 only.

No axis-general implementation or runtime toggle is admitted in this version.
The unchanged NN is run once by callers. No labels/robot/history argument.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from . import VERSION
from ..cube_position_measurement_20261103_v8.prediction import (
    predict_current_position as legacy_predict,
    _check_probabilities,PROPOSAL_THRESHOLDS,
)
from ..cube_position_measurement_20261103_v8.evaluation import validate_prediction_route as validate_legacy_route
from ..near_goal_cube_geometry_scope12_20261103_v1 import measure_current_cube_scope12,VERSION as SCOPE_GEOMETRY_VERSION


def prediction_route(prediction):
    return 'scope12_EF' if prediction.get('scope12_prediction_version')==VERSION else 'legacy_v8'


def predict_current_position(inputs,crops,probabilities):
    p=_check_probabilities(probabilities)
    legacy=legacy_predict(inputs,crops,p)
    return _after_legacy(inputs,crops,p,legacy)


def _after_legacy(inputs,crops,p,legacy):
    if legacy['position_measurement_available']:
        return legacy  # Preserve the entire original object/route/support tree.
    proposals=[]
    for threshold in PROPOSAL_THRESHOLDS:
        mask=(p[:,1]>=threshold)&crops['valid_crop']
        measurement=measure_current_cube_scope12(inputs,crops,mask)
        if type(measurement.get('refiner_available'))is not bool:
            raise ValueError('Explicit original EF availability required')
        proposals.append(dict(threshold=threshold,geometry_available=measurement['refiner_available'],
            reasons=list(measurement['reasons']),measured_points=measurement['measured_points'],
            visible_views=measurement['visible_views'],geometry_confidence=measurement['geometry_confidence']))
        if measurement['refiner_available']or threshold==PROPOSAL_THRESHOLDS[-1]:
            available=measurement['refiner_available']
            return dict(measurement=measurement,threshold=threshold,
                selected_threshold=threshold if available else None,
                proposal_thresholds=PROPOSAL_THRESHOLDS,thresholds_tried=[r['threshold']for r in proposals],
                proposal_trace=proposals,proposal_route=('scope12_first_0p995'if threshold==.995 else'scope12_fallback_0p9')if available else'UNKNOWN',
                legacy_thresholds_tried=deepcopy(legacy['thresholds_tried']),
                legacy_proposal_trace=deepcopy(legacy['proposal_trace']),legacy_proposal_route='UNKNOWN',
                position_measurement_available=available,centre=measurement['centre']if available else None,
                raw_Cube_mask=mask,raw_mask_qualified=False,geometry_version=SCOPE_GEOMETRY_VERSION,
                completion=False,adaptive_public_proposal_version=VERSION,scope12_prediction_version=VERSION,
                scope_radius_m=.12,axis_general_runtime_enabled=False,route_uses_GT_or_future=False)
    raise RuntimeError('Fixed scope12 proposal prefix exhausted without output')


def _prediction_sha(prediction):
    def clean(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:clean(v)for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v)for v in value]
        return value
    value={k:v for k,v in prediction.items()if k not in ('scene_id','support_array_index','legacy_audit')}
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def predict_with_legacy_audit(inputs,crops,probabilities):
    """Qualification-only public audit, with one complete legacy evaluation."""
    p=_check_probabilities(probabilities)
    legacy=legacy_predict(inputs,crops,p)
    result=_after_legacy(inputs,crops,p,legacy)
    audit=dict(legacy_position_measurement_available=legacy['position_measurement_available'],
        legacy_centre=deepcopy(legacy['centre']),legacy_selected_threshold=legacy['selected_threshold'],
        legacy_prediction_sha256=_prediction_sha(legacy),selected_prediction_sha256=_prediction_sha(result),
        legacy_pass_returned_exact_object=(result is legacy)if legacy['position_measurement_available']else False)
    validate_legacy_audit(result,audit)
    return result,audit


def validate_legacy_audit(prediction,audit):
    available=audit.get('legacy_position_measurement_available')
    if (type(available)is not bool or type(audit.get('legacy_pass_returned_exact_object'))is not bool
        or audit.get('selected_prediction_sha256')!=_prediction_sha(prediction)):
        raise ValueError('Exact frozen public legacy comparison required')
    if available:
        if (prediction_route(prediction)!='legacy_v8' or prediction['position_measurement_available']is not True
            or audit['legacy_pass_returned_exact_object']is not True
            or audit.get('legacy_prediction_sha256')!=audit['selected_prediction_sha256']
            or audit.get('legacy_centre')!=prediction['centre']
            or audit.get('legacy_selected_threshold')!=prediction['selected_threshold']):
            raise ValueError('Every complete legacy pass must remain exactly unchanged')
    elif (prediction_route(prediction)!='scope12_EF' or audit['legacy_pass_returned_exact_object']is not False
        or audit.get('legacy_centre')is not None or audit.get('legacy_selected_threshold')is not None):
        raise ValueError('Scope fallback requires complete legacy UNKNOWN, no old centre')


def validate_prediction_route(prediction):
    if prediction_route(prediction)=='legacy_v8':
        validate_legacy_route(prediction)
        if prediction['position_measurement_available']is not True:
            raise ValueError('Legacy UNKNOWN cannot bypass the scope12 public prefix')
        return
    available=prediction['measurement'].get('refiner_available')
    trace=prediction.get('proposal_trace');tried=prediction.get('thresholds_tried')
    old=prediction.get('legacy_proposal_trace')
    if (type(available)is not bool or prediction.get('position_measurement_available')is not available
        or prediction.get('scope12_prediction_version')!=VERSION
        or prediction.get('adaptive_public_proposal_version')!=VERSION
        or prediction.get('raw_mask_qualified')is not False or prediction.get('completion')is not False
        or prediction.get('geometry_version')!=SCOPE_GEOMETRY_VERSION
        or prediction['measurement'].get('geometry_profile')!=SCOPE_GEOMETRY_VERSION
        or prediction.get('scope_radius_m')!=.12 or prediction.get('axis_general_runtime_enabled')is not False
        or prediction.get('route_uses_GT_or_future')is not False
        or prediction.get('legacy_thresholds_tried')!=[.995,.9]
        or prediction.get('legacy_proposal_route')!='UNKNOWN'
        or not isinstance(old,list)or len(old)!=2
        or [r['threshold']for r in old]!=[.995,.9]or any(r['geometry_available']is not False for r in old)
        or not isinstance(trace,list)or not 1<=len(trace)<=2
        or tried!=[r['threshold']for r in trace]
        or list(prediction.get('proposal_thresholds',()))!=[.995,.9]):
        raise ValueError('Exact legacyUNKNOWN then fixed scope12 EF prefix required; no axis route')
    if tried==[.995]:
        ok=available is True and prediction.get('proposal_route')=='scope12_first_0p995'and trace[0]['geometry_available']is True
    elif tried==[.995,.9]:
        ok=(trace[0]['geometry_available']is False and trace[1]['geometry_available']is available
            and prediction.get('proposal_route')==('scope12_fallback_0p9'if available else'UNKNOWN'))
    else:ok=False
    if (not ok or prediction.get('threshold')!=tried[-1]
        or prediction.get('selected_threshold')!=(tried[-1]if available else None)
        or (not available and prediction.get('centre')is not None)):
        raise ValueError('No proposal reordering, uncertified axis or unknown policy centre')
    if available:
        centre=np.asarray(prediction.get('centre'),float)
        measured=np.asarray(prediction['measurement'].get('centre'),float)
        if (centre.shape!=(3,)or not np.isfinite(centre).all()or measured.shape!=(3,)
            or not np.array_equal(centre,measured)):
            raise ValueError('Finite exact current measurement centre required')
