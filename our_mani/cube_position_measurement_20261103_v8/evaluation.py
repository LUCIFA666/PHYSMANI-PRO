"""Current-position qualification; raw-mask qualification stays separate.

Predictions use only current public RGBD. Labels enter after prediction freeze
to score position and the pixels actually used by the geometry decoder.
"""
import hashlib
import json
import numpy as np

from ..near_goal_depth_refiner_20261103.data import make_measurement as legacy_measurement
from ..near_goal_depth_refiner_20261103_v4.evaluation import (
    BASELINE_ELIGIBILITY_PROTOCOL, conservative_pixel_metrics, _sum_pixels,
    _position_record, _aggregate_positions,
)
from ..near_goal_cube_geometry_20261103_v1 import measure_current_cube, VERSION as GEOMETRY_VERSION
from .data import PARTITION_IDS, SHOT_IDS
from .prediction import predict_current_position, PROPOSAL_THRESHOLDS, VERSION as PROPOSAL_VERSION
from ..cube_position_measurement_20261103_v7.evaluation import position_gate as normative_position_gate

QUALITY_PROTOCOL = 'current_Cube_position_measurement_used_support_coverage_v7'
PROPOSAL_RULE = 'highest fixed threshold with currently available F geometry, otherwise UNKNOWN'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _samples(samples):
    expected = [(p, p+'/'+s, 'calib') for p in PARTITION_IDS for s in SHOT_IDS]
    if [(s['parent_id'], s['scene_id'], s['role']) for s in samples] != expected:
        raise ValueError('All eight new position-calibration parents and all64 ordered shots required')


def freeze_legacy_eligibility(samples):
    samples = list(samples)
    _samples(samples)
    rows = []
    for sample in samples:
        old = legacy_measurement(sample['inputs'], sample['crops'], sample['semantic'] == 1)
        rows.append(dict(scene_id=sample['scene_id'], eligible=bool(old['refiner_available'])))
    result = dict(protocol=BASELINE_ELIGIBILITY_PROTOCOL, scope='position_calibration8',
        records=rows, required_eligible=sum(r['eligible'] for r in rows),
        frozen_before_NN_predictions=True)
    result['content_sha256'] = canonical_sha(result)
    return result


def _eligibility(eligibility):
    value = dict(eligibility)
    digest = value.pop('content_sha256', None)
    if (digest != canonical_sha(value) or value.get('protocol') != BASELINE_ELIGIBILITY_PROTOCOL
            or value.get('scope') != 'position_calibration8'
            or value.get('frozen_before_NN_predictions') is not True
            or [r['scene_id'] for r in value['records']] != [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]
            or any(type(r['eligible']) is not bool for r in value['records'])
            or value['required_eligible'] != sum(r['eligible'] for r in value['records'])):
        raise ValueError('Exact pre-prediction legacy observable ledger required')




def aggregate_support(rows, key):
    counts = dict(TP=0, FP=0, U=0)
    for row in rows:
        metric = row[key]
        for name in counts:
            value = metric[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Nonnegative integer actual-support pixel counts required')
            counts[name] += int(value)
    total = sum(counts.values())
    return dict(**counts, predicted_positive_count=total,
        conservative_precision_lowerbound=None if not total else counts['TP']/total,
        known_precision=None if not counts['TP']+counts['FP'] else counts['TP']/(counts['TP']+counts['FP']),
        pixel_samples_independent=False, statistical_confidence_bound=False)


def position_gate(positions, point_pixels, stencil_pixels, *, expected_samples=64):
    """Unchanged normative V7 numeric gates; only proposal selection differs."""
    decision = normative_position_gate(positions, point_pixels, stencil_pixels,
                                       expected_samples=expected_samples)
    decision.pop('fixed_threshold')
    decision.update(proposal_thresholds=list(PROPOSAL_THRESHOLDS),
                    adaptive_public_proposal_version=PROPOSAL_VERSION)
    return decision


def validate_prediction_route(prediction):
    trace = prediction.get('proposal_trace', [])
    tried = prediction.get('thresholds_tried')
    available = prediction['measurement']['refiner_available']
    route = prediction.get('proposal_route')
    if (prediction.get('adaptive_public_proposal_version') != PROPOSAL_VERSION
            or list(prediction.get('proposal_thresholds', [])) != list(PROPOSAL_THRESHOLDS)
            or prediction.get('route_uses_GT_or_future') is not False
            or prediction.get('position_measurement_available') is not available
            or not 1 <= len(trace) <= 2 or tried != [r['threshold'] for r in trace]):
        raise ValueError('Exact current public adaptive-proposal trace required')
    if tried == [.995]:
        valid = (route == 'first_0p995' and available is True
                 and trace[0]['geometry_available'] is True
                 and prediction['threshold'] == .995 and prediction['selected_threshold'] == .995)
    elif tried == [.995, .9]:
        valid = (trace[0]['geometry_available'] is False
                 and trace[1]['geometry_available'] is available and prediction['threshold'] == .9
                 and ((available and route == 'fallback_0p9' and prediction['selected_threshold'] == .9)
                      or (not available and route == 'UNKNOWN' and prediction['selected_threshold'] is None)))
    else:
        valid = False
    if not valid or (not available and prediction.get('centre') is not None):
        raise ValueError('No ground-truth route choice, extra threshold or unknown centre fallback')



def score_frozen_positions(samples, predictions, eligibility):
    """Call only after all current predictions are written and byte-frozen."""
    from ..cube_position_measurement_20261103_v7.support_quality import score_current_support
    samples, predictions = list(samples), list(predictions)
    _samples(samples)
    _eligibility(eligibility)
    if len(predictions) != len(samples):
        raise ValueError('No missing predictions or scene filtering')
    records, supports, raw_metrics = [], [], []
    for sample, prediction, fixed in zip(samples, predictions, eligibility['records']):
        if prediction['scene_id'] != sample['scene_id']:
            raise ValueError('Original fixed current-position prediction identity required')
        validate_prediction_route(prediction)
        m = prediction['measurement']
        baseline = dict(refiner_available=fixed['eligible'])
        records.append(_position_record(sample, m, baseline))
        support = score_current_support(sample, m)
        if bool(support['geometry_available']) != bool(m['refiner_available']):
            raise ValueError('Support availability must be the frozen measurement result')
        supports.append(support)
        raw_metrics.append(conservative_pixel_metrics(sample['semantic'],
            np.asarray(prediction['raw_Cube_mask']), sample['crops']['valid_crop']))
    positions = _aggregate_positions(records)
    if positions['oracle_geometry_eligible'] != eligibility['required_eligible']:
        raise ValueError('New geometry cannot shrink original eligibility')
    used = [s for s in supports if s['geometry_available']]
    used_metrics = {name: aggregate_support(used, key)
                    for name, key in (('points', 'point_pixels'), ('stencils', 'stencil_pixels'))}
    all_metrics = {name: aggregate_support(supports, key)
                   for name, key in (('points', 'point_pixels'), ('stencils', 'stencil_pixels'))}
    decision = position_gate(positions, used_metrics['points'], used_metrics['stencils'])
    return dict(quality_protocol=QUALITY_PROTOCOL, decision=decision, positions=positions,
        used_available_support=used_metrics, all_predictions_support_diagnostic=all_metrics,
        raw_mask_pixels_diagnostic_only=_sum_pixels(raw_metrics),
        records=records, per_scene_support=supports,
        by_parent={p: _aggregate_positions([r for r in records if r['parent_id'] == p]) for p in PARTITION_IDS},
        GT_labels_only_posthoc=True, threshold_selected_from_fresh=False, TEST_read=False)
