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

QUALITY_PROTOCOL = 'current_Cube_position_measurement_used_support_coverage_v7'
THRESHOLD = .995


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _samples(samples):
    expected = [(p, p+'/'+s, 'calib') for p in PARTITION_IDS for s in SHOT_IDS]
    if [(s['parent_id'], s['scene_id'], s['role']) for s in samples] != expected:
        raise ValueError('All four new position-calibration parents and all32 ordered shots required')


def freeze_legacy_eligibility(samples):
    samples = list(samples)
    _samples(samples)
    rows = []
    for sample in samples:
        old = legacy_measurement(sample['inputs'], sample['crops'], sample['semantic'] == 1)
        rows.append(dict(scene_id=sample['scene_id'], eligible=bool(old['refiner_available'])))
    result = dict(protocol=BASELINE_ELIGIBILITY_PROTOCOL, scope='position_calibration4',
        records=rows, required_eligible=sum(r['eligible'] for r in rows),
        frozen_before_NN_predictions=True)
    result['content_sha256'] = canonical_sha(result)
    return result


def _eligibility(eligibility):
    value = dict(eligibility)
    digest = value.pop('content_sha256', None)
    if (digest != canonical_sha(value) or value.get('protocol') != BASELINE_ELIGIBILITY_PROTOCOL
            or value.get('scope') != 'position_calibration4'
            or value.get('frozen_before_NN_predictions') is not True
            or [r['scene_id'] for r in value['records']] != [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]
            or any(type(r['eligible']) is not bool for r in value['records'])
            or value['required_eligible'] != sum(r['eligible'] for r in value['records'])):
        raise ValueError('Exact pre-prediction legacy observable ledger required')


def predict_current_position(inputs, crops, probabilities):
    """No label, robot, history, ground-truth pose or future argument exists."""
    p = np.asarray(probabilities)
    if (p.shape != (4, 2, 64, 64) or p.dtype != np.float32 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Unchanged normalized current four-view float32 probabilities required')
    raw = (p[:, 1] >= THRESHOLD) & crops['valid_crop']
    measurement = measure_current_cube(inputs, crops, raw)
    return dict(measurement=measurement, threshold=THRESHOLD, geometry_version=GEOMETRY_VERSION,
        position_measurement_available=measurement['refiner_available'],
        centre=measurement['centre'], raw_Cube_mask=raw,
        raw_mask_qualified=False, completion=False)


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


def position_gate(positions, point_pixels, stencil_pixels, *, expected_samples=32):
    """Empirical coordinate and used-support gates, not raw-mask certification."""
    reasons = []
    if positions['samples'] != expected_samples or positions['oracle_geometry_eligible'] <= 0:
        reasons.append('entire_fixed_observable_cohort_required')
    coverage = positions['oracle_eligible_coverage']
    if coverage is None or not np.isfinite(coverage) or coverage < .90:
        reasons.append('fixed_legacy_coverage_below_90percent')
    for key in ('qualified_visible_centre_error', 'all_available_centre_error'):
        metric = positions[key]
        if (metric['count'] <= 0 or metric['p95_m'] is None or metric['max_m'] is None
                or not np.isfinite([metric['p95_m'], metric['max_m']]).all()
                or metric['p95_m'] > .005 or metric['max_m'] > .010):
            reasons.append(key+'_exceeds_5mm_p95_or_10mm_max')
    for name, metric in (('actual_D_points', point_pixels), ('used_3x3_stencil_union', stencil_pixels)):
        total = metric['TP']+metric['FP']+metric['U']
        if total <= 0 or metric['TP']/total < .995:
            reasons.append(name+'_conservative_precision_below_995')
    for group in ('sphere_only', 'cube_far'):
        negative = positions['negatives'][group]
        if negative['samples'] <= 0 or negative['falseavailable'] != 0:
            reasons.append(group+'_missing_or_falseavailable')
    if positions['owner_unobserved_falseavailable'] != 0:
        reasons.append('unobserved_owner_falseavailable')
    return dict(quality_protocol=QUALITY_PROTOCOL, position_measurement_qualified=not reasons,
        raw_mask_model_qualified=False, grasp_qualified=False, Native_control_qualified=False,
        failure_reasons=reasons, fixed_threshold=THRESHOLD,
        empirical_internal_known_axisaligned_50mm_Cube_only=True,
        global_statistical_or_physical_bound=False)


def score_frozen_positions(samples, predictions, eligibility):
    """Call only after all current predictions are written and byte-frozen."""
    from .support_quality import score_current_support
    samples, predictions = list(samples), list(predictions)
    _samples(samples)
    _eligibility(eligibility)
    if len(predictions) != len(samples):
        raise ValueError('No missing predictions or scene filtering')
    records, supports, raw_metrics = [], [], []
    for sample, prediction, fixed in zip(samples, predictions, eligibility['records']):
        if prediction['scene_id'] != sample['scene_id'] or prediction['threshold'] != THRESHOLD:
            raise ValueError('Original fixed current-position prediction identity required')
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
