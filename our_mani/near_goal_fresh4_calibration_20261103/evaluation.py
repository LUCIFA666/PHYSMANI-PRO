"""Fresh4 raw995 calibration gates; old v4 verdict remains unchanged.

All32 scenes are retained. Original-geometry true-mask eligibility is frozen
before predictions, rather than inferred from the new F decoder availability.
"""
import hashlib
import json
import numpy as np
from ..near_goal_depth_refiner_20261103.data import make_measurement as legacy_make_measurement
from ..near_goal_depth_refiner_20261103_v4.evaluation import (
    THRESHOLD_GRID, QUALITY_PROTOCOL, MIN_ORACLE_ELIGIBLE_COVERAGE,
    BASELINE_ELIGIBILITY_PROTOCOL, GEOMETRY_VERSION, evaluate_probabilities,
)
from .data import PARTITION_IDS, SHOT_IDS

def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _validate_samples(samples):
    expected = [(p,p+'/'+s,'calib') for p in PARTITION_IDS for s in SHOT_IDS]
    actual = [(s['parent_id'],s['scene_id'],s['role']) for s in samples]
    if actual != expected:
        raise ValueError('Exact independent four-parent/all32shot calibration ledger required')

def freeze_legacy_eligibility(samples):
    samples = list(samples)
    _validate_samples(samples)
    rows = []
    for sample in samples:
        m = legacy_make_measurement(sample['inputs'],sample['crops'],sample['semantic']==1)
        rows.append(dict(scene_id=sample['scene_id'],eligible=bool(m['refiner_available'])))
    value = dict(protocol=BASELINE_ELIGIBILITY_PROTOCOL,scope='fresh4_calibration',
        records=rows,frozen_before_NN_predictions=True,GT_not_forwarded=True,
        required_eligible=sum(r['eligible'] for r in rows))
    value['content_sha256'] = _canonical_sha(value)
    return value

def _validate_eligibility(eligibility):
    e = dict(eligibility)
    digest = e.pop('content_sha256',None)
    expected = [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]
    if (digest != _canonical_sha(e) or e.get('protocol') != BASELINE_ELIGIBILITY_PROTOCOL
            or e.get('scope') != 'fresh4_calibration' or e.get('frozen_before_NN_predictions') is not True
            or [r['scene_id'] for r in e.get('records',[])] != expected
            or any(type(r['eligible']) is not bool for r in e['records'])
            or e['required_eligible'] != sum(r['eligible'] for r in e['records'])):
        raise ValueError('Exact pre-prediction legacy eligibility ledger required')

def _check_report_eligibility(report,eligibility):
    _validate_eligibility(eligibility)
    actual = [(r['scene_id'],r['oracle_geometry_eligible']) for r in report['records']]
    expected = [(r['scene_id'],r['eligible']) for r in eligibility['records']]
    if (actual != expected or report['positions']['samples'] != 32
            or report['positions']['oracle_geometry_eligible'] != eligibility['required_eligible']
            or report.get('fresh_eligibility_sha256') != eligibility['content_sha256']):
        raise ValueError('No filtering or new-geometry eligibility substitution is allowed')

def evaluate_fresh_probabilities(samples, probabilities, threshold, eligibility):
    samples = list(samples)
    _validate_samples(samples)
    _validate_eligibility(eligibility)
    report = evaluate_probabilities(samples,probabilities,threshold)
    report.pop('expected_full32_baseline_eligible')
    report.pop('expected_calibration_baseline_eligible')
    report.update(scope='fresh4_calibration',fresh_eligibility_sha256=eligibility['content_sha256'],
        required_fresh_calibration_legacy_eligible=eligibility['required_eligible'])
    _check_report_eligibility(report,eligibility)
    return report

def choose_fresh_calibrated_threshold(reports, eligibility):
    _validate_eligibility(eligibility)
    required_eligible = sum(r["eligible"] for r in eligibility["records"])
    if required_eligible <= 0:
        raise ValueError("Nonempty original legacy observable set required")
    for report in reports:
        _check_report_eligibility(report, eligibility)
    if tuple(r['threshold'] for r in reports) != THRESHOLD_GRID:
        raise ValueError('entire exact fixed threshold grid required; no new sweep')
    decisions = []
    for report in reports:
        pix, pos = report['pixels'], report['positions']
        lower, p95 = pix['conservative_precision_lowerbound'], pos['qualified_visible_centre_error']['p95_m']
        negative = pos['negatives']
        passed = bool(pix['predicted_positive_count'] > 0 and lower is not None and lower >= .995
            and pos['oracle_geometry_eligible'] == required_eligible
            and report.get('baseline_eligibility_protocol') == BASELINE_ELIGIBILITY_PROTOCOL
            and report.get('geometry_version') == GEOMETRY_VERSION
            and pos['oracle_eligible_coverage'] is not None
            and pos['oracle_eligible_coverage'] >= MIN_ORACLE_ELIGIBLE_COVERAGE
            and p95 is not None and p95 <= .005
            and pos['owner_unobserved_falseavailable'] == 0
            and all(negative[g]['samples'] > 0 and negative[g]['falseavailable'] == 0
                    for g in ('sphere_only', 'cube_far')))
        decisions.append(dict(threshold=report['threshold'], passed=passed,
            conservative_precision_lowerbound=lower, known_precision=pix['known_precision'],
            predicted_positive_count=pix['predicted_positive_count'], unknown_Cube_predictions=pix['U'],
            owner_unobserved_falseavailable=pos['owner_unobserved_falseavailable'],
            oracle_geometry_eligible=pos['oracle_geometry_eligible'],
            oracle_eligible_coverage=pos['oracle_eligible_coverage'], qualified_centre_p95_m=p95))
    selected = next((r['threshold'] for r in decisions if r['passed']), None)
    return dict(quality_protocol=QUALITY_PROTOCOL, selected_threshold=selected,
        geometry_version=GEOMETRY_VERSION,
        baseline_eligibility_protocol=BASELINE_ELIGIBILITY_PROTOCOL,
        required_calibration_baseline_eligible=required_eligible,
        calibration_passed=selected is not None, grid=decisions,
        criterion='lowest fixedgrid meeting nonzero conservativePPV>=.995, eligible>0/coverage>=.90, bothnegativegroups present/FPR0, owner-unobserved falseavailable0, centreP95<=.005m',
        empirical_internal_TRAIN_only=True, global_statistical_guarantee=False)
