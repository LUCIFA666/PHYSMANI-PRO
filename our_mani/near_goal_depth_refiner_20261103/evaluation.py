"""Partial-label conservative PPV and current-depth oracle; no task success."""
from collections import Counter

import numpy as np

from .data import make_measurement, LABEL_PROTOCOL

THRESHOLD_GRID = (.5, .7, .9, .95, .99)
QUALITY_PROTOCOL = 'binary_depth_influence_conservative_PPV_coverage_v1'
MIN_ORACLE_ELIGIBLE_COVERAGE = .90


def position_stats(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('finite entire position-error ledger required')
    return dict(count=len(values), mean_m=None if not len(values) else float(values.mean()),
                p50_m=None if not len(values) else float(np.quantile(values, .5)),
                p95_m=None if not len(values) else float(np.quantile(values, .95)),
                max_m=None if not len(values) else float(values.max()))


def conservative_pixel_metrics(labels, pred_cube, domain=None):
    labels, pred_cube = np.asarray(labels), np.asarray(pred_cube)
    if (labels.dtype != np.uint8 or pred_cube.dtype != np.bool_ or labels.shape != pred_cube.shape
            or not set(np.unique(labels)) <= {0, 1, 255}):
        raise ValueError('binary0/1/255 labels and aligned bool predictions required')
    if domain is None:
        domain = np.ones_like(pred_cube)
    domain = np.asarray(domain)
    if domain.shape != labels.shape or domain.dtype != np.bool_:
        raise ValueError('explicit public observed-image domain required')
    tp = int((domain & pred_cube & (labels == 1)).sum())
    fp = int((domain & pred_cube & (labels == 0)).sum())
    u = int((domain & pred_cube & (labels == 255)).sum())
    fn = int((domain & ~pred_cube & (labels == 1)).sum())
    denominator = tp+fp+u
    return dict(TP=tp, FP=fp, U=u, FN=fn, predicted_positive_count=denominator,
        known_positive_count=int((domain & (labels == 1)).sum()),
        known_zero_count=int((domain & (labels == 0)).sum()),
        UNKNOWN_count=int((domain & (labels == 255)).sum()),
        known_precision=None if tp+fp == 0 else tp/(tp+fp),
        conservative_precision_lowerbound=None if denominator == 0 else tp/denominator,
        label_completion_precision_upperbound=None if denominator == 0 else (tp+u)/denominator,
        known_recall=None if tp+fn == 0 else tp/(tp+fn),
        known_IoU=None if tp+fp+fn == 0 else tp/(tp+fp+fn),
        known_Dice=None if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn),
        bounds_are_statistical_confidence=False)


def _sum_pixels(metrics):
    result = {key: sum(x[key] for x in metrics) for key in (
        'TP', 'FP', 'U', 'FN', 'predicted_positive_count', 'known_positive_count',
        'known_zero_count', 'UNKNOWN_count')}
    tp, fp, u, fn = (result[k] for k in ('TP', 'FP', 'U', 'FN'))
    result.update(known_precision=None if tp+fp == 0 else tp/(tp+fp),
        conservative_precision_lowerbound=None if tp+fp+u == 0 else tp/(tp+fp+u),
        label_completion_precision_upperbound=None if tp+fp+u == 0 else (tp+u)/(tp+fp+u),
        known_recall=None if tp+fn == 0 else tp/(tp+fn),
        known_IoU=None if tp+fp+fn == 0 else tp/(tp+fp+fn),
        known_Dice=None if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn),
        bounds_are_statistical_confidence=False)
    return result


def probability_to_observation(inputs, crops, probabilities, threshold):
    p = np.asarray(probabilities)
    if (p.shape != (4, 2, 64, 64) or not np.isfinite(p).all() or (p < 0).any()
            or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('finite normalized4x2x64x64 probabilities required')
    if isinstance(threshold, (bool, np.bool_)) or threshold not in THRESHOLD_GRID:
        raise ValueError('one fixed-grid confidence threshold required')
    raw = (p[:, 1] >= threshold) & crops['valid_crop']
    roles = np.full(raw.shape, -1, np.int8)
    roles[(p[:, 0] >= threshold) & crops['valid_crop']] = 0
    roles[raw] = 1
    return dict(semantic_roles=roles, probabilities=p, raw_Cube_mask=raw,
        measurement=make_measurement(inputs, crops, raw), threshold=float(threshold),
        label_semantics=LABEL_PROTOCOL, completion=False,
        complete_visible_instance_claimed=False, Native_control_qualified=False)


def _position_record(sample, measurement, oracle):
    gt = np.asarray(sample['cube_center_world'], float)
    if gt.shape != (3,) or not np.isfinite(gt).all():
        raise ValueError('finite labels-only Cube centre required to score fixed scene')
    available, eligible = measurement['refiner_available'], oracle['refiner_available']
    error = None if not available else float(np.linalg.norm(np.asarray(measurement['centre'])-gt))
    return dict(parent_id=sample['parent_id'], scene_id=sample['scene_id'],
        role=sample['role'], condition=sample['condition'],
        raw_positive_supervision_pixels=int((sample['semantic'] == 1).sum()),
        owner_supervision_unobserved=not bool((sample['semantic'] == 1).any()),
        physical_or_visual_absence_not_inferred_from_zero_owner=True,
        absence_from_all_camera_frusta_by_construction=sample.get('absence_from_all_camera_frusta_by_construction', False),
        oracle_geometry_eligible=eligible, refiner_available=available,
        owner_unobserved_falseavailable=bool(not (sample['semantic'] == 1).any() and available),
        qualified_centre_error_m=error if eligible else None,
        all_available_GT_centre_error_m=error, measurement=measurement,
        GT_used_only_to_score=True)


def _aggregate_positions(rows):
    eligible = [r for r in rows if r['oracle_geometry_eligible']]
    errors = [r['qualified_centre_error_m'] for r in rows if r['qualified_centre_error_m'] is not None]
    return dict(samples=len(rows), oracle_geometry_eligible=len(eligible),
        oracle_eligible_refiner_available=sum(r['refiner_available'] for r in eligible),
        oracle_eligible_coverage=None if not eligible else sum(r['refiner_available'] for r in eligible)/len(eligible),
        qualified_visible_centre_error=position_stats(errors),
        all_available_centre_error=position_stats([r['all_available_GT_centre_error_m'] for r in rows
                                                  if r['all_available_GT_centre_error_m'] is not None]),
        owner_supervision_unobserved=sum(r['owner_supervision_unobserved'] for r in rows),
        owner_unobserved_falseavailable=sum(r['owner_unobserved_falseavailable'] for r in rows),
        UNKNOWN_reasons=dict(Counter(reason for r in rows for reason in r['measurement']['reasons'])),
        negatives={group: dict(samples=sum(r['condition'] == group for r in rows),
            falseavailable=sum(r['condition'] == group and r['refiner_available'] for r in rows))
            for group in ('sphere_only', 'cube_far')})


def oracle_diagnostic(dataset):
    """True raw owner -> one erosion -> same actual-depth centre, before any fit."""
    records = []
    for sample in dataset:
        oracle = make_measurement(sample['inputs'], sample['crops'], sample['semantic'] == 1)
        records.append(_position_record(sample, oracle, oracle))
    summary = _aggregate_positions(records)
    p95 = summary['qualified_visible_centre_error']['p95_m']
    passed = bool(summary['oracle_geometry_eligible'] > 0 and p95 is not None and p95 <= .005
                  and summary['owner_unobserved_falseavailable'] == 0
                  and all(x['falseavailable'] == 0 for x in summary['negatives'].values()))
    return dict(schema='raw_depth_influence_trueMask_eroded_oracle_v1',
        records=records, aggregate=summary,
        by_role={role: _aggregate_positions([r for r in records if r['role'] == role])
                 for role in ('fit', 'selection', 'calib')},
        by_condition={group: _aggregate_positions([r for r in records if r['condition'] == group])
                      for group in sorted({r['condition'] for r in records})},
        ready_for_segmentation_fit=passed, gate_is_necessary_not_sufficient=True,
        network_qualified=False, unknown_not_filled_from_GT=True, TEST_read=False)


def evaluate_probabilities(samples, probabilities, threshold):
    samples, probabilities = list(samples), list(probabilities)
    if not samples or len(samples) != len(probabilities):
        raise ValueError('entire fixed nonempty role sample ledger required')
    pixels, records, parent_pixels, condition_pixels = [], [], {}, {}
    camera_pixels = {str(i): [] for i in range(4)}
    reason_predictions = {}
    nll, brier, confidence, correct = [], [], [], []
    unknown_positive, unknown_negative, unknown_abstain = 0, 0, 0
    for sample, probs in zip(samples, probabilities):
        obs = probability_to_observation(sample['inputs'], sample['crops'], probs, threshold)
        metric = conservative_pixel_metrics(sample['semantic'], obs['raw_Cube_mask'], sample['crops']['valid_crop'])
        pixels.append(metric)
        parent_pixels.setdefault(sample['parent_id'], []).append(metric)
        condition_pixels.setdefault(sample['condition'], []).append(metric)
        for camera in range(4):
            camera_pixels[str(camera)].append(conservative_pixel_metrics(sample['semantic'][camera],
                obs['raw_Cube_mask'][camera], sample['crops']['valid_crop'][camera]))
        oracle = make_measurement(sample['inputs'], sample['crops'], sample['semantic'] == 1)
        records.append(_position_record(sample, obs['measurement'], oracle))
        truth, roles = sample['semantic'], obs['semantic_roles']
        unknown = (truth == 255) & sample['crops']['valid_crop']
        unknown_positive += int((unknown & (roles == 1)).sum())
        unknown_negative += int((unknown & (roles == 0)).sum())
        unknown_abstain += int((unknown & (roles < 0)).sum())
        reasons = sample.get('unknown_reason_masks')
        if reasons is None:
            reasons = {'unspecified_UNKNOWN': unknown}
        reason_union = np.zeros_like(unknown)
        for reason, mask in reasons.items():
            mask = np.asarray(mask)
            if mask.shape != unknown.shape or mask.dtype != np.bool_ or np.any(mask & ~unknown):
                raise ValueError('Labels-only UNKNOWN reason masks must be aligned subsets')
            reason_union |= mask
            entry = reason_predictions.setdefault(reason, dict(total=0, Cube=0, known_zero=0, abstained=0,
                per_camera={str(i): dict(total=0, Cube=0, known_zero=0, abstained=0) for i in range(4)}))
            for key, selected in (('total', mask), ('Cube', mask & (roles == 1)),
                                  ('known_zero', mask & (roles == 0)), ('abstained', mask & (roles < 0))):
                entry[key] += int(selected.sum())
                for camera in range(4):
                    entry['per_camera'][str(camera)][key] += int(selected[camera].sum())
        if np.any(unknown & ~reason_union):
            raise ValueError('Every observed UNKNOWN pixel must retain an explicit reason')
        valid = (truth != 255) & sample['crops']['valid_crop']
        p = np.moveaxis(np.asarray(probs), 1, -1)[valid]
        target = truth[valid]
        if len(target):
            nll.extend((-np.log(np.maximum(p[np.arange(len(target)), target], 1e-12))).tolist())
            brier.extend(((p-np.eye(2)[target])**2).sum(1).tolist())
            confidence.extend(p.max(1).tolist())
            correct.extend((p.argmax(1) == target).tolist())
    conf, ok = np.asarray(confidence), np.asarray(correct)
    ece = 0.
    for lo in np.linspace(0, 1, 11)[:-1]:
        selected = (conf >= lo) & (conf < lo+.1 if lo < .9 else conf <= 1)
        if selected.any():
            ece += float(selected.mean()*abs(conf[selected].mean()-ok[selected].mean()))
    return dict(threshold=float(threshold), pixels=_sum_pixels(pixels),
        per_parent_pixels={k: _sum_pixels(v) for k, v in parent_pixels.items()},
        per_condition_pixels={k: _sum_pixels(v) for k, v in condition_pixels.items()},
        per_camera_pixels={k: _sum_pixels(v) for k, v in camera_pixels.items()},
        UNKNOWN_predictions=dict(Cube=unknown_positive, known_zero=unknown_negative, abstained=unknown_abstain),
        UNKNOWN_predictions_by_reason=reason_predictions, UNKNOWN_reason_masks_can_overlap=True,
        positions=_aggregate_positions(records), records=records,
        by_condition={group: _aggregate_positions([r for r in records if r['condition'] == group])
                      for group in sorted({r['condition'] for r in records})},
        probability_calibration_known_labels_only=dict(
            NLL=None if not nll else float(np.mean(nll)), Brier=None if not brier else float(np.mean(brier)),
            ECE10=None if not len(conf) else ece, partial_labels=True,
            statistical_error_bound_claimed=False, pixels_not_independent_samples=True),
        GT_labels_only_posthoc=True, Native_task_success_claimed=False, TEST_read=False)


def choose_calibrated_threshold(reports):
    if tuple(r['threshold'] for r in reports) != THRESHOLD_GRID:
        raise ValueError('entire exact fixed threshold grid required; no new sweep')
    decisions = []
    for report in reports:
        pix, pos = report['pixels'], report['positions']
        lower, p95 = pix['conservative_precision_lowerbound'], pos['qualified_visible_centre_error']['p95_m']
        negative = pos['negatives']
        passed = bool(pix['predicted_positive_count'] > 0 and lower is not None and lower >= .995
            and pos['oracle_geometry_eligible'] > 0 and pos['oracle_eligible_coverage'] is not None
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
        calibration_passed=selected is not None, grid=decisions,
        criterion='lowest fixedgrid meeting nonzero conservativePPV>=.995, eligible>0/coverage>=.90, bothnegativegroups present/FPR0, owner-unobserved falseavailable0, centreP95<=.005m',
        empirical_internal_TRAIN_only=True, global_statistical_guarantee=False)
