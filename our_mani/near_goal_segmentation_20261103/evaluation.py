"""Offline TRAIN-role diagnostics and fixed calibration grid; never task success."""
from collections import Counter
import numpy as np
from .data import current_cube_measurement,CLASS_NAMES

THRESHOLD_GRID=(.5,.7,.9,.95,.99)
NEAR_SCOPE_M=.08
MIN_ORACLE_ELIGIBLE_COVERAGE=.90
QUALITY_PROTOCOL='near_goal_calibration_with_oracle_eligible_coverage_v2'


def position_stats(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean_m=float(a.mean()),p50_m=float(np.median(a)),p95_m=float(np.percentile(a,95)),max_m=float(a.max())) if len(a) else dict(count=0,mean_m=None,p50_m=None,p95_m=None,max_m=None)


def probability_to_observation(inputs,crops,probabilities,threshold):
    if float(threshold) not in THRESHOLD_GRID:raise ValueError('only predeclared confidence grid')
    p=np.asarray(probabilities,float)
    if p.shape!=(4,3,64,64) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('finite normalized current3class crop probabilities required')
    roles=p.argmax(1).astype(np.int8);roles[(p.max(1)<threshold)|~crops['valid_crop']]=-1
    measurement=current_cube_measurement(inputs,crops,roles==1)
    return dict(semantic_roles=roles,probabilities=p,threshold=float(threshold),measurement=measurement,
        unknown_pixel_mask=roles==-1,refiner_available=measurement['refiner_available'],
        current_visible_pixels_only=True,object_completion=False,orientation_observed=False,
        near_scope_m=NEAR_SCOPE_M,legacy_free_probe_route_modified=False)


def _position_record(sample,measurement,*,oracle_geometry_eligible=False):
    truth=sample['semantic'];visible=int((truth==1).sum());present=sample['cube_present']
    gt=np.asarray(sample['cube_center_world'],float)
    # Label-derived eligibility is posthoc only, never an input/availability gate.
    expected=bool(present and visible>0 and np.isfinite(gt).all()
        and np.linalg.norm(gt-sample['inputs']['public_sphere_center'])<=NEAR_SCOPE_M)
    error=None
    if measurement['refiner_available'] and present and visible>0:
        error=float(np.linalg.norm(np.asarray(measurement['centre'])-gt))
    return dict(parent_id=sample['parent_id'],scene_id=sample['scene_id'],condition=sample['condition'],
        expected_near_visible_from_labels=expected,oracle_geometry_eligible=bool(oracle_geometry_eligible),visible_cube_pixels=visible,
        fully_invisible=visible==0,refiner_available=measurement['refiner_available'],centre_error_m=error,
        measurement=measurement,fully_invisible_correctly_unknown=bool(visible==0 and not measurement['refiner_available']),
        GT_used_only_to_score=True)


def _aggregate_positions(rows):
    expected=[r for r in rows if r['expected_near_visible_from_labels']]
    qualified=[r for r in rows if r['refiner_available'] and r['centre_error_m'] is not None]
    eligible=[r for r in rows if r['oracle_geometry_eligible']]
    negatives={family:[r for r in rows if r['condition']==family] for family in ('sphere_only','cube_far')}
    return dict(samples=len(rows),expected_near_visible=len(expected),qualified_visible_centres=len(qualified),
        oracle_geometry_eligible=len(eligible),oracle_eligible_refiner_available=sum(r['refiner_available'] for r in eligible),
        oracle_eligible_coverage=None if not eligible else sum(r['refiner_available'] for r in eligible)/len(eligible),
        qualified_visible_centre_error=position_stats([r['centre_error_m'] for r in qualified]),
        expected_visible_coverage=None if not expected else sum(r['refiner_available'] for r in expected)/len(expected),
        fully_invisible=sum(r['fully_invisible'] for r in rows),
        fully_invisible_falseavailable=sum(r['fully_invisible'] and r['refiner_available'] for r in rows),
        negatives={k:dict(samples=len(v),falseavailable=sum(r['refiner_available'] for r in v)) for k,v in negatives.items()},
        UNKNOWN_reasons=dict(Counter(reason for r in rows for reason in r['measurement']['reasons'])))


def oracle_diagnostic(dataset):
    """Native labels select real current pixels ONLY offline; no GT-centre fill."""
    records=[]
    for sample in dataset:
        measurement=current_cube_measurement(sample['inputs'],sample['crops'],sample['semantic']==1)
        records.append(_position_record(sample,measurement,oracle_geometry_eligible=measurement['refiner_available'] and bool((sample['semantic']==1).any())))
    summary=_aggregate_positions(records);p95=summary['qualified_visible_centre_error']['p95_m']
    passed=p95 is not None and p95<=.005 and summary['fully_invisible_falseavailable']==0
    return dict(schema='near_goal_visible_mask_oracle_before_any_NN_fit_v1',records=records,aggregate=summary,
        by_condition={k:_aggregate_positions([r for r in records if r['condition']==k]) for k in sorted({r['condition'] for r in records})},
        by_role={role:_aggregate_positions([r for r,s in zip(records,dataset.records) if s['role']==role]) for role in ('fit','selection','calib')},
        oracle_centre_gate_passed=passed,ready_for_segmentation_fit=passed,
        gate_is_necessary_not_sufficient=True,geometric_bottleneck_if_failed=not passed,
        unknown_not_filled_from_GT=True,Native_started=False,network_fit_started=False,TEST_read=False)


def evaluate_probabilities(samples,probabilities,threshold):
    samples=list(samples);probabilities=list(probabilities)
    if len(samples)!=len(probabilities):raise ValueError('entire fixed role sample ledger required')
    confusion=np.zeros((3,4),np.int64);records=[];brier=[];nll=[];confidence=[];correct=[]
    parents={};conditions={}
    for sample,probs in zip(samples,probabilities):
        obs=probability_to_observation(sample['inputs'],sample['crops'],probs,threshold)
        truth=sample['semantic'];valid=truth!=255;pred=obs['semantic_roles'];column=np.where(pred<0,3,pred)
        local=np.bincount((truth[valid]*4+column[valid]).ravel(),minlength=12).reshape(3,4)
        confusion+=local;parents.setdefault(sample['parent_id'],np.zeros((3,4),np.int64));parents[sample['parent_id']]+=local
        conditions.setdefault(sample['condition'],np.zeros((3,4),np.int64));conditions[sample['condition']]+=local
        oracle=current_cube_measurement(sample['inputs'],sample['crops'],truth==1)
        records.append(_position_record(sample,obs['measurement'],oracle_geometry_eligible=oracle['refiner_available'] and bool((truth==1).any())))
        pv=np.moveaxis(np.asarray(probs),1,-1)[valid];tv=truth[valid]
        if len(tv):
            onehot=np.eye(3)[tv];brier.extend(((pv-onehot)**2).sum(1).tolist())
            nll.extend((-np.log(np.maximum(pv[np.arange(len(tv)),tv],1e-12))).tolist())
            confidence.extend(pv.max(1).tolist());correct.extend((pv.argmax(1)==tv).tolist())
    def metrics(cm):
        result={}
        for cls,name in enumerate(CLASS_NAMES):
            tp=int(cm[cls,cls]);fp=int(cm[:,cls].sum()-tp);fn=int(cm[cls].sum()-tp)
            result[name]=dict(TP=tp,FP=fp,FN=fn,precision=None if tp+fp==0 else tp/(tp+fp),
                recall=None if tp+fn==0 else tp/(tp+fn),IoU=None if tp+fp+fn==0 else tp/(tp+fp+fn),
                dice=None if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
        return result
    conf=np.asarray(confidence);ok=np.asarray(correct);ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        inside=(conf>=lo)&(conf<lo+.1 if lo<.9 else conf<=1)
        if inside.any():ece+=float(inside.mean()*abs(conf[inside].mean()-ok[inside].mean()))
    summary=_aggregate_positions(records)
    return dict(threshold=float(threshold),pixels=metrics(confusion),confusion_rows_true_cols_bg_cube_goal_unknown=confusion.tolist(),
        per_parent_pixels={k:metrics(v) for k,v in parents.items()},per_condition_pixels={k:metrics(v) for k,v in conditions.items()},
        pixel_unknown_counts_by_true_class=confusion[:,3].tolist(),positions=summary,records=records,
        by_condition={k:_aggregate_positions([r for r in records if r['condition']==k]) for k in sorted({r['condition'] for r in records})},
        probability_calibration=dict(pixel_Brier=None if not brier else float(np.mean(brier)),
            pixel_NLL=None if not nll else float(np.mean(nll)),ECE10=None if not len(conf) else ece,
            calibrated_statistical_error_bound_claimed=False,correlated_pixels_not_independent_samples=True),
        labels_only_posthoc=True,Native_task_success_claimed=False,TEST_read=False)


def choose_calibrated_threshold(reports):
    if tuple(r['threshold'] for r in reports)!=THRESHOLD_GRID:raise ValueError('exact fixed threshold grid, no other sweep')
    decisions=[]
    for report in reports:
        p=report['pixels']['cube']['precision'];pos=report['positions'];n=pos['negatives'];p95=pos['qualified_visible_centre_error']['p95_m']
        passed=bool(p is not None and p>=.995 and all(n[k]['samples']>0 and n[k]['falseavailable']==0 for k in ('sphere_only','cube_far'))
            and p95 is not None and p95<=.005 and pos['fully_invisible_falseavailable']==0
            and pos['oracle_geometry_eligible']>0 and pos['oracle_eligible_coverage'] is not None
            and pos['oracle_eligible_coverage']>=MIN_ORACLE_ELIGIBLE_COVERAGE)
        decisions.append(dict(threshold=report['threshold'],passed=passed,cube_pixel_precision=p,qualified_visible_centre_p95_m=p95,oracle_geometry_eligible=pos['oracle_geometry_eligible'],
            oracle_eligible_coverage=pos['oracle_eligible_coverage']))
    selected=next((r['threshold'] for r in decisions if r['passed']),None)
    return dict(quality_protocol=QUALITY_PROTOCOL,selected_threshold=selected,calibration_passed=selected is not None,grid=decisions,
        criterion='lowest fixedgrid with aggregateCubepixelprecision>=.995, both negativegroups present/zero falseavailable, qualifiedvisiblecentreP95<=.005m,oracleeligible>0 and coverage>=.90',
        empirical_internal_TRAIN_only=True,global_statistical_guarantee=False)
