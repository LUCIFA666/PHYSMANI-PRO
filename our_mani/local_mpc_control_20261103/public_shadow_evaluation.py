"""Offline public-observation/forecast position evaluation; never a policy input.

Only actual timestamp matches are labels. No interpolation, nearest-outside-
tolerance clamping, future observation input, model fitting or task-success
qualification occurs here. Missing contact labels cannot certify free motion.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


SCHEMA='public_shadow_position_timestamp_evaluation_20261103_v1'
FULL_HORIZONS_MS=tuple(range(50,801,50))
H8_MS=tuple(range(100,801,100))


def _time(value):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError('actual finite nonnegative timestamp required')
    result=float(value)
    if not np.isfinite(result) or result<0:raise ValueError('actual finite nonnegative timestamp required')
    return result


class TimeIndex:
    """Exact actual samples within declared float tolerance, never interpolated."""
    def __init__(self,rows,key,tolerance):
        entries=[(_time(key(row)),row) for row in rows]
        entries.sort(key=lambda pair:pair[0])
        self.times=[time for time,_ in entries];self.rows=[row for _,row in entries]
        self.tolerance=tolerance
        if any(b-a<=2*tolerance for a,b in zip(self.times,self.times[1:])):
            raise ValueError('duplicate or ambiguous actual timestamps cannot be silently deduplicated')
    def get(self,timestamp):
        at=bisect_left(self.times,timestamp)
        options=[i for i in (at-1,at) if 0<=i<len(self.times) and abs(self.times[i]-timestamp)<=self.tolerance]
        return self.rows[options[0]] if len(options)==1 else None


def _xyz(value,name):
    try:value=np.asarray(value,dtype=np.float64)
    except (TypeError,ValueError):return None,'invalid_'+name
    if value.shape!=(3,):return None,'missing_or_invalid_shape_'+name
    if not np.isfinite(value).all():return None,'nonfinite_'+name
    return value,None


def _tracker(result,anchor,tolerance):
    obj=result.get('object_observation')
    if not isinstance(obj,dict):return None,None,'missing_current_RedBox_observation'
    try:observed=_time(obj.get('timestamp'))
    except (ValueError,TypeError):return None,None,'invalid_tracker_timestamp'
    if abs(observed-anchor)>tolerance:return None,None,'stale_tracker_observation'
    if obj.get('visible') is not True:return None,None,'tracker_not_observed_visible'
    p,reason=_xyz((obj.get('pose') or {}).get('position'),'tracker_position')
    v,vreason=_xyz(obj.get('linear_velocity'),'tracker_velocity')
    return p,v,reason or vreason


def _series_at(times,positions,endpoint,tolerance,name):
    if not isinstance(times,(list,tuple,np.ndarray)) or not isinstance(positions,(list,tuple,np.ndarray)):
        return None,'missing_'+name
    finite_times=[]
    try:
        for i,time in enumerate(times):finite_times.append((_time(time),i))
    except (ValueError,TypeError):return None,'invalid_'+name+'_timestamps'
    if any(b[0]-a[0]<=2*tolerance for a,b in zip(finite_times,finite_times[1:])):
        return None,'duplicate_or_unsorted_'+name+'_timestamps'
    found=[i for time,i in finite_times if abs(time-endpoint)<=tolerance]
    if len(found)!=1:return None,'missing_exact_'+name+'_endpoint'
    if found[0]>=len(positions):return None,'missing_'+name+'_position'
    return _xyz(positions[found[0]],name+'_position')


def _full_forecast_at(result,endpoint,tolerance):
    forecast=result.get('forecast')
    if not isinstance(forecast,dict):return None,'missing_full_GRU_forecast'
    ids=forecast.get('object_ids',[])
    if not isinstance(ids,(list,tuple)) or ids.count('target')!=1:return None,'missing_or_duplicate_target_forecast_id'
    target=ids.index('target');values=forecast.get('positions')
    if not isinstance(values,(list,tuple,np.ndarray)):return None,'missing_full_GRU_positions'
    positions=[]
    for row in values:
        try:positions.append(row[target])
        except (TypeError,IndexError):positions.append(None)
    return _series_at(forecast.get('times'),positions,endpoint,tolerance,'full_GRU')


def _interval(labels,anchor,endpoint,sensor_dt,tolerance):
    steps=int(round((endpoint-anchor)/sensor_dt))
    if steps<0 or abs(endpoint-anchor-steps*sensor_dt)>tolerance:
        return dict(strict_free=False,preheld_only=False,strict_status='invalid_horizon_clock',preheld_status='invalid_horizon_clock',
                    missing_actual_ticks=1,unknown_contact_ticks=0,unknown_held_ticks=0,contact_ticks=0,held_ticks=0)
    missing=unknown_contact=unknown_held=contact=held=0
    for tick in range(steps+1):
        row=labels.get(anchor+tick*sensor_dt)
        if row is None:missing+=1;continue
        value=row.get('held')
        if type(value) is not bool:unknown_held+=1
        elif value:held+=1
        value=row.get('contact')
        if type(value) is not bool:unknown_contact+=1
        elif value:contact+=1
    strict=missing==unknown_contact==unknown_held==contact==held==0
    preheld=missing==unknown_held==held==0
    strict_status=('eligible' if strict else 'censored_contact_or_held' if contact or held else
                   'unknown_missing_actual_ticks' if missing else 'unknown_held_label' if unknown_held else 'unknown_contact_label')
    preheld_status=('eligible' if preheld else 'censored_held' if held else 'unknown_missing_actual_ticks' if missing else 'unknown_held_label')
    return dict(strict_free=strict,preheld_only=preheld,strict_status=strict_status,preheld_status=preheld_status,
                missing_actual_ticks=missing,unknown_contact_ticks=unknown_contact,unknown_held_ticks=unknown_held,
                contact_ticks=contact,held_ticks=held)


def _comparison(head,anchor,endpoint,prediction,prediction_reason,label,interval,horizon_ms):
    target,reason=_xyz(label.get('object_position') if label else None,'GT_position')
    errors=[]
    if prediction_reason:errors.append(prediction_reason)
    if label is None:errors.append('unobserved_exact_endpoint_no_interpolation')
    elif reason:errors.append(reason)
    error=None if prediction is None or target is None else float(np.linalg.norm(prediction-target))
    return dict(head=head,anchor_time=anchor,endpoint_time=endpoint,horizon_ms=horizon_ms,
                predicted_xyz=None if prediction is None else prediction.tolist(),
                actual_xyz=None if target is None else target.tolist(),error_m=error,
                missing_or_invalid_reasons=errors,eligibility=deepcopy(interval))


def _metrics(rows):
    values=np.asarray([r['error_m'] for r in rows if r['error_m'] is not None],float)
    expected=len(rows);valid=len(values);complete=expected>0 and expected==valid
    count3=int(np.count_nonzero(values<=.03+1e-12));count5=int(np.count_nonzero(values<=.05+1e-12))
    return dict(expected_comparisons=expected,observed_finite_comparisons=valid,missing_or_invalid_comparisons=expected-valid,
        statistics_population='all_finite_observed_pairs_no_error_based_filtering',cohort_complete=complete,
        mean_m=float(values.mean()) if valid else None,p50_m=float(np.quantile(values,.5)) if valid else None,
        p95_m=float(np.quantile(values,.95)) if valid else None,max_m=float(values.max()) if valid else None,
        fraction_le_3cm=count3/expected if expected else None,fraction_le_5cm=count5/expected if expected else None,
        samples_le_3cm=count3,samples_le_5cm=count5,
        all_samples_le_5cm=bool(complete and count5==expected),
        mean_le_3cm=bool(complete and float(values.mean())<=.03+1e-12),
        position_scope_gate_passed=bool(complete and count5==expected and float(values.mean())<=.03+1e-12),
        invalid_reasons=dict(Counter(reason for row in rows for reason in row['missing_or_invalid_reasons'])))


def _scopes(rows):
    strict=[r for r in rows if r['eligibility']['strict_free']]
    preheld=[r for r in rows if r['eligibility']['preheld_only']]
    unknown=sum(r['eligibility']['strict_status'].startswith('unknown') or r['eligibility']['strict_status']=='invalid_horizon_clock' for r in rows)
    strict_stats=_metrics(strict)
    # Unknown contact/gaps may never silently make the selected free subset
    # pass a full-scope gate, even if some known-free comparisons are excellent.
    strict_stats['eligible_cohort_position_gate_passed']=strict_stats['position_scope_gate_passed']
    if unknown:
        strict_stats['position_scope_gate_passed']=False
        strict_stats['all_samples_le_5cm']=False
        strict_stats['mean_le_3cm']=False
    strict_stats.update(scope='actual_contact_false_and_held_false_at_every_observed_50ms_tick_through_endpoint',
        eligibility_unknown_comparisons=int(unknown),contact_censored_comparisons=sum(r['eligibility']['contact_ticks']>0 for r in rows),
        held_censored_comparisons=sum(r['eligibility']['held_ticks']>0 for r in rows),
        eligibility_status_counts=dict(Counter(r['eligibility']['strict_status'] for r in rows)),
        eligible_cohort_only_not_task_or_contact_success=True)
    return dict(all_raw=_metrics(rows),strict_free_precontact=strict_stats,
        auxiliary_preheld_only_contact_unverified=dict(_metrics(preheld),scope='held_false_through_endpoint_contact_not_required_or_certified',
            strict_free_motion_certified=False,eligibility_status_counts=dict(Counter(r['eligibility']['preheld_status'] for r in rows))))


def evaluate_rows(public_outputs,evaluation_labels,*,sensor_dt=.05,timestamp_tolerance=1e-7):
    """Pure offline timestamp join, with a complete emitted-forecast ledger.

    Forecast cohort membership uses only public `free_forecast_available` at
    issuance; future GT only annotates/censors evaluation, never policy input.
    Unavailable/warm-up anchors are counted separately. Each issued forecast
    expects all H8 and16 full future points, including unobserved final tails.
    """
    if sensor_dt!=.05 or not np.isfinite(timestamp_tolerance) or not 0<timestamp_tolerance<sensor_dt/4:
        raise ValueError('declared actual50ms sensor grid and small timestamp tolerance required')
    public=deepcopy(list(public_outputs));truth=deepcopy(list(evaluation_labels))
    if not public:raise ValueError('empty public observation cohort is not a passed evaluation')
    outputs=TimeIndex(public,lambda r:r['result']['timestamp'],timestamp_tolerance)
    labels=TimeIndex(truth,lambda r:r['time'],timestamp_tolerance)
    observations=[];forecast_rows=[];issued=0;unavailable=Counter();mode_counts=Counter();consistency=[]
    # GT-only actual frames expose missing public outputs instead of vanishing.
    union=list(outputs.times)
    union.extend(t for t in labels.times if outputs.get(t) is None)
    for at in sorted(union):
        row=outputs.get(at);label=labels.get(at)
        if row is None:
            p=None;reason='missing_public_output_at_actual_label_timestamp'
        else:
            result=row['result'];p,_,reason=_tracker(result,at,timestamp_tolerance)
            # Observation-position validity does not require a velocity estimate.
            if p is not None and reason in ('missing_or_invalid_shape_tracker_velocity','nonfinite_tracker_velocity','invalid_tracker_velocity'):reason=None
        observations.append(_comparison('RedBox_position',at,at,p,reason,label,
            _interval(labels,at,at,sensor_dt,timestamp_tolerance),0))
    for row,at in zip(outputs.rows,outputs.times):
        result=row['result'];mode_counts[str(result.get('mode','UNKNOWN'))]+=1
        if result.get('free_forecast_available') is not True:
            unavailable[str(result.get('reason') or 'not_emitted')]+=1;continue
        issued+=1
        p,v,tracker_reason=_tracker(result,at,timestamp_tolerance)
        for horizon in FULL_HORIZONS_MS:
            endpoint=at+horizon/1000.;label=labels.get(endpoint)
            interval=_interval(labels,at,endpoint,sensor_dt,timestamp_tolerance)
            full,full_reason=_full_forecast_at(result,endpoint,timestamp_tolerance)
            cv=p+v*(horizon/1000.) if p is not None and v is not None and tracker_reason is None else None
            cv_reason=tracker_reason if cv is None else None
            forecast_rows.append(_comparison('GRU_full_future',at,endpoint,full,full_reason,label,interval,horizon))
            forecast_rows.append(_comparison('CV_full_future',at,endpoint,cv,cv_reason,label,interval,horizon))
            if horizon in H8_MS:
                h8,h8_reason=_series_at(result.get('endpoint_times'),result.get('positions'),endpoint,timestamp_tolerance,'H8_GRU')
                forecast_rows.append(_comparison('GRU_H8',at,endpoint,h8,h8_reason,label,interval,horizon))
                forecast_rows.append(_comparison('CV_H8',at,endpoint,cv,cv_reason,label,interval,horizon))
                if h8 is not None and full is not None:
                    consistency.append(dict(anchor_time=at,horizon_ms=horizon,difference_m=float(np.linalg.norm(h8-full))))
    metrics={}
    for head in ('GRU_H8','CV_H8','GRU_full_future','CV_full_future'):
        rows=[r for r in forecast_rows if r['head']==head]
        horizons=H8_MS if head.endswith('H8') else FULL_HORIZONS_MS
        metrics[head]=dict(aggregate=_scopes(rows),by_horizon_ms={str(h):_scopes([r for r in rows if r['horizon_ms']==h]) for h in horizons})
    return dict(schema=SCHEMA,complete=True,scope='offline_public_position_and_forecast_accuracy_only_not_policy_or_task_success',
        Native_task_success_claimed=False,precision_qualified=False,formal_TEST_used=False,
        labels_used_for_offline_evaluation_only=True,future_GT_used_as_policy_input=False,
        actual_sensor_dt=sensor_dt,timestamp_tolerance=timestamp_tolerance,label_interpolation_used=False,
        full_future_horizons_ms=list(FULL_HORIZONS_MS),H8_horizons_ms=list(H8_MS),
        forecast_cohort='all_publicly_emitted_free_forecasts_no_GT_or_success_selection',
        public_rows=len(public),label_rows=len(truth),observation_expected_timestamps=len(observations),
        emitted_free_forecast_anchors=issued,unavailable_forecast_anchors=len(public)-issued,
        unavailable_forecast_reasons=dict(unavailable),public_mode_counts=dict(mode_counts),
        RedBox_position=_scopes(observations),forecast=metrics,
        H8_full_forecast_consistency=dict(compared=len(consistency),max_difference_m=max((x['difference_m'] for x in consistency),default=None),
                                          mismatches_over_1e_7m=sum(x['difference_m']>1e-7 for x in consistency)),
        sample_ledger=observations+forecast_rows)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_shadow_directory(directory,*,output_dir=None):
    """Read an explicitly non-TEST shadow, optionally write a fresh artifact.

    Scope/run guards are read before any label file. Source files are never
    rewritten. A new output directory must not already exist.
    """
    directory=Path(directory).resolve(strict=True)
    scope_path=directory/'scope.json';scope=json.loads(scope_path.read_text())
    if scope.get('formal_TEST_used') is not False:raise ValueError('explicit non-TEST shadow scope required before label reads')
    run_path=directory.parent/'run_inputs.json'
    if run_path.exists() and json.loads(run_path.read_text()).get('TEST_read') is not False:
        raise ValueError('formal TEST shadow source rejected before labels are read')
    paths=[scope_path,directory/'public_outputs.jsonl',directory/'evaluation_labels.jsonl']
    identity={str(path):_sha(path) for path in paths}
    public=[json.loads(line) for line in paths[1].read_text().splitlines() if line.strip()]
    labels=[json.loads(line) for line in paths[2].read_text().splitlines() if line.strip()]
    result=evaluate_rows(public,labels)
    if identity!={str(path):_sha(path) for path in paths}:raise ValueError('shadow source changed during evaluation')
    result['source_identity']=identity
    result['source_scope']=scope
    if output_dir is not None:
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
        ledger=result.pop('sample_ledger')
        ledger_path=output/'sample_ledger.jsonl'
        with ledger_path.open('x') as stream:
            for row in ledger:stream.write(json.dumps(row,allow_nan=False)+'\n')
        result['sample_ledger_file']=dict(path=str(ledger_path.resolve()),sha256=_sha(ledger_path),rows=len(ledger))
        result['evaluator_sha256']=_sha(__file__)
        with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result
