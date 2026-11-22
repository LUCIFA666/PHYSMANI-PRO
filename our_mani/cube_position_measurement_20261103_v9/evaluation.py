"""Scope qualification: fixed required24/annulus16, all80 safety.

Hand/axis diagnostics cannot be relabelled into scope coverage. Full56near
coverage is a separate descriptive/composite flag, not hidden by this module.
"""
import hashlib,json
import numpy as np
from .data import PARTITION_IDS,SHOT_IDS,POSITION_BANDS
from .prediction import validate_prediction_route,prediction_route,validate_legacy_audit
from ..near_goal_depth_refiner_20261103.data import make_measurement as legacy_measurement
from ..near_goal_depth_refiner_20261103_v4.evaluation import (
    BASELINE_ELIGIBILITY_PROTOCOL,_position_record,_aggregate_positions,position_stats)
from ..cube_position_measurement_20261103_v8.evaluation import aggregate_support
from ..cube_position_measurement_20261103_v7.support_quality import score_current_support

QUALITY_PROTOCOL='independent_scope12_EF_required24_annulus16_all80_position_v9'


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _samples(samples):
    expected=[(p,p+'/'+s,'calib')for p in PARTITION_IDS for s in SHOT_IDS]
    if [(s['parent_id'],s['scene_id'],s['role'])for s in samples]!=expected:raise ValueError('Entire exact new80 required')
    if any(s.get('position_band')!=POSITION_BANDS[s['scene_id'].split('/')[-1]]for s in samples):raise ValueError('Fixed80 bands changed')


def freeze_legacy_eligibility(samples):
    samples=list(samples);_samples(samples);rows=[]
    for s in samples:
        old=legacy_measurement(s['inputs'],s['crops'],s['semantic']==1)
        shot=s['scene_id'].split('/')[-1]
        rows.append(dict(scene_id=s['scene_id'],eligible=bool(old['refiner_available']),position_band=s['position_band'],
            legacy_control=shot=='shot02',scope_annulus=shot in ('shot07','shot08'),
            required_position=shot in ('shot02','shot07','shot08'),planned_near=s['condition']=='near_goal'))
    value=dict(protocol=BASELINE_ELIGIBILITY_PROTOCOL,scope='position_calibration8_v9',records=rows,
        required_eligible=sum(r['eligible']for r in rows),control_denominator=8,annulus_denominator=16,required_position_denominator=24,
        full_near_denominator=56,frozen_before_NN_predictions=True)
    value['content_sha256']=canonical_sha(value);return value


def _eligibility(ledger):
    value=dict(ledger);digest=value.pop('content_sha256',None)
    if (digest!=canonical_sha(value)or value.get('scope')!='position_calibration8_v9'
        or value.get('protocol')!=BASELINE_ELIGIBILITY_PROTOCOL or value.get('frozen_before_NN_predictions')is not True
        or value.get('control_denominator')!=8 or value.get('annulus_denominator')!=16
        or value.get('required_position_denominator')!=24 or value.get('full_near_denominator')!=56):
        raise ValueError('Fixed preforward legacy/control/annulus ledger required')
    expected=[p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]
    if [r['scene_id']for r in value['records']]!=expected:raise ValueError('All80 ordered eligibility rows')
    for r in value['records']:
        s=r['scene_id'].split('/')[-1]
        if (type(r.get('eligible'))is not bool or r.get('position_band')!=POSITION_BANDS[s]
            or r.get('legacy_control')is not(s=='shot02')or r.get('scope_annulus')is not(s in ('shot07','shot08'))
            or r.get('required_position')is not(s in ('shot02','shot07','shot08'))
            or r.get('planned_near')is not(s in SHOT_IDS[2:9])):raise ValueError('No swapped scope/control denominators')
    if value['required_eligible']!=sum(r['eligible']for r in value['records']):raise ValueError('Legacy denominator changed')


def _group(rows):
    return dict(samples=len(rows),available=sum(r['refiner_available']for r in rows),
        coverage=None if not rows else sum(r['refiner_available']for r in rows)/len(rows),
        centre_error=position_stats([r['all_available_GT_centre_error_m']for r in rows if r['all_available_GT_centre_error_m']is not None]),
        scene_ids=[r['scene_id']for r in rows])


def _groups(records):
    controls=[r for r in records if r['legacy_control']]
    control=_group(controls)
    control.update(baseline_available=sum(r['legacy_baseline_available']for r in controls),
        baseline_centre_error=position_stats([r['legacy_baseline_GT_centre_error_m']for r in controls
            if r['legacy_baseline_GT_centre_error_m']is not None]),
        legacy_passes_preserved=sum(r['legacy_baseline_available']and r['legacy_pass_exact']for r in controls),
        legacy_pass_nonregression=all(not r['legacy_baseline_available']or r['legacy_pass_exact']for r in controls),
        independent_coverage_gate=False)
    return dict(required24=_group([r for r in records if r['required_position']]),legacy_control=control,
        scope_annulus=_group([r for r in records if r['scope_annulus']]),
        all56_near=_group([r for r in records if r['planned_near']]))


def position_gate(positions,point_pixels,stencil_pixels,*,expected_samples=80):
    reasons=[]
    if type(positions.get('samples'))is not int or positions['samples']!=expected_samples:reasons.append('all80_required')
    groups=positions.get('scope_validation',{})
    for key,count in (('required24',24),('scope_annulus',16)):
        g=groups.get(key,{})
        if (type(g.get('samples'))is not int or g['samples']!=count or type(g.get('available'))is not int
            or not 0<=g['available']<=count or isinstance(g.get('coverage'),(bool,np.bool_))
            or not isinstance(g.get('coverage'),(int,float))or not np.isfinite(g['coverage'])
            or abs(g['coverage']-g['available']/count)>1e-15 or g['coverage']<.9):
            reasons.append(key+'_fixed_coverage_below90')
        m=g.get('centre_error',{})
        if (m.get('count',0)<=0 or m.get('p95_m')is None or m.get('max_m')is None
            or not np.isfinite([m['p95_m'],m['max_m']]).all()or m['p95_m']>.005 or m['max_m']>.010):
            reasons.append(key+'_position_exceeds5mm_p95_or10mm_max')
    control=groups.get('legacy_control',{})
    if (type(control.get('samples'))is not int or control['samples']!=8
        or control.get('independent_coverage_gate')is not False
        or control.get('legacy_pass_nonregression')is not True
        or type(control.get('baseline_available'))is not int
        or type(control.get('legacy_passes_preserved'))is not int
        or not 0<=control['baseline_available']<=8
        or control['legacy_passes_preserved']!=control['baseline_available']):
        reasons.append('legacy_control_missing_or_oldpass_regression')
    m=positions.get('all_available_centre_error',{})
    if (m.get('count',0)<=0 or m.get('p95_m')is None or m.get('max_m')is None
        or not np.isfinite([m['p95_m'],m['max_m']]).all()or m['p95_m']>.005 or m['max_m']>.010):reasons.append('all_available_position_exceeds5mm_p95_or10mm_max')
    for key,m in (('actual_D_points',point_pixels),('used_stencil_union',stencil_pixels)):
        if any(type(m.get(k))is not int or m[k]<0 for k in ('TP','FP','U')):raise ValueError('Integer nonnegative TP/FP/U required')
        n=m['TP']+m['FP']+m['U']
        if n<=0 or m['TP']/n<.995:reasons.append(key+'_conservative_PPV_below995')
    for key,count in (('sphere_only',16),('cube_far',8)):
        n=positions.get('negatives',{}).get(key,{})
        if n.get('samples')!=count or n.get('falseavailable')!=0:reasons.append(key+'_missing_or_falseavailable')
    if positions.get('owner_unobserved_falseavailable')!=0:reasons.append('owner_unobserved_falseavailable')
    passed=not reasons;near=groups.get('all56_near',{})
    composite=bool(passed and near.get('samples')==56 and near.get('coverage')is not None and near['coverage']>=.9)
    return dict(quality_protocol=QUALITY_PROTOCOL,position_measurement_qualified=passed,scope12_EF_qualified=passed,
        axis_general_qualified=False,composite_global_coverage_qualified=composite,
        active_certified_routes=['legacy_v8','scope12_EF']if passed else[],model_qualified=False,raw_mask_model_qualified=False,
        grasp_qualified=False,Native_control_qualified=False,force_or_contact_qualified=False,
        failure_reasons=reasons,scope_radius_m=.12,proposal_thresholds=[.995,.9],
        empirical_known_axisaligned_opaque50mm_current_position_only=True,global_statistical_or_physical_bound=False)


def validate_report_integrity(report,eligibility):
    """Rebuild derived summaries from the complete frozen rows at load time."""
    _eligibility(eligibility);records=report.get('records',[])
    if [(r.get('parent_id'),r.get('scene_id'),r.get('role'))for r in records]!=[(p,p+'/'+s,'calib')for p in PARTITION_IDS for s in SHOT_IDS]:
        raise ValueError('All80 scored scenes in fixed order required')
    for r,fixed in zip(records,eligibility['records']):
        for key in ('legacy_control','scope_annulus','required_position','planned_near','position_band'):
            if r.get(key)!=fixed[key] or (key!='position_band'and type(r.get(key))is not bool):raise ValueError('Fixed scored group changed')
        if r.get('oracle_geometry_eligible')is not fixed['eligible']or type(r.get('refiner_available'))is not bool:
            raise ValueError('Availability/old eligibility must retain exact bool identities')
        if (type(r.get('legacy_baseline_available'))is not bool or type(r.get('legacy_pass_exact'))is not bool
            or(r['legacy_baseline_available']and(r['legacy_pass_exact']is not True or r['refiner_available']is not True
                or r['legacy_baseline_GT_centre_error_m']!=r['all_available_GT_centre_error_m']))):
            raise ValueError('Frozen complete legacy pass comparison changed')
    expected=_aggregate_positions(records)
    expected['scope_validation']=_groups(records)
    if expected!=report.get('positions'):raise ValueError('Derived scope and allavailable position report changed')
    support=report.get('per_scene_support',[])
    if len(support)!=80 or any(s.get('geometry_available')is not r['refiner_available']for s,r in zip(support,records)):
        raise ValueError('Every rejected/available support row must match output availability')
    used=[s for s in support if s['geometry_available']]
    metrics={name:aggregate_support(used,key)for name,key in (('points','point_pixels'),('stencils','stencil_pixels'))}
    if metrics!=report.get('used_available_support'):raise ValueError('Used support totals changed')
    decision=position_gate(expected,metrics['points'],metrics['stencils'])
    if decision!=report.get('decision'):raise ValueError('Independent scope-only decision changed')
    return decision


def score_frozen_positions(samples,predictions,eligibility):
    samples=list(samples);_samples(samples);_eligibility(eligibility)
    if len(predictions)!=80:raise ValueError('All80 frozen public predictions required')
    records=[];supports=[]
    for s,p,fixed in zip(samples,predictions,eligibility['records']):
        if p['scene_id']!=s['scene_id']:raise ValueError('No reordered/dropped prediction')
        validate_prediction_route(p);validate_legacy_audit(p,p['legacy_audit']);m=p['measurement']
        r=_position_record(s,m,dict(refiner_available=fixed['eligible']))
        r.update(position_band=s['position_band'],legacy_control=fixed['legacy_control'],scope_annulus=fixed['scope_annulus'],
            required_position=fixed['required_position'],planned_near=fixed['planned_near'],active_prediction_route=prediction_route(p),
            legacy_baseline_available=p['legacy_audit']['legacy_position_measurement_available'],
            legacy_pass_exact=p['legacy_audit']['legacy_pass_returned_exact_object'],
            legacy_baseline_GT_centre_error_m=(r['all_available_GT_centre_error_m']
                if p['legacy_audit']['legacy_position_measurement_available']else None))
        records.append(r);supports.append(score_current_support(s,m))
    positions=_aggregate_positions(records)
    if positions['oracle_geometry_eligible']!=eligibility['required_eligible']:raise ValueError('Original legacy ledger changed')
    positions['scope_validation']=_groups(records)
    available=[r for r in supports if r['geometry_available']]
    used={name:aggregate_support(available,key)for name,key in (('points','point_pixels'),('stencils','stencil_pixels'))}
    all_support={name:aggregate_support(supports,key)for name,key in (('points','point_pixels'),('stencils','stencil_pixels'))}
    decision=position_gate(positions,used['points'],used['stencils'])
    return dict(quality_protocol=QUALITY_PROTOCOL,positions=positions,records=records,per_scene_support=supports,
        used_available_support=used,all_predictions_support_diagnostic=all_support,decision=decision,
        by_parent={p:_aggregate_positions([r for r in records if r['parent_id']==p])for p in PARTITION_IDS},
        by_position_band={band:_aggregate_positions([r for r in records if r['position_band']==band])for band in sorted(set(POSITION_BANDS.values()))},
        route_counts={route:sum(r['active_prediction_route']==route for r in records)for route in ('legacy_v8','scope12_EF')},
        threshold_selected_from_fresh=False,axis_general_qualification_attempted=False,
        interval_radii_are_error_CI=False,GT_labels_only_posthoc=True,TEST_read=False)
