"""Posthoc scorer: validate the complete public prediction freeze before labels.

This cannot operate on arbitrary cases, TEST, active capture or NN fit outputs.
GT is permitted only for the fixed closed V59 development observer replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_v60_far_consistency_20261103 import BASE,CASES,SOURCES,roster


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def validate_freeze(predictions):
    """No label file may be opened until this entire function succeeds."""
    out=Path(predictions).resolve(strict=True); freeze=read(out/'public_prediction_freeze.json')
    cases=roster() # closure metadata only
    expected={'settings.json'}|{n+s for n in CASES for s in ('.predictions.jsonl','.current_components.npz')}
    if (freeze.get('complete') is not True or freeze.get('cases')!=list(CASES)
        or set(freeze.get('prediction_and_config_files_sha256',{}))!=expected
        or set(freeze.get('case_frames',{}))!=set(CASES)
        or set(freeze.get('frame_rosters',{}))!=set(CASES)
        or freeze.get('GT_labels_opened_during_prediction') is not False):
        raise ValueError('Exact complete four-case public prediction roster required before GT')
    for name,digest in freeze['prediction_and_config_files_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('Frozen prediction or config changed')
    settings=read(out/'settings.json')
    if (settings.get('cases')!=list(CASES) or settings.get('NN_weights_loaded') is not False
        or settings.get('minimum_red')!=100 or settings.get('complete_geometry') is not False
        or settings.get('half_side_m')!=.025 or settings.get('point_axis_development_allowance_m')!=.006
        or settings.get('near_scope_m')!=.08 or settings.get('tolerance_or_threshold_search') is not False):
        raise ValueError('Fixed no-NN geometry-only replay contract required')
    expected_sources={str(ROOT/name) for name in SOURCES}
    expected_public={str(row['camera_calibration']) for row in cases.values()}
    expected_public.update(str(path) for row in cases.values() for path in row['frames'])
    if (set(freeze.get('source_sha256',{}))!=expected_sources
        or settings.get('source_sha256')!=freeze['source_sha256']
        or set(freeze.get('public_input_sources_sha256',{}))!=expected_public):
        raise ValueError('Exact whole source and public input SHA roster required before GT')
    for filename,digest in {**freeze['source_sha256'],**freeze['public_input_sources_sha256']}.items():
        if sha(filename)!=digest:raise ValueError('Frozen public source/current implementation changed')
    result={}
    for name in CASES:
        rows=lines(out/(name+'.predictions.jsonl')); count=len(cases[name]['frames'])
        if (len(rows)!=count or freeze['case_frames'][name]!=count
            or len(freeze['frame_rosters'][name])!=count):raise ValueError('Full cohort frame count required')
        with np.load(out/(name+'.current_components.npz'),allow_pickle=False) as clouds:
            if set(clouds.files)!={k+s for k in ('legacy','conditional') for s in ('_points','_offsets')}:
                raise ValueError('Complete actual-component arrays required')
            for k in ('legacy','conditional'):
                pts,offsets=clouds[k+'_points'],clouds[k+'_offsets']
                if pts.ndim!=2 or pts.shape[1]!=3 or offsets.shape!=(count+1,) or offsets[0]!=0 or offsets[-1]!=len(pts) or np.any(np.diff(offsets)<0):
                    raise ValueError('Component offsets/order mismatch')
                for i,row in enumerate(rows):
                    p=pts[offsets[i]:offsets[i+1]]
                    if row[k].get('raw_available') and hashlib.sha256(p.tobytes()).hexdigest()!=row[k].get('raw_component_sha256'):
                        raise ValueError('Current component differs from frozen replay')
        previous=None
        for i,(row,expected_row,path) in enumerate(zip(rows,freeze['frame_rosters'][name],cases[name]['frames'])):
            if (row['case']!=name or row['index']!=i or row['input_path']!=str(path)
                or row['input_sha256']!=expected_row['input_sha256'] or row['input_sha256']!=sha(path)
                or expected_row['index']!=i or expected_row['timestamp']!=row['timestamp']
                or not np.isfinite(row['timestamp'])
                or (previous is not None and abs(row['timestamp']-previous-.05)>1e-6)):
                raise ValueError('Exact causal same-frame rows required')
            previous=row['timestamp']
        result[name]=rows
    return out,freeze,result


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=None if not len(a) else float(a.mean()),
        p95=None if not len(a) else float(np.percentile(a,95)),max=None if not len(a) else float(a.max()))


def summarize(rows):
    near=[r for r in rows if r.get('GT_centre_distance_to_public_sphere_m',np.inf)<=.08]
    accepted=[r for r in rows if r['route']=='FAR_RAW_CONSISTENT']
    mid=[r for r in rows if .08<r.get('GT_centre_distance_to_public_sphere_m',-1)<=.08+np.sqrt(3)*.025]
    return dict(frames=len(rows),GT_missing=sum(not r['GT_available'] for r in rows),
        far_consistent_accepted=len(accepted),unknown=sum(r['route']=='UNKNOWN' for r in rows),
        all_current_NN_measurements_unavailable=True,
        GT_near_frames=len(near),GT_near_incorrectly_routed_far=sum(r['route']=='FAR_RAW_CONSISTENT' for r in near),
        GT_near_remaining_unknown=sum(r['route']=='UNKNOWN' for r in near),
        accepted_GT_centre_outside_compatible_box=sum(r.get('GT_centre_in_compatible_box') is False for r in accepted),
        accepted_axis_error_allowance_violations=sum(r.get('maximum_component_axis_excursion_beyond_GT_cube_m',np.inf)>.006 for r in accepted),
        accepted_raw_centre_error_m=stats([r['raw_centre_error_m'] for r in accepted if r.get('raw_centre_error_m') is not None]),
        all_legacy_confident_centre_error_m=stats([r['legacy_centre_error_m'] for r in rows if r.get('legacy_confident') and r.get('legacy_centre_error_m') is not None]),
        newly_accepted_beyond_fixed_halfdiagonal_gate=sum(not r['fixed_halfdiagonal_test_passed'] for r in accepted),
        GT_8_to_12p33cm_frames=len(mid),GT_8_to_12p33cm_accepted=sum(r['route']=='FAR_RAW_CONSISTENT' for r in mid),
        GT_8_to_12p33cm_accepted_but_halfdiagonal_rejected=sum(r['route']=='FAR_RAW_CONSISTENT' and not r['fixed_halfdiagonal_test_passed'] for r in mid),
        rejected_current_component_still_had_raw_centre=sum(r['route']=='UNKNOWN' and r['raw_available'] for r in rows),
        raw_component_unavailable=sum(not r['raw_available'] for r in rows),
        legacy_tracker_wall_seconds=stats([r['legacy_wall_seconds'] for r in rows]),
        conditional_tracker_and_farcheck_wall_seconds=stats([r['conditional_wall_seconds'] for r in rows]))


def score(predictions,output):
    # This full validation finishes for all four cases before the first GT open.
    frozen,freeze,predictions=validate_freeze(predictions)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    GT_sources={}; allrows=[];percase={}
    for name,pred in predictions.items():
        initial_path=BASE/name/'initial_snapshot.json';ticks_path=BASE/name/'native_ticks.jsonl'
        GT_sources[str(initial_path)]=sha(initial_path);GT_sources[str(ticks_path)]=sha(ticks_path)
        initial=read(initial_path)['state']; ticks=lines(ticks_path)
        actual=[initial]+ticks;bytime={}
        for state in actual:
            now=float(state['time']);key=int(round(now/.05))
            if abs(now-key*.05)>1e-8 or key in bytime:raise ValueError('Unique actual50ms GT ledger required')
            bytime[key]=state
        firstclose=None;previous=1.
        for state in actual:
            requested=float(state['requested_opening'])
            if previous>=.5 and requested<.5 and firstclose is None:firstclose=float(state['time'])
            previous=requested
        rows=[]
        with np.load(frozen/(name+'.current_components.npz'),allow_pickle=False) as z:
            points=z['conditional_points'];offsets=z['conditional_offsets']
            for p in pred:
                c=p['conditional'];legacy=p['legacy'];i=p['index'];now=p['timestamp']
                t=bytime.get(int(round(now/.05)))
                r=dict(case=name,index=i,timestamp=now,GT_available=t is not None,
                    route=c['route'],raw_available=c['raw_available'],
                    public_initial_sphere=p['initial_public_sphere'],
                    fixed_halfdiagonal_test_passed=c.get('fixed_halfdiagonal_test_passed',False),
                    far_reason=c.get('far_consistency',{}).get('reason',c.get('raw_error')),
                    legacy_confident=bool(legacy.get('raw_confidence',0.)>=.5),
                    legacy_wall_seconds=legacy['wall_seconds'],conditional_wall_seconds=c['wall_seconds'],
                    cohort='preclose' if firstclose is None or now<firstclose-1e-8 else 'postclose')
                if t is not None:
                    centre=np.asarray(t['object_position'],float);sphere=p['initial_public_sphere']
                    if sphere is not None:r['GT_centre_distance_to_public_sphere_m']=float(np.linalg.norm(centre-np.asarray(sphere)))
                    if legacy.get('raw_available'):r['legacy_centre_error_m']=float(np.linalg.norm(np.asarray(legacy['raw_centre'])-centre))
                    if c.get('raw_available'):
                        r['raw_centre_error_m']=float(np.linalg.norm(np.asarray(c['raw_centre'])-centre))
                        proof=c['far_consistency'];lo=proof['centre_lower_m'];hi=proof['centre_upper_m']
                        r['GT_centre_in_compatible_box']=bool(lo is not None and hi is not None and np.all(centre>=np.asarray(lo)-1e-12) and np.all(centre<=np.asarray(hi)+1e-12))
                        pc=points[offsets[i]:offsets[i+1]]
                        r['maximum_component_axis_excursion_beyond_GT_cube_m']=float(np.maximum(np.abs(pc-centre)-.025,0).max())
                        local=(pc-centre)@Rotation.from_quat(t['object_quaternion']).as_matrix()
                        q=np.abs(local)-.025
                        sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
                        r['component_distance_to_actual_oriented_Cube_surface_m']=stats(np.abs(sdf))
                rows.append(r)
        percase[name]=dict(summary=summarize(rows),by_phase={k:summarize([r for r in rows if r['cohort']==k]) for k in ('preclose','postclose')})
        allrows+=rows
    for path,digest in GT_sources.items():
        if sha(path)!=digest:raise ValueError('GT file changed during posthoc scoring')
    with (out/'posthoc_rows.jsonl').open('x') as stream:
        for row in allrows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(schema='V60_far_consistency_closed_V59_posthoc_v1',
        prediction_freeze=dict(path=str(frozen/'public_prediction_freeze.json'),sha256=sha(frozen/'public_prediction_freeze.json')),
        all_four_public_outputs_validated_before_GT=True,GT_sources_sha256=GT_sources,
        total=summarize(allrows),per_case=percase,
        phase_summaries={k:summarize([r for r in allrows if r['cohort']==k]) for k in ('preclose','postclose')},
        NN_weights_loaded=False,Native_started=False,fit_started=False,TEST_read=False,
        tolerance_or_threshold_selected_using_GT=False,production_binding_approved=False,
        limits=['Fixed oldrobottrajectory observerreplay, notnewclosedloop orV60NNperformance.',
            'GT distance uses original actualCube centre relative to initial publicSphere, onlyposthoc.',
            'Farconsistency isconditional on association/axisalignedknownCube/6mmdevelopmentpointallowance; wrongcomponent neednotfailmath.',
            'Cube GTsurface distance isgeometryproxy only, notsemanticdepthownership/force/contactgroundtruth.',
            'Allnear NNmeasurements absent bydesign; UNKNOWN frames retained. No fallbackmintedmeasurement.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'scoring_source.py').write_bytes(Path(__file__).read_bytes())
    return dict(output=str(out),total=report['total'],report_sha256=sha(out/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True)
    p.add_argument('--score',action='store_true');p.add_argument('--output')
    args=p.parse_args()
    if args.score:
        if not args.output:p.error('Fresh --output required for posthoc score')
        print(json.dumps(score(args.predictions,args.output)))
    else:
        out,freeze,rows=validate_freeze(args.predictions)
        print(json.dumps(dict(all_public_outputs_validated_before_any_GT=True,case_frames={n:len(v) for n,v in rows.items()},prediction_freeze_sha256=sha(out/'public_prediction_freeze.json'))))
