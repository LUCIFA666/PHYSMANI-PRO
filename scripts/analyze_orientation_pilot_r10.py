"""Read-only TRAIN orientation metrics; never write observability annotations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import pilot_orientation_registration_r10 as pilot


def statistics(values,*,scale=1.):
    x=np.asarray(values,float)*scale
    if not len(x):return dict(count=0,mean=None,median=None,p90=None,maximum=None)
    pilot.require(np.isfinite(x).all(),'nonfinite metric cannot be silently summarized')
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p90=float(np.quantile(x,.9)),maximum=float(x.max()))


def longest_gap(flags):
    maximum=run=0
    for present in flags:
        run=0 if present else run+1;maximum=max(maximum,run)
    return maximum


def summarize_body(rows):
    fits=[r['raw_fit_metric_only'] for r in rows if r['raw_fit_metric_only'] is not None]
    valid=[r for r in fits if r['fit_valid']]
    clear=[r for r in valid if not r['rotation_ambiguous']]
    physical=[r for r in fits if r['rotation_errors'] is not None]
    result=dict(frames=len(rows),raw_fitted_frames=len(fits),valid_raw_frames=len(valid),
        valid_nonambiguous_raw_frames=len(clear),
        directly_visible_frames=sum(r['causal_registration'].get('directly_visible',False) for r in rows),
        registration_valid_frames=sum(r['causal_registration'].get('registration_valid',False) for r in rows),
        missing_raw_fit_frames=len(rows)-len(fits),
        longest_invalid_registration_gap_frames=longest_gap([r['causal_registration'].get('registration_valid',False) for r in rows]),
        rotation_ambiguous_frames=sum(r['rotation_ambiguous'] for r in fits),
        rotation_ambiguity_fraction_of_raw_fits=sum(r['rotation_ambiguous'] for r in fits)/len(fits) if fits else None,
        full_orientation_verified=False,
        observed_points=statistics([r['causal_registration'].get('observed_point_count',0) for r in rows]),
        position_error_mm_all_raw_fits=statistics([r['position_error_m'] for r in fits],scale=1000),
        position_error_mm_valid_raw_fits=statistics([r['position_error_m'] for r in valid],scale=1000),
        raw_registration_rms_mm=statistics([r['rms_m'] for r in fits],scale=1000),
        raw_SO3_error_degrees=statistics([r['rotation_errors']['raw_SO3_error_rad'] for r in physical],scale=180/np.pi),
        valid_nonambiguous_SO3_error_degrees=statistics([r['rotation_errors']['raw_SO3_error_rad'] for r in clear
                                                      if r['rotation_errors'] is not None],scale=180/np.pi),axes={})
    for axis in ('local_x','local_y','local_z'):
        result['axes'][axis]={kind:statistics([r['rotation_errors']['axes'][axis][kind] for r in physical],scale=180/np.pi)
            for kind in ('directed_error_rad','undirected_error_rad')}
        result['axes'][axis]['reported_summary_units']='degrees; original per-frame metrics retain radians'
    return result


def analyze(directory,plan_path):
    directory=Path(directory).resolve();plan=pilot.verify_plan(plan_path)
    queue_path=directory/'queue_report.json';queue=json.loads(queue_path.read_text())
    execution_path=directory/'execution_inputs.json';execution=json.loads(execution_path.read_text())
    pilot.require(queue['complete'] and queue['expected']==14 and len(queue['results'])==14,'full 14-case queue evidence required')
    pilot.require(execution['plan_sha256']==plan['plan_sha256'] and execution['plan_file_sha256']==pilot.sha(plan_path)
        and execution['pilot_source_sha256']==pilot.sha(pilot.__file__) and execution['workers']==2
        and execution['GPU_disabled'] is True and execution['teacher_used'] is False,'pilot execution identity differs')
    bodies={};reports={};infra=[]
    for case in queue['results']:
        task=case['task']
        if not case['completed']:
            infra.append(case);continue
        path=ROOT/case['report'];pilot.require(pilot.sha(path)==case['report_sha256'],'native report changed')
        report=json.loads(path.read_text());reports[task]=dict(path=str(path.relative_to(ROOT)),sha256=pilot.sha(path))
        pilot.require(report['complete'] and report['task']==task and report['plan_sha256']==plan['plan_sha256']
            and report['TRAIN_only'] is True and report['TEST_used'] is False and report['teacher_used'] is False
            and report['GPU_used'] is False and report['annotation_changes_written'] is False,'native report scope differs')
        sensor=report['causal_sensor_observations'];sensor_path=ROOT/sensor['path']
        pilot.require(pilot.sha(sensor_path)==sensor['sha256'],'causal sensor archive changed')
        with np.load(sensor_path,allow_pickle=False) as arrays:
            pilot.require(len(arrays['times'])==report['actual_frames'] and not any(
                term in key.lower() for key in arrays for term in ('target','truth','outcome','success')), 'GT entered causal sensor archive')
            for camera in ('front','8','16','36'):
                pilot.require(arrays['rgb_'+camera].shape==(report['actual_frames'],256,256,3)
                    and arrays['depth_'+camera].shape==(report['actual_frames'],256,256)
                    and arrays['mask_'+camera].shape==(report['actual_frames'],256,256,3), 'four-view sensor coverage differs')
        task_rows={}
        for variant in report['variants']:
            all_rows=report['frames'];ids=all_rows[0]['metric_only']['body_ids'];variant_rows={}
            for role in ids:
                values=[f['metric_only']['registration_errors'][variant][role] for f in all_rows]
                variant_rows[role]=summarize_body(values)
            task_rows[variant]=variant_rows
        bodies[task]=dict(planned_frames=report['planned_frames'],actual_frames=report['actual_frames'],
            counted_hold_ticks=report['counted_hold_ticks'],shortened_coverage=report['shortened_coverage'],variants=task_rows,
            CAD_symmetry_diagnostic=report['CAD_symmetry_diagnostic'])
    return dict(version='r10_TRAIN_orientation_pilot_metric_summary_v1',complete=True,TRAIN_only=True,TEST_used=False,
        expected_cases=14,completed_cases=len(reports),infrastructure_cases=infra,task_bodies=bodies,
        source_sha256={str(p.relative_to(ROOT)):pilot.sha(p) for p in (queue_path,execution_path,Path(plan_path),Path(__file__))},
        native_reports=reports,annotation_changes_written=False,automatic_annotation_promotion=False,
        orientation_verified_for_policy=False,interpretation=[
            'candidate_full is an unverified hypothesis, including frames where the tracker itself reports observable=true',
            'raw SO3/axis errors include ambiguous fits and are metric-only; valid_nonambiguous subset is separately reported',
            'sphere physical-rotation metrics remain absent; identity gauge is not ground-truth orientation',
            'axis errors use corresponding native CAD local axes; low undirected error does not establish polarity or full twist',
            'one TRAIN source and short unmanipulated hold sequence cannot establish across-source/occlusion/grasp observability',
            'continuous/discrete CAD symmetries still require review even if a finite-source SO3 error is small'])


def fmt(x):return '--' if x is None else f'{x:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    report=analyze(ROOT/args.directory,ROOT/args.plan);out=(ROOT/args.out).resolve();pilot.write(out,report)
    text=['R10 TRAIN orientation registration pilot：原始误差/覆盖诊断，不自动更新方向注解',
        f"completed={report['completed_cases']}/14; infrastructure={len(report['infrastructure_cases'])}; no teacher/GPU/TEST",'']
    for task,data in report['task_bodies'].items():
        text.append(task+f" | frames={data['actual_frames']}/{data['planned_frames']}")
        for variant,roles in data['variants'].items():
            text.append('  '+variant)
            for role,r in roles.items():
                axes='; '.join(name+':'+fmt(value['directed_error_rad']['p90'])+'/'+fmt(value['undirected_error_rad']['p90'])
                              for name,value in r['axes'].items())
                text.append(f"    {role}: valid {r['valid_raw_frames']}/{r['frames']}, ambiguity {r['rotation_ambiguous_frames']}/{r['raw_fitted_frames']}, "
                    f"gap {r['longest_invalid_registration_gap_frames']}f, position p90 {fmt(r['position_error_mm_valid_raw_fits']['p90'])}mm, "
                    f"raw SO3 p90 {fmt(r['raw_SO3_error_degrees']['p90'])}deg, axes directed/undirected p90 deg [{axes}]")
        text.append('')
    text.extend(report['interpretation']);text_path=out.with_suffix('.txt')
    pilot.require(not text_path.exists(),'preserve existing summary text');text_path.write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(complete=True,completed_cases=report['completed_cases'],infrastructure_cases=len(report['infrastructure_cases']),
        annotation_changes_written=False,summary=str(out.relative_to(ROOT))),ensure_ascii=False))


if __name__=='__main__':main()
