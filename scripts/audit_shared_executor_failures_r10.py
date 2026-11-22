"""Read fixed TRAIN traces to distinguish planning, contact and dispatch failures."""
from pathlib import Path
from collections import Counter
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_continuous_executor_pilots_r10 import summarize
from our_mani.multitask.evaluation import atomic_json,file_sha
BASE=ROOT/'artifacts/framework/multitask_r10'


def case_audit(path):
    case=json.loads(path.read_text())
    summary=summarize(path)
    actor=case.get('actor_metrics') or {}; m=actor.get('continuous_reference_feedforward') or {}
    timeline=actor.get('executor_public_timeline',{}).get('events',[])
    first=summary['first_violation']; near=[]
    if first:
        t=first['time']
        near=[r for r in timeline if t-.200001<=r['time']<=t+.050001]
    delays={}
    rows={round(r['time'],6):r for r in timeline}
    for delay in (.2,.5,1.):
        errors=[]
        for row in timeline:
            future=rows.get(round(row['time']+delay,6))
            if future is None: continue
            for body,predictions in row.get('forecast',{}).items():
                actual=future.get('public_bodies',{}).get(body)
                prediction=next((p for p in predictions if abs(p['delay']-delay)<1e-7),None)
                if actual and prediction:
                    errors.append(float(np.linalg.norm(np.asarray(prediction['position'])-actual['pose_xyzw'][:3])))
        delays[str(delay)]=dict(public_observation_pairs=len(errors),mean_position_error_m=float(np.mean(errors)) if errors else None,
            p90_position_error_m=float(np.quantile(errors,.9)) if errors else None)
    guard=actor.get('button_standoff',{}); ge=guard.get('events',[])
    score=lambda r:float(np.linalg.norm(r['relative_xy']))+guard['control_interval_seconds']*float(np.linalg.norm(r['relative_velocity_xy']))
    valid=[r for r in ge if r['relative_xy'] is not None and r['relative_velocity_xy'] is not None]
    best=min(valid,key=score) if valid else None
    roll=m.get('current_acquire_rolling_events',[])
    phase_events=[r for r in actor.get('public_events',[]) if r.get('reason') in
        ('advance_to_next_TRAIN_fitted_skill','no_feasible_fitted_interaction_candidate')]
    reasons=Counter(t.get('reason') for r in roll for t in r.get('trials',[]) if not t.get('feasible'))
    return dict(**summary,first_violation_public_context=near,
        delayed_public_forecast_error=delays,
        best_button_alignment=None if best is None else dict(event=best,envelope_m=score(best),
            required_contact_radius_m=guard['contact_radius_m']),
        button_descent_permitted_ticks=sum(r['descent_permitted'] for r in ge),
        rolling_queries=len(roll),rolling_commits=sum(r.get('committed') is True for r in roll),
        rolling_rejection_reasons=dict(reasons),
        phase_events=phase_events,
        last_public_state=None if not timeline else timeline[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',action='append',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(BASE) or out.exists():raise ValueError('fresh audit required')
    cohorts={}
    for cohort in args.cohort:
        if not cohort.replace('_','').isalnum():raise ValueError('simple cohort name')
        cohorts[cohort]=[case_audit(p) for p in sorted((BASE/cohort).rglob('case.json'))]
    atomic_json(out/'report.json',dict(cohorts=cohorts,source_sha256=file_sha(__file__),
        delayed_comparison_is_TRAIN_diagnostic_only=True,additional_native_calls=0,formal_VAL_TEST=False,TEST_read=False))
    for cohort,rows in cohorts.items():
        print(json.dumps(dict(cohort=cohort,rows=[{k:r[k] for k in ('task','method','native_success',
            'actual_motion_pass','button_descent_permitted_ticks','rolling_queries','rolling_commits',
            'delayed_public_forecast_error','best_button_alignment')} for r in rows])))


if __name__=='__main__':main()
