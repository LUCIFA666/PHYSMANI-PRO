"""Snapshot each TRAIN repair cohort independently; never merge retries as scores."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_continuous_executor_pilots_r10 import summarize
from our_mani.multitask.evaluation import atomic_json,file_sha
BASE=ROOT/'artifacts/framework/multitask_r10'
EXPECTED={'continuous_bridge_v402':6,'moving_button_v404':4,'high_peg_insert_v396':2,
    'joint16_remove_v405':2,'task_geometry_v407':6,'registered_tasks_v408':22,
    'task_constraint_v409':6,'buzz_command_v412':4,'button_corridor_v414':4,
    'focused_ring_v415':1,'buzz_handoff_v417':4,'buzz_physical_v418':4,
    'grasp_compensated_v420':2,'basketball_executor_v421':4}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(BASE) or out.exists():raise ValueError('fresh audit snapshot required')
    cohorts={}
    for label,expected in EXPECTED.items():
        rows=[]
        for p in sorted((BASE/label).rglob('case.json')):
            r=summarize(p);d=json.loads(p.read_text());a=d.get('actor_metrics')or{}
            r['completed_public_phases']=[e['phase'] for e in a.get('observable_skill_events',[]) if e.get('event')=='observable_skill_completed']
            r['selected_phases']=[e['phase'] for e in a.get('candidate_selection_events',[])]
            r['source_raw_episode_id']=d.get('source_raw_episode_id',d.get('parent_raw_episode_id'))
            r['first_violation']=None if r['first_violation'] is None else dict(time=r['first_violation']['time'],violations=r['first_violation']['violations'])
            rows.append(r)
        cohorts[label]=dict(expected_cases=expected,completed=len(rows),pending=expected-len(rows),
            runtime_clean=sum(r['runtime_clean'] for r in rows),motion_pass=sum(r['actual_motion_pass'] is True for r in rows),
            native_success=sum(r['native_success'] for r in rows),cases=rows)
    owner=ROOT/'runtime/framework/native_resource_owner_v401'
    status=json.loads((owner/'status.json').read_text());launch=json.loads((owner/'launch.json').read_text())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),cohorts=cohorts,
        live_owner=dict(pid=launch['pid'],start_ticks=launch['start_ticks'],active_jobs=status['active_jobs'],
            completed_jobs=len(status['completed']),rejected_requests=len(status['rejected'])),
        shared_context_checkpoint='6447ac755032b022d18884936e02a90d8ec2f2b219b5d42fc140cdd1618cd017',
        dynamics_TRAIN_parent_count=1600,formal_VAL_TEST_started=False,
        exactly_three_methods=['cv_geometry','context_geometry','physmani'],
        baseline_algorithm_unchanged=True,TRAIN_smoke_is_not_formal_score=True,
        source_sha256=file_sha(__file__),TEST_read=False)
    atomic_json(out/'report.json',report)
    print(json.dumps({k:{f:v for f,v in r.items() if f!='cases'} for k,r in cohorts.items()}))


if __name__=='__main__':main()
