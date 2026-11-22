"""Current TRAIN cohorts; support compact receipts and measured block windows."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.three_method_release16 import measured_motion
BASE=ROOT/'artifacts/framework/multitask_r10'
EXPECTED={'block_fresh_tail_v451':16,'cup_contact_speed_v443':4,
 'remove_uncertain_translation_v445':4,'core10_fixed_four_v446':54,
 'baseline_windows_v437':13,'remove_translation_v440':4}

def summarize(path):
 d=json.loads(path.read_text())
 if d['split']!='train' or d.get('formal_test_used') or d.get('TEST_used_for_fitting'):
  raise ValueError('only nonfitting TRAIN smoke evidence')
 a=d.get('actor_metrics')or{};m=a.get('continuous_reference_feedforward')or{}
 events=d.get('all_stage_counted_joint_motion',{}).get('events',m.get('observed_events',[]))
 violation=next((e for e in events if e.get('violations') or e.get('speed_passed') is False
  or e.get('position_passed') is False or e.get('acceleration_passed') is False),None)
 return dict(task=d['task'],raw=d['raw_episode_id'],method=d['method'],
  runtime_clean=d.get('complete') is True and d.get('cleanup_completed') is True and d.get('infrastructure_error') is None,
  motion_pass=measured_motion(d),native_success=d['official_task_success'],
  first_violation=violation,
  completed_public_phases=[e['phase'] for e in a.get('observable_skill_events',[]) if e.get('event')=='observable_skill_completed'],
  source=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
 out=(ROOT/args.out).resolve()
 if not out.is_relative_to(BASE) or out.exists():raise ValueError('fresh snapshot required')
 cohorts={}
 for name,n in EXPECTED.items():
  rows=[summarize(p) for p in sorted((BASE/name).rglob('case.json'))]
  cohorts[name]=dict(expected=n,completed=len(rows),pending=n-len(rows),runtime_clean=sum(r['runtime_clean'] for r in rows),
   motion_pass=sum(r['motion_pass'] is True for r in rows),native_success=sum(r['native_success'] for r in rows),cases=rows)
 owner=ROOT/'runtime/framework/native_resource_owner_v401';s=json.loads((owner/'status.json').read_text());l=json.loads((owner/'launch.json').read_text())
 atomic_json(out/'report.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),cohorts=cohorts,
  live_owner=dict(pid=l['pid'],start_ticks=l['start_ticks'],active_jobs=s['active_jobs'],completed=len(s['completed'])),
  formal_VAL_TEST_started=False,TEST_read=False,methods=['cv_geometry','context_geometry','physmani'],source_sha256=file_sha(__file__)))
 print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in cohorts.items()}))
if __name__=='__main__':main()
