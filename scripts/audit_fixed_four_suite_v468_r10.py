"""Exact full16 fixed-four grid; runtime/motion readiness is separate from success."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha,digest
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.three_method_release16 import policy_fingerprint,measured_motion
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
BASE=ROOT/'artifacts/framework/multitask_r10'


def planned_cases():
 roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text());values=validate_roster(roster,'train')
 fixed={t:[r for r in values if r['task']==t][:4] for t in TASKS}
 core=json.loads((BASE/'core10_fixed_four_v446_setup/plan.json').read_text())
 reuse={(r['task'],r['raw'],r['method']):ROOT/r['case']['path'] for r in core['reused_cases']}
 first_baseline={}
 for setup in ['physmani_windows_v194_setup','block_physmani_v348_setup','baseline_windows_v437_setup']:
  for r in json.loads((BASE/setup/'CPU_acceptance.json').read_text())['actual_argv_preflights']:
   first_baseline[r['task']]=ROOT/r['argv'][r['argv'].index('--out')+1]/'case.json'
 expected=[]
 for task in TASKS:
  for index,row in enumerate(fixed[task]):
   raw=row['raw_episode_id']
   for method in METHODS:
    if method=='physmani':
     path=first_baseline[task] if index==0 else BASE/'baseline_fixed_four_v452'/'development_demo_only'/task/raw/method/'case.json'
    elif (task,raw,method) in reuse:path=reuse[(task,raw,method)]
    else:
     if task in BLOCKS:label='block_next_sample_v459'
     elif task.startswith('place_cups_on_rotating_frame'):label='cup_next_sample_v462' if index==0 else 'cup_fixed_four_v469'
     elif task.startswith('remove_cups_from_rotating_frame'):label='remove_uncertain_translation_v445' if index==0 else 'remove_fixed_four_v466'
     elif task=='moving_basketball_in_hoop_high_speed' and raw=='episode1' and method=='cv_geometry':label='core_input_edges_v465'
     else:label='core10_fixed_four_v446'
     path=BASE/label/'development_demo_only'/task/raw/method/'case.json'
    expected.append((task,raw,method,row,path))
 if len(expected)!=192 or len(set((t,r,m) for t,r,m,_,_ in expected))!=192:raise ValueError('exact full16x4x3 grid')
 return expected


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=(ROOT/args.out).resolve()
 if out.exists() or not out.is_relative_to(BASE):raise ValueError('fresh audit namespace required')
 pointer=BASE/'executor_suite_routes_v467/routes.json';routes=json.loads(pointer.read_text());policy=policy_fingerprint(routes)
 for route in routes['routes']:
  for key in ('configuration','frozen_inputs','worker','consumer'):
   p=route[key]
   if file_sha(ROOT/p['path'])!=p['sha256']:raise ValueError('candidate policy pointer changed')
 route_map={(r['task'],r['method']):r for r in routes['routes']};rows=[]
 for task,raw,method,source,path in planned_cases():
  status='pending';case=None;motion=None;native=None;error=None
  if path.exists():
   case=json.loads(path.read_text());identity=case['identity'];expected=route_map[(task,method)]
   if (case['task']!=task or case['method']!=method or case['split']!='train'
    or identity.get('source_row_sha256')!=digest(source)
    or identity.get('config_sha256')!=expected['configuration']['sha256']
    or case.get('TEST_used_for_fitting') is not False or case.get('formal_test_used')):
    raise ValueError('case differs from fixed original row or exact frozen current configuration: '+str(path))
   clean=case.get('complete') is True and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
   motion=measured_motion(case);native=case['official_task_success']
   if type(motion) is not bool or type(native) is not bool:raise ValueError('explicit measured window and native result required')
   status='completed' if clean else 'infrastructure_failure'
  elif (path.parent/'infrastructure_error.json').exists():status='infrastructure_failure'
  rows.append(dict(task=task,raw=raw,method=method,status=status,motion_pass=motion,native_success=native,
   case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path) if path.exists() else None)))
 groups=[]
 for task in TASKS:
  for method in METHODS:
   selected=[r for r in rows if r['task']==task and r['method']==method]
   complete=[r for r in selected if r['status']=='completed'];motion=sum(r['motion_pass'] is True for r in complete)
   groups.append(dict(task=task,method=method,denominator=4,completed=len(complete),motion_pass=motion,
    native_success=sum(r['native_success'] is True for r in complete),
    runtime_motion_ready=len(complete)==4 and (motion==4 or method=='physmani'),
    native_success_is_not_a_release_threshold=True))
 ours=[r for r in groups if r['method']!='physmani'];baseline=[r for r in groups if r['method']=='physmani']
 fully_ready=all(r['runtime_motion_ready'] for r in groups)
 training=BASE/'joint16_training_acceptance_v438/report.json'
 report=dict(schema='r10_exact16_fixed_four_three_method_readiness_audit_v468',created_utc=datetime.now(timezone.utc).isoformat(),
  routes=dict(path=str(pointer.relative_to(ROOT)),sha256=file_sha(pointer)),policy_sha256=policy,
  training_acceptance=dict(path=str(training.relative_to(ROOT)),sha256=file_sha(training)),
  methods=list(METHODS),tasks=list(TASKS),source_count=64,method_case_count=192,
  completed=sum(r['status']=='completed' for r in rows),groups=groups,cases=rows,
  ours_task_ready_count=sum(all(r['runtime_motion_ready'] for r in ours if r['task']==t) for t in TASKS),
  ours_method_ready_count=sum(r['runtime_motion_ready'] for r in ours),
  baseline_method_complete_count=sum(r['runtime_motion_ready'] for r in baseline),
  all16_fixed_four_runtime_motion_ready=fully_ready,formal_release_granted=False,
  native_failures_preserved=True,baseline_motion_violations_reported_without_algorithm_changes=True,
  TEST_read=False,formal_VAL_TEST_started=False,source_sha256=file_sha(__file__))
 atomic_json(out/'report.json',report)
 print(json.dumps({k:report[k] for k in ('completed','method_case_count','ours_task_ready_count','ours_method_ready_count',
  'baseline_method_complete_count','all16_fixed_four_runtime_motion_ready')}))
if __name__=='__main__':main()
