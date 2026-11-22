"""After-exit development diagnostics; original Native/P1 judge unchanged.

All declared cases must be closed0 and byte-source verified before ANY truth
trajectory is opened. Public evidence is reported beside actual task metrics.
"""
from datetime import datetime
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_case
from our_mani.local_mpc_control_20261103.public_grasp import PublicGraspConfig,PublicLiftHoldMetric
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from our_mani.contracts import ObjectState,Pose


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointer(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def lines(path):
 with Path(path).open()as stream:
  for line in stream:
   if line.strip():yield json.loads(line)
def resolve(path):
 p=Path(path);return (p if p.is_absolute()else ROOT/p).resolve(strict=True)
def _non_test(path):
 if any('test' in p.casefold()for p in Path(path).parts):raise ValueError('No TEST cases or plans in development evaluator')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def preflight(plan_path):
 path=resolve(plan_path);_non_test(path);plan=read(path);jobs=plan.get('jobs',[])
 if not isinstance(jobs,list)or not jobs:raise ValueError('Fixed nonempty plan.jobs required')
 prepared=[];names=set();live_cache={};source_identity=None;archive_checks=0
 for job in jobs:
  case=resolve(job['path']);_non_test(case)
  name=job.get('id',job.get('case_id',case.name))
  if name in names:raise ValueError('Duplicate declared case')
  names.add(name);owner_path=case.with_name(case.name+'.owner.actual_exit.json')
  owner=read(owner_path);worker=read(case/'worker_exit.json');result=read(case/'result.json');run=read(case/'run_inputs.json')
  if (type(owner.get('actual_process_returncode'))is not int or owner['actual_process_returncode']!=0
   or owner.get('actual_wait_completed')is not True or owner.get('source_unchanged')is not True
   or owner.get('stop_reason')is not None or Path(owner.get('output','')).resolve()!=case
   or worker.get('native_closed')is not True or worker.get('exit_code')!=0
   or result.get('complete')is not True or result.get('source_unchanged')is not True
   or run.get('TEST_read')is not False):raise ValueError('All cases require actualclosed0/wait/source before any GT: '+name)
  for filename in ('initial_snapshot.json','native_ticks.jsonl'):
   if not(case/filename).is_file():raise ValueError('Missing completed trajectory file')
  sources=run.get('source_sha256',{});freeze=read(case/'execution_source_freeze.json')
  if not sources or freeze.get('files_sha256')!=sources:raise ValueError('Same actual frozen parent source map required')
  if source_identity is None:source_identity=sources
  elif sources!=source_identity:raise ValueError('One unchanged method/source graph across fixed cohort required')
  for relative,digest in sources.items():
   rel=Path(relative)
   if rel.is_absolute()or '..'in rel.parts:raise ValueError('Repository-relative source identity required')
   archived=case/'source_snapshot'/relative
   if sha(archived)!=digest:raise ValueError('Archived source changed: '+relative)
   archive_checks+=1
   if relative not in live_cache:
    if sha(ROOT/relative)!=digest:raise ValueError('Current source differs from captured freeze: '+relative)
    live_cache[relative]=digest
  config=run['config']
  if config.get('formal_evaluation')is True:raise ValueError('This helper only handles explicit development diagnostics')
  negative=any(config.get(k)is True for k in ('counterfactual_no_proxy','diagnostic_no_proxy_attachment'))
  prepared.append(dict(id=name,case=case,owner=owner,result=result,run=run,negative=negative,
   metadata=dict(owner=pointer(owner_path),worker=pointer(case/'worker_exit.json'),result=pointer(case/'result.json'),run_inputs=pointer(case/'run_inputs.json'))))
 return path,prepared,dict(complete=True,all_actual_exits0_and_nativeclosed=True,cohort_cases=len(prepared),
  source_files_per_case=len(source_identity),archived_files_verified=archive_checks,unique_live_files_verified=len(live_cache),
  common_source_map_sha256=canonical(source_identity),GT_opened_before_all_case_gate=False)


def typed_pose(value):return None if value is None else Pose(value['position'],value['rotation'])
def typed_object(value):
 if value is None:return None
 return ObjectState(value['object_id'],value['timestamp'],typed_pose(value['pose']),value['points_world'],
  value.get('linear_velocity',[0,0,0]),value.get('angular_velocity',[0,0,0]),value.get('confidence',1.),value.get('visible',True))
def typed_attachment(value):
 result=dict(value);event=result.get('evidence')
 if isinstance(event,dict):
  result['evidence']=PublicMotionEvidence(event['event'],event['timestamp'],tuple(event['sources']),typed_pose(event.get('tcp_relative_pose')),event.get('orientation_observable',False))
 return result


def summarize_public(case,result):
 metric=PublicLiftHoldMetric(PublicGraspConfig(**result['public_lift_hold']['criteria']))
 longest=0.;ever_hold=False;ever_attachment=False;first_attachment=None;raw_probe_ever=False;frames=0;last=None
 hold_ledger={}
 for packet in lines(case/'public_shadow/public_outputs.jsonl'):
  row=packet['result'];robot=PublicRobotSensors.from_mapping(packet['public_robot']);attachment=typed_attachment(row.get('attachment_estimate',{}))
  observed=dict(row,object_observation=typed_object(row.get('object_observation')))
  latest=metric.observe(observed,robot,attachment,typed_object(row.get('public_preclose_visual_baseline')))
  hold_ledger[round(robot.timestamp,8)]=latest
  longest=max(longest,latest['public_hold_seconds']);ever_hold|=latest['public_lift_hold_observed'];frames+=1
  evidence=attachment.get('evidence');confirmed=attachment.get('mode')=='ATTACHED_PUBLIC' and getattr(evidence,'event',None)=='attachment_confirmed'
  ever_attachment|=confirmed
  if confirmed and first_attachment is None:first_attachment=robot.timestamp
  raw=row.get('original_probe_estimator_diagnostic',{})
  raw_probe_ever|=raw.get('mode')=='ATTACHED_PUBLIC' and raw.get('evidence',{}).get('event')=='attachment_confirmed'
  last=row
 goal=result.get('public_goal_completion',{}).get('latest',{})
 final=result['public_lift_hold']['latest'];replay=metric.latest
 # A frozen result's public-mode and its effective attachment-mode are distinct.
 return dict(real_public_frames=frames,ever_effective_public_attachment_confirmed=ever_attachment,
  first_effective_confirmation_time=first_attachment,ever_original_probe_diagnostic_confirmed=raw_probe_ever,
  final_effective_attachment_mode=result.get('effective_attachment_final',{}).get('mode'),
  final_public_motion_mode=None if last is None else last.get('mode'),
  final_attachment_confirmed=bool(final.get('current_public_attachment_confirmed')),
  public_lift_hold_final=bool(result['public_lift_hold_observed']),public_lift_hold_ever=ever_hold,
  longest_public_lift_hold_seconds=longest,public_hold_replay_final=bool(replay and replay['public_lift_hold_observed']),
  public_goal_complete_final=bool(goal.get('public_grasp_goal_complete')),public_goal_complete_ever=bool(goal.get('ever_public_grasp_goal_complete')),
  public_metric_implementation='Original PublicLiftHoldMetric replay on every recorded public50ms frame, current visible body and typed public evidence, no GT.',
  replay_final_matches_logged=bool(replay and replay['public_lift_hold_observed']==final['public_lift_hold_observed'] and abs(replay['public_hold_seconds']-final['public_hold_seconds'])<1e-8))


def quantiles(values):
 a=np.asarray(values,float)
 if a.ndim!=1 or not len(a)or not np.isfinite(a).all()or(a<0).any():raise ValueError('Complete finite nonnegative wall samples required')
 return dict(samples=len(a),p50_seconds=float(np.quantile(a,.5)),p95_seconds=float(np.quantile(a,.95)),
  maximum_seconds=float(a.max()),over100ms_count=int((a>.1).sum()),over100ms_fraction=float((a>.1).mean()),real_time_qualified=False)


def evaluate_plan(plan_path,output):
 path,prepared,source_audit=preflight(plan_path)
 out=Path(output).resolve()
 if out.exists():raise FileExistsError('New immutable report output required')
 if any(x['case']==out or x['case']in out.parents for x in prepared):raise ValueError('Do not write inside a case')
 summaries=[];all_wall=[];positive_wall=[]
 # All closure/source verification completed above. Only now open GT.
 for item in prepared:
  case=item['case'];report,_=evaluate_case(case);m=report['metrics'];result=item['result'];config=item['run']['config']
  samples=result['full_cycle_wall_seconds_samples']
  if len(samples)!=result['cycles']:raise ValueError('Every actual cycle wall sample required')
  all_wall.extend(samples)
  if not item['negative']:positive_wall.extend(samples)
  logs=list(case.glob('public_candidate_position_v*_decisions.jsonl'))
  if len(logs)!=1:raise ValueError('Exact current public decision file required')
  cycle_count=0;tick_sum=0;no_unexpected=True
  for row in lines(logs[0]):
   e=row['executor'];cycle_count+=1;tick_sum+=e['executed_ticks']
   no_unexpected&=e['executed_ticks']==2 and e['unexpected_physics_ticks']==0 and e['tick_calls_completed']==2
  expected_ticks=int(round(config['simulation_seconds']/config['physics_dt']))
  budget=dict(config_seconds=config['simulation_seconds'],config_expected_ticks=expected_ticks,
   actual_ticks=m['recorded_controlled_ticks'],actual_cycles=cycle_count,receipt_tick_sum=tick_sum,
   every_cycle_2ticks_without_unexpected=bool(no_unexpected),full_budget_consumed=m['recorded_controlled_ticks']==expected_ticks)
  summaries.append(dict(id=item['id'],case=str(case),expected_negative_no_proxy=item['negative'],
   original_native_task_and_hold=m,actual_goalAny_and_strict_final_hold=m['strict_current_joint_condition']['final'],
   public=summarize_public(case,result),cycle_wall=quantiles(samples),budget=budget,
   owner_elapsed_wall_seconds=(datetime.fromisoformat(item['owner']['finished_utc'])-datetime.fromisoformat(item['owner']['started_utc'])).total_seconds(),
   original_benchmark_scope=report['benchmark'],source_inputs=report['source_files'],metadata=item['metadata']))
 positives=[x for x in summaries if not x['expected_negative_no_proxy']];negatives=[x for x in summaries if x['expected_negative_no_proxy']]
 aggregate=dict(positive_cases=len(positives),positive_original_goalAny_and_strict_final_hold=sum(x['actual_goalAny_and_strict_final_hold']for x in positives),
  all_cases=len(summaries),all_cases_original_goalAny_and_strict_final_hold=sum(x['actual_goalAny_and_strict_final_hold']for x in summaries),
  positive_public_final_lift_hold=sum(x['public']['public_lift_hold_final']for x in positives),positive_public_ever_lift_hold=sum(x['public']['public_lift_hold_ever']for x in positives),
  positive_public_final_goal_complete=sum(x['public']['public_goal_complete_final']for x in positives),
  negative_cases=len(negatives),negative_ever_effective_public_false_confirmation=sum(x['public']['ever_effective_public_attachment_confirmed']for x in negatives),
  negative_ever_public_lift_hold=sum(x['public']['public_lift_hold_ever']for x in negatives),negative_ever_public_goal_complete=sum(x['public']['public_goal_complete_ever']for x in negatives))
 result=dict(schema='closed_public_position_development_cohort_evaluation_v1',plan=pointer(path),source_audit=source_audit,
  cases=summaries,aggregate=aggregate,pooled_all_cycle_wall=quantiles(all_wall),pooled_positive_cycle_wall=None if not positive_wall else quantiles(positive_wall),
  metric_contract='Original Native success any and original P1 .5s/.04m/.02m currenthold kept separate from finalnativeflag/publicproof. Negative expected failures are not pooled as successful task trials.',
  GT_only_posthoc=True,policy_import_permitted=False,Native_started=False,fit_started=False,TEST_read=False,formal_qualification=False,
  evaluator=pointer(__file__),original_judge=pointer(ROOT/'our_mani/local_mpc_control_20261103/native_task_evaluation.py'))
 out.mkdir(parents=True,exist_ok=False)
 with(out/'report.json').open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(report=pointer(out/'report.json'),aggregate=aggregate,wall=result['pooled_all_cycle_wall'])))
 return result


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
 evaluate_plan(a.plan,a.output);return 0
if __name__=='__main__':raise SystemExit(main())
