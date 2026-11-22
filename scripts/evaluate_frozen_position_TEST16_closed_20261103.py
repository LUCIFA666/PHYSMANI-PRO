"""Evaluation-only generated reserved TEST16, strictly after ALL actual exits.

This does not change the original development-only evaluate_case TEST guard.
The original evaluate_states judge is called only after every case/source gate.
No model, threshold, case, or success-criterion selection is performed here.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_public_position_closed_cases_20261103 as common
from scripts.evaluate_public_position_closed_cases_with_alignment_20261103 import alignment_from_records
from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_states

SCHEMA='V64_reserved_TEST16_evaluation_only_execution_plan_v1'
PARTITION=ROOT/'config/framework/P4_new_heldout_TEST_partition_v59_20261103_v1.json'
PARTITION_SHA='c917d28f7889258b8961a3e462bf511dd9819d1ee01df9f63951998b15c59234'
PROTOCOL=ROOT/'artifacts/framework/local_mpc_20261102/V64_reserved_TEST16_evaluation_only_protocol_v1/protocol.json'
PROTOCOL_SHA='17ed236d7da53c7b04a552e237e452d0415fd0a7adfd0d3f5ede6bc9a9962136'
IDENTITY_FIELDS=('id','parent_id','seed','scene','canonical_motion_sha256')


def checked_pointer(value):
 if not isinstance(value,dict)or set(value)!={'path','sha256'}:raise ValueError('Exact file pointer required')
 path=common.resolve(value['path'])
 if common.sha(path)!=value['sha256']:raise ValueError('Evidence pointer SHA mismatch')
 return path

def validate_jobs(jobs,assigned):
 if len(jobs)!=16 or len(assigned)!=16:raise ValueError('All16 preassigned jobs required, no outcomes-based subset')
 for index,(job,original)in enumerate(zip(jobs,assigned)):
  if any(job.get(k)!=original.get(k)for k in IDENTITY_FIELDS):raise ValueError('Reserved parent/seed/scene/motion/order changed')
  if type(job.get('seed'))is not int or job['seed']!=261103500+index:raise ValueError('Exact numeric seed order500..515')
 if len({j['path']for j in jobs})!=16:raise ValueError('Sixteen distinct actual output directories required')


def preflight(plan_path):
 path=common.resolve(plan_path);plan=common.read(path)
 if (plan.get('schema')!=SCHEMA or plan.get('evaluation_only')is not True
  or plan.get('raw_model_qualified')is not False or plan.get('formal_pose_real_time_force_qualified')is not False
  or plan.get('all16_regardless_of_task_outcomes')is not True
  or plan.get('public_hold_canary_gate',{}).get('passed')is not False
  or plan.get('public_hold_canary_gate',{}).get('status')!='FAIL_preserved'):
  raise ValueError('Explicit evaluation-only plan preserving failed public gate required')
 partition=checked_pointer(plan['partition']);protocol=checked_pointer(plan['protocol'])
 if partition!=PARTITION.resolve()or common.sha(partition)!=PARTITION_SHA or protocol!=PROTOCOL.resolve()or common.sha(protocol)!=PROTOCOL_SHA:
  raise ValueError('Only the metadata-preassigned unused reserved partition and fixed protocol')
 fixed=common.read(partition);method=common.read(protocol);jobs=plan.get('jobs',[]);validate_jobs(jobs,fixed['jobs'])
 freeze_path=checked_pointer(plan['source_freeze']);freeze=common.read(freeze_path);sources=freeze.get('files_sha256',{})
 material_path=checked_pointer(plan['materialization_receipt']);material=common.read(material_path)
 if (material.get('schema')!='V64_reserved_TEST16_evaluation_materialization_receipt_v1'
  or material.get('complete')is not True or material.get('evaluation_only')is not True
  or material.get('public_hold_gate_passed')is not False or material.get('case_configs_generated')is not True
  or material.get('TEST_data_generated')is not False
  or any(material.get(k)!=plan[k]for k in ('partition','protocol','source_freeze'))
  or not isinstance(sources,dict)or not sources):raise ValueError('Completed same-source evaluation-only materialization receipt required')
 checked_pointer(material['authorization']);checked_pointer(material['pending_materialization'])
 configs=material.get('configs',[])
 if len(configs)!=16 or any(any(c.get(k)!=j[k]for k in IDENTITY_FIELDS)for c,j in zip(configs,jobs)):
  raise ValueError('Every original materialized configuration required in source order')
 for job,cfgrow,original in zip(jobs,configs,fixed['jobs']):
  config_path=checked_pointer(cfgrow['config'])
  if config_path!=common.resolve(job['config']):raise ValueError('Actual job configuration path differs')
  cfg=common.read(config_path)
  if (cfg.get('seed')!=original['seed'] or any(cfg.get(k)!='test'for k in ('dataset_use','data_split','evaluation_split'))
   or cfg.get('P4_collection_metadata',{}).get('split')!='test'
   or cfg.get('P4_collection_metadata',{}).get('parent_id')!=original['parent_id']
   or cfg.get('initial_target_position',[])[:2]!=original['initial_xy']
   or cfg.get('uniform_velocity')!=original['initial_velocity_m_s']
   or cfg.get('changed_velocity')!=original['changed_velocity_m_s']
   or cfg.get('speed_change_at_tick')!=original['speed_change_at_tick']):raise ValueError('Actual frozen TEST role/scenario/config assignment changed')
  if sources.get(str(config_path.relative_to(ROOT)))!=cfgrow['config']['sha256']:raise ValueError('Every materialized config must be frozen before execution')
 # All16 closure metadata first; no Native truth is opened in this phase.
 prepared=[]
 for job,cfgrow in zip(jobs,configs):
  case=common.resolve(job['path']);owner_path=case.with_name(case.name+'.owner.actual_exit.json')
  owner=common.read(owner_path);worker=common.read(case/'worker_exit.json');result=common.read(case/'result.json');run=common.read(case/'run_inputs.json')
  if (type(owner.get('actual_process_returncode'))is not int or owner['actual_process_returncode']!=0
   or owner.get('actual_wait_completed')is not True or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None
   or Path(owner.get('output','')).resolve()!=case or owner.get('case_id')!=job['id']
   or worker.get('exit_code')!=0 or worker.get('native_closed')is not True
   or result.get('complete')is not True or result.get('source_unchanged')is not True
   or run.get('source_sha256')!=sources or run.get('config')!=common.read(checked_pointer(cfgrow['config']))
   or common.sha(case/'execution_source_freeze.json')!=plan['source_freeze']['sha256']):
   raise ValueError('All16 actualclosed0/samecase/frozenconfig/source required before any TEST truth')
  prepared.append(dict(job=job,case=case,owner=owner,result=result,run=run))
 live_checked=0;archive_checked=0
 for relative,digest in sources.items():
  rel=Path(relative)
  if rel.is_absolute()or '..'in rel.parts:raise ValueError('Repository-relative source graph required')
  if common.sha(ROOT/relative)!=digest:raise ValueError('Current frozen source changed')
  live_checked+=1
  for row in prepared:
   if common.sha(row['case']/'source_snapshot'/relative)!=digest:raise ValueError('Archived frozen source changed')
   archive_checked+=1
 return path,plan,prepared,dict(all16_actualclosed0=True,all16_sourceverified=True,GT_read_before_all16_gate=False,
  live_source_files_verified=live_checked,archived_source_files_verified=archive_checked,source_map_sha256=common.canonical(sources),
  materialization=common.pointer(material_path),fixed_partition=common.pointer(partition),protocol=common.pointer(protocol))


def evaluate_plan(plan_path,output):
 path,plan,prepared,audit=preflight(plan_path)
 out=Path(output).resolve()
 if out.exists():raise FileExistsError('New output only')
 if any(row['case']==out or row['case']in out.parents for row in prepared):raise ValueError('Independent report outside original cases')
 cases=[];wall=[];align=[]
 for row in prepared:
  case=row['case'];initial=common.read(case/'initial_snapshot.json')['state'];ticks=list(common.lines(case/'native_ticks.jsonl'))
  metrics,_=evaluate_states(initial,ticks)
  logs=list(case.glob('public_candidate_position_v*_decisions.jsonl'))
  if len(logs)!=1:raise ValueError('Unique actual public decision log')
  decisions=list(common.lines(logs[0]));A=alignment_from_records(initial,ticks,decisions)
  if A['available']:align.append(A['primary_distance_m'])
  samples=row['result']['full_cycle_wall_seconds_samples']
  if len(samples)!=row['result']['cycles']:raise ValueError('Every actual fullcycle wall sample required')
  wall.extend(samples)
  case_summary=dict(id=row['job']['id'],parent_id=row['job']['parent_id'],seed=row['job']['seed'],scene=row['job']['scene'],
   original_native_task_and_hold=metrics,actual_goalAny_and_strict_final_hold=metrics['strict_current_joint_condition']['final'],
   public=common.summarize_public(case,row['result']),first_close_alignment_A=A,cycle_wall=common.quantiles(samples),
   controlled_ticks=len(ticks),cycle_receipt_ticks=sum(d['executor']['executed_ticks']for d in decisions),
   all_executor_cycles_2ticks_no_unexpected=all(d['executor']['executed_ticks']==2 and d['executor']['unexpected_physics_ticks']==0 for d in decisions))
  cases.append(case_summary)
 values=np.asarray(align,float)
 report=dict(schema='frozen_V64_generated_TEST16_evaluation_only_afterexit_v1',plan=common.pointer(path),source_gate=audit,cases=cases,
  aggregate=dict(planned=16,actualclosed=16,scored=16,original_goalAny_and_strict_final_hold=sum(x['actual_goalAny_and_strict_final_hold']for x in cases),
   public_final_lift_hold=sum(x['public']['public_lift_hold_final']for x in cases),public_ever_lift_hold=sum(x['public']['public_lift_hold_ever']for x in cases),
   public_final_goal_complete=sum(x['public']['public_goal_complete_final']for x in cases)),
  firstclose_A=dict(declared=16,available=len(values),missing=16-len(values),mean_m=None if not len(values)else float(values.mean()),p95_m=None if not len(values)else float(np.quantile(values,.95)),max_m=None if not len(values)else float(values.max()),over3cm_count=int((values>.03).sum()),over5cm_count=int((values>.05).sum())),
  pooled_fullcycle_wall=common.quantiles(wall),public_hold_canary_gate=plan['public_hold_canary_gate'],evaluation_only=True,
  raw_model_qualified=False,formal_pose_real_time_force_qualified=False,grasp_qualification_granted=False,
  original_evaluate_case_TEST_guard_unchanged=True,judge='original native_task_evaluation.evaluate_states, original Nativeflag and P1currenthold; no changed criterion',
  official_PhysMani_TEST=False,GT_labels_used_only_after_all16_exit=True,Native_started=False,fit_started=False,threshold_or_model_selected_from_TEST=False,
  evaluator=common.pointer(__file__))
 out.mkdir(parents=True,exist_ok=False)
 with(out/'report.json').open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(report=common.pointer(out/'report.json'),aggregate=report['aggregate'],firstclose_A=report['firstclose_A'])))
 return report


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
 evaluate_plan(a.plan,a.output);return 0
if __name__=='__main__':raise SystemExit(main())
