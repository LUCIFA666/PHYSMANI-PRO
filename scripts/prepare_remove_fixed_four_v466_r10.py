"""Expand unchanged445 cup-removal policy to all fixed four original parents."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.three_method_release16 import measured_motion
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
 base=ROOT/'artifacts/framework/multitask_r10';old=base/'remove_uncertain_translation_v445_setup';setup=base/'remove_fixed_four_v466_setup'
 if setup.exists():raise FileExistsError('preserve preparation')
 receipt=json.loads((old/'CPU_acceptance.json').read_text());rows=receipt['actual_argv_preflights']
 tasks=['remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'];methods=['cv_geometry','context_geometry']
 if receipt['complete'] is not True or {(r['task'],r['method']) for r in rows}!={(t,m) for t in tasks for m in methods} or len(rows)!=4:
  raise ValueError('exact original first-source445 CPU proof required')
 roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
 fixed={t:[r for r in roster if r['task']==t][:4] for t in tasks};reused=[]
 for row in rows:
  if row['raw']!=fixed[row['task']][0]['raw_episode_id']:raise ValueError('original first source only')
  argv=row['argv'];p=ROOT/argv[argv.index('--out')+1]/'case.json';d=json.loads(p.read_text())
  if d.get('complete') is not True or d.get('cleanup_completed') is not True or d.get('infrastructure_error') is not None or measured_motion(d) is not True:
   raise ValueError('first-source policy still needs runtime/motion repair')
  config=ROOT/argv[argv.index('--config')+1]
  if d['identity']['config_sha256']!=file_sha(config):raise ValueError('case did not run the same exact configuration')
  reused.append(dict(task=row['task'],raw=row['raw'],method=row['method'],path=str(p.relative_to(ROOT)),sha256=file_sha(p),
   native_success=d['official_task_success'],outcome_filtering=False))
 owner=ROOT/'runtime/framework/native_resource_owner_v401';launch=json.loads((owner/'launch.json').read_text())
 def live():
  stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
  if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks'] or json.loads((owner/'status.json').read_text())['stop_requested']:
   raise ValueError('sole owner unavailable')
 live();files=json.loads((old/'frozen_inputs.json').read_text())['files'];files[str(Path(__file__))]=file_sha(__file__)
 for original_rows in fixed.values():
  for row in original_rows:files[str(Path(row['path']).resolve())]=file_sha(row['path'])
 verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
  original_config_policy_models_unchanged=True,formal_VAL_TEST=False))
 prior=[json.loads(p.read_text()) for directory in (owner,ROOT/'runtime/framework/native_resource_owner_v19') for p in (directory/'requests').glob('*.json')]
 jobs=[]
 for row in rows:
  for source in fixed[row['task']][1:]:
   task,raw,method=row['task'],source['raw_episode_id'],row['method'];argv=list(row['argv'])
   argv[argv.index('--raw-episode-id')+1]=raw;argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
   out=base/'remove_fixed_four_v466'/'development_demo_only'/task/raw/method;argv[argv.index('--out')+1]=str(out.relative_to(ROOT))
   if out.exists() or any('--out' in p['argv'] and p['argv'][p['argv'].index('--out')+1]==argv[-1] for p in prior):raise FileExistsError('existing output/request')
   request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id='remove_fixed_four_v466_'+task+'_'+raw+'_'+method.split('_')[0],
    argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
   validate_request(request);jobs.append((dict(task=task,raw=raw,method=method,argv=argv),request))
 if len(jobs)!=12:raise ValueError('exactly twelve missing fixed-source method cases')
 atomic_json(setup/'plan.json',dict(expected_cases=16,new_cases=12,reused_cases=reused,planned_native_cases=[r for _,r in jobs],
  all_failures_retained=True,original_policy_unchanged=True,formal_VAL_TEST=False))
 def check(job):
  row,request=job;r=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True);return row,request,r
 passed=[];failed=[];queued=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(check,j) for j in jobs]):
   row,request,r=future.result();name=request['job_id']
   if r.returncode:
    atomic_json(setup/(name+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,native_started=False,automatic_retry=False));failed.append(row)
    print(json.dumps(dict(CPU_failed=name)),flush=True);continue
   proof=json.loads(r.stdout);assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
   atomic_json(setup/(name+'_CPU_preflight.json'),proof);passed.append(row);live();validate_request(request)
   atomic_json(owner/'requests'/('0043_'+name+'.json'),request);queued.append(name)
   atomic_json(setup/(name+'_queued.json'),dict(queued_once=True,owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],automatic_retry=False))
   print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),queued=len(queued),total=12)),flush=True)
 verify_inputs(ROOT,files)
 atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,formal_VAL_TEST=False))
 atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],automatic_retry=False))
 print(json.dumps(dict(complete=not failed,queued=len(queued),reused=4,failed=len(failed))),flush=True)
if __name__=='__main__':main()
