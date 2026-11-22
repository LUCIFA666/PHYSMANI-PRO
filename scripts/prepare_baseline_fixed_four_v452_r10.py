"""Complete original baseline fixed-four grid, reusing every first-source request."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
 base=ROOT/'artifacts/framework/multitask_r10';setup=base/'baseline_fixed_four_v452_setup'
 if setup.exists():raise FileExistsError('never overwrite or automatically repeat preparation')
 owner=ROOT/'runtime/framework/native_resource_owner_v401';launch=json.loads((owner/'launch.json').read_text())
 def live():
  stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
  if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks'] or json.loads((owner/'status.json').read_text())['stop_requested']:
   raise ValueError('sole current native owner unavailable')
 live();core=base/'physmani_windows_v194_setup';block=base/'block_physmani_v348_setup'
 # Reuse includes pending first-source jobs: pending means wait, never relaunch.
 reused=[]
 for old in (core,block,base/'baseline_windows_v437_setup'):
  receipt=json.loads((old/'CPU_acceptance.json').read_text())
  if receipt['complete'] is not True:raise ValueError('original actual-argv CPU acceptance required')
  for row in receipt['actual_argv_preflights']:
   if row['method']!='physmani':raise ValueError('original baseline only')
   reused.append(row)
 tasks=sorted(set(TASK_SPECS)|set(BLOCKS))
 if len(tasks)!=16 or len(reused)!=16 or len({r['task'] for r in reused})!=16:
  raise ValueError('exactly sixteen distinct first-source baseline requests must be reused')
 roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
 fixed={t:[r for r in roster if r['task']==t][:4] for t in tasks}
 if any(len(rows)!=4 for rows in fixed.values()):raise ValueError('original fixed four required')
 for row in reused:
  if row['raw']!=fixed[row['task']][0]['raw_episode_id']:raise ValueError('reuse must be original first source')
 prior=[json.loads(p.read_text()) for directory in (owner,ROOT/'runtime/framework/native_resource_owner_v19')
  for p in (directory/'requests').glob('*.json')]
 for row in reused:
  if not any(p['argv']==row['argv'] for p in prior):raise ValueError('reused request missing from owner history')
 files={};excluded={}
 for old in (base/'baseline_windows_v437_setup',block):
  for name,sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
   if Path(name).name=='updtChck.dat' and '/runtime/CoppeliaSim_Edu_' in name:
    excluded[name]=sha;continue
   if name in files and files[name]!=sha:raise ValueError('frozen input conflict: '+name)
   files[name]=sha
 for rows in fixed.values():
  for row in rows:files[str(Path(row['path']).resolve())]=file_sha(row['path'])
 files[str(Path(__file__))]=file_sha(__file__)
 verify_inputs(ROOT,files);setup.mkdir()
 atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
  excluded_mutable_update_checker_state=excluded,old_locks_unchanged=True,formal_VAL_TEST=False))
 jobs=[]
 for task in tasks:
  isblock=task in BLOCKS;config=block/'configuration.json' if isblock else core/'configuration.json'
  worker='scripts/run_task_executor_smoke_block_physmani_v347_r10.py' if isblock else 'scripts/run_task_executor_smoke_physmani_v192_r10.py'
  for row in fixed[task][1:]:
   raw=row['raw_episode_id'];out=base/'baseline_fixed_four_v452'/'development_demo_only'/task/raw/'physmani'
   argv=['bash','scripts/run_aligned_gpu.sh',worker,'--config',str(config.relative_to(ROOT)),
    '--manifest','artifacts/framework/protocol/train_manifest.json','--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
    '--task',task,'--raw-episode-id',raw,'--split','train','--method','physmani']
   if not isblock:argv+=['--replica','0']
   argv+=['--development-demo','--out',str(out.relative_to(ROOT))]
   request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
    job_id='baseline_fixed_four_v452_'+task+'_'+raw+'_physmani',argv=argv,use_gpu=True,
    timeout_seconds=1800,source_sha256=file_sha(ROOT/worker))
   validate_request(request)
   if out.exists() or any('--out' in p['argv'] and p['argv'][p['argv'].index('--out')+1]==argv[-1] for p in prior):
    raise FileExistsError('output exists or already submitted')
   jobs.append((dict(task=task,raw=raw,method='physmani',argv=argv),request))
 if len(jobs)!=48:raise ValueError('sixteen fixed-four tasks minus sixteen reused first sources')
 atomic_json(setup/'planned_jobs.json',dict(jobs=[r for _,r in jobs],reused_first_source_requests=reused,
  total_baseline_grid=64,new_cases=48,original_algorithm_weights_RML_unchanged=True,
  all_original_fixed_sources_retained=True,automatic_retry=False,formal_VAL_TEST=False))
 def check(job):
  row,request=job;r=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
  return row,request,r
 passed=[];failed=[];queued=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(check,j) for j in jobs]):
   row,request,r=future.result();name=request['job_id']
   if r.returncode:
    atomic_json(setup/(name+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,
     native_started=False,automatic_retry=False));failed.append(row);print(json.dumps(dict(CPU_failed=name)),flush=True);continue
   proof=json.loads(r.stdout)
   assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
   atomic_json(setup/(name+'_CPU_preflight.json'),proof);passed.append(row);live();validate_request(request)
   atomic_json(owner/'requests'/('0036_'+name+'.json'),request);queued.append(name)
   atomic_json(setup/(name+'_queued.json'),dict(queued_once=True,job_id=name,owner_pid=launch['pid'],
    owner_start_ticks=launch['start_ticks'],automatic_retry=False))
   print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),queued=len(queued),total=48,task=row['task'])),flush=True)
 verify_inputs(ROOT,files)
 atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
  reused_first_source_requests=reused,formal_VAL_TEST=False))
 atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,owner_pid=launch['pid'],
  owner_start_ticks=launch['start_ticks'],automatic_retry=False,formal_VAL_TEST=False))
 print(json.dumps(dict(prepared=not failed,checks=len(passed),queued=len(queued),failed=len(failed))),flush=True)
if __name__=='__main__':main()
