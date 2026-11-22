"""Recover only the pre-native topology failure, preserving its original receipt."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
 base=ROOT/'artifacts/framework/multitask_r10';old=base/'core10_fixed_four_v446_setup';setup=base/'core_input_edges_v465_setup'
 if setup.exists():raise FileExistsError('preserve preparation')
 failures=list(old.glob('*CPU_failed.json'))
 if len(failures)!=1:raise ValueError('explicitly diagnosed single pre-native failure only')
 failure=json.loads(failures[0].read_text())
 if failure.get('native_started') is not False or 'canonical input topology remained unstable' not in failure['stderr']:
  raise ValueError('not the diagnosed validation topology failure')
 name=failures[0].name.removesuffix('_CPU_failed.json');planned=json.loads((old/'plan.json').read_text())['planned_native_cases']
 matches=[r for r in planned if r['job_id']==name]
 if len(matches)!=1:raise ValueError('exact old planned job required')
 request=matches[0];argv=list(request['argv']);original=argv[2]
 if original!='scripts/run_task_executor_smoke_basketball_r10.py':raise ValueError('preserve original basketball policy')
 oldout=ROOT/argv[argv.index('--out')+1]
 if oldout.exists():raise ValueError('pre-native failure unexpectedly has an output')
 owner=ROOT/'runtime/framework/native_resource_owner_v401';launch=json.loads((owner/'launch.json').read_text())
 for p in (owner/'requests').glob('*.json'):
  prev=json.loads(p.read_text())
  if prev['argv']==argv:raise ValueError('never relaunch an existing native request')
 files=json.loads((old/'frozen_inputs.json').read_text())['files']
 for n in ('our_mani/multitask/frozen_input_edges_v463.py','scripts/run_task_executor_smoke_index_stable_v464_r10.py',
  'scripts/prepare_failed_core_input_v465_r10.py','tests/test_frozen_input_edges_v463.py'):
  files[str(ROOT/n)]=file_sha(ROOT/n)
 verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
 argv[2]='scripts/run_task_executor_smoke_index_stable_v464_r10.py'
 argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
 argv[argv.index('--out')+1]=str((base/'core_input_edges_v465'/'development_demo_only'/
  argv[argv.index('--task')+1]/argv[argv.index('--raw-episode-id')+1]/argv[argv.index('--method')+1]).relative_to(ROOT))
 argv+=['--original-worker',original]
 request=dict(request,job_id=name.replace('core10_fixed_four_v446','core_input_edges_v465'),argv=argv,source_sha256=file_sha(ROOT/argv[2]))
 validate_request(request)
 atomic_json(setup/'planned_jobs.json',dict(jobs=[request],old_failure=dict(path=str(failures[0].relative_to(ROOT)),sha256=file_sha(failures[0])),
  original_task_parent_method_config_unchanged=True,automatic_retry=False,formal_VAL_TEST=False))
 result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
 if result.returncode:
  atomic_json(setup/'CPU_failed.json',dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode,native_started=False));result.check_returncode()
 proof=json.loads(result.stdout);assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
 verify_inputs(ROOT,files);atomic_json(setup/'CPU_acceptance.json',dict(complete=True,proof=proof,actual_argv=argv,formal_VAL_TEST=False))
 stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
 if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks']:raise ValueError('sole owner changed')
 atomic_json(owner/'requests'/('0050_'+request['job_id']+'.json'),request)
 atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,job_id=request['job_id'],owner_pid=launch['pid'],
  owner_start_ticks=launch['start_ticks'],automatic_retry=False))
 print(json.dumps(dict(complete=True,queued_once=True,old_failure_preserved=True)),flush=True)
if __name__=='__main__':main()
