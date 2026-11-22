"""Freeze original buzz sources after missing command refresh was diagnosed."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.buzz_command_executor_native import SOURCES
from our_mani.multitask.command_reference_service import binding,TASK_PHASES
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10'
    old,setup=base/'registered_tasks_v408_setup',base/'buzz_command_v412_setup'
    if setup.exists():raise FileExistsError('preserve old CPU/native attempt')
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    jobs=[r for r in json.loads((old/'planned_jobs.json').read_text())['jobs'] if
        r['argv'][r['argv'].index('--task')+1] in TASK_PHASES]
    if len(jobs)!=4:raise ValueError('same fixed first four task/method cases')
    for r in jobs:
        proof=json.loads((old/(r['job_id']+'_CPU_preflight.json')).read_text())
        assert proof['complete'] and not proof['native_started']
    for p in (*SOURCES,'scripts/run_task_executor_smoke_buzz_command_r10.py',
        'scripts/prepare_buzz_command_pilot_r10.py','tests/test_command_reference_service.py'):
        files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir()
    for task in TASK_PHASES:
        c=json.loads((old/(task+'_configuration.json')).read_text())
        c['command_reference_service']=binding(ROOT,c,task)
        c.update(execution_scope='TRAIN_only_buzz_moving_command_reference_v412',formal_comparison_ready=False)
        path=setup/(task+'_configuration.json');atomic_json(path,c);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_buzz_command412_lock',formal_VAL_TEST=False))
    checks=[]
    for old_job in jobs:
        argv=list(old_job['argv']);task=argv[argv.index('--task')+1];raw=argv[argv.index('--raw-episode-id')+1];method=argv[argv.index('--method')+1]
        argv[2]='scripts/run_task_executor_smoke_buzz_command_r10.py'
        argv[argv.index('--config')+1]=str((setup/(task+'_configuration.json')).relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((base/'buzz_command_v412'/'development_demo_only'/task/raw/method).relative_to(ROOT))
        request=dict(old_job,argv=argv,job_id='buzz_command_v412_'+task+'_'+raw+'_'+method.split('_')[0],
            source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request);checks.append((dict(task=task,raw=raw,method=method,argv=argv),request))
    def check(job):
        row,request=job
        result=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
        return row,request,result
    accepted=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,j) for j in checks]):
            row,request,result=future.result()
            if result.returncode:
                atomic_json(setup/(request['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,native_started=False))
            result.check_returncode();proof=json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(request['job_id']+'_CPU_preflight.json'),proof);accepted.append(row)
            print(json.dumps(dict(CPU_pass=len(accepted),total=len(checks),task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,
        original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=len(accepted),frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
