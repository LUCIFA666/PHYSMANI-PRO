"""Freeze unchanged high-peg first source for both predictors."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.remove_continuous_executor import binding,SOURCES,TASKS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'registered_tasks_v408_setup';setup=base/'remove_continuous_v431_setup'
    if setup.exists():raise FileExistsError('preserve earlier setup')
    rows=[r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'] if r['task'] in TASKS]
    if len(rows)!=4:raise ValueError('same four original task/method cases')
    files=json.loads((old/'frozen_inputs.json').read_text())['files'];configs={}
    for task in TASKS:
        c=json.loads((old/(task+'_configuration.json')).read_text());c.update(binding(ROOT,c,task))
        c.update(execution_scope='TRAIN_only_reference4_native6_remove_v431',formal_comparison_ready=False)
        configs[task]=c
    for p in (*SOURCES,'scripts/run_task_executor_smoke_remove_continuous_r10.py','scripts/prepare_remove_continuous_pilot_r10.py',
        'tests/test_remove_continuous_executor.py','tests/test_remove_physical_capture.py'):
        files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir()
    for task,c in configs.items():
        path=setup/(task+'_configuration.json');atomic_json(path,c);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,schema='r10_remove_continuous431_lock',formal_VAL_TEST=False))
    jobs=[]
    for row in rows:
        argv=list(row['argv']);task,raw,method=row['task'],row['raw'],row['method']
        argv[2]='scripts/run_task_executor_smoke_remove_continuous_r10.py'
        argv[argv.index('--config')+1]=str((setup/(task+'_configuration.json')).relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((base/'remove_continuous_v431'/'development_demo_only'/task/raw/method).relative_to(ROOT))
        request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id='remove_continuous_v431_'+task+'_'+raw+'_'+method.split('_')[0],
            argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request);jobs.append((dict(row,argv=argv),request))
    def check(job):
        row,request=job;result=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
        return row,request,result
    accepted=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,j) for j in jobs]):
            row,request,result=future.result()
            if result.returncode:atomic_json(setup/(request['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,native_started=False))
            result.check_returncode();proof=json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(request['job_id']+'_CPU_preflight.json'),proof);accepted.append(row)
            print(json.dumps(dict(CPU_pass=len(accepted),total=len(jobs),task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=len(accepted),frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
