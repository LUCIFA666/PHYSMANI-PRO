"""Freeze four unchanged TRAIN first sources for contact-corridor regression."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.button_corridor_executor_native import binding,SOURCES,TASKS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10'
    old,setup=base/'task_constraint_v409_setup',base/'button_corridor_v414_setup'
    if setup.exists():raise FileExistsError('preserve prior attempt')
    rows=[r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'] if r['task'] in TASKS]
    if len(rows)!=4:raise ValueError('same four original fixed first sources')
    c=json.loads((ROOT/rows[0]['argv'][rows[0]['argv'].index('--config')+1]).read_text())
    c.pop('task_geometry_executor',None);c.pop('task_constraint_executor',None)
    c['button_corridor_executor']={t:binding(ROOT,c,t) for t in TASKS}
    c.update(execution_scope='TRAIN_only_full_contact_corridor_v414',formal_comparison_ready=False)
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    for p in (*SOURCES,'scripts/run_task_executor_smoke_button_corridor_r10.py',
              'scripts/prepare_button_corridor_pilot_r10.py','tests/test_button_contact_corridor.py'):
        files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',c)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_button_corridor414_lock',formal_VAL_TEST=False))
    jobs=[]
    for row in rows:
        argv=list(row['argv']);task,raw,method=row['task'],row['raw'],row['method']
        argv[2]='scripts/run_task_executor_smoke_button_corridor_r10.py'
        argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((base/'button_corridor_v414'/'development_demo_only'/task/raw/method).relative_to(ROOT))
        request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
            job_id='button_corridor_v414_'+task+'_'+raw+'_'+method.split('_')[0],argv=argv,
            use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request);jobs.append((dict(task=task,raw=raw,method=method,argv=argv),request))
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
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,
        original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=len(accepted),frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
