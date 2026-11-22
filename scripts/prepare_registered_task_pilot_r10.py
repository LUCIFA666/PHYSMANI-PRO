"""Freeze eleven fixed TRAIN adapters, preflight two at a time and queue once."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.task_executor_registry import ADAPTERS, binding
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from scripts.run_native_resource_owner_v19_r10 import validate_request
BASE = ROOT/'artifacts/framework/multitask_r10'
OWNER = ROOT/'runtime/framework/native_resource_owner_v401'


def owner_identity():
    launch = json.loads((OWNER/'launch.json').read_text())
    stat = Path('/proc/'+str(launch['pid'])+'/stat').read_text()
    if int(stat.rsplit(')',1)[1].split()[19]) != launch['start_ticks']:
        raise ValueError('resource owner PID identity changed')
    if json.loads((OWNER/'status.json').read_text())['stop_requested']:
        raise ValueError('resource owner stopped')
    return launch


def main():
    setup = BASE/'registered_tasks_v408_setup'
    if setup.exists():
        raise FileExistsError('do not restart or overwrite a previous preparation')
    launch = owner_identity()
    files, configs = {}, {}
    first = {r['task']:r['fixed_smoke_source'] for r in
        json.loads((BASE/'all16_executor_joint_train_v1/plan.json').read_text())['tasks']}
    for task, route in ADAPTERS.items():
        old = BASE/route.setup
        for p,sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
            if p in files and files[p] != sha:
                raise ValueError('inherited immutable inputs disagree: '+p)
            files[p] = sha
        c = json.loads((old/'configuration.json').read_text())
        if task.startswith('moving_basketball'):
            library = InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
            for predictor in ('cv','context'):
                identity = persistent_reference_continuation_identity(ROOT,c,task,predictor,library,
                    CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
                c['continuation_identities'].setdefault(task,{})[predictor] = identity['sha256']
        c['shared_task_executor'] = binding(ROOT,c,task)
        c.update(execution_scope='TRAIN_only_registered_continuous_executor_v408',formal_comparison_ready=False)
        if c['context_checkpoint']['sha256'] != '6447ac755032b022d18884936e02a90d8ec2f2b219b5d42fc140cdd1618cd017':
            raise ValueError('all tasks must use the same trained joint16 dynamics')
        configs[task] = c
        for p,sha in c['shared_task_executor']['sources_sha256'].items():
            files[str(ROOT/p)] = sha
    for p in ('scripts/run_task_executor_smoke_registered_r10.py',
              'scripts/prepare_registered_task_pilot_r10.py',
              'tests/test_task_executor_registry.py','tests/test_reference_constraint_authority.py'):
        files[str(ROOT/p)] = file_sha(ROOT/p)
    verify_inputs(ROOT,files)
    setup.mkdir()
    for task,c in configs.items():
        path = setup/(task+'_configuration.json')
        atomic_json(path,c); files[str(path)] = file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_registered408_lock',formal_VAL_TEST=False))
    jobs = []
    for task in ADAPTERS:
        raw = first[task]
        for method in ('cv_geometry','context_geometry'):
            argv = ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_registered_r10.py',
                '--config',str((setup/(task+'_configuration.json')).relative_to(ROOT)),
                '--manifest','artifacts/framework/protocol/train_manifest.json',
                '--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
                '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,
                '--replica','0','--development-demo','--out',str((BASE/'registered_tasks_v408'/
                    'development_demo_only'/task/raw/method).relative_to(ROOT))]
            request = dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
                job_id='registered_tasks_v408_'+task+'_'+raw+'_'+method.split('_')[0],argv=argv,
                use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
            validate_request(request)
            if (ROOT/argv[-1]).exists():
                raise FileExistsError('fresh output required')
            jobs.append((dict(task=task,raw=raw,method=method,argv=argv),request))
    atomic_json(setup/'planned_jobs.json',dict(jobs=[r for _,r in jobs],
        original_fixed_TRAIN_sources=True,automatic_retry=False,formal_VAL_TEST=False))
    def preflight(job):
        row,request = job
        result = subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
        return row,request,result
    passed, failed, queued = [], [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(preflight,job) for job in jobs]
        for future in as_completed(futures):
            row,request,result = future.result(); job = request['job_id']
            if result.returncode:
                error = dict(**row,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    native_started=False,automatic_retry=False)
                atomic_json(setup/(job+'_CPU_failed.json'),error); failed.append(error)
                print(json.dumps(dict(CPU_failed=job,native_started=False)),flush=True)
                continue
            proof = json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(job+'_CPU_preflight.json'),proof); passed.append(row)
            current = owner_identity()
            if (current['pid'],current['start_ticks']) != (launch['pid'],launch['start_ticks']):
                raise ValueError('owner changed during CPU preparation')
            validate_request(request)
            for old in (ROOT/'runtime/framework/native_resource_owner_v19',OWNER):
                for p in (old/'requests').glob('*.json'):
                    prior = json.loads(p.read_text()); argv = prior['argv']
                    if '--out' in argv and argv[argv.index('--out')+1] == row['argv'][-1]:
                        raise ValueError('output already submitted')
            atomic_json(OWNER/'requests'/('0040_'+job+'.json'),request)
            queued.append(job)
            atomic_json(setup/(job+'_queued.json'),dict(queued_once=True,job_id=job,
                owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],automatic_retry=False))
            print(json.dumps(dict(CPU_pass=len(passed),CPU_failed=len(failed),queued=job,total=len(jobs))),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,
        failed_preflights=failed,original_fixed_TRAIN_sources=True,formal_VAL_TEST=False))
    atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,
        owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],automatic_retry=False,
        CPU_failed=len(failed),formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,passed=len(passed),failed=len(failed),queued=len(queued),
        frozen_inputs=len(files))),flush=True)


if __name__ == '__main__':
    main()
