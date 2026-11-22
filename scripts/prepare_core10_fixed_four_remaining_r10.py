"""Complete the fixed-four TRAIN grid for ten unchanged core task policies.

Reuse every completed case with the exact existing policy and original source,
including failures. Cups and blocks remain separate repair cohorts. This is
development qualification, never formal VAL/TEST or successful-source selection.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.executor_suite16_v447 import route,CHECKPOINT_SHA
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'core10_fixed_four_v446_setup'
    if setup.exists():raise FileExistsError('preserve previous preparation')
    tasks=sorted(t for t in TASK_SPECS if not t.startswith(('place_cups','remove_cups')))
    if len(tasks)!=10:raise ValueError('exact unchanged core10 task cohort')
    owner=ROOT/'runtime/framework/native_resource_owner_v401';launch=json.loads((owner/'launch.json').read_text())
    def live():
        stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
        if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks'] or json.loads((owner/'status.json').read_text())['stop_requested']:
            raise ValueError('sole current owner unavailable')
    live();roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    files={};existing={};pointers={}
    for task in tasks:
        r=route(task,'cv_geometry');old=base/r.setup;config=base/r.configuration(task)
        c=json.loads(config.read_text())
        if c['context_checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('same trained joint16 dynamics required')
        pointers[task]=dict(configuration=str(config.relative_to(ROOT)),configuration_sha256=file_sha(config),worker='scripts/'+r.worker+'.py')
        for name,sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
            if name in files and files[name]!=sha:raise ValueError('immutable policy dependencies disagree: '+name)
            files[name]=sha
        for row in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']:
            if row.get('task',task)!=task:continue
            argv=row['argv']
            if argv[2]!=pointers[task]['worker']:raise ValueError('actual frozen worker differs from route')
            case=ROOT/argv[argv.index('--out')+1]/'case.json'
            if not case.exists():continue
            record=json.loads(case.read_text());identity=record['identity']
            if (identity['config_sha256']!=pointers[task]['configuration_sha256'] or record.get('complete') is not True
                or record.get('cleanup_completed') is not True or record.get('infrastructure_error') is not None):
                raise ValueError('prior fixed case cannot be silently replaced or promoted')
            raw=record.get('source_raw_episode_id',record.get('parent_raw_episode_id'))
            if raw!=row['raw'] or record['method']!=row['method']:raise ValueError('prior original source/method differs')
            key=(task,raw,row['method']);existing[key]=dict(task=task,raw=raw,method=row['method'],
                case=dict(path=str(case.relative_to(ROOT)),sha256=file_sha(case)),
                native_success=record['official_task_success'],reused_without_retry=True)
    excluded={name:sha for name,sha in files.items() if Path(name).name=='updtChck.dat' and '/runtime/CoppeliaSim_Edu_' in name}
    for name in excluded:del files[name]
    for path in (Path(__file__),ROOT/'our_mani/multitask/executor_suite16_v447.py'):
        files[str(path)]=file_sha(path)
    verify_inputs(ROOT,files);setup.mkdir()
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        excluded_mutable_update_checker_state=excluded,original_policy_configurations_unchanged=True))
    old_requests=[json.loads(p.read_text()) for d in (owner,ROOT/'runtime/framework/native_resource_owner_v19') for p in (d/'requests').glob('*.json')]
    reused=[];jobs=[]
    for task in tasks:
        first=[r for r in roster if r['task']==task][:4]
        if len(first)!=4:raise ValueError('fixed four original TRAIN sources required')
        for source in first:
            raw=source['raw_episode_id']
            for method in ('cv_geometry','context_geometry'):
                key=(task,raw,method)
                if key in existing:reused.append(existing[key]);continue
                policy=pointers[task];out=base/'core10_fixed_four_v446'/'development_demo_only'/task/raw/method
                argv=['bash','scripts/run_local_sim.sh',policy['worker'],'--config',policy['configuration'],
                    '--manifest','artifacts/framework/protocol/train_manifest.json',
                    '--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
                    '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,
                    '--replica','0','--development-demo','--out',str(out.relative_to(ROOT))]
                request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
                    job_id='core10_fixed_four_v446_'+task+'_'+raw+'_'+method.split('_')[0],argv=argv,use_gpu=False,
                    timeout_seconds=1800,source_sha256=file_sha(ROOT/policy['worker']))
                validate_request(request)
                if out.exists() or any('--out' in r['argv'] and r['argv'][r['argv'].index('--out')+1]==argv[-1] for r in old_requests):
                    raise FileExistsError('new case output already used or submitted')
                jobs.append((dict(task=task,raw=raw,method=method,argv=argv),request))
    if len(reused)+len(jobs)!=80:raise ValueError('exact80-case fixed-four two-method core10 grid required')
    atomic_json(setup/'plan.json',dict(tasks=tasks,methods=['cv_geometry','context_geometry'],expected_cases=80,
        reused_cases=reused,planned_native_cases=[r for _,r in jobs],source_policy_pointers=pointers,
        all_failures_retained=True,source_outcome_filtering=False,formal_VAL_TEST=False,
        full16_qualification_not_claimed=True,automatic_retry=False))
    print(json.dumps(dict(reused=len(reused),new_cases=len(jobs),expected=80)),flush=True)
    def preflight(job):
        row,request=job;r=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
        return row,request,r
    passed=[];failed=[];queued=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(preflight,j) for j in jobs]):
            row,request,r=future.result();name=request['job_id']
            if r.returncode:
                atomic_json(setup/(name+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,
                    native_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=name)),flush=True);continue
            proof=json.loads(r.stdout);assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(name+'_CPU_preflight.json'),proof);passed.append(row);live();validate_request(request)
            atomic_json(owner/'requests'/('0050_'+name+'.json'),request);queued.append(name)
            atomic_json(setup/(name+'_queued.json'),dict(queued_once=True,job_id=name,owner_pid=launch['pid'],
                owner_start_ticks=launch['start_ticks'],automatic_retry=False))
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),queued=len(queued),total=len(jobs),task=row['task'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        original_fixed_sources=True,all_failures_retained=True,formal_VAL_TEST=False))
    atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,owner_pid=launch['pid'],
        owner_start_ticks=launch['start_ticks'],automatic_retry=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=not failed,queued=len(queued),reused=len(reused),failed=len(failed))),flush=True)


if __name__=='__main__':main()
