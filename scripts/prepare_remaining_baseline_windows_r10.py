"""Fill thirteen missing baseline window traces on their original first TRAIN rows.

The ordinary bin and two blocks already have complete measured-window records.
No policy, model, RML parameter or source selection changes in this fill.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'physmani_windows_v194_setup';setup=base/'baseline_windows_v437_setup'
    if setup.exists():raise FileExistsError('never overwrite prior preparation')
    owner=ROOT/'runtime/framework/native_resource_owner_v401'
    launch=json.loads((owner/'launch.json').read_text())
    def live():
        stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
        if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks'] or json.loads((owner/'status.json').read_text())['stop_requested']:
            raise ValueError('current sole resource owner unavailable')
    live()
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    # CoppeliaSim writes this update-check timestamp during ordinary launches.
    # It is application state, not a physics binary, script, policy or model.
    # Keep the old lock intact and disclose the single mutable-file exclusion.
    excluded={name:sha for name,sha in files.items() if Path(name).name=='updtChck.dat'
        and '/runtime/CoppeliaSim_Edu_' in name}
    for name in excluded:del files[name]
    files[str(Path(__file__))]=file_sha(__file__)
    verify_inputs(ROOT,files);setup.mkdir()
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_baseline_remaining13_original_policy_measured_windows437_lock',formal_VAL_TEST=False,
        excluded_mutable_update_checker_state=excluded,old_input_lock_unchanged=True))
    roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    first={t:next(r['raw_episode_id'] for r in roster if r['task']==t) for t in TASK_SPECS}
    tasks=sorted(set(TASK_SPECS)-{'put_rubbish_in_moving_bin'})
    if len(tasks)!=13:raise ValueError('fourteen original tasks minus measured ordinary bin')
    prior=[json.loads(p.read_text()) for directory in (owner,ROOT/'runtime/framework/native_resource_owner_v19')
        for p in (directory/'requests').glob('*.json')]
    jobs=[]
    for task in tasks:
        raw=first[task];out=base/'baseline_windows_v437'/'development_demo_only'/task/raw/'physmani'
        argv=['bash','scripts/run_aligned_gpu.sh','scripts/run_task_executor_smoke_physmani_v192_r10.py',
            '--config',str((old/'configuration.json').relative_to(ROOT)),
            '--manifest','artifacts/framework/protocol/train_manifest.json',
            '--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
            '--task',task,'--raw-episode-id',raw,'--split','train','--method','physmani',
            '--replica','0','--development-demo','--out',str(out.relative_to(ROOT))]
        request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
            job_id='baseline_windows_v437_'+task+'_'+raw+'_physmani',argv=argv,use_gpu=True,
            timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request)
        if out.exists() or any('--out' in p['argv'] and p['argv'][p['argv'].index('--out')+1]==argv[-1] for p in prior):
            raise FileExistsError('output exists or is already submitted')
        jobs.append((dict(task=task,raw=raw,method='physmani',argv=argv),request))
    atomic_json(setup/'planned_jobs.json',dict(jobs=[r for _,r in jobs],original_fixed_first_sources=True,
        baseline_algorithm_unchanged=True,automatic_retry=False,formal_VAL_TEST=False))
    def preflight(job):
        row,request=job;r=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
        return row,request,r
    passed=[];failed=[];queued=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(preflight,j) for j in jobs]):
            row,request,r=future.result();name=request['job_id']
            if r.returncode:
                atomic_json(setup/(name+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,
                    returncode=r.returncode,native_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=name)),flush=True);continue
            proof=json.loads(r.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(name+'_CPU_preflight.json'),proof);passed.append(row)
            live();validate_request(request)
            # Each native job is committed exactly once, after its actual argv check.
            atomic_json(owner/'requests'/('0035_'+name+'.json'),request);queued.append(name)
            atomic_json(setup/(name+'_queued.json'),dict(queued_once=True,job_id=name,owner_pid=launch['pid'],
                owner_start_ticks=launch['start_ticks'],automatic_retry=False))
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),queued=len(queued),total=13,task=row['task'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,
        failed_preflights=failed,native_started=bool(queued),formal_VAL_TEST=False))
    atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,
        owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],automatic_retry=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=not failed,queued=len(queued),failed=len(failed))),flush=True)


if __name__=='__main__':main()
