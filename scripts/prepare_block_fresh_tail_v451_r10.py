"""Both block tasks, all fixed four original TRAIN sources, with fresh safety tails."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.block_fresh_tail_v448 import binding,SOURCES
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'joint16_block_support_v318_setup';setup=base/'block_fresh_tail_v451_setup'
    if setup.exists():raise FileExistsError('preserve prior setup')
    owner=ROOT/'runtime/framework/native_resource_owner_v401';launch=json.loads((owner/'launch.json').read_text())
    def live():
        stat=Path('/proc/'+str(launch['pid'])+'/stat').read_text()
        if int(stat.rsplit(')',1)[1].split()[19])!=launch['start_ticks'] or json.loads((owner/'status.json').read_text())['stop_requested']:
            raise ValueError('sole current resource owner unavailable')
    live();c=json.loads((old/'configuration.json').read_text());c['block_fresh_tail_v448']=binding(ROOT)
    c.update(execution_scope='TRAIN_only_fixed_four_fresh_counted_safety_tail_v451',formal_comparison_ready=False)
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    rows=[r for task in sorted(BLOCKS) for r in [x for x in roster if x['task']==task][:4]]
    if len(rows)!=8:raise ValueError('two tasks each fixed four original TRAIN parents')
    for row in rows:files[str(Path(row['path']).resolve())]=file_sha(row['path'])
    for p in (*SOURCES,'our_mani/multitask/joint16_block_fresh_tail_episode_v449.py',
        'scripts/run_task_executor_smoke_block_fresh_tail_v450_r10.py','scripts/prepare_block_fresh_tail_v451_r10.py',
        'tests/test_block_fresh_tail_v448.py'):
        files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',c)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    jobs=[]
    # The previously failed fixed source is checked first. Every other fixed
    # source stays in this same cohort; no success filtering or source swap.
    rows.sort(key=lambda r:(r['raw_episode_id']!='episode105',r['task'],r['raw_episode_id']))
    for row in rows:
        task,raw=row['task'],row['raw_episode_id']
        for method in ('context_geometry','cv_geometry'):
            out=base/'block_fresh_tail_v451'/'development_demo_only'/task/raw/method
            argv=['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_block_fresh_tail_v450_r10.py',
                '--config',str((setup/'configuration.json').relative_to(ROOT)),
                '--manifest','artifacts/framework/protocol/train_manifest.json',
                '--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
                '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,
                '--development-demo','--out',str(out.relative_to(ROOT))]
            request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id='block_fresh_tail_v451_'+task+'_'+raw+'_'+method.split('_')[0],
                argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
            validate_request(request)
            if out.exists():raise FileExistsError('preserve prior case')
            jobs.append((dict(task=task,raw=raw,method=method,argv=argv),request))
    atomic_json(setup/'planned_jobs.json',dict(jobs=[r for _,r in jobs],expected_cases=16,all_original_fixed_sources_retained=True,
        diagnostic_failed_source_not_replaced=True,automatic_retry=False,formal_VAL_TEST=False))
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
            proof=json.loads(r.stdout);assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(name+'_CPU_preflight.json'),proof);passed.append(row);live();validate_request(request)
            atomic_json(owner/'requests'/('0032_'+name+'.json'),request);queued.append(name)
            atomic_json(setup/(name+'_queued.json'),dict(queued_once=True,job_id=name,owner_pid=launch['pid'],
                owner_start_ticks=launch['start_ticks'],automatic_retry=False))
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),queued=len(queued),total=16,raw=row['raw'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        all_original_fixed_sources_retained=True,formal_VAL_TEST=False))
    atomic_json(setup/'queued_current_owner.json',dict(queued_once=True,request_jobs=queued,owner_pid=launch['pid'],
        owner_start_ticks=launch['start_ticks'],automatic_retry=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=not failed,checks=len(passed),queued=len(queued),failed=len(failed))),flush=True)


if __name__=='__main__':main()
