"""Freeze the same first TRAIN button sources for the PhysMani RML adaptation."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.physmani_geometry_executor_v517 import binding,SOURCES
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10'
    old,setup=base/'button_corridor_v414_setup',base/'physmani_geometry_v520_setup'
    if setup.exists():raise FileExistsError('preserve existing RML preparation')
    rows=json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
    if len(rows)!=4 or {r['method'] for r in rows}!={'cv_geometry','context_geometry'}:
        raise ValueError('same first TRAIN source of each of two tasks and two methods')
    original=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    for row in rows:
        first=next(r for r in original if r['task']==row['task'])
        if first['raw_episode_id']!=row['raw']:
            raise ValueError('no outcome-selected source replacement')
    config=json.loads((old/'configuration.json').read_text())
    config['physmani_geometry_executor_v517']=binding(ROOT,config['baseline_repo'])
    config.update(execution_scope='TRAIN_only_geometry_intents_on_original_PhysMani_RML_v520',
                  formal_comparison_ready=False)
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    files.update(config['physmani_geometry_executor_v517']['native_sources_sha256'])
    for name in (*SOURCES,'our_mani/multitask/physmani_geometry_policy_v518.py',
                 'our_mani/multitask/native_window_observer_v191.py',
                 'scripts/run_task_executor_smoke_physmani_geometry_v519_r10.py',
                 'scripts/prepare_physmani_geometry_v520_r10.py',
                 'tests/test_physmani_geometry_executor_v517.py',
                 'tests/test_physmani_geometry_policy_v518.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    verify_inputs(ROOT,files)
    setup.mkdir();atomic_json(setup/'configuration.json',config)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    jobs=[]
    for row in rows:
        argv=list(row['argv'])
        argv[2]='scripts/run_task_executor_smoke_physmani_geometry_v519_r10.py'
        argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((base/'physmani_geometry_v520'/'development_demo_only'/
            row['task']/row['raw']/row['method']).relative_to(ROOT))
        name='physmani_geometry_v520_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0]
        validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,
            argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2])))
        jobs.append(dict(row,argv=argv,job_id=name))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    accepted=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,r) for r in jobs]):
            row,result=future.result()
            if result.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,
                    stderr=result.stderr,native_started=False,automatic_retry=False))
            result.check_returncode();proof=json.loads(result.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:
                raise ValueError('actual CPU-only native entrypoint proof required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);accepted.append(row)
            print(json.dumps(dict(CPU_pass=len(accepted),total=4,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,
        original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False,
        original_PhysMani_neural_baseline_modified=False))
    print(json.dumps(dict(prepared=True,checks=4,frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
