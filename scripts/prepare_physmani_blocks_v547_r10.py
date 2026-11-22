"""Freeze first-source block RML smokes without changing completed R09 cases."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.physmani_geometry_executor_v517 import binding,SOURCES as RML_SOURCES
from our_mani.multitask.common_rml_protocol_v530 import binding as common_binding
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from scripts.run_task_executor_smoke_physmani_blocks_v546_r10 import SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'joint16_block_support_v318_setup'
    setup=base/'physmani_blocks_v547_setup'
    if setup.exists():raise FileExistsError('preserve every frozen block RML preparation')
    config=json.loads((old/'configuration.json').read_text());files=json.loads((old/'frozen_inputs.json').read_text())['files']
    config['physmani_geometry_executor_v517']=binding(ROOT,config['baseline_repo'])
    config['common_author_RML_profile']=common_binding(ROOT,config['baseline_repo'])
    config.update(execution_scope='TRAIN_public_blocks_on_author_RML_v540',formal_comparison_ready=False)
    files.update(config['physmani_geometry_executor_v517']['native_sources_sha256'])
    files.update({str(Path(config['baseline_repo'])/n):s for n,s in config['common_author_RML_profile']['native_sources_sha256'].items()})
    for name in (*SOURCES,*RML_SOURCES,'scripts/prepare_physmani_blocks_v547_r10.py','tests/test_physmani_block_v538.py','tests/test_physmani_block_receipt_v541.py','tests/test_physmani_block_record_v545.py',
                 'our_mani/multitask/baseline_measured_windows_v345.py','our_mani/multitask/native_window_observer_v191.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    excluded={n:s for n,s in files.items() if Path(n).name=='updtChck.dat' and '/runtime/CoppeliaSim_Edu_' in n}
    for name in excluded:del files[name]
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',config)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        excluded_mutable_update_checker_state=excluded))
    roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes'];jobs=[]
    worker='scripts/run_task_executor_smoke_physmani_blocks_v546_r10.py'
    for task in BLOCKS:
        row=next(r for r in roster if r['task']==task);raw=row['raw_episode_id']
        for method in ('context_geometry','cv_geometry'):
            argv=['bash','scripts/run_local_sim.sh',worker,'--config',str((setup/'configuration.json').relative_to(ROOT)),
                '--manifest','artifacts/framework/protocol/train_manifest.json','--frozen-inputs',
                str((setup/'frozen_inputs.json').relative_to(ROOT)),'--task',task,'--raw-episode-id',raw,
                '--split','train','--method',method,'--development-demo','--out',
                str((base/'physmani_blocks_v547'/'development_demo_only'/task/raw/method).relative_to(ROOT))]
            name='physmani_blocks_v547_'+task+'_'+raw+'_'+method.split('_')[0]
            validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,
                argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/worker)))
            jobs.append(dict(task=task,raw=raw,method=method,argv=argv,job_id=name))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,r) for r in jobs]):
            row,r=future.result()
            if r.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,automatic_retry=False,native_started=False))
                r.check_returncode()
            proof=json.loads(r.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('CPU-only actual argv required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),total=4,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=passed,
        first_original_TRAIN_source_only=True,historical_R09_results_modified=False,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,checks=4,frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
