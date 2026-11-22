"""Restore all four first-source cup-placement RML public input consumers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.rml_cup_support_v695 import binding
from our_mani.multitask.rml_case_preflight_v698 import SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'rml_suite16_smoke_v677_setup';setup=base/'cup_input_repair_v700_setup'
    if setup.exists():raise FileExistsError('preserve the held-cup input repair attempt')
    files=json.loads((old/'frozen_inputs.json').read_text())['files'];configs={}
    tasks=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
    worker='scripts/run_task_executor_smoke_rml_cup_support_v697_r10.py'
    for name in (*SOURCES,worker,'scripts/prepare_cup_input_repair_v700_r10.py','tests/test_rml_cup_support_v699.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    for task in tasks:
        c=json.loads((old/f'{task}_ours_configuration.json').read_text())
        c['RML_cup_public_support_v695']={task:binding(ROOT,c,task)}
        c.update(execution_scope='TRAIN_restore_public_held_cup_support_in_direct_RML_v700',formal_comparison_ready=False)
        configs[task]=c
    verify_inputs(ROOT,files);setup.mkdir()
    for task,c in configs.items():
        path=setup/f'{task}_ours_configuration.json';atomic_json(path,c);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        original_failed_cup_runs_retained='rml_suite16_smoke_v677',same_fixed_original_sources=True,
        original_public_support_math_restored=True,actuator_model_native_RML_and_success_unchanged=True))
    jobs=[]
    old_jobs=json.loads((old/'plan.json').read_text())['planned']
    for task in tasks:
        for method in ('cv_geometry','context_geometry'):
            row=next(j for j in old_jobs if j['task']==task and j['method']==method);argv=list(row['argv']);argv[2]=worker
            argv[argv.index('--config')+1]=str((setup/f'{task}_ours_configuration.json').relative_to(ROOT))
            argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
            argv[argv.index('--out')+1]=str((base/'cup_input_repair_v700'/'development_demo_only'/task/row['raw']/method).relative_to(ROOT))
            name=f'cup_input_repair_v700_{task}_{row["raw"]}_{method.split("_")[0]}'
            validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,argv=argv,
                use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/worker)))
            jobs.append(dict(task=task,raw=row['raw'],method=method,argv=argv,job_id=name))
    atomic_json(setup/'plan.json',dict(planned=jobs,expected_cases=4,source_outcome_filtering=False,
        both_original_cup_tasks_and_both_predictors=True,automatic_retry=False,formal_VAL_TEST=False))
    def check(j):return j,subprocess.run(j['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[];failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            row,r=f.result()
            if r.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,native_started=False))
                failed.append(row);print(json.dumps(dict(CPU_failed=row['job_id'])),flush=True);continue
            proof=json.loads(r.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('actual CPU-only receipt required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),total=4,method=row['method'],task=row['task'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        native_started=False,GPU_started=False,formal_VAL_TEST=False))
    if failed:raise ValueError('CPU failure preserved; do not queue')


if __name__=='__main__':main()
