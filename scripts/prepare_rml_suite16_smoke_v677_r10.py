"""Freeze first-original-TRAIN full16 three-method native runtime qualification.

This checks the actual future formal consumer on TRAIN before executor freeze
and the requested new joint16 training. No formal source is selected or run.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.rml_case_preflight_v663 import SOURCES
from our_mani.multitask.common_rml_protocol_v530 import binding as common_binding
from scripts.run_native_resource_owner_v19_r10 import validate_request
from our_mani.multitask.rml_policy_components_v669 import binding as latest_policy
from our_mani.multitask.physmani_remove_intercept_v667 import binding as intercept_binding


def main():
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'rml_suite16_smoke_v677_setup'
    if setup.exists():raise FileExistsError('preserve every full16 consumer qualification attempt')
    core=base/'direct_components_v662_setup'
    if json.loads((core/'CPU_acceptance.json').read_text()).get('complete') is not True:
        raise ValueError('actual current fourteen-task component loader CPU qualification required')
    other=[base/'physmani_blocks_v547_setup',base/'physmani_windows_v194_setup',base/'block_physmani_v348_setup']
    files={}
    for directory in [core,*other]:
        for name,sha in json.loads((directory/'frozen_inputs.json').read_text())['files'].items():
            if Path(name).name=='updtChck.dat' and '/runtime/CoppeliaSim_Edu_' in name:continue
            canonical=str((ROOT/name).resolve())
            if canonical in files and files[canonical]!=sha:raise ValueError('immutable consumer inputs conflict: '+canonical)
            files[canonical]=sha
    worker='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'
    for name in (*SOURCES,worker,'scripts/prepare_rml_suite16_smoke_v677_r10.py','tests/test_rml_case_preflight_v665.py','tests/test_physmani_remove_intercept_v668.py',
                 'our_mani/multitask/input_stat_cache_v79.py','our_mani/multitask/frozen_input_edges_v463.py'):
        path=ROOT/name;files[str(path)]=file_sha(path)
    roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json';roster=validate_roster(json.loads(roster_path.read_text()),'train')
    files[str(roster_path)]=file_sha(roster_path)
    for task in TASKS:
        original=next(r for r in roster if r['task']==task)
        path=Path(original['path']).resolve()
        files[str(path)]=file_sha(path)
    configs={}
    for task in TASKS:
        own_path=other[0]/'configuration.json' if task in BLOCKS else core/(task+'_configuration.json')
        own=json.loads(own_path.read_text())
        if task not in BLOCKS:
            if task.startswith('remove_cups_from_rotating_frame'):
                own['remove_RML_request_intercept_v667']={task:intercept_binding(ROOT,task)}
            own['direct_RML_policy_components_v669']={task:latest_policy(ROOT,own,task)}
        configs[task,'ours']=own
        baseline=json.loads((other[2] if task in BLOCKS else other[1]).joinpath('configuration.json').read_text())
        baseline['common_author_RML_profile']=common_binding(ROOT,baseline['baseline_repo'])
        baseline.update(execution_scope='original_PhysMani_full16_common_RML_runtime_smoke_v677',formal_comparison_ready=False)
        configs[task,'physmani']=baseline
    for c in configs.values():
        for name,sha in c['common_author_RML_profile']['native_sources_sha256'].items():
            path=Path(c['baseline_repo'])/name;files[str(path.resolve())]=sha
    verify_inputs(ROOT,files);setup.mkdir()
    for (task,method),c in configs.items():
        path=setup/f'{task}_{method}_configuration.json';atomic_json(path,c);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        qualification_before_requested_new_joint16_training=True,actual_future_full16_consumer_on_TRAIN=True,
        historical_R09_results_modified=False))
    jobs=[]
    for task in TASKS:
        raw=next(r['raw_episode_id'] for r in roster if r['task']==task)
        for method in METHODS:
            scope='physmani' if method=='physmani' else 'ours'
            argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',worker,
                '--config',str((setup/f'{task}_{scope}_configuration.json').relative_to(ROOT)),
                '--manifest',str(roster_path.relative_to(ROOT)),'--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
                '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,'--development-demo',
                '--out',str((base/'rml_suite16_smoke_v677'/'development_demo_only'/task/raw/method).relative_to(ROOT))]
            name=f'rml_suite16_smoke_v677_{task}_{raw}_{method.split("_")[0]}'
            validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,
                argv=argv,use_gpu=method=='physmani',timeout_seconds=1800,source_sha256=file_sha(ROOT/worker)))
            jobs.append(dict(task=task,raw=raw,method=method,argv=argv,job_id=name))
    atomic_json(setup/'plan.json',dict(tasks=list(TASKS),methods=list(METHODS),planned=jobs,
        expected_cases=48,first_original_TRAIN_source_only=True,source_outcome_filtering=False,
        automatic_retry=False,formal_VAL_TEST=False,new_unified_training_started=False))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[];failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,r) for r in jobs]):
            row,result=future.result()
            if result.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,
                    returncode=result.returncode,native_started=False,GPU_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=row['job_id'],returncode=result.returncode)),flush=True);continue
            proof=json.loads(result.stdout)
            if proof.get('complete') is not True or proof.get('native_started') is not False or proof.get('GPU_started') is not False:
                raise ValueError('actual CPU-only full16 worker preflight required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),total=48,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        native_started=False,GPU_started=False,formal_VAL_TEST=False,actual_full16_consumer_checked=True))
    print(json.dumps(dict(prepared=not failed,checks=len(passed),failed=len(failed),frozen_inputs=len(files))),flush=True)
    if failed:raise ValueError('CPU failures preserved; do not enqueue this incomplete batch')


if __name__=='__main__':main()
