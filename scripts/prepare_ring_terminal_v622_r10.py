"""Freeze first original TRAIN source for every core14 task on common author RML."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.executor_suite16_v447 import route,CHECKPOINT_SHA
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.physmani_ring_compensation_v580 import binding as compensation_binding
from our_mani.multitask.physmani_ring_release_v590 import binding as release_binding
from our_mani.multitask.physmani_ring_frontend_v570 import binding as gauge_binding
from our_mani.multitask.physmani_geometry_executor_v517 import binding,SOURCES
from our_mani.multitask.physmani_terminal_opening_v620 import binding as terminal_binding
from our_mani.multitask.common_rml_protocol_v530 import binding as profile_binding
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_task_executor_smoke_ring_terminal_v621_r10 import EXTRA_SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'ring_terminal_v622_setup'
    if setup.exists():raise FileExistsError('preserve every previous skill-RML preparation')
    roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    tasks=['insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed']
    files={};configs={}
    for task in tasks:
        r=route(task,'cv_geometry');old=base/r.setup
        config=json.loads((base/r.configuration(task)).read_text())
        if config['context_checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('same trained joint16 model required')
        for name,sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
            if name in files and files[name]!=sha:raise ValueError('immutable component source conflict: '+name)
            files[name]=sha
        config['physmani_geometry_executor_v517']=binding(ROOT,config['baseline_repo'])
        config['common_author_RML_profile']=profile_binding(ROOT,config['baseline_repo'])
        config['public_ring_frontend_v570']={task:gauge_binding(ROOT,config,task)}
        config['public_ring_compensation_v580']={task:compensation_binding(ROOT,config['baseline_repo'],task)}
        config['author_ring_terminal_release_v590']={task:release_binding(ROOT,config['baseline_repo'],task)}
        prior=config['public_ring_compensation_v580'][task]['prior']
        files[str(ROOT/prior['path'])]=prior['sha256']
        config['terminal_opening_v620']=terminal_binding(ROOT)
        config.update(execution_scope='TRAIN_first_core14_public_skills_on_author_RML_v536',formal_comparison_ready=False)
        configs[task]=config
        files.update(config['physmani_geometry_executor_v517']['native_sources_sha256'])
        files.update({str(Path(config['baseline_repo'])/k):v for k,v in config['common_author_RML_profile']['native_sources_sha256'].items()})
    excluded={n:s for n,s in files.items() if Path(n).name=='updtChck.dat' and '/runtime/CoppeliaSim_Edu_' in n}
    for name in excluded:del files[name]
    for name in (*SOURCES,*EXTRA_SOURCES,'scripts/prepare_ring_terminal_v622_r10.py',
        'tests/test_physmani_skill_v533.py','tests/test_physmani_ring_frontend_v570.py','tests/test_physmani_ring_compensation_v585.py','tests/test_physmani_ring_release_v590.py','tests/test_common_rml_release_v531.py','tests/test_physmani_terminal_opening_v620.py',
        'our_mani/multitask/three_method_release_v531.py','our_mani/multitask/formal_suite_admission_v532.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    verify_inputs(ROOT,files);setup.mkdir()
    for task,config in configs.items():
        path=setup/(task+'_configuration.json');atomic_json(path,config);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        excluded_mutable_update_checker_state=excluded))
    jobs=[]
    worker='scripts/run_task_executor_smoke_ring_terminal_v621_r10.py'
    for task in tasks:
        source=next(r for r in roster if r['task']==task);raw=source['raw_episode_id']
        for method in ('context_geometry','cv_geometry'):
            argv=['bash','scripts/run_local_sim.sh',worker,'--config',str((setup/(task+'_configuration.json')).relative_to(ROOT)),
                '--manifest','artifacts/framework/protocol/train_manifest.json','--frozen-inputs',
                str((setup/'frozen_inputs.json').relative_to(ROOT)),'--task',task,'--raw-episode-id',raw,
                '--split','train','--method',method,'--replica','0','--development-demo','--out',
                str((base/'ring_terminal_v622'/'development_demo_only'/task/raw/method).relative_to(ROOT))]
            name='ring_terminal_v622_'+task+'_'+raw+'_'+method.split('_')[0]
            validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
                job_id=name,argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/worker)))
            jobs.append(dict(task=task,raw=raw,method=method,argv=argv,job_id=name))
    atomic_json(setup/'plan.json',dict(task_count=2,method_cases=4,planned=jobs,
        first_original_TRAIN_source_only=True,source_outcome_filtering=False,automatic_retry=False,formal_VAL_TEST=False))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[];failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,row) for row in jobs]):
            row,result=future.result()
            if result.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,
                    returncode=result.returncode,native_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=row['job_id'],returncode=result.returncode)),flush=True);continue
            proof=json.loads(result.stdout)
            if proof.get('complete') is not True or proof.get('native_started') is not False or proof.get('GPU_started') is not False:
                raise ValueError('actual argv CPU-only preflight required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),total=4,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,
        failed_preflights=failed,original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False,
        original_PhysMani_neural_baseline_modified=False))
    print(json.dumps(dict(prepared=not failed,checks=len(passed),failed=len(failed),frozen_inputs=len(files))),flush=True)
    if failed:raise ValueError('preserved failed CPU preflights; native queue not released')


if __name__=='__main__':main()
