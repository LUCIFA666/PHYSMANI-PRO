"""Freeze the same current core14 RML policies with direct qualified components."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.rml_component_domain_v640 import for_train
from our_mani.multitask.rml_component_factory_v643 import load_library,identity
from our_mani.multitask.rml_policy_components_v660 import binding as policy_binding,SOURCES
from our_mani.multitask.physmani_geometry_executor_v517 import SOURCES as RML_SOURCES
from scripts.run_task_executor_smoke_direct_components_v661_r10 import EXTRA_SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request

BLUEPRINTS={
 'push_moving_button':'physmani_skills_v537_setup',
 'insert_onto_rotating_peg':'ring_release_contact_v659_setup',
 'moving_basketball_in_hoop':'physmani_basketball_release_v560_setup',
 'put_rubbish_in_moving_bin':'bin_clearance_v652_setup',
 'place_cups_on_rotating_frame':'stage_lookahead_v596_setup',
 'beat_the_rotating_buzz':'buzz_terminal_v625_setup',
 'remove_cups_from_rotating_frame':'remove_source_roles_v655_setup'}


def main():
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'direct_components_v662_setup'
    if setup.exists():raise FileExistsError('preserve every component qualification attempt')
    manifest=ROOT/'artifacts/framework/protocol/train_manifest.json';roster=json.loads(manifest.read_text())['episodes']
    files={};configs={}
    for task in sorted(TASK_SPECS):
        old=base/BLUEPRINTS[TASK_SPECS[task].family]
        accepted=json.loads((old/'CPU_acceptance.json').read_text())
        if accepted.get('complete') is not True:raise ValueError('each current policy blueprint needs actual CPU proof')
        row=next(r for r in accepted['actual_argv_preflights'] if r['task']==task)
        argv=row['argv'];c=json.loads((ROOT/argv[argv.index('--config')+1]).read_text())
        prior_files=json.loads((ROOT/argv[argv.index('--frozen-inputs')+1]).read_text())['files']
        for name,sha in prior_files.items():
            if name in files and files[name]!=sha:raise ValueError('immutable current component inputs conflict: '+name)
            files[name]=sha
        c['direct_RML_policy_components_v660']={task:policy_binding(ROOT,c,task)}
        c.update(execution_scope='TRAIN_fixed_original_sources_direct_RML_components_v662',formal_comparison_ready=False)
        configs[task]=c
    files[str(manifest)]=file_sha(manifest)
    for name in (*SOURCES,*RML_SOURCES,*EXTRA_SOURCES,'scripts/prepare_direct_components_v662_r10.py',
                 'scripts/run_task_executor_smoke_direct_components_v661_r10.py','tests/test_rml_component_domain_v644.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    for task,c in configs.items():
        first=next(r for r in roster if r['task']==task)
        c['direct_RML_component_identity_v643']={task:{}}
        for method in ('cv_geometry','context_geometry'):
            domain=for_train(ROOT,files,task,first['raw_episode_id'],method)
            library=load_library(ROOT,c,task,domain)
            c['direct_RML_component_identity_v643'][task][method]=identity(ROOT,c,task,method,library)
    verify_inputs(ROOT,files);setup.mkdir()
    for task,c in configs.items():
        p=setup/(task+'_configuration.json');atomic_json(p,c);files[str(p)]=file_sha(p)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        actual_actor_factory='rml_component_factory_v643',legacy_actor_constructors_installed=False,
        strong_stat_SHA_reuse_and_canonical_edge_index=True,declared_input_set_or_SHA_checks_removed=False))
    jobs=[];worker='scripts/run_task_executor_smoke_direct_components_v661_r10.py'
    for task in sorted(configs):
        source=next(r for r in roster if r['task']==task);raw=source['raw_episode_id']
        for method in ('cv_geometry','context_geometry'):
            argv=['bash','scripts/run_local_sim.sh',worker,'--config',str((setup/(task+'_configuration.json')).relative_to(ROOT)),
                '--manifest',str(manifest.relative_to(ROOT)),'--frozen-inputs',str((setup/'frozen_inputs.json').relative_to(ROOT)),
                '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,'--replica','0','--development-demo',
                '--out',str((base/'direct_components_v662'/'development_demo_only'/task/raw/method).relative_to(ROOT))]
            name='direct_components_v662_'+task+'_'+raw+'_'+method.split('_')[0]
            validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,argv=argv,
                use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(ROOT/worker)))
            jobs.append(dict(task=task,raw=raw,method=method,argv=argv,job_id=name))
    atomic_json(setup/'plan.json',dict(task_count=14,method_cases=28,planned=jobs,first_original_TRAIN_source_only=True,
        source_outcome_filtering=False,automatic_retry=False,formal_VAL_TEST=False,
        two_block_routes_remain_existing_public_RML_v547=True))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[];failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(check,r) for r in jobs]):
            row,result=f.result()
            if result.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,
                    returncode=result.returncode,native_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=row['job_id'],returncode=result.returncode)),flush=True);continue
            proof=json.loads(result.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('actual CPU-only argv proof required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),total=28,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        original_fixed_TRAIN_sources=True,native_started=False,GPU_started=False,formal_VAL_TEST=False,
        actual_direct_actor_model_loaders_checked=True,VAL_TEST_relabelled_as_TRAIN=False))
    print(json.dumps(dict(prepared=not failed,checks=len(passed),failed=len(failed),frozen_inputs=len(files))),flush=True)
    if failed:raise ValueError('CPU failures retained; no native batch admission')


if __name__=='__main__':main()
