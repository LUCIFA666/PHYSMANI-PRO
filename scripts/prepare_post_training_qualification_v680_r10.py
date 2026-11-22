"""Bind the fresh joint16 model to frozen executors and exact fixed-four TRAIN.

The16 observed baseline first cases are reused with a transparent input-scope
extension; their neural policy, configuration, native runtime and RML stay the
same. All128 geometry cases use the newly trained shared checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.rml_component_domain_v640 import for_train
from our_mani.multitask.rml_component_factory_v643 import load_library,identity
from our_mani.multitask.common_rml_protocol_v530 import fingerprint
from our_mani.multitask.rml_policy_components_v702 import binding as policy_binding
from our_mani.multitask.rml_case_preflight_v710 import SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request


def point(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'rml_suite16_smoke_v677_setup'
    setup=base/'post_training_qualification_v680_setup'
    if setup.exists():raise FileExistsError('preserve the final model/executor qualification')
    accepted_path=base/'joint16_after_executor_training_acceptance_v672/report.json';trained=json.loads(accepted_path.read_text())
    if (trained.get('training_complete') is not True or trained.get('fresh_training_after_executor_repairs') is not True
        or trained['training']['TRAIN_original_parents']!=1600 or trained['training']['completed_epochs']!=30):
        raise ValueError('requested new complete16-task training required before final qualification')
    verify_inputs(ROOT,trained['inputs_sha256'])
    files=json.loads((base/'cup_input_repair_v712_setup/frozen_inputs.json').read_text())['files']
    for name,sha in json.loads((base/'remove_time_scene_v706_setup/frozen_inputs.json').read_text())['files'].items():
        if name in files and files[name]!=sha:raise ValueError('immutable current component inputs conflict')
        files[name]=sha
    for path in [accepted_path,ROOT/trained['checkpoint']['path'],ROOT/'scripts/prepare_post_training_qualification_v680_r10.py',
                 ROOT/'scripts/run_task_executor_smoke_rml_inputs_v709_r10.py',ROOT/'tests/test_rml_split_source_inputs_v684.py']:
        files[str(path)]=file_sha(path)
    for name in SOURCES:files[str(ROOT/name)]=file_sha(ROOT/name)
    checkpoint=ROOT/trained['checkpoint']['path'];training_sources=checkpoint.parent/'training_sources.json'
    files[str(training_sources)]=file_sha(training_sources)
    roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json';roster=validate_roster(json.loads(roster_path.read_text()),'train')
    fixed={t:[r for r in roster if r['task']==t][:4] for t in TASKS}
    for rows in fixed.values():
        for row in rows:files[str(Path(row['path']).resolve())]=file_sha(row['path'])
    configs={}
    for task in TASKS:
        current=(base/'cup_input_repair_v712_setup' if task.startswith('place_cups_on_rotating_frame') else
            base/'remove_time_scene_v706_setup' if task.startswith('remove_cups_from_rotating_frame') else old)
        c=json.loads((current/f'{task}_ours_configuration.json').read_text())
        c['context_checkpoint']=trained['checkpoint']
        if 'joint16_context_checkpoint' in c:c['joint16_context_checkpoint']=trained['checkpoint']
        c.pop('joint16_development_checkpoint_v157',None)
        c.update(execution_scope='full16_frozen_RML_new_joint16_training_v680',formal_comparison_ready=True)
        if task not in BLOCKS:
            c['direct_RML_policy_components_v702']={task:policy_binding(ROOT,c,task)}
            c['direct_RML_component_identity_v643']={task:{}}
            for method in METHODS[:2]:
                domain=for_train(ROOT,files,task,fixed[task][0]['raw_episode_id'],method)
                library=load_library(ROOT,c,task,domain)
                c['direct_RML_component_identity_v643'][task][method]=identity(ROOT,c,task,method,library)
        configs[task]=c
    verify_inputs(ROOT,files);setup.mkdir()
    for task,c in configs.items():
        path=setup/f'{task}_ours_configuration.json';atomic_json(path,c);files[str(path)]=file_sha(path)
    lock=setup/'frozen_inputs.json';atomic_json(lock,dict(files=files,approved_for_execution=True,
        immutable_executables_models_and_TRAIN_qualification_sources=True,
        formal_split_data_locked_separately_after_admission=True,formal_VAL_TEST=False))
    worker=ROOT/'scripts/run_task_executor_smoke_rml_inputs_v709_r10.py';routes=[]
    for task in TASKS:
        for method in METHODS:
            config=old/f'{task}_physmani_configuration.json' if method=='physmani' else setup/f'{task}_ours_configuration.json'
            c=json.loads(config.read_text())
            routes.append(dict(task=task,method=method,GPU_required=method=='physmani',configuration=point(config),
                frozen_inputs=point(lock),worker=point(worker),consumer=point(ROOT/'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'),
                source_preflight=point(ROOT/'our_mani/multitask/rml_case_preflight_v710.py'),
                common_RML_profile_sha256=fingerprint(c['common_author_RML_profile'])))
    route_path=setup/'routes.json';atomic_json(route_path,dict(schema='r10_exact16_frozen_RML_fresh_joint16_routes_v680',
        tasks=list(TASKS),methods=list(METHODS),routes=routes,shared_context_checkpoint_sha256=trained['checkpoint']['sha256'],
        common_author_RML_profile=configs[TASKS[0]]['common_author_RML_profile']))
    old_jobs=json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'];jobs=[];reused=[]
    for task in TASKS:
        for source in fixed[task]:
            raw=source['raw_episode_id']
            for method in METHODS:
                if method=='physmani' and source is fixed[task][0]:
                    previous=next(j for j in old_jobs if j['task']==task and j['method']=='physmani')
                    case_path=ROOT/previous['argv'][previous['argv'].index('--out')+1]/'case.json'
                    if not case_path.exists():raise ValueError('the original baseline first case must already be completed')
                    reused.append(dict(task=task,raw=raw,method=method,case=point(case_path),original_argv=previous['argv'],
                        original_input_lock=point(old/'frozen_inputs.json'),input_scope_extension_only=True,
                        native_runtime_unchanged=point(ROOT/'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py')))
                    continue
                r=next(r for r in routes if r['task']==task and r['method']==method)
                argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',str(worker.relative_to(ROOT)),
                    '--config',r['configuration']['path'],'--manifest',str(roster_path.relative_to(ROOT)),
                    '--frozen-inputs',str(lock.relative_to(ROOT)),'--task',task,'--raw-episode-id',raw,'--split','train',
                    '--method',method,'--development-demo','--out',
                    str((base/'post_training_qualification_v680'/'development_demo_only'/task/raw/method).relative_to(ROOT))]
                name=f'post_training_v680_{task}_{raw}_{method.split("_")[0]}'
                validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,argv=argv,
                    use_gpu=method=='physmani',timeout_seconds=1800,source_sha256=file_sha(worker)))
                jobs.append(dict(task=task,raw=raw,method=method,argv=argv,job_id=name))
    if len(jobs)!=176 or len(reused)!=16:raise ValueError('exact full16x4x3 grid required')
    atomic_json(setup/'plan.json',dict(planned=jobs,reused=reused,expected_method_cases=192,new_cases=176,
        existing_observed_baseline_cases=16,all_failures_retained=True,source_outcome_filtering=False,
        automatic_retry=False,formal_VAL_TEST=False,routes=point(route_path),training_acceptance=point(accepted_path)))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[];failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,r) for r in jobs]):
            row,result=future.result()
            if result.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=result.stdout,stderr=result.stderr,
                    returncode=result.returncode,native_started=False,automatic_retry=False));failed.append(row)
                print(json.dumps(dict(CPU_failed=row['job_id'])),flush=True);continue
            proof=json.loads(result.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('actual CPU-only argv proof required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),failed=len(failed),total=176,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=not failed,actual_argv_preflights=passed,failed_preflights=failed,
        reused_observed_baseline_cases=reused,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=not failed,CPU_checks=len(passed),failed=len(failed))),flush=True)


if __name__=='__main__':main()
