"""One-shot source-bound CPU full14 fitter after accepted actual400 proof."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,release_preflight as release
from our_mani.multitask.evaluation import atomic_json


def run(recipe_path,consumer_path,status_path,log_path):
    inputs=release._Inputs(ROOT);recipe=inputs.read(recipe_path);consumer_file=inputs.file(consumer_path)
    consumer=json.loads(consumer_file.read_text())
    a.require(consumer.get('schema')=='r10_actual400_combined_command_goal_v4_semantic_v6_consumer_audit_v1'
        and consumer.get('actual_teacher_sources_rebuilt')==400 and consumer.get('distinct_original_parent_groups')==200
        and consumer.get('producer_consumer_all_source_proofs_identical') is True
        and consumer.get('combined_goal_and_semantic_views_rebuilt') is True
        and len(consumer.get('negative_cases_rejected',[]))==36
        and len(set(consumer.get('negative_cases_rejected',[])))==36
        and consumer.get('native_GPU_or_collection_started') is False
        and consumer.get('formal_model_policy_release') is False,'complete actual400 dual-typed consumer acceptance required')
    config_path=inputs.pointer(recipe['fitting_configuration']);config=json.loads(config_path.read_text())
    for field in ('fitting_core_source_snapshot','fitting_producer_source_snapshot'):inputs.source_snapshot(config[field])
    producer=inputs.read_pointer(consumer['production_fit'])
    a.require(producer.get('all675_original_deposit_commands_retained') is True
        and len(producer['tasks'])==2 and all(not row['failed_fit_keys'] and 'library' in row
            for row in producer['tasks'].values()),'both complete TRAIN400 skill fits required before full14')
    inputs.pointer(recipe['actual_bundle']);inputs.pointer(recipe['actual_complete3000_bundle_audit'])
    audit=inputs.read_pointer(recipe['actual_complete3000_bundle_audit'])
    a.require(audit.get('primary_complete2800') is True and audit.get('auxiliary_complete200') is True
        and audit.get('combined_acquisitions')==3000 and audit.get('original_TRAIN_parents')==1400
        and audit.get('teacher_rollouts')==1600 and audit.get('original_Context_hold_rollouts')==1400,
        'the actual full3000 bundle audit must pass')
    bundle_proof=inputs.read_pointer(audit['bundle_audit'])
    a.require(bundle_proof.get('audit_passed') is True and bundle_proof.get('complete') is True,
        'the nested full3000 bundle proof is incomplete/failed')
    argv=recipe['argv'];a.require(isinstance(argv,list) and argv[:2]==[str(ROOT/'.conda-env/bin/python'),
        'scripts/fit_multitask_interaction_artifacts_r10.py'] and len(argv)==12,
        'exact one-shot CPU fitter argv required')
    expected={'--collector-manifest':recipe['actual_bundle']['path'],'--fit-config':recipe['fitting_configuration']['path'],
        '--demonstrations':'artifacts/framework/multitask_r10/demonstrations/manifest.json',
        '--original-prior':'artifacts/framework/multitask_r10/original_TRAIN_action_phase_priors.json',
        '--out':'artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv4_semanticv6_v1'}
    a.require(dict(zip(argv[2::2],argv[3::2]))==expected and len(set(argv[2::2]))==5,'full14 recipe argv scope differs')
    out=a.local(ROOT,expected['--out']);a.require(not out.exists(),'fresh full14 model output directory required')
    status=a.local(ROOT,status_path);log=a.local(ROOT,log_path)
    a.require(not status.exists() and not log.exists(),'one-shot fresh execution status/log required')
    inputs.rehash_all();initial=dict(inputs.files)
    a.require(recipe['environment']==dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),
        'exact reviewed CPU-only execution environment required')
    env=dict(os.environ,**recipe['environment'])
    log.parent.mkdir(parents=True,exist_ok=True)
    state=dict(schema='r10_full14_actual3000_CPU_fit_execution_v1',started_utc=datetime.now(timezone.utc).isoformat(),
        recipe=a.pointer(ROOT,a.local(ROOT,recipe_path)),actual400_consumer=a.pointer(ROOT,consumer_file),
        configuration=recipe['fitting_configuration'],argv=argv,environment=recipe['environment'],
        source_and_input_files_sha256=initial,supervisor_pid=os.getpid(),native_GPU_or_collection_started=False,
        formal_model_policy_release=False,status='starting')
    with log.open('w') as stream:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:start=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:start=None
        state.update(status='running' if child.poll() is None else 'exited_before_running_snapshot',
            pid=child.pid,proc_starttime=start)
        atomic_json(status,state);code=child.wait()
    state.update(process_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
    try:
        inputs.rehash_all()
        a.require(code==0,'full14 CPU fitter exited unsuccessfully')
        result=out/'report.json';a.require(result.is_file(),'full14 fitter output report is missing')
        report=a.read(result);state.update(status='complete',report=a.pointer(ROOT,result),fitting_status=report.get('status'),
            native_TRAIN_parent_sources=report.get('native_TRAIN_parent_sources'))
    except Exception as error:
        state.update(status='failed_preserved_no_retry',error=repr(error));atomic_json(status,state);raise
    atomic_json(status,state);return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','consumer','status','log'):p.add_argument('--'+name,required=True)
    v=p.parse_args();print(json.dumps(run(v.recipe,v.consumer,v.status,v.log),ensure_ascii=False))
