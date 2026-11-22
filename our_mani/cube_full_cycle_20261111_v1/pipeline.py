"""Actual serial collection -> fit -> generated validation -> frozen test owner."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import ROOT,OUT,PLAN,NATIVE_PYTHON,plan,read,write,sha,pointer,status,stopped,training_sources,verify,utc


def stage_worker(stage):
    p=plan()
    if stage=='motion-compile':
        from .datasets import compile_motion
        compile_motion(p)
    elif stage=='motion-fit':
        from .fit import fit
        fit(p,'motion')
    elif stage=='public-replay':
        from .datasets import replay_public
        replay_public(p,OUT/'motion_fit/selected.pt')
    elif stage=='candidate-compile':
        from .datasets import compile_candidate
        compile_candidate(p,OUT/'motion_fit/selected.pt')
    elif stage=='candidate-fit':
        from .fit import fit
        fit(p,'candidate',OUT/'motion_fit/selected.pt')
    elif stage=='runtime-check':
        from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
        from ..cube_native_contact_feedback_live_20261104_v1.candidate_runtime import NativeContactMatchedInferenceV1
        import torch
        torch.set_num_threads(2)
        m=OUT/'motion_fit/selected.pt';c=OUT/'candidate_fit/selected.pt'
        CubeResidualMotionDynamics(m,expected_sha256=sha(m))
        NativeContactMatchedInferenceV1(c,expected_sha256=sha(c),forecast_sha256=sha(m))
        write(OUT/'runtime_check.json',dict(complete=True,motion=pointer(m),candidate=pointer(c),runtime_guards_unchanged=True))
    else:raise ValueError(stage)
    return 0


def run_child(p,name,argv,timeout=14400):
    """Wait on the real child; never infer completion from PID disappearance."""
    stopped(p)
    if not name.startswith(('train_','validation_','test_')):status(name,complete=False)
    journal=OUT/'stage_exits';journal.mkdir(exist_ok=True)
    request=journal/(name+'_request.json');write(request,dict(argv=argv,started_utc=utc(),timeout_seconds=timeout))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    start=time.monotonic();reason=None;stopping=None
    received=[];previous={s:signal.signal(s,lambda n,f:received.append(n))for s in(signal.SIGTERM,signal.SIGINT)}
    process=None
    try:
        with(journal/(name+'.log')).open('x')as log:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll()is None:
                if stopping is None:
                    try:stopped(p)
                    except InterruptedError:reason='scoped_STOP';stopping=time.monotonic()
                    if received:reason='signal';stopping=time.monotonic()
                    if time.monotonic()-start>timeout:reason='timeout';stopping=time.monotonic()
                    if stopping is not None:
                        if not(OUT/'STOP').exists():(OUT/'STOP').write_text(reason+'\n')
                        process.send_signal(signal.SIGTERM)
                if stopping is not None and time.monotonic()-stopping>45:
                    os.killpg(process.pid,signal.SIGKILL)
                time.sleep(.5)
            code=process.wait()
        write(journal/(name+'.json'),dict(complete=code==0 and reason is None,actual_process_returncode=code,
            actual_wait_completed=True,elapsed_seconds=time.monotonic()-start,stop_reason=reason))
        if code or reason:raise RuntimeError(name+' exited '+str(code)+'; inspect '+str(journal/(name+'.log')))
        return code
    finally:
        if process is not None and process.poll()is None:
            process.terminate()
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for sig,handler in previous.items():signal.signal(sig,handler)


def make_cohort_plan(p,role):
    if role not in('train','validation','test'):raise ValueError(role)
    selected=('fit','selection')if role=='train'else(role,)
    cases=deepcopy([c for c in p['cases']if c['role']in selected])
    if role!='train':
        m=OUT/'motion_fit/selected.pt';candidate=OUT/'candidate_fit/selected.pt'
        for case in cases:
            case['config'].update(corrected_forecast_checkpoint=m.relative_to(ROOT).as_posix(),corrected_forecast_checkpoint_sha256=sha(m),
                candidate_transformer_checkpoint=candidate.relative_to(ROOT).as_posix(),candidate_transformer_checkpoint_sha256=sha(candidate))
    output=OUT/(role+'_native_plan.json')
    write(output,dict(schema='Cube_full_source_Native_cohort_v1',scope='cube_full_cycle_native',training_authorized=True,
        worker_module='our_mani.cube_full_cycle_20261111_v1.native_worker',historical_global_stop=p['historical_global_stop'],
        timeout_seconds=3600,cases=cases,formal_VAL_TEST=False,source_split=role,
        original_source_coverage=True,official_archived_TEST_resets=role=='test'))
    return output,cases


def native_summary(folder,cases):
    results=[]
    for c in cases:
        receipt=read(folder/c['id']/'actual_exit.json')
        if not receipt['complete']or not receipt['actual_wait_completed']or receipt['unreaped_descendants']:
            raise RuntimeError('Native owner not cleanly complete '+c['id'])
        r=read(folder/c['id']/'episode/result.json')
        results.append(dict(parent_id=c['parent_id'],task=c['task'],raw_episode_id=c['raw_episode_id'],variation=c['variation'],
            seed=c['seed'],success=r['episode_success'],actual_seconds=r['actual_ticks']*.05,
            first_success_time=r['native_task_judge']['first_success_time'],stop_reason=r['stop_reason'],
            result=pointer(folder/c['id']/'episode/result.json')))
    per_speed={}
    for kind in('ordinary','high_speed'):
        sub=[r for r in results if r['task'].endswith('high_speed')==(kind=='high_speed')]
        count=sum(r['success']for r in sub)
        per_speed[kind]=dict(successes=count,episodes=len(sub),success_rate=count/len(sub))
    return dict(episodes=len(results),successes=sum(r['success']for r in results),per_speed=per_speed,cases=results)


def run_cohort(p,role):
    from shutil import disk_usage
    verify(read(OUT/'native_sources.json'))
    path,cases=make_cohort_plan(p,role)
    folder=OUT/('collection_v1'if role=='train'else role);folder.mkdir(exist_ok=False)
    write(folder/'request.json',dict(source_count=len(cases),plan=pointer(path),fresh_Native_process_per_source=True,automatic_retry=False))
    completed=[]
    for index,c in enumerate(cases):
        stopped(p);verify(read(OUT/'native_sources.json'))
        if disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than8GiB free; preserve captures and stop before next Native source')
        status(role+'_Native',complete=False,completed_sources=index,total_sources=len(cases),current_source=c['parent_id'],
            current_task=c['task'],variation=c['variation'])
        name=role+'_'+str(index).zfill(3)
        run_child(p,name,[NATIVE_PYTHON,'-B','scripts/cube_full_native_owner_20261111_v1.py',
            '--plan',str(path),'--case-id',c['id'],'--output',str(folder/c['id'])],timeout=3700)
        owner=read(folder/c['id']/'actual_exit.json');closed=read(folder/c['id']/'episode/worker_exit.json')
        if not owner['complete']or owner['actual_process_returncode']!=0 or not owner['actual_wait_completed']or owner['unreaped_descendants']or not closed['native_closed']:
            raise RuntimeError('Native source shutdown/actual wait failed '+c['id'])
        completed.append(c['parent_id'])
        r=read(folder/c['id']/'episode/result.json')
        write(folder/(str(index).zfill(3)+'_done.json'),dict(source_id=c['parent_id'],success=r['episode_success'],owner_exit=pointer(folder/c['id']/'actual_exit.json')))
        status(role+'_Native',complete=False,completed_sources=index+1,total_sources=len(cases),last_source=c['parent_id'],last_success=r['episode_success'])
    receipt=dict(complete=True,sources=len(cases),all_case_owners_actual_waited=True,all_case_workers_closed=True,
        original_assigned_sources=completed,source_plan=pointer(path),automatic_retries=0)
    write(folder/'collection_receipt.json',receipt)
    return native_summary(folder,cases)


def verify_test_inputs(p):
    """After complete VAL only, hash every one of the80 released reset sources."""
    from collections import Counter
    original=p['official_test_hash_manifest']
    if sha(original['path'])!=original['sha256']:raise RuntimeError('Original verified TEST metadata changed')
    release=read(OUT/'validation_release.json')
    if not release['complete']or release['sources']!=40:raise RuntimeError('Full40 VAL release required')
    old=read(original['path']);all_rows=old['episodes'];episodes=[];native={}
    for c in [c for c in p['cases']if c['role']=='test']:
        stopped(p)
        matches=[r for r in all_rows if r['task']==c['task']and r['raw_episode_id']==c['raw_episode_id']]
        if len(matches)!=1:raise ValueError('Original TEST metadata row missing')
        row=deepcopy(matches[0])
        if row['variation']!=c['variation']or row['official_episode_index']!=c['official_episode_index']or Path(row['path']).resolve()!=Path(c['source_path']).resolve():
            raise ValueError('Official source identity/variation/index mismatch')
        required=row['source_required_files']
        if len(required)!=19:raise ValueError('All19 original reset/calibration/first-frame files required')
        for name,item in required.items():
            path=Path(row['path'])/name
            if path.resolve()!=Path(item['path']).resolve()or path.stat().st_size!=item['bytes']or sha(path)!=item['sha256']:
                raise RuntimeError('Original TEST file changed '+str(path))
        alias=Path(c['official_alias_path'])
        if not alias.is_symlink()or alias.resolve()!=Path(row['path']).resolve():raise RuntimeError('Author variation alias changed')
        row['official_alias_path']=str(alias);episodes.append(row)
        for suffix in('tasks/'+c['task']+'.py','task_ttms/'+c['task']+'.ttm'):
            path=Path(c['config']['native_repo'])/'third_party/RLBench/rlbench'/suffix
            native[str(path)]=sha(path)
    if len(episodes)!=80 or len({(r['task'],r['raw_episode_id'])for r in episodes})!=80:raise RuntimeError('Full80 unique TEST rows required')
    write(OUT/'test_input_manifest.json',dict(complete=True,episodes=episodes,native_inputs_sha256=native,
        source_metadata=original,full40_VAL_release=pointer(OUT/'validation_release.json'),TEST_data_used_for_fitting=False,
        all_TEST_payload_hashes_selected_after_completed_VAL=True))


def execute():
    p=plan()
    if read(OUT/'tests.json')['passed']is not True or read(OUT/'gpu_preflight.json')['complete']is not True:raise RuntimeError('Passed full-cycle checks and actual GPU preflight required')
    write(OUT/'pipeline_request.json',dict(user_authorization=p['user_authorization'],partition=pointer(PLAN),started_utc=utc(),
        old_queues_resumed=False,full_sources=dict(TRAIN=200,VAL=40,TEST=80),Native_GPU_overlap=False))
    sources=training_sources();write(OUT/'pipeline_sources.json',sources)
    try:
        verify(sources)
        from ..cube_native_contact_worker_20261104_v1.contracts import native_sources_for_plan
        native=native_sources_for_plan(ROOT/'config/framework/cube_native_contact_worker_20261104_v1/plan.json')
        write(OUT/'native_sources.json',native);verify(native);summaries={}
        summaries['TRAIN_collection']=run_cohort(p,'train');verify(sources)
        for stage in('motion-compile','motion-fit','public-replay','candidate-compile','candidate-fit','runtime-check'):
            run_child(p,stage,[NATIVE_PYTHON,'-B','-m','our_mani.cube_full_cycle_20261111_v1.pipeline','--stage',stage],timeout=14400)
            verify(sources)
        frozen=dict(frozen_utc=utc(),partition=pointer(PLAN),motion=pointer(OUT/'motion_fit/selected.pt'),
            candidate=pointer(OUT/'candidate_fit/selected.pt'),position_refiner=p['model_initialization']['position_refiner_checkpoint_sha256'],
            source_sha256=sources,selection_rule=p['validation_selection_rule'],test_payloads_read=False)
        write(OUT/'frozen_models_before_validation.json',frozen)
        summaries['validation']=run_cohort(p,'validation');verify(sources)
        write(OUT/'validation_release.json',dict(complete=True,sources=40,summary=summaries['validation'],
            frozen_models=pointer(OUT/'frozen_models_before_validation.json'),VAL_used_for_refitting=False))
        for key in('motion','candidate'):
            if sha(frozen[key]['path'])!=frozen[key]['sha256']:raise RuntimeError('Frozen model changed after VAL')
        write(OUT/'frozen_models_before_test.json',dict(frozen,validation_release=pointer(OUT/'validation_release.json'),validation_used_for_model_selection=False))
        verify_test_inputs(p)
        summaries['test']=run_cohort(p,'test');verify(sources);stopped(p)
        verify(read(OUT/'native_sources.json'))
        report=dict(complete=True,finished_utc=utc(),scope='PhysMani ordinary/high-speed Cube complete source cycle',
            task_time_limit_seconds=60,source_interpretation=p['source_interpretation'],
            source_coverage=dict(TRAIN=200,VAL=40,official_archived_TEST=80),training_partition=dict(fit=160,TRAIN_internal_selection=40),
            motion_training=read(OUT/'motion_fit/fit_status.json'),candidate_training=read(OUT/'candidate_fit/fit_status.json'),
            runtime_check=read(OUT/'runtime_check.json'),tests=read(OUT/'tests.json'),gpu_preflight=read(OUT/'gpu_preflight.json'),native=summaries,
            success_authority=p['success_authority'],model_pair_frozen_before_validation_and_test=True,
            validation_used_for_refitting=False,test_used_for_selection=False,official_archived_TEST_reset_coverage_complete=True,
            VAL_is_full_archived_scene_replay=False,original_paper_benchmark_comparable=False,old_models_preserved=True,
            default_deployed_pair_changed=False,source_unchanged=True,Native_GPU_overlap=False,old_queues_resumed=False)
        write(OUT/'report.json',report);status('complete',complete=True,report=pointer(OUT/'report.json'))
        print(json.dumps(dict(complete=True,validation=summaries['validation']['per_speed'],test=summaries['test']['per_speed'])),flush=True)
        return 0
    except BaseException as error:
        write(OUT/'pipeline_failure.json',dict(complete=False,type=type(error).__name__,error=str(error),automatic_retry=False))
        status('failed',complete=False,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute',action='store_true');group.add_argument('--stage')
    args=parser.parse_args()
    return execute()if args.execute else stage_worker(args.stage)

if __name__=='__main__':raise SystemExit(main())
