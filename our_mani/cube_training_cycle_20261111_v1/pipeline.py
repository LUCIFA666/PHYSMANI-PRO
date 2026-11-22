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
    stopped(p);status(name,complete=False)
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


def evaluation_plan(p,role):
    if role not in('validation','test'):raise ValueError(role)
    cases=deepcopy([c for c in p['cases']if c['role']==role])
    m=OUT/'motion_fit/selected.pt';candidate=OUT/'candidate_fit/selected.pt'
    for case in cases:
        case['config'].update(corrected_forecast_checkpoint=m.relative_to(ROOT).as_posix(),corrected_forecast_checkpoint_sha256=sha(m),
            candidate_transformer_checkpoint=candidate.relative_to(ROOT).as_posix(),candidate_transformer_checkpoint_sha256=sha(candidate))
    output=OUT/(role+'_plan.json')
    write(output,dict(schema='Cube_train_cycle_generated_evaluation_v1',scope='cube_train_cycle_native',training_authorized=True,
        worker_module='our_mani.cube_training_cycle_20261111_v1.evaluation_worker',historical_global_stop=p['historical_global_stop'],
        timeout_seconds=14400,cases=cases,models_frozen_before_evaluation=True,formal_VAL_TEST=False,
        role=role,private_TRAIN_supervision_produced=False))
    return output


def native_summary(folder,cases):
    if not read(folder/'actual_exit.json')['complete']:raise RuntimeError('Actual Native owner incomplete')
    results=[]
    for c in cases:
        r=read(folder/c['id']/'result.json')
        results.append(dict(parent_id=c['parent_id'],task=c['task'],seed=c['seed'],success=r['episode_success'],
            actual_seconds=r['actual_ticks']*.05,first_success_time=r['native_task_judge']['first_success_time']))
    per_speed={}
    for kind in('ordinary','high_speed'):
        sub=[r for r in results if r['task'].endswith('high_speed')==(kind=='high_speed')]
        count=sum(r['success']for r in sub)
        per_speed[kind]=dict(successes=count,episodes=len(sub),success_rate=count/len(sub))
    return dict(episodes=len(results),successes=sum(r['success']for r in results),per_speed=per_speed,cases=results)


def execute():
    p=plan()
    from .datasets import assert_collection_closed
    assert_collection_closed()
    write(OUT/'pipeline_request.json',dict(user_authorization=p['user_authorization'],partition=pointer(PLAN),started_utc=utc(),
        old_queues_resumed=False,official_archived_VAL_TEST_used=False,Native_GPU_overlap=False))
    sources=training_sources();write(OUT/'pipeline_sources.json',sources)
    try:
        verify(sources)
        for stage in('motion-compile','motion-fit','public-replay','candidate-compile','candidate-fit','runtime-check'):
            run_child(p,stage,[NATIVE_PYTHON,'-B','-m','our_mani.cube_training_cycle_20261111_v1.pipeline','--stage',stage])
            verify(sources)
        frozen=dict(frozen_utc=utc(),partition=pointer(PLAN),motion=pointer(OUT/'motion_fit/selected.pt'),
            candidate=pointer(OUT/'candidate_fit/selected.pt'),position_refiner=p['model_initialization']['position_refiner_checkpoint_sha256'],
            source_sha256=sources,selection_rule=p['validation_selection_rule'],test_payloads_read=False)
        write(OUT/'frozen_models_before_validation.json',frozen)
        summaries={}
        for role in('validation','test'):
            verify(sources)
            for k in('motion','candidate'):
                if sha(frozen[k]['path'])!=frozen[k]['sha256']:raise RuntimeError('Frozen model changed')
            if role=='test':write(OUT/'frozen_models_before_test.json',dict(frozen,validation_read_for_weight_selection=False,validation=pointer(OUT/'validation/actual_exit.json')))
            path=evaluation_plan(p,role)
            run_child(p,role,[NATIVE_PYTHON,'-B','scripts/cube_cycle_native_owner_20261111_v1.py','--plan',str(path),'--output',str(OUT/role)])
            summaries[role]=native_summary(OUT/role,[c for c in p['cases']if c['role']==role])
        summaries['TRAIN_collection']=native_summary(OUT/'collection_v1',[c for c in p['cases']if c['role']in('fit','selection')])
        verify(sources)
        report=dict(complete=True,finished_utc=utc(),scope='ordinary/high-speed Cube only',task_time_limit_seconds=60,
            motion_training=read(OUT/'motion_fit/fit_status.json'),candidate_training=read(OUT/'candidate_fit/fit_status.json'),
            runtime_check=read(OUT/'runtime_check.json'),tests=read(OUT/'tests.json'),gpu_preflight=read(OUT/'gpu_preflight.json'),native=summaries,success_authority=p['success_authority'],
            model_pair_frozen_before_validation_and_test=True,validation_used_for_refitting=False,test_used_for_selection=False,
            official_archived_VAL_TEST_evaluated=False,small_generated_sample=True,old_models_preserved=True,default_deployed_pair_changed=False,
            source_unchanged=True,global_STOP_preserved=True,Native_GPU_overlap=False,old_queues_resumed=False)
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
