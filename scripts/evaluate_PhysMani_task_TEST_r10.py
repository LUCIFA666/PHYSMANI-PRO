"""New normal-basketball official TEST entrypoint; no VAL permission reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import baseline_task_test as test
from our_mani.multitask import baseline_component as component
from our_mani.multitask.evaluation import atomic_json,digest,file_sha,verify_inputs,validate_completed_case,case_identity
from scripts import evaluate_multitask_case_r10 as evaluator

VERSION='r10_PhysMani_official_task_TEST_wrapper_v1'
PLAN='r10_PhysMani_original560_normal_basketball_task40_TEST_queue_v1'
require=component.require


def read(path):return json.loads(Path(path).read_text())
def absolute(path):return (ROOT/path).resolve()
def pointer(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=file_sha(path))


def collect_case(args,*,backend_factory=None,runner=None,release_checker=None):
    prepared=test.metadata_preflight(args)
    reference=pointer(absolute(args.execution_contract))
    verified=test.load_TEST_contract(ROOT,reference,prepared)
    backend=(backend_factory or test.TestProjectedBaselineBackend)(verified=verified,execution_contract=reference,root=ROOT)
    out=prepared['out'];out.mkdir(parents=True,exist_ok=True);jit_receipt=None
    native_requested=False
    try:
        backend.start_jit_observer(prepared)
        test.authorize_task_TEST(prepared,checker=release_checker)
        # CPU model proof is complete. Only now can run_case request official
        # reset/policy. It still rechecks every actual lock/config input.
        native_requested=True
        (runner or evaluator.run_case)(prepared,backend=backend)
        backend._imports()
        jit_receipt=backend.close_jit_observer()
        state_receipt=backend.finalize_runtime_state()
        require(backend.witness is not None and backend.witness.get('complete') is True,
            'TEST projected runtime/import witness missing')
        evaluator.verify_preflight(prepared)
        case=out/'case.json';base=read(case);validate_completed_case(base,prepared['identity'])
        require(base.get('entrypoint_sha256')==file_sha(evaluator.__file__),'original TEST evaluator source changed')
        original=out/'evaluate_run_case_receipt.json';require(not original.exists(),'never overwrite prior TEST base receipt')
        case.rename(original)
        record=deepcopy(base)
        record['baseline_task_TEST_execution']=dict(schema=VERSION,execution_contract=reference,
            actual_wrapper=pointer(__file__),original_evaluator=pointer(evaluator.__file__),
            actual_full560_manifest=pointer(prepared['manifest_path']),selected_original_task_sources=40,
            full_three_method_task_VAL_release=deepcopy(prepared['r10_validation_release']),
            base_run_case_receipt=pointer(original),base_run_case_inputs=pointer(out/'inputs.json'),
            dependency_witness_after_cleanup=deepcopy(backend.witness),
            stock_JIT_execution=jit_receipt,mutable_runtime_state_execution=state_receipt,
            actual_split='test',VAL_permission_used_for_TEST=False,development_reset_hook_used=False,
            original_record_fields_rewritten=False,TEST_fitting_calibration_or_source_selection_used=False)
        atomic_json(case,record);return record
    except BaseException as error:
        try:jit_receipt=backend.close_jit_observer(error)
        except BaseException as cleanup_error:
            atomic_json(out/'TEST_JIT_cleanup_error.json',dict(error=repr(cleanup_error),complete=False))
        atomic_json(out/'baseline_task_TEST_infrastructure_receipt.json',dict(schema=VERSION,
            identity=prepared['identity'],error_class=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            native_environment_requested=native_requested,actual_wrapper=pointer(__file__),execution_contract=reference,
            stock_JIT_execution=jit_receipt,runtime_state=backend.infrastructure_runtime_state(),
            complete=False,counted_as_policy_failure=False,automatic_retry=False))
        raise


def prepare_plan(config_path,manifest_path,lock_path,contract_path,*,plan_path,output):
    config_path,manifest_path,lock_path,contract_path=map(absolute,(config_path,manifest_path,lock_path,contract_path))
    config,manifest,lock,contract=map(read,(config_path,manifest_path,lock_path,contract_path))
    require(manifest_path==ROOT/test.OFFICIAL_MANIFEST,'TEST plan must reference original full560 manifest')
    rows=test.official_task_rows(manifest,test.NORMAL_BASKETBALL)
    test.structural_release(config,test.NORMAL_BASKETBALL,lock['files'])
    require(contract.get('schema')==test.CONTRACT and contract.get('approved_before_TEST_execution') is True
        and contract.get('task')==test.NORMAL_BASKETBALL and contract.get('split')=='test'
        and contract.get('original_task_rows_sha256')==digest(rows),'new TEST contract/entire40 roster required')
    verify_inputs(ROOT,lock['files']);files=component.normalized(ROOT,lock['files'])
    for path in (config_path,manifest_path,contract_path,Path(__file__),Path(test.__file__),Path(evaluator.__file__)):
        require(files.get(str(path.resolve()))==file_sha(path),'TEST plan executable/input not frozen')
    jobs=[]
    for row in rows:
        identity=case_identity(split='test',task=row['task'],raw_episode_id=row['raw_episode_id'],method='physmani',
            source_row=row,config_sha256=file_sha(config_path),inputs_sha256=file_sha(lock_path),replica=0)
        jobs.append(dict(job_id=identity['sha256'][:20],row=row,identity=identity,replica=0,exploration_seed=None,record_demo=False))
    value=dict(version=PLAN,created_utc=datetime.now(timezone.utc).isoformat(),split='test',task=test.NORMAL_BASKETBALL,
        config=pointer(config_path),manifest=pointer(manifest_path),frozen_inputs=pointer(lock_path),
        execution_contract=pointer(contract_path),parent_source=pointer(__file__),child_source=pointer(__file__),
        output=str(absolute(output).relative_to(ROOT)),full_official_manifest_sources=560,expected_cases=40,
        methods=['physmani'],jobs=jobs,case_isolation='fresh_simulator_process_per_episode',
        TEST_fitting_calibration_or_selection_used=False,development_reset_hook_used=False,automatic_retry=False)
    value['sha256']=digest(value);atomic_json(absolute(plan_path),value);return value


def verify_plan(plan_path):
    path=absolute(plan_path);plan=read(path)
    require(plan.get('version')==PLAN and plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan['split']=='test' and plan['task']==test.NORMAL_BASKETBALL
        and plan['expected_cases']==len(plan['jobs'])==40 and plan['full_official_manifest_sources']==560
        and plan['methods']==['physmani'] and plan['TEST_fitting_calibration_or_selection_used'] is False
        and plan.get('case_isolation')=='fresh_simulator_process_per_episode'
        and plan.get('development_reset_hook_used') is False and plan.get('automatic_retry') is False,
        'exact full560/original normal-task40 TEST plan required')
    for key in ('config','manifest','frozen_inputs','execution_contract','parent_source','child_source'):
        component.pointer(ROOT,plan[key])
    require(plan['parent_source']==plan['child_source']==pointer(__file__),'new TEST queue source not frozen')
    require(absolute(plan['manifest']['path'])==ROOT/test.OFFICIAL_MANIFEST,'TEST queue manifest must remain original full560')
    require(absolute(plan['output']).is_relative_to(ROOT/'artifacts/framework/multitask_r10'),
        'TEST outputs must stay in the owned artifacts namespace')
    manifest=read(ROOT/plan['manifest']['path']);rows=test.official_task_rows(manifest,test.NORMAL_BASKETBALL)
    require([job['row'] for job in plan['jobs']]==rows,'TEST plan omitted/replaced/reordered original40')
    config=read(ROOT/plan['config']['path']);lock=read(ROOT/plan['frozen_inputs']['path']);verify_inputs(ROOT,lock['files'])
    test.structural_release(config,test.NORMAL_BASKETBALL,lock['files'])
    for job in plan['jobs']:
        identity=case_identity(split='test',task=job['row']['task'],raw_episode_id=job['row']['raw_episode_id'],
            method='physmani',source_row=job['row'],config_sha256=plan['config']['sha256'],
            inputs_sha256=plan['frozen_inputs']['sha256'],replica=0)
        require(job['identity']==identity and job['job_id']==identity['sha256'][:20]
            and job['replica']==0 and job['exploration_seed'] is None and job['record_demo'] is False,
            'TEST job identity/exploration/source changed')
    return plan


def worker_command(plan,job,out):
    return ['bash','scripts/run_aligned_gpu.sh',plan['child_source']['path'],'--collect-case',
        '--config',plan['config']['path'],'--manifest',plan['manifest']['path'],
        '--frozen-inputs',plan['frozen_inputs']['path'],'--execution-contract',plan['execution_contract']['path'],
        '--split','test','--task',job['row']['task'],'--raw-episode-id',job['row']['raw_episode_id'],
        '--method','physmani','--replica','0','--out',str(out.relative_to(ROOT))]


def run_queue(plan_path,workers,*,case_timeout=3600):
    plan=verify_plan(plan_path);require(type(workers) is int and 1<=workers<=4,'one to four TEST workers required')
    require(type(case_timeout) is int and case_timeout>0,'positive TEST infrastructure wall timeout required')
    stop_file=ROOT/'runtime/framework/multitask_r10_STOP'
    output=ROOT/plan['output'];output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'queue_owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A stopped attempt remains immutable. Resume is only for already
        # canonical completed rows, never an infrastructure retry.
        for job in plan['jobs']:
            require(not (output/'cases'/job['job_id']).exists()
                or (output/'results'/(job['job_id']+'.json')).is_file(),
                'no retry of prior incomplete TEST attempt')
            require(not (output/'queue_errors'/(job['job_id']+'.json')).exists(),
                'prior TEST queue error must remain infrastructure; no retry')
        stopped=threading.Event();lock=threading.Lock();launch_lock=threading.Lock();outcomes={};running=set()
        def publish():
            with lock:
                status=dict(schema=PLAN,plan_sha256=plan['sha256'],expected=40,
                    completed=sum(r['status']=='completed' for r in outcomes.values()),
                    errors=[r for r in outcomes.values() if r['status']=='infrastructure_error'],
                    complete=len(outcomes)==40 and all(r['status']=='completed' for r in outcomes.values()) and not stop_file.exists(),
                    results=list(outcomes.values()),running=sorted(running),
                    remaining_unattempted=40-len(outcomes)-len(running),
                    dispatch_stopped_on_infrastructure=stopped.is_set(),stop_requested=stop_file.exists(),automatic_retry=False)
                atomic_json(output/'status.json',status,overwrite=True)
            return status
        def one(job):
            jid=job['job_id'];base=output/'cases'/jid;canonical=output/'results'/(jid+'.json')
            command=None;process=None
            try:
                if canonical.exists():
                    record=read(canonical);validate_completed_case(record,job['identity']);parent=record['parent_case_process_receipt']
                    raw=component.pointer(ROOT,dict(path=parent['child_path'],sha256=parent['child_sha256']))
                    require(read(raw)=={k:v for k,v in record.items() if k!='parent_case_process_receipt'},'TEST readonly resume chain changed')
                    require(parent['command']==worker_command(plan,job,raw.parent)
                        and parent['plan_sha256']==plan['sha256'] and parent['returncode']==0,
                        'TEST readonly resume actual parent command/plan/exit changed')
                    result=dict(job_id=jid,status='completed',resumed_read_only=True)
                else:
                    if stopped.is_set() or stop_file.exists():return
                    verify_plan(plan_path)
                    with launch_lock:
                        if stopped.is_set() or stop_file.exists():return
                        require(not base.exists(),'no retry of prior incomplete TEST attempt')
                        base.mkdir(parents=True);out=base/'attempt0000';command=worker_command(plan,job,out)
                        with (base/'native.log').open('xb') as stream:
                            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        with lock:running.add(jid)
                    publish()
                    try:code=process.wait(timeout=case_timeout)
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        raise RuntimeError('TEST child infrastructure wall timeout; physical10s budget unchanged')
                    require(code==0 and (out/'case.json').is_file(),'TEST child did not produce complete original raw')
                    record=read(out/'case.json');validate_completed_case(record,job['identity'])
                    require(type(record['process_id']) is int and record['process_id']>0,
                        'actual child PID must be positive; launcher and Python PID are independent')
                    record['parent_case_process_receipt']=dict(child_path=str((out/'case.json').relative_to(ROOT)),
                        child_sha256=file_sha(out/'case.json'),child_process_id=record['process_id'],launcher_process_id=process.pid,
                        command=command,plan_sha256=plan['sha256'],parent_source_sha256=file_sha(__file__),
                        returncode=code,queue_process_id=os.getpid(),
                        log_path=str((base/'native.log').relative_to(ROOT)),log_sha256=file_sha(base/'native.log'))
                    verify_plan(plan_path);atomic_json(canonical,record);result=dict(job_id=jid,status='completed')
            except BaseException:
                with launch_lock:stopped.set()
                error=dict(job_id=jid,command=command,launcher_process_id=None if process is None else process.pid,
                    returncode=None if process is None else process.returncode,traceback=traceback.format_exc(),
                    counted_as_policy_failure=False,automatic_retry=False)
                atomic_json(output/'queue_errors'/(jid+'.json'),error)
                result=dict(job_id=jid,status='infrastructure_error')
            with lock:
                running.discard(jid);outcomes[jid]=result
            publish()
        todo=iter(plan['jobs'])
        publish()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active={pool.submit(one,next(todo)) for _ in range(workers)}
            while active:
                done,active=wait(active,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                if not stopped.is_set() and not stop_file.exists():
                    for _ in done:
                        job=next(todo,None)
                        if job is not None:active.add(pool.submit(one,job))
        return publish()


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--collect-case',action='store_true');mode.add_argument('--prepare-plan',action='store_true')
    mode.add_argument('--check',action='store_true');mode.add_argument('--run-queue',action='store_true')
    mode.add_argument('--prepare-contract',action='store_true')
    for key in ('config','manifest','frozen-inputs','execution-contract','plan','output','out','raw-episode-id','original-VAL-plan','ROOT-review'):
        parser.add_argument('--'+key)
    parser.add_argument('--split',default='test');parser.add_argument('--task',default=test.NORMAL_BASKETBALL)
    parser.add_argument('--method',default='physmani');parser.add_argument('--replica',type=int,default=0)
    parser.add_argument('--exploration-seed',type=int);parser.add_argument('--record-demo',action='store_true')
    parser.add_argument('--demo-only',action='store_true');parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--case-timeout',type=int,default=3600)
    args=parser.parse_args()
    if args.prepare_contract:
        result=test.prepare_TEST_runtime_contract(ROOT,args.config,args.manifest,args.frozen_inputs,args.original_VAL_plan,
            output=args.output,ROOT_review=args.ROOT_review)
        print(json.dumps(result));return
    if args.collect_case:result=collect_case(args);print(json.dumps(dict(complete=result['complete'])));return
    if args.prepare_plan:
        result=prepare_plan(args.config,args.manifest,args.frozen_inputs,args.execution_contract,plan_path=args.plan,output=args.output)
        print(json.dumps(dict(plan_sha256=result['sha256'],cases=40)));return
    if args.check:result=verify_plan(args.plan);print(json.dumps(dict(plan_sha256=result['sha256'],cases=40,native_started=False)));return
    result=run_queue(args.plan,args.workers,case_timeout=args.case_timeout);print(json.dumps(result))
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
