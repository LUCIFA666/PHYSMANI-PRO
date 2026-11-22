"""Separate fixed all-parent basketball replica1 TRAIN teacher worker.

Preparation/checking are CPU metadata only. Native execution requires a fresh
ROOT-reviewed registration; never modifies the original K1 roster or receipts.
The unchanged collector and accepted owned hooks execute each fixed case once.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
import collect_multitask_recovery_r10 as recovery
from our_mani.multitask import teacher_supplement as supplement

collection=recovery.collection
lineage=recovery.lineage
require=supplement.require
pointer=lambda path:lineage.pointer(ROOT,path)
read=lineage.read


def prepare(draft_report_path,setup,output):
    setup=collection.destination(setup);output=collection.destination(output)
    require(not setup.exists() and not output.exists(),'fresh auxiliary production-candidate setup/output required')
    draft_report=read(draft_report_path)
    draft=collection.verify_plan(recovery.checked(draft_report['plan']))
    proposal=read(recovery.checked(draft_report['proposal']))
    primary=collection.verify_plan(recovery.checked(draft_report['primary_plan']))
    reviewed=collection.verify_plan(recovery.checked(proposal['shared_profile_reference_plan']))
    require(draft['protocol'].get('draft') is True and draft.get('native_execution_ready') is False,
        'original unapproved draft must stay planning-only')
    require(draft['cases']==supplement.expected_cases(primary) and reviewed['cases']==primary['cases'],
        'exact all100 x2tasks replica1 derivation required')
    setup.mkdir()
    protocol=deepcopy(reviewed['protocol']);protocol.pop('recovery_input',None)
    protocol.update(draft=False,native_execution_authorized=True)
    protocol_path=setup/'protocol.json';collection.write_json(protocol_path,protocol)
    plan=deepcopy(draft)
    plan.update(created_utc=collection.utc(),protocol=protocol,protocol_path=str(protocol_path),
        auxiliary_teacher_scope=supplement.PLAN_SCOPE,native_execution_ready=True,
        auxiliary_collection=str(output.relative_to(ROOT)))
    inputs=dict(draft['inputs'])
    additions=[Path(draft_report_path).resolve(),recovery.checked(draft_report['source_archive']),protocol_path,
        ROOT/'artifacts/framework/multitask_r10/demonstrations/manifest.json',
        ROOT/'artifacts/framework/protocol/train_manifest.json',ROOT/'artifacts/framework/protocol/validation_manifest.json']
    additions.extend(ROOT/name for name in supplement.OWNED_SOURCES)
    for path in additions:
        inputs[str(path.resolve())]=collection.sha(path)
    for key in supplement.HOOKS:
        inputs.update(recovery.permission_dependencies(protocol[key]))
    plan['inputs']=inputs;plan.pop('plan_sha256',None);plan['plan_sha256']=collection.canonical_sha(plan)
    plan_path=setup/'plan.json';collection.write_json(plan_path,plan);collection.verify_plan(plan_path)
    entries=[]
    for name,checksum in sorted(inputs.items()):
        source=Path(name)
        if source.suffix not in ('.py','.sh'):
            continue
        raw=source.read_bytes();require(collection.sha(source)==checksum,'auxiliary source changed before archive')
        target=setup/'sources'/checksum/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        entries.append(dict(original_path=str(source),**pointer(target)))
    collection.verify_plan(plan_path)
    archive=setup/'source_snapshot.json'
    collection.write_json(archive,dict(version=lineage.SOURCE_SCHEMA,plan=dict(**pointer(plan_path),internal_sha256=plan['plan_sha256']),
        source_count=len(entries),entries=entries,all_plan_inputs_verified_before_and_after=True,
        source_files_modified=False,case_records_or_labels_modified=False))
    registry=dict(schema=supplement.REGISTRATION_SCHEMA,draft_preparation_report=pointer(draft_report_path),
        primary_plan=draft_report['primary_plan'],shared_profile_plan=proposal['shared_profile_reference_plan'],
        execution=dict(plan=pointer(plan_path),source_snapshot=pointer(archive),collection=str(output.relative_to(ROOT))),
        execution_ready=False,fixed_budget=deepcopy(supplement.BUDGET))
    registry_path=setup/'execution_registration_candidate.json';collection.write_json(registry_path,registry)
    context=supplement.verify_registration(ROOT,pointer(registry_path),live=True)
    report=dict(schema='r10_fixed_auxiliary_teacher_production_candidate_preparation_v1',plan=pointer(plan_path),
        source_snapshot=pointer(archive),execution_registration=pointer(registry_path),case_count=200,
        execution_ready=False,ROOT_first_native_job_review_required=True,native_or_GPU_started=False,
        original_draft_report=pointer(draft_report_path),draft_and_primary_outputs_modified=False,
        first_fixed_case_id=plan['cases'][0]['case_id'],original_primary_K1_and_Context_coverage_changed=False,
        actual_archived_source_count=len(context['current']['sources']))
    collection.write_json(setup/'preparation_report.json',report)
    return report


def approve_registration(candidate,new_path):
    """The caller's explicit ROOT review produces a new immutable registry."""
    context=supplement.verify_registration(ROOT,pointer(candidate),live=True)
    require(context['registry']['execution_ready'] is False,'candidate registry already approved')
    new_path=collection.destination(new_path)
    require(not new_path.exists() and not context['current']['collection'].exists(),
        'approve once before any auxiliary native output; never rewrite an old registry')
    registry=deepcopy(context['registry']);registry['execution_ready']=True
    collection.write_json(new_path,registry)
    supplement.verify_registration(ROOT,pointer(new_path),live=True,require_ready=True)
    return dict(execution_registration=pointer(new_path),native_or_GPU_started=False)


def collect_case(plan_path,case_id,output,guard_report,registration_path):
    context=supplement.verify_registration(ROOT,pointer(registration_path),live=True,require_ready=True)
    require(context['current']['path']==Path(plan_path).resolve() and case_id in context['expected_cases'],
        'CLI auxiliary plan/case differs from typed registration')
    case=context['expected_cases'][case_id]
    expected=context['current']['collection']/'infrastructure_attempts'/case_id/'0/output'
    require(Path(output).resolve()==expected,'auxiliary child needs its explicit single attempt0 output')
    out=collection.case_directory(output,case)
    require(not out.exists(),'no hidden child resume or automatic auxiliary retry')
    captures=dict(simulations=[],paths=[]);started=collection.utc();record=None;failure=None
    try:
        with recovery.scoped_owned_hooks(context['plan'],captures):
            record=collection.collect_native_case(plan_path,case_id,output,guard_report)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        hooks=dict(version=supplement.RUNNER_VERSION,execution_kind='fixed_auxiliary_TRAIN_teacher_replica1_first_execution',
            case_id=case_id,plan=context['current']['registration']['plan'],plan_sha256=context['plan']['plan_sha256'],
            execution_registration=context['registry_pointer'],primary_plan=context['registry']['primary_plan'],
            actual_runner=pointer(Path(__file__)),started_utc=started,finished_utc=collection.utc(),
            permissions={key:deepcopy(context['plan']['protocol'][key]) for key in supplement.HOOKS},
            reset_recovery_receipts=[deepcopy(getattr(env,'reset_recovery_receipt',None)) for env in captures['simulations']],
            RML_path_recovery_receipts=[deepcopy(path.recovery_receipt) for path in captures['paths']],error=failure,
            complete=bool(record is not None and failure is None),first_execution=True,is_infrastructure_recovery=False,
            previous_infrastructure_failure_required=False,outcome_selected=False,primary_outputs_modified=False,
            policy_failure_label_assigned=False,automatic_retry=False)
        hooks['complete']=bool(hooks['complete'] and len(hooks['reset_recovery_receipts'])==1
            and isinstance(hooks['reset_recovery_receipts'][0],dict) and hooks['reset_recovery_receipts'][0].get('complete') is True
            and all(path.get('close_complete') is True for path in hooks['RML_path_recovery_receipts']))
        collection.write_json(out/'teacher_supplement_hook_receipt.json',hooks)
    require(hooks['complete'],'auxiliary raw shared reset/RML witnesses did not close')
    base=out/'base_collector_receipt.json';(out/'case.json').rename(base)
    record=deepcopy(record)
    record['teacher_supplement_execution']=dict(version=supplement.RUNNER_VERSION,
        hook_receipt=pointer(out/'teacher_supplement_hook_receipt.json'),base_collector_receipt=pointer(base),
        execution_registration=context['registry_pointer'])
    for path in (base,out/'teacher_supplement_hook_receipt.json'):
        record['files_sha256'][str(path.relative_to(ROOT))]=collection.sha(path)
    collection.write_json(out/'case.json',record)
    supplement.validate_result(ROOT,out/'case.json',context,case)
    return record


def worker_command(plan_path,cid,output,guard_report,registration_path):
    return ['bash',str(ROOT/'scripts/run_local_sim.sh'),str(Path(__file__).resolve()),'--collect-case',
        '--plan',str(Path(plan_path).resolve()),'--case-id',cid,'--output',str(Path(output).resolve()),
        '--guard-report',str(Path(guard_report).resolve()),'--execution-registration',str(Path(registration_path).resolve())]


def run_queue(plan_path,output,guard_report,registration_path,*,case_ids=None,workers=2,case_timeout=900):
    context=supplement.verify_registration(ROOT,pointer(registration_path),live=True,require_ready=True)
    require(Path(plan_path).resolve()==context['current']['path'] and Path(output).resolve()==context['current']['collection'],
        'auxiliary parent plan/output differs from typed registration')
    require(type(workers) is int and 1<=workers<=8 and type(case_timeout) is int and case_timeout>0,
        'explicit CPU workers1..8 and positive wall timeout required')
    known=context['expected_cases']
    if case_ids is None:
        case_ids=list(known)
    require(case_ids and len(case_ids)==len(set(case_ids)) and set(case_ids)<=set(known),'fixed auxiliary queue IDs must be unique planned cases')
    output=collection.destination(output);gate=collection.require_r09_release(guard_report)
    for parent in (output/'infrastructure_attempts').glob('*/0/parent_receipt.json'):
        require(read(parent).get('status')!='infrastructure_error','unresolved auxiliary infra requires an explicit new registration/lineage; no automatic retry or continuation')
    selected=[case for cid,case in known.items() if cid in set(case_ids)]
    for case in selected:
        canonical=collection.case_directory(output,case)/'case.json'
        attempt_root=output/'infrastructure_attempts'/case['case_id']
        require(canonical.exists() or not attempt_root.exists(),
            'unfinished/inflight auxiliary attempt must be reviewed before dispatch; no automatic retry')
    def one(case):
        canonical=collection.case_directory(output,case)/'case.json'
        if canonical.exists():
            supplement.validate_result(ROOT,canonical,context,case)
            parent=output/'infrastructure_attempts'/case['case_id']/'0/parent_receipt.json'
            require(parent.exists() and read(parent).get('status')=='completed','auxiliary completed case lacks parent commit')
            return dict(case_id=case['case_id'],status='resumed_verified')
        if collection.COLLECTION_STOP.exists():
            return dict(case_id=case['case_id'],status='cancelled_stop_marker',policy_failure_label_assigned=False)
        attempt=output/'infrastructure_attempts'/case['case_id']/'0'
        require(not attempt.parent.exists(),'no automatic auxiliary retry of an unfinished/inflight case')
        attempt.mkdir(parents=True);attempt_output=attempt/'output'
        command=worker_command(plan_path,case['case_id'],attempt_output,guard_report,registration_path)
        started=collection.utc();status='infrastructure_error';detail='';code=None;pid=None
        try:
            with (attempt/'native.log').open('wb') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                        PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                pid=process.pid
                try:
                    code=process.wait(timeout=case_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid,signal.SIGKILL);process.wait()
                    detail='auxiliary wall timeout; partial evidence retained without policy label'
            if code==0:
                fresh=supplement.verify_registration(ROOT,pointer(registration_path),live=True,require_ready=True)
                child=collection.case_directory(attempt_output,case)/'case.json'
                record=deepcopy(supplement.validate_result(ROOT,child,fresh,case))
                require(type(pid) is int and pid>0 and record['process_id']!=pid,'independent native Python child and parent Popen PIDs required')
                launcher=ROOT/'scripts/run_local_sim.sh'
                record['parent_process_receipt']=dict(launcher=str(launcher.relative_to(ROOT)),launcher_sha256=collection.sha(launcher),
                    process_id=pid,command=command,child_case_path=str(child.relative_to(ROOT)),child_case_sha256=collection.sha(child))
                collection.write_json(canonical,record);status='completed'
        except Exception as error:
            detail=''.join(traceback.format_exception(error))
        receipt=dict(version=supplement.RUNNER_VERSION,case_id=case['case_id'],status=status,detail=detail,returncode=code,
            process_id=pid,process_started=pid is not None,started_utc=started,finished_utc=collection.utc(),command=command,attempt=0,
            policy_failure_label_assigned=False,gate=gate,execution_registration=context['registry_pointer'],
            actual_plan=context['current']['registration']['plan'],automatic_retries=False)
        collection.write_json(attempt/'parent_receipt.json',receipt)
        return receipt
    results=[];iterator=iter(selected);inflight={};stopped=False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(workers,len(selected))):
            case=next(iterator);inflight[pool.submit(one,case)]=case['case_id']
        while inflight:
            done,_=wait(inflight,return_when=FIRST_COMPLETED)
            for future in done:
                inflight.pop(future)
                try:result=future.result()
                except Exception as error:result=dict(status='infrastructure_error',detail=repr(error),policy_failure_label_assigned=False)
                results.append(result)
                if result['status'] not in ('completed','resumed_verified'):
                    stopped=True
            if not stopped:
                for _ in done:
                    case=next(iterator,None)
                    if case is not None:inflight[pool.submit(one,case)]=case['case_id']
    proof=supplement.audit_collection(ROOT,pointer(registration_path),require_complete=False)
    collection.write_json(output/'manifest.json',proof['acquisition_projection'],overwrite=True)
    report=dict(version=supplement.RUNNER_VERSION,created_utc=collection.utc(),receipts=results,workers=workers,
        stopped_dispatch_after_first_infrastructure_error=stopped,inflight_jobs_allowed_to_finish=True,
        expected=200,completed=proof['acquisition_projection']['completed'],complete=proof['complete'],
        pending_case_ids=proof['acquisition_projection']['missing_case_ids'],automatic_retries=False,
        execution_registration=context['registry_pointer'],auxiliary_component_only=True,primary_K1_modified=False,GPU_used=False)
    name=collection.canonical_sha(report)[:16]+'.json';collection.write_json(output/'queue_reports'/name,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','approve-registration','check','collect-case','run-queue'):
        actions.add_argument('--'+action,action='store_true')
    parser.add_argument('--draft-report',type=Path);parser.add_argument('--setup',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--execution-registration',type=Path);parser.add_argument('--approved-registration',type=Path)
    parser.add_argument('--case-id');parser.add_argument('--queue-case-id',action='append')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--case-timeout',type=int,default=900)
    parser.add_argument('--guard-report',type=Path,default=ROOT/'artifacts/framework/aligned_r09/final_report.json')
    args=parser.parse_args()
    def need(*keys):
        for key in keys:
            if getattr(args,key) is None:parser.error('required --'+key.replace('_','-'))
    if args.prepare:
        need('draft_report','setup','output');value=prepare(args.draft_report,args.setup,args.output)
    elif args.approve_registration:
        need('execution_registration','approved_registration');value=approve_registration(args.execution_registration,args.approved_registration)
    elif args.check:
        need('execution_registration');context=supplement.verify_registration(ROOT,pointer(args.execution_registration),live=True)
        value=dict(valid=True,fixed_cases=200,execution_ready=context['registry']['execution_ready'],native_or_GPU_started=False)
    elif args.collect_case:
        need('plan','case_id','output','execution_registration')
        record=collect_case(args.plan,args.case_id,args.output,args.guard_report,args.execution_registration)
        value=dict(case_id=record['case_id'],status=record['status'])
    else:
        need('plan','output','execution_registration');value=run_queue(args.plan,args.output,args.guard_report,args.execution_registration,
            case_ids=args.queue_case_id,workers=args.workers,case_timeout=args.case_timeout)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
