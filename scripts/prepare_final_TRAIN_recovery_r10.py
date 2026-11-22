"""Freeze the final infrastructure-only roster after the original queue ends.

This prepares a new plan, source archive and registration. It never executes
native work, replaces a canonical case, changes physics permissions, selects
task outcomes, or automatically retries a failed recovery.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from scripts import collect_multitask_recovery_r10 as recovery
from scripts import freeze_recovery_registration_r10 as freezer
from scripts.audit_acquisition_lineage_r10 import observed_manifest


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def live_plan_processes(plan_paths):
    """Inspect actual argv, including orphan workers, rather than a lock file."""
    paths={str(Path(path).resolve()) for path in plan_paths};found=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            argv=[x.decode(errors='replace') for x in (directory/'cmdline').read_bytes().split(b'\0') if x]
            raw=(directory/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if fields[0]=='Z' or '--plan' not in argv:continue
            index=argv.index('--plan')
            if index+1>=len(argv):continue
            cwd=(directory/'cwd').resolve()
            plan=str((cwd/argv[index+1]).resolve())
            if plan not in paths:continue
            if not any(name in arg for arg in argv for name in
                    ('collect_multitask_training_r10.py','collect_multitask_recovery_r10.py')):continue
            found.append(dict(pid=int(directory.name),state=fields[0],start_ticks=fields[19],argv=argv))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def require_terminal(report,snapshot,live):
    a.require(not live,'original TRAIN/recovery native queue or worker is still alive; no final roster yet')
    a.require(snapshot['original_queue_pid_after']['alive'] is True
        and snapshot['original_queue_pid_after']['identity_matches'] is True,
        'an independently observed actual live-generation snapshot is required')
    a.require(report.get('expected')==report.get('selected_expected')==2800
        and report.get('unchanged_full_plan_subset') is False
        and report.get('teacher_training_only') is True and report.get('GPU_used') is False
        and report.get('automatic_retries') is False and len(report.get('receipts',[]))==2800
        and all(row.get('status') in ('completed','resumed_verified','infrastructure_error')
            for row in report['receipts'])
        and a.instant(report['created_utc'])>=a.instant(snapshot['finished_utc']),
        'full2800 terminal parent report must postdate the last verified live generation; stale/subset reports forbidden')


def known_RML_failure(parent,log):
    terminal=[line.strip() for line in log.splitlines()
        if re.match(r'^[\w.]+(?:Error|Exception):',line)]
    return (parent.get('status')=='infrastructure_error' and parent.get('returncode')==1
        and parent.get('policy_failure_label_assigned') is False
        and terminal==['RuntimeError: [RML错误] 超过最大步数 10000，强制退出']
        and '_get_rml_handle' in log)


def require_new_recovery_cases(missing,previously_declared):
    a.require(not set(missing)&set(previously_declared),
        'a previously registered recovery remains incomplete; separate cause review and explicit new lineage required, no automatic retry')


def prepare(registrations_path,last_live_snapshot,template_plan_path,out,collection):
    registrations_path=a.local(ROOT,registrations_path);registrations=a.read(registrations_path)
    snapshot_path=a.local(ROOT,last_live_snapshot);snapshot=a.read(snapshot_path)
    original=registrations[0]
    # This early guard has no writes and does not hash the entire dataset.
    live=live_plan_processes([a.local(ROOT,row['plan']['path']) for row in registrations])
    a.require(not live,'original TRAIN/recovery native queue or worker is still alive; no final roster yet')
    queue_path=a.local(ROOT,original['collection'])/'queue_report.json'
    terminal_pointer=a.pointer(ROOT,queue_path)
    terminal=a.read(queue_path);require_terminal(terminal,snapshot,live)
    out=a.local(ROOT,out);collection=a.local(ROOT,collection)
    a.require(out.is_relative_to(ROOT/'artifacts/framework/multitask_r10')
        and collection.is_relative_to(ROOT/'artifacts/framework/multitask_r10')
        and not out.exists() and not collection.exists(),'fresh final recovery setup and output required')
    files={};contexts=[]
    for registration in registrations:
        current=a.load_archived_plan(ROOT,registration,files)
        if 'predecessor_plan_sha256' in registration:
            previous=next(c for c in contexts if c['plan']['plan_sha256']==registration['predecessor_plan_sha256'])
            a.verify_recovery_transition(ROOT,previous,current,files)
        contexts.append(current)
    train=ROOT/'artifacts/framework/protocol/train_manifest.json'
    validation=ROOT/'artifacts/framework/protocol/validation_manifest.json'
    manifest=observed_manifest(ROOT,registrations,train,validation)
    proof=a.audit_acquisition_lineage(ROOT,manifest,require_complete=False)
    missing=sorted(proof['missing_case_ids'])
    a.require(missing,'original2800 is already complete; no empty recovery plan needed')
    previously_declared=set()
    for current in contexts[1:]:
        payload=a.read(a.checked_pointer(ROOT,current['plan']['protocol']['recovery_input'],files))
        previously_declared.update(payload['case_ids'])
    require_new_recovery_cases(missing,previously_declared)
    a.require(not live_plan_processes([current['path'] for current in contexts]),'a registered native worker appeared during final snapshot')
    raw_failures=[];logs=[];selected=[];base=contexts[0]
    by_id={row['case_id']:row for row in base['plan']['cases']}
    for cid in missing:
        a.require(all(not (current['collection']/'cases/train'/by_id[cid]['task']/cid/'case.json').exists()
            for current in contexts),'refusing to replace a canonical original/recovery case')
        parents=sorted((base['collection']/'infrastructure_attempts'/cid).glob('*/parent_receipt.json'))
        a.require(parents,'unattempted case cannot be labelled an infrastructure failure')
        accepted=[]
        for path in parents:
            parent=a.read(path);log_path=path.with_name('native.log')
            a.require(log_path.is_file() and known_RML_failure(parent,log_path.read_text(errors='replace')),
                'final roster contains an unresolved error outside the accepted RML symptom: '+cid)
            ptr=a.pointer(ROOT,path)
            a.verify_attempt_parent(ROOT,ptr,base,cid,files,status='infrastructure_error')
            raw_failures.append(ptr);accepted.append(ptr);logs.append(a.pointer(ROOT,log_path))
        selected.append(dict(case_id=cid,task=by_id[cid]['task'],parent_source_group=by_id[cid]['parent_source_group'],
            original_fixed_case=by_id[cid],failed_parent_receipts=accepted))
    a.require(a.pointer(ROOT,queue_path)==terminal_pointer,'terminal report changed during final snapshot')
    template=recovery.collection.verify_plan(a.local(ROOT,template_plan_path))
    for relative in recovery.OWNED_SOURCES:
        source=str((ROOT/relative).resolve())
        a.require(template['inputs'].get(source)==a.file_sha(source),'accepted recovery executable changed: '+relative)
    from our_mani.multitask.development_reset_recovery import load_permission
    physical=load_permission(template['protocol']['development_reset_profile'],root=ROOT,
        native_repo=template['protocol']['baseline_repo'])
    physical_inputs=[]
    for name,checksum in physical['native_inputs_sha256'].items():
        path=(Path(template['protocol']['baseline_repo'])/name).resolve()
        a.require(template['inputs'].get(str(path))==checksum and a.file_sha(path)==checksum,
            'accepted native physical reset source changed or was never bound: '+str(path))
        physical_inputs.append(path)
    out.mkdir(parents=True)
    write(out/'original_registration.json',original)
    write(out/'terminal_lineage_snapshot.json',manifest)
    write(out/'terminal_lineage_audit.json',proof)
    selection=dict(schema='r10_final_post_terminal_known_RML_infrastructure_roster_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),original_terminal_report=terminal_pointer,
        last_actual_live_snapshot=a.pointer(ROOT,snapshot_path),actual_registration_list=a.pointer(ROOT,registrations_path),
        prior_lineage=a.pointer(ROOT,out/'terminal_lineage_snapshot.json'),strict_completed=len(manifest['completed_acquisitions']),
        fixed_case_ids=missing,selected_cases=selected,failed_parent_receipts=raw_failures,native_logs=logs,
        future_failures_admitted=False,task_outcomes_used_for_selection=False,automatic_retry=False,
        original_or_recovery_canonical_replaced=False,native_or_GPU_started=False)
    write(out/'fixed_remaining_roster.json',selection)
    payload=dict(schema=recovery.INPUT_SCHEMA,created_utc=selection['created_utc'],original_plan=original['plan'],
        failed_parent_receipts=raw_failures,case_ids=missing,
        development_reset_profile=template['protocol']['development_reset_profile'],
        RML_factory_hook=template['protocol']['RML_factory_hook'],infrastructure_only=True,
        source_snapshots=[row['source_snapshot'] for row in registrations],
        reason='One fixed remaining known-RML roster after actual original full2800 queue terminal. Preserve all original parents, seeds, variations, budgets, failures and canonical outcomes; no future admission, scene resampling or automatic retry.')
    write(out/'recovery_input.json',payload)
    plan=recovery.prepare_recovery_plan(original,physical_permission=payload['development_reset_profile'],
        rml_permission=payload['RML_factory_hook'],recovery_input=a.pointer(ROOT,out/'recovery_input.json'),
        protocol_output=out/'protocol.json',plan_output=out/'plan.json',extra_inputs=physical_inputs)
    frozen=freezer.freeze(out/'original_registration.json',out/'plan.json',collection,out/'frozen')
    context=recovery.verify_execution(out/'plan.json',out/'frozen/execution_registration.json',output=collection,case_ids=missing)
    new_registrations=registrations+[context['current']['registration']]
    write(out/'registrations_with_final_recovery.json',new_registrations)
    result=dict(schema='r10_final_TRAIN_infrastructure_recovery_preparation_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),plan=a.pointer(ROOT,out/'plan.json'),
        execution_registration=a.pointer(ROOT,out/'frozen/execution_registration.json'),
        fixed_roster=a.pointer(ROOT,out/'fixed_remaining_roster.json'),registrations=a.pointer(ROOT,out/'registrations_with_final_recovery.json'),
        fixed_case_ids=missing,preparation_script=a.pointer(ROOT,Path(__file__)),
        original_cases=len(plan['cases']),native_or_GPU_started=False,first_native_job_review_pending=True)
    result['source_snapshot']=a.pointer(ROOT,out/'frozen/source_snapshot.json')
    write(out/'preparation_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registrations',required=True)
    parser.add_argument('--last-live-snapshot',required=True)
    parser.add_argument('--accepted-recovery-plan',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--collection',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.registrations,args.last_live_snapshot,args.accepted_recovery_plan,args.out,args.collection),ensure_ascii=False))
