"""Read-only TRAIN2800/VAL275 committed-count preflight, safe during collection.

This is an honest snapshot, not a composite manifest or an array/model release.
Use a fresh --out on every invocation. Strict lineage/array audits still follow.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as audit
from our_mani.multitask.validation_lineage import verify_validation_execution_registration


def inventory(plans,split):
    return sorted((str(path),audit.file_sha(path)) for plan in plans
        for path in (plan['collection']/'cases'/split).glob('*/*/case.json'))


def count_commits(plans,roster,split,files):
    initial=inventory(plans,split);by_collection={str(plan['collection']):plan for plan in plans}
    known={};verified={};pending=[];errors=[];counts=Counter()
    for name,checksum in initial:
        path=Path(name)
        try:
            record=audit.read(path);cid=record.get('case_id')
            audit.require(cid in roster and cid not in known,'unknown or duplicate completed identity across actual plans')
            known[cid]=name;case=roster[cid]
            plan=next(value for prefix,value in by_collection.items() if path.is_relative_to(Path(prefix)))
            audit.require(path==plan['collection']/'cases'/split/case['task']/cid/'case.json'
                and record.get('status')=='completed' and record.get('plan_sha256')==plan['plan']['plan_sha256']
                and record.get('split')==split and record.get('teacher_actions_used') is case['teacher_used'],
                'canonical actual plan/task/behavior namespace differs')
            process=record['parent_process_receipt'];output=Path(audit.command_argument(process['command'],'--output')).resolve()
            parent_path=output.parent/'parent_receipt.json'
            if not parent_path.is_file():pending.append(dict(case_id=cid,reason='canonical published; raw parent commit still pending'));continue
            parent=audit.read(parent_path)
            if parent.get('status')!='completed':pending.append(dict(case_id=cid,reason='raw parent is not committed completed',status=parent.get('status')));continue
            audit.verify_attempt_parent(ROOT,audit.pointer(ROOT,parent_path),plan,cid,files,status='completed')
            audit.require(parent['command']==process['command'] and parent['process_id']==process['process_id'],
                'completed parent/process binding differs')
            raw=audit.checked_pointer(ROOT,dict(path=process['child_case_path'],sha256=process['child_case_sha256']),files)
            audit.require(audit.read(raw)=={key:value for key,value in record.items() if key!='parent_process_receipt'},
                'canonical changed its actual raw child')
            launcher=str(audit.local(ROOT,process['launcher']))
            audit.require(launcher in plan['sources'] and plan['sources'][launcher]['sha256']==process['launcher_sha256'],
                'historical launcher is not bound to its actual source archive')
            audit.checked_pointer(ROOT,dict(path=name,sha256=checksum),files)
            verified[cid]=dict(path=name,sha256=checksum,actual_plan_sha256=plan['plan']['plan_sha256'])
            counts[(case['task'],case['behavior'])]+=1
        except (ValueError,KeyError,OSError,StopIteration,TypeError) as error:
            errors.append(dict(path=name,error_type=type(error).__name__,error=str(error)))
    final=inventory(plans,split)
    return dict(expected=len(roster),canonical_snapshot=len(initial),committed_and_raw_verified=len(verified),
        stable_canonical_inventory=initial==final,all_commits_present=len(verified)==len(roster),
        pending_raw_parent_commits=pending,errors=errors,missing_case_ids=sorted(set(roster)-set(verified)),
        per_task_behavior_counts={task:{behavior:count for (name,behavior),count in counts.items() if name==task}
            for task in sorted({case['task'] for case in roster.values()})},verified_committed_receipts=verified)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-registrations',required=True)
    parser.add_argument('--validation-registration',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();out=audit.local(ROOT,args.out)
    audit.require(not out.exists(),'fresh count-preflight output directory required; never overwrite earlier snapshots')
    files={};registrations_path=audit.local(ROOT,args.train_registrations)
    registrations=audit.read(audit.checked_pointer(ROOT,audit.pointer(ROOT,registrations_path),files))
    plans=[];by_sha={}
    for index,registration in enumerate(registrations):
        current=audit.load_archived_plan(ROOT,registration,files);key=current['plan']['plan_sha256']
        audit.require(key not in by_sha,'duplicate actual TRAIN plan registration')
        if index:
            previous=by_sha[registration['predecessor_plan_sha256']]
            audit.verify_recovery_transition(ROOT,previous,current,files)
        by_sha[key]=current;plans.append(current)
    audit.require(plans and len(plans[0]['plan']['cases'])==2800,'full original TRAIN2800 roster required')
    val_context=verify_validation_execution_registration(ROOT,args.validation_registration,files)
    train=audit.read(audit.local(ROOT,val_context['registry']['original_TRAIN_parents']['path']))
    val=audit.read(audit.local(ROOT,val_context['registry']['original_VAL_parents']['path']))
    audit.validate_fixed_roster(plans[0]['plan'],train,val)
    audit.require(all(plan['plan']['cases']==plans[0]['plan']['cases'] for plan in plans),'actual TRAIN roster differs across recovery plans')
    train_roster={case['case_id']:case for case in plans[0]['plan']['cases']}
    val_roster={case['case_id']:case for case in val_context['actual']['plan']['cases']}
    started=datetime.now(timezone.utc).isoformat()
    train_counts=count_commits(plans,train_roster,'train',files)
    val_counts=count_commits([val_context['actual']],val_roster,'validation',files)
    source_and_read_files_stable=all(Path(name).is_file() and audit.file_sha(name)==checksum for name,checksum in files.items())
    ready=source_and_read_files_stable and all(row['stable_canonical_inventory'] and row['all_commits_present']
        and not row['pending_raw_parent_commits'] and not row['errors'] for row in (train_counts,val_counts))
    result=dict(schema='r10_read_only_full_acquisition_commit_count_preflight_v1',snapshot_started_utc=started,
        snapshot_finished_utc=datetime.now(timezone.utc).isoformat(),ready_for_strict_complete_lineage_audits=ready,
        expected_TRAIN_cases=2800,expected_VAL_cases=275,TRAIN=train_counts,VAL=val_counts,
        actual_plan_source_archives_and_transitions_verified=True,source_and_read_files_stable=source_and_read_files_stable,
        full_output_hashes_or_array_validation_proven=False,scientific_context_or_phase_coverage_proven=False,
        strict_require_complete_TRAIN_and_VAL_audits_still_required=True,full_live_composite_constructed=False,
        native_or_GPU_or_fitting_started=False,counts_do_not_filter_native_success_or_censored_attempts=True,
        verified_files_sha256=files)
    out.mkdir(parents=True);path=out/'report.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(ready_for_strict_audits=ready,TRAIN_committed=train_counts['committed_and_raw_verified'],
        VAL_committed=val_counts['committed_and_raw_verified'],report=audit.pointer(ROOT,path))))
    if args.require_complete and not ready:raise SystemExit(2)


if __name__=='__main__':main()
