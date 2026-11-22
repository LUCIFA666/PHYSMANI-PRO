"""Build a read-only partial historical lineage snapshot or audit a composite."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.acquisition_lineage import (SCHEMA,canonical_sha,pointer,read,local,
    audit_acquisition_lineage,command_argument,instant)

BASE=ROOT/'artifacts/framework/multitask_r10'


def parent_receipt_for_raw_child(root,record):
    child=local(root,record['parent_process_receipt']['child_case_path'])
    output=next((parent for parent in child.parents if parent.name=='output'),None)
    if output is None:raise ValueError('raw child is not in a preserved infrastructure attempt output')
    parent=output.parent/'parent_receipt.json'
    if not parent.exists():raise ValueError('canonical child awaits raw parent commit; retry this read-only snapshot later')
    return parent


def observed_manifest(root,registrations,train_parents,validation_parents):
    root=Path(root).resolve();plan_objects={}
    for registration in registrations:
        plan=read(local(root,registration['plan']['path']));plan_objects[plan['plan_sha256']]=(registration,plan)
    base_key=next(iter(plan_objects));base=plan_objects[base_key][1];infra={};completed=[]
    for key,(registration,plan) in plan_objects.items():
        collection=local(root,registration['collection'])
        for path in sorted((collection/'infrastructure_attempts').glob('*/*/parent_receipt.json')):
            record=read(path)
            if record.get('status')!='infrastructure_error':continue
            item=dict(actual_plan_sha256=key,parent_receipt=pointer(root,path),native_log=pointer(root,path.with_name('native.log')),
                started_utc=record['started_utc'],finished_utc=record['finished_utc'])
            infra.setdefault(record['case_id'],[]).append(item)
        for path in sorted((collection/'cases'/plan['split']).glob('*/*/case.json')):
            record=read(path);completed.append(dict(case_id=record['case_id'],actual_plan_sha256=key,
                canonical_receipt=pointer(root,path),parent_receipt=pointer(root,parent_receipt_for_raw_child(root,record))))
    for row in completed:
        row['preceding_infrastructure_attempts']=sorted(infra.get(row['case_id'],[]),
            key=lambda item:(instant(item['finished_utc']),item['parent_receipt']['path']))
    manifest=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),base_plan_sha256=base_key,
        original_TRAIN_parents=pointer(root,train_parents),original_VAL_parents=pointer(root,validation_parents),
        actual_plans=registrations,completed_acquisitions=completed,
        coverage_contract=dict(original_TRAIN_parents=1400,original_VAL_parents=275,minimum_completed_phase_parents=8),
        case_records_or_labels_rewritten=False)
    manifest['lineage_sha256']=canonical_sha(manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',help='audit an already constructed immutable composite manifest')
    parser.add_argument('--registrations',help='JSON actual_plans list; defaults to the originalv11 only')
    parser.add_argument('--base-plan',default=str(BASE/'TRAIN_full_collection_plan_v11.json'))
    parser.add_argument('--source-snapshot',default=str(BASE/'TRAIN_full_collection_v11_executed_sources/manifest.json'))
    parser.add_argument('--collection',default=str(BASE/'TRAIN_full_collection_v11'))
    parser.add_argument('--train-parents',default=str(ROOT/'artifacts/framework/protocol/train_manifest.json'))
    parser.add_argument('--validation-parents',default=str(ROOT/'artifacts/framework/protocol/validation_manifest.json'))
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--out',default=str(BASE/('historical_lineage_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))))
    args=parser.parse_args();output=local(ROOT,args.out)
    if output.exists():raise ValueError('fresh audit directory required; preserve prior receipts and snapshots')
    output.mkdir(parents=True)
    if args.manifest:
        manifest=read(local(ROOT,args.manifest));manifest_input=args.manifest
    else:
        registrations=read(local(ROOT,args.registrations)) if args.registrations else [dict(
            plan=pointer(ROOT,args.base_plan),source_snapshot=pointer(ROOT,args.source_snapshot),
            collection=str(local(ROOT,args.collection).relative_to(ROOT)))]
        manifest=observed_manifest(ROOT,registrations,args.train_parents,args.validation_parents)
        path=output/'observed_lineage_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        manifest_input=path
    report=audit_acquisition_lineage(ROOT,manifest_input,require_complete=args.require_complete)
    report['audit_module']=pointer(ROOT,ROOT/'our_mani/multitask/acquisition_lineage.py')
    report['audit_script']=pointer(ROOT,Path(__file__))
    path=output/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'report.txt').write_text('\n'.join([
        report['status'],f"Original fixed roster: {report['expected_acquisitions']}",
        f"Verified original completed receipts: {report['verified_completed_acquisitions']}",
        f"Missing/pending: {len(report['missing_case_ids'])}",
        'Historical plan/source archives, raw child/parent receipts, output hashes and preceding infra chains verified.',
        'Original records/labels are preserved. This is lineage acceptance only; no model/phase/VAL/TEST release.'])+'\n')
    print(json.dumps(dict(status=report['status'],verified=report['verified_completed_acquisitions'],expected=report['expected_acquisitions'],
        report=pointer(ROOT,path))))


if __name__=='__main__':main()
