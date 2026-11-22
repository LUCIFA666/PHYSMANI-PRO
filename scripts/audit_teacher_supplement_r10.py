"""Audit complete fixed200 teacher data and optionally bind primary2800.

The bundle adds teacher replica1 to interaction fitting only. Context retains
its original primary K1 manifest. This command performs no native execution,
fitting, candidate selection, or formal policy release.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask import teacher_supplement as supplement


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def audit(auxiliary_manifest,out,primary_manifest=None):
    out=lineage.local(ROOT,out)
    lineage.require(out.is_relative_to(ROOT/'artifacts/framework/multitask_r10')
        and not out.exists(),'fresh project audit output directory required')
    aux=lineage.local(ROOT,auxiliary_manifest)
    aux_pointer=lineage.pointer(ROOT,aux)
    proof=supplement.audit_manifest(ROOT,aux,require_complete=True)
    out.mkdir(parents=True)
    write(out/'auxiliary_audit.json',proof)
    result=dict(schema='r10_complete_fixed_teacher_supplement_cli_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        auxiliary_manifest=aux_pointer,auxiliary_complete200=True,
        auxiliary_audit=lineage.pointer(ROOT,out/'auxiliary_audit.json'),
        original_Context_primary_K1_changed=False,native_or_GPU_started=False,
        fitting_performed=False,formal_policy_release=False)
    if primary_manifest is not None:
        primary=lineage.local(ROOT,primary_manifest)
        bundle=dict(version=supplement.BUNDLE_VERSION,
            primary_manifest=lineage.pointer(ROOT,primary),
            auxiliary_teacher_manifest=aux_pointer,primary_Context_view_unchanged=True)
        write(out/'bundle.json',bundle)
        bundled=supplement.audit_bundle(ROOT,out/'bundle.json',require_complete=True)
        write(out/'bundle_audit.json',bundled)
        result.update(bundle=lineage.pointer(ROOT,out/'bundle.json'),
            bundle_audit=lineage.pointer(ROOT,out/'bundle_audit.json'),
            primary_complete2800=True,combined_acquisitions=3000,
            original_TRAIN_parents=1400,teacher_rollouts=1600,
            original_Context_hold_rollouts=1400)
    lineage.require(lineage.pointer(ROOT,aux)==aux_pointer,
        'actual auxiliary manifest changed during complete-data audit')
    write(out/'report.json',result)
    return dict(report=lineage.pointer(ROOT,out/'report.json'),
        auxiliary_complete200=True,bundle=result.get('bundle'),
        formal_policy_release=False,native_or_GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auxiliary-manifest',required=True)
    parser.add_argument('--primary-manifest')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.auxiliary_manifest,args.out,args.primary_manifest),ensure_ascii=False))
