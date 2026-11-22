"""Read-only PhysMani task VAL component; never starts native/GPU work."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.baseline_task_component import validate_baseline_task_component,COMPONENT
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.specs import TASK_SPECS


def pointer(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('component metadata pointer must stay inside project')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=tuple(TASK_SPECS))
    for key in ('plan','execution-contract','common-config','common-lock','out'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--canonical-val-manifest',default='artifacts/framework/protocol/validation_manifest.json')
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        parser.error('choose a new output path under desktop multitask_r10')
    try:
        result=validate_baseline_task_component(ROOT,args.task,pointer(args.plan),pointer(args.execution_contract),
            pointer(args.common_config),pointer(args.common_lock),pointer(args.canonical_val_manifest))
    except Exception as error:
        pending=dict(schema=COMPONENT,task=args.task,complete=False,baseline_task_component_reusable=False,
            errors=[repr(error)],formal_validation_release_issued=False,formal_TEST_permitted=False,
            infrastructure_errors_are_not_policy_failures=True,native_or_GPU_launched=False)
        atomic_json(out,pending);print(json.dumps(pending,ensure_ascii=False));raise SystemExit(1)
    atomic_json(out,result)
    print(json.dumps(dict(complete=True,task=args.task,sources=result['sources'],out=str(out.relative_to(ROOT)),
        actual_other_task_completed_snapshot=result['actual_other_task_completed_snapshot'],
        global_queue_complete_claimed=False,formal_TEST_permitted=False,native_or_GPU_launched=False),ensure_ascii=False))


if __name__=='__main__':main()
