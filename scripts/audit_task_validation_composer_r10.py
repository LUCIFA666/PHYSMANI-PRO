"""Prepare or audit a full-five-method task VAL composition; no native launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,digest,file_sha
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.task_release import DEFAULT_VAL,DEFAULT_EXPERIMENT,_canonical_task
from our_mani.multitask.task_validation_composer import COMPOSITION,finalize_task_composite_validation_release
from our_mani.multitask.validation_release import Proof


def pointer(name):
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('composition pointers must be existing files inside desktop project')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=tuple(TASK_SPECS))
    parser.add_argument('--out',required=True)
    parser.add_argument('--canonical-val-manifest',default=DEFAULT_VAL)
    parser.add_argument('--experiment-plan',default=DEFAULT_EXPERIMENT)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-composition',action='store_true')
    mode.add_argument('--composition')
    for field in ('ours-plan','baseline-plan','baseline-execution-contract','coverage','resource-budget-receipt'):
        parser.add_argument('--'+field)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        parser.error('choose a new output path under desktop multitask_r10')
    if args.prepare_composition:
        if any(getattr(args,key) is None for key in ('ours_plan','baseline_plan','baseline_execution_contract')):
            parser.error('preparation requires ours-plan, baseline-plan and baseline-execution-contract')
        canonical,_=_canonical_task(Proof(ROOT),args.canonical_val_manifest,'validation',args.task)
        result=dict(schema=COMPOSITION,task=args.task,ours_queue_plan=pointer(args.ours_plan),
            baseline_queue_plan=pointer(args.baseline_plan),baseline_execution_contract=pointer(args.baseline_execution_contract),
            canonical_validation_manifest=pointer(args.canonical_val_manifest),raw_identities_rewritten=False,
            expected_source_cases=len(canonical),expected_method_cases=5*len(canonical),
            resource_budget_receipt=None if args.resource_budget_receipt is None else pointer(args.resource_budget_receipt))
        result['sha256']=digest(result);atomic_json(out,result)
        print(json.dumps(dict(composition=pointer(out),formal_TEST_permitted=False,native_or_GPU_launched=False)))
    else:
        if args.coverage is None:parser.error('audit requires the actual task coverage certificate')
        result=finalize_task_composite_validation_release(ROOT,args.task,pointer(args.composition),pointer(args.coverage),out,
            canonical_validation_manifest=args.canonical_val_manifest,experiment_plan=args.experiment_plan)
        print(json.dumps(result,ensure_ascii=False))
        if not result.get('complete'):raise SystemExit(1)


if __name__=='__main__':main()
