"""CPU-only per-task model coverage, five-way VAL release and TEST admission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,verify_inputs,file_sha
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.task_release import (audit_task_model_coverage,
    finalize_task_validation_release,finalize_task_composite_validation_release,
    require_task_test_release,DEFAULT_TRAIN,DEFAULT_VAL,DEFAULT_TEST,DEFAULT_EXPERIMENT)


def read(name):return json.loads((ROOT/name).read_text())
def pointer(name):
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('release input must remain under desktop project')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('coverage','finalize-val','finalize-component-val','check-test'))
    parser.add_argument('--task',required=True,choices=tuple(TASK_SPECS))
    parser.add_argument('--config')
    parser.add_argument('--frozen-inputs')
    parser.add_argument('--context-training-view')
    parser.add_argument('--queue-plan')
    parser.add_argument('--composition')
    parser.add_argument('--coverage-certificate')
    parser.add_argument('--validation-release')
    parser.add_argument('--out',required=True)
    parser.add_argument('--train-manifest',default=DEFAULT_TRAIN)
    parser.add_argument('--validation-manifest',default=DEFAULT_VAL)
    parser.add_argument('--test-manifest',default=DEFAULT_TEST)
    parser.add_argument('--experiment-plan',default=DEFAULT_EXPERIMENT)
    args=parser.parse_args()
    output=(ROOT/args.out).resolve()
    if not output.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or output.exists():
        parser.error('use a new output path in desktop multitask_r10')
    if args.operation=='coverage':
        if not args.config or not args.context_training_view:parser.error('coverage needs --config and --context-training-view')
        lock=read(args.frozen_inputs)['files'] if args.frozen_inputs else None
        if lock is not None:verify_inputs(ROOT,lock)
        result=audit_task_model_coverage(ROOT,args.task,read(args.config),context_training_view=pointer(args.context_training_view),
            train_source_manifest=args.train_manifest,validation_source_manifest=args.validation_manifest,frozen_files=lock)
        atomic_json(output,result)
    elif args.operation=='finalize-val':
        if not args.queue_plan or not args.coverage_certificate:parser.error('finalize-val needs --queue-plan and --coverage-certificate')
        result=finalize_task_validation_release(ROOT,args.task,args.queue_plan,pointer(args.coverage_certificate),output,
            canonical_validation_manifest=args.validation_manifest,experiment_plan=args.experiment_plan)
    elif args.operation=='finalize-component-val':
        if not args.composition or not args.coverage_certificate:parser.error('finalize-component-val needs --composition and --coverage-certificate')
        result=finalize_task_composite_validation_release(ROOT,args.task,pointer(args.composition),pointer(args.coverage_certificate),output,
            canonical_validation_manifest=args.validation_manifest,experiment_plan=args.experiment_plan)
    else:
        if not args.config or not args.frozen_inputs or not args.validation_release:parser.error('check-test needs config, lock and validation-release pointer JSON')
        lock=read(args.frozen_inputs)['files'];verify_inputs(ROOT,lock)
        result=require_task_test_release(ROOT,args.task,pointer(args.validation_release),read(args.config),lock,
            official_test_manifest=args.test_manifest)
        atomic_json(output,result)
    print(json.dumps(dict(operation=args.operation,task=args.task,out=str(output.relative_to(ROOT)),
        passed=result.get('formal_model_coverage_passed',result.get('protocol_acceptance_passed',result.get('passed',False))),
        native_or_GPU_launched=False),ensure_ascii=False))


if __name__=='__main__':main()
