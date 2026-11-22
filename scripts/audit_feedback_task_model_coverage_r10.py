"""Audit actual feedback-policy models on CPU; never collect or optimize."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.feedback_model_coverage import audit_feedback_task_model_coverage
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.task_release import DEFAULT_TRAIN,DEFAULT_VAL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=tuple(TASK_SPECS))
    parser.add_argument('--config',required=True)
    parser.add_argument('--context-training-view',required=True)
    parser.add_argument('--train-manifest',default=DEFAULT_TRAIN)
    parser.add_argument('--validation-manifest',default=DEFAULT_VAL)
    parser.add_argument('--frozen-inputs')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        parser.error('preserve prior evidence; choose a new output under desktop multitask_r10')
    config_path=(ROOT/args.config).resolve();config_sha=file_sha(config_path)
    protocol=json.loads(config_path.read_text())
    files=None if args.frozen_inputs is None else json.loads((ROOT/args.frozen_inputs).read_text())['files']
    view=(ROOT/args.context_training_view).resolve()
    result=audit_feedback_task_model_coverage(ROOT,args.task,protocol,
        context_training_view=dict(path=str(view.relative_to(ROOT)),sha256=file_sha(view)),
        train_source_manifest=args.train_manifest,validation_source_manifest=args.validation_manifest,frozen_files=files)
    if file_sha(config_path)!=config_sha:
        result['formal_model_coverage_passed']=False
        result['errors'].append(dict(component='config_stability',error='actual configuration changed during coverage audit'))
    atomic_json(out,result)
    print(json.dumps(dict(task=args.task,formal_model_coverage_passed=result['formal_model_coverage_passed'],
        errors=result['errors'],report=dict(path=str(out.relative_to(ROOT)),sha256=file_sha(out)),
        native_or_GPU_launched=False),ensure_ascii=False))
    if not result['formal_model_coverage_passed']:raise SystemExit(1)


if __name__=='__main__':main()
