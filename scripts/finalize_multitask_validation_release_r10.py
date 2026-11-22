"""Finalize complete frozen R10 VAL evidence; otherwise save a pending report."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from our_mani.multitask.validation_release import finalize_validation_release
from our_mani.multitask.evaluation import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-plan',required=True)
    parser.add_argument('--coverage-certificate',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--canonical-validation-manifest',default='artifacts/framework/protocol/validation_manifest.json')
    parser.add_argument('--experiment-plan',default='config/framework/multitask_r10_experiment_plan.json')
    args=parser.parse_args();path=(ROOT/args.coverage_certificate).resolve()
    result=finalize_validation_release(ROOT,args.queue_plan,
        dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path) if path.is_file() else None),args.out,
        canonical_validation_manifest=args.canonical_validation_manifest,experiment_plan=args.experiment_plan)
    print(json.dumps(dict(complete=result['complete'],formal_test_permitted=result['formal_test_permitted'],
        output=args.out if result['complete'] else result.get('pending_report'),
        errors=result.get('errors',[]),actual_R10_TEST_read=False)))
    if not result['complete']:raise SystemExit(2)


if __name__=='__main__':main()
