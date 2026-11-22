"""Publish full fixed R10 official TEST/70 demos, optionally index old R09."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from our_mani.multitask.test_report import finalize_official_test,combine_published_blocks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-plan');parser.add_argument('--out',required=True)
    parser.add_argument('--official-manifest',default='artifacts/framework/multitask_r10/official_test_manifest.json')
    parser.add_argument('--combined-out')
    parser.add_argument('--combine-existing',action='store_true',help='Recheck an already published --out and create only the combined index')
    args=parser.parse_args()
    if args.combine_existing:
        if args.queue_plan or not args.combined_out:parser.error('--combine-existing requires --combined-out and no --queue-plan')
        combined=combine_published_blocks(ROOT,args.out,args.combined_out)
        print(json.dumps(dict(complete=True,report=args.out,combined_index=args.combined_out,
            formal_run_count_only=combined['formal_run_count_only'],demos=combined['demo_completed'],
            native_or_GPU_launched=False,model_fitting_performed=False),ensure_ascii=False))
        return
    if not args.queue_plan:parser.error('--queue-plan is required unless --combine-existing is selected')
    report=finalize_official_test(ROOT,args.queue_plan,args.out,official_manifest=args.official_manifest)
    result=dict(complete=report['complete'],report=args.out if report['complete'] else report.get('pending_report'),
        errors=report.get('errors',[]),native_or_GPU_launched=False,model_fitting_performed=False)
    if report['complete'] and args.combined_out:
        combined=combine_published_blocks(ROOT,args.out,args.combined_out)
        result.update(combined_index=args.combined_out,formal_run_count_only=combined['formal_run_count_only'],demos=combined['demo_completed'])
    print(json.dumps(result,ensure_ascii=False))
    if not report['complete']:raise SystemExit(2)


if __name__=='__main__':main()
