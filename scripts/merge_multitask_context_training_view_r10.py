"""Verify fixed TRAIN/VAL acquisitions and expose a complete causal context view."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.training_view import merge_training_view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest',action='append',required=True)
    parser.add_argument('--validation-manifest',action='append',required=True)
    parser.add_argument('--train-sources',default='artifacts/framework/protocol/train_manifest.json')
    parser.add_argument('--validation-sources',default='artifacts/framework/protocol/validation_manifest.json')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10'):
        parser.error('all new outputs must stay inside desktop multitask_r10')
    report=merge_training_view(ROOT,args.train_manifest,args.validation_manifest,
        args.train_sources,args.validation_sources,out)
    print(json.dumps(dict(training_view_ready=report['training_view_ready'],
        formal_comparison_ready=False,coverage_errors=report['coverage_errors'],
        report=str(out/'report.json')),ensure_ascii=False))
    if not report['training_view_ready']: raise SystemExit(2)


if __name__=='__main__': main()
