"""Fit development interaction/colour/layout artifacts from verified TRAIN only.

All fit thresholds are explicit in --fit-config. This entry never starts a
simulator, learns from VAL/TEST, replaces the dynamics/critic trainers or marks
the formal comparison ready. The existing 1400-source offline action-phase
prior and actual generated TRAIN parent coverage are reported separately.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.artifact_fitting import fit_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collector-manifest',required=True)
    parser.add_argument('--demonstrations',required=True)
    parser.add_argument('--original-prior',required=True)
    parser.add_argument('--fit-config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10'):
        parser.error('new fitting outputs must remain in desktop multitask_r10')
    report=fit_artifacts(ROOT,args.collector_manifest,args.demonstrations,args.original_prior,args.fit_config,out)
    print(json.dumps(dict(status=report['status'],formal_comparison_ready=False,
        native_TRAIN_parent_sources=report['native_TRAIN_parent_sources'],
        original_action_phase_prior_sources=report['original_action_phase_prior_binding']['original_TRAIN_source_count'],
        report=str(out/'report.json')),ensure_ascii=False))


if __name__=='__main__': main()
