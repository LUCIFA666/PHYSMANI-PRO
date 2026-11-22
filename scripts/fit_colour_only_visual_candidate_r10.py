#!/usr/bin/env python3
"""Run one fixed TRAIN-only colour candidate, without native/GPU execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.colour_only_candidate import run


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    base=ROOT/'artifacts/framework/multitask_r10'
    result=run(ROOT,ROOT/args.output,
        proof_path=base/'TRAIN_primary3000_counted_highcup100_bundle_v1/bundle_audit.json',
        annotation_path=base/'TRAIN_colour_visual_patch_annotation_candidate_v1/annotation.json',
        peer_path=base/'TRAIN_colour_visual_patch_independent_peer_v1/report.json',
        config_path=base/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json')
    print(json.dumps({k:result[k] for k in ('actual_fit_complete','total_colour_parent_sources',
        'crossfit_valid','crossfit_passed','macro_heldout_parent_correct_and_supported_fraction')}))
