#!/usr/bin/env python3
"""Explicit TRAIN diagnostic with frozen weights and measured press/retract execution."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    from our_mani.executor_contract_repair_20261112_v6.common import OUT,read,write
    from our_mani.executor_contract_repair_20261112_v6.owner import run_case
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('push_moving_button','push_moving_button_high_speed'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh output directory')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    calibration=next(p for p in read(OUT/'calibration_lock.json')if f'/{args.task}/'in p)
    case=dict(task=args.task,seed=args.seed,variation=0,calibration_episode=calibration,
        role='TRAIN_executor_diagnostic',control_ticks=2,barrier_rate=5.,reference_frequency=6.,max_chunks=600,
        proposal_selection='feasible_bank',task_specific_execution_prior='explicit_button_press_cycle',
        checkpoint=str(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v1/head_fit/selected.pt'),
        checkpoint_sha256='65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af')
    write(output/'case.json',case)
    result,exit_receipt=run_case(output/'case.json',output/'Native_run',mode='frozen_policy')
    summary=dict(run_completed=exit_receipt['complete'],task_success=None if result is None else result['episode_success'],
        stop_reason=None if result is None else result['stop_reason'],actual_wait_completed=exit_receipt['actual_wait_completed'],
        formal_VAL_TEST=False,model_weights_changed=False,task_specific_execution_prior=True)
    write(output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if exit_receipt['complete']else 1

if __name__=='__main__':raise SystemExit(main())
