#!/usr/bin/env python3
"""Run the explicit repaired executor profile on a fresh generated TRAIN scene."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from our_mani.executor_contract_repair_20261112_v4.common import OUT, read, write
    from our_mani.executor_contract_repair_20261112_v4.owner import run_case
    calibrations=read(OUT/'calibration_lock.json')
    tasks=sorted({Path(p).parts[-5]for p in calibrations})
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=tasks,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh output directory')
    parser.add_argument('--mode',choices=('frozen_policy','robot_probe'),default='frozen_policy')
    parser.add_argument('--max-chunks',type=int,default=120,help='Policy horizon; 120 chunks = 60 physical seconds')
    parser.add_argument('--proposal-selection',choices=('feasible_bank','primary'),default='feasible_bank')
    args=parser.parse_args()
    if not 1<=args.max_chunks<=120:parser.error('--max-chunks must be 1..120')
    calibration=next(p for p in calibrations if f'/{args.task}/'in p)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    case=dict(task=args.task,seed=args.seed,variation=0,calibration_episode=calibration,
              role='TRAIN_executor_diagnostic',max_chunks=args.max_chunks,
              proposal_selection=args.proposal_selection,
              checkpoint=str(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v1/head_fit/selected.pt'),
              checkpoint_sha256='65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af')
    write(output/'case.json',case)
    result,receipt=run_case(output/'case.json',output/'Native_run',mode=args.mode)
    summary=dict(run_completed=receipt['complete'],task_success=None if result is None else result.get('episode_success'),
                 probe_passed=None if result is None else result.get('probe_passed'),
                 stop_reason=None if result is None else result.get('stop_reason'),
                 actual_wait_completed=receipt['actual_wait_completed'],formal_VAL_TEST=False)
    write(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if receipt['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
