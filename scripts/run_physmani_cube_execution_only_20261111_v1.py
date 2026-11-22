"""Run the repaired Cube executor on generated TRAIN/development conditions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.task_time_limits_20261111_v1 import TaskBudget


def prepare(task,seed,output,seconds=None):
    budget=TaskBudget() if seconds is None else TaskBudget(seconds)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh output directory')
    plan=json.loads((ROOT/'config/framework/highspeed_control_20261111_v1/frozen_validation6_16s_v2.json').read_text())
    native='pick_moving_target_on_the_table'+('_high_speed' if task=='high_speed' else '')
    case=deepcopy(next(c for c in plan['cases'] if c['task']==native))
    case['seed']=case['config']['seed']=seed;case['id']=f'manual_{task}_{seed}'
    case['config'].update(task_time_limit_seconds=budget.seconds,budget_ticks=budget.ticks,
        evaluation_role='generated_TRAIN_manual_repaired_executor_run')
    case['config']['P4_collection_metadata']=dict(parent_id=case['id'],split='generated_TRAIN_development',official_VAL_or_TEST_source=False)
    plan['cases']=[case];plan.pop('cohort_contract',None)
    plan['worker_module']='our_mani.highspeed_control_20261111_v1.native_validation_v18'
    case['config']['active_controller_contract']['worker']='native_validation_v18'
    plan['time_limit_authorization']='user: all tasks including legacy16; expand16seconds to60seconds'
    plan['task_budget']=budget.receipt();plan['entrypoint']='scripts/run_physmani_cube.py'
    directory=ROOT/'runtime/framework/highspeed_control_20261111_v1/manual_requests'
    directory.mkdir(parents=True,exist_ok=True);path=directory/(uuid.uuid4().hex+'.json')
    with path.open('x') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2)
    return path,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('ordinary','high_speed'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-limit-seconds',type=float,default=TaskBudget().seconds)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();plan,output=prepare(args.task,args.seed,args.output,args.time_limit_seconds)
    print(json.dumps(dict(plan=str(plan),output=str(output),simulation_seconds=args.time_limit_seconds),ensure_ascii=False),flush=True)
    if args.prepare_only:return 0
    from scripts.validate_highspeed_control_20261111_v3 import supervise
    return supervise(plan,output)


if __name__=='__main__':raise SystemExit(main())
