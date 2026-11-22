"""Fixed TRAIN ring rollout with calibrated grasp and live command frames."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_joint16_v174_r10 import parser
from our_mani.multitask.task_executor_registry import driver,verify as verify_registry
from our_mani.multitask.live_grasp_executor_native import LiveGraspBackend,verify
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args=parser().parse_args()
    if args.method not in ('cv_geometry','context_geometry'):raise ValueError('geometry methods only')
    module=driver(args.task);prepared=module.preflight(args);verify_registry(ROOT,prepared);overlay=verify(ROOT,prepared)
    checked_subset(ROOT,prepared['files'],{str(Path(__file__)):file_sha(__file__)})
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=prepared['identity'],moving_command_executor=overlay)))
        return
    record=module.run_prepared(prepared,backend=LiveGraspBackend(args.task))
    atomic_json(prepared['out']/'live_grasp_entrypoint.json',dict(complete=True,entrypoint_sha256=file_sha(__file__),
        moving_command_executor=overlay,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],case=str(prepared['out']/'case.json'))))


if __name__=='__main__':main()
