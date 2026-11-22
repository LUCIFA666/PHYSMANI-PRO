"""Original fixed TRAIN high-peg episodes with reference4 reserve and unchanged native6."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_joint16_v174_r10 import parser
from our_mani.multitask.task_executor_registry import driver,verify as verify_registry
from our_mani.multitask.cup_receding_bridge_v485 import CupRecedingBridgeBackend,verify
from our_mani.multitask.lossless_case_archive import run_archived_case
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args=parser().parse_args()
    if args.method not in ('cv_geometry','context_geometry'):raise ValueError('two original geometry methods')
    module=driver(args.task);prepared=module.preflight(args);verify_registry(ROOT,prepared);overlay=verify(ROOT,prepared)
    checked_subset(ROOT,prepared['files'],{str(Path(__file__)):file_sha(__file__)})
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=prepared['identity'],cup_receding_bridge_v485=overlay)))
        return
    record=module.run_prepared(prepared,backend=CupRecedingBridgeBackend(args.task),run_case=run_archived_case)
    atomic_json(prepared['out']/'cup_receding_bridge_entrypoint_v486.json',dict(complete=True,entrypoint_sha256=file_sha(__file__),
        cup_receding_bridge_v485=overlay,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],case=str(prepared['out']/'case.json'))))


if __name__=='__main__':main()
