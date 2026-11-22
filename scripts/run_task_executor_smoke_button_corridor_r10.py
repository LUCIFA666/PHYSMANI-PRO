"""Original fixed TRAIN button rollout with contact-path descent admission."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_button_reference_v388_r10 as original
from our_mani.multitask.button_corridor_executor_native import ButtonCorridorBackend,verify
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args=original.parser().parse_args()
    if args.method not in ('cv_geometry','context_geometry'):raise ValueError('original geometry methods only')
    prepared=original.preflight(args)
    checked_subset(ROOT,prepared['files'],{str(Path(__file__)):file_sha(__file__)})
    overlay=verify(ROOT,prepared)
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,
            identity=prepared['identity'],button_corridor_executor=overlay)))
        return
    record=original.run_prepared(prepared,backend=ButtonCorridorBackend())
    atomic_json(prepared['out']/'button_corridor_entrypoint.json',dict(complete=True,
        entrypoint_sha256=file_sha(__file__),button_corridor_executor=overlay,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],case=str(prepared['out']/'case.json'))))


if __name__=='__main__':main()
