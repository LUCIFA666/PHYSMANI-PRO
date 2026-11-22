"""TRAIN-only original rollout with common reference constraint authority."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_task_geometry_r10 import driver
from our_mani.multitask.task_geometry_executor_native import BUTTONS, verify_overlay
from our_mani.multitask.task_constraint_executor_native import TaskConstraintBackend, verify
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args = driver(BUTTONS[0]).parser().parse_args()
    if args.method == 'physmani':
        raise ValueError('unchanged author baseline has its own executor')
    selected = driver(args.task); prepared = selected.preflight(args)
    checked_subset(ROOT,prepared['files'],{str(Path(__file__)):file_sha(__file__)})
    verify_overlay(ROOT,prepared); overlay = verify(ROOT,prepared)
    if args.check:
        print(json.dumps(dict(complete=True,check_only=True,native_started=False,
            GPU_started=False,identity=prepared['identity'],task_constraint_executor=overlay)))
        return
    record = selected.run_prepared(prepared,backend=TaskConstraintBackend(args.task))
    atomic_json(prepared['out']/'task_constraint_entrypoint.json',dict(complete=True,
        entrypoint_sha256=file_sha(__file__),task_constraint_executor=overlay,
        native_success=record['official_task_success'],formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],
        case=str(prepared['out']/'case.json'))))


if __name__ == '__main__':
    main()
