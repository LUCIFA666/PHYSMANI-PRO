"""Actual shared-core TRAIN entrypoint using each task's frozen rollout driver."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.multitask.task_geometry_executor_native import BUTTONS, TASKS, TaskGeometryBackend, verify_overlay
from our_mani.multitask.evaluation import file_sha, atomic_json
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def driver(task):
    if task not in TASKS:
        raise ValueError('unsupported task geometry pilot')
    if task in BUTTONS:
        from scripts import run_task_executor_smoke_button_reference_v388_r10 as module
    else:
        from scripts import run_task_executor_smoke_bin_physical_v239_r10 as module
    return module


def main():
    args = driver(BUTTONS[0]).parser().parse_args()
    if args.method == 'physmani':
        raise ValueError('this ablation changes our geometry executor only')
    selected = driver(args.task)
    prepared = selected.preflight(args)
    checked_subset(ROOT, prepared['files'], {str(Path(__file__)):file_sha(__file__)})
    overlay = verify_overlay(ROOT, prepared)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False,
            GPU_started=False, identity=prepared['identity'], task_geometry_executor=overlay)))
        return
    record = selected.run_prepared(prepared, backend=TaskGeometryBackend(args.task))
    atomic_json(prepared['out']/'task_geometry_executor_entrypoint.json', dict(complete=True,
        path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__),
        task_geometry_executor=overlay, native_success=record['official_task_success'], formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True, native_success=record['official_task_success'],
        case=str(prepared['out']/'case.json'))))


if __name__ == '__main__':
    main()
