"""Fixed TRAIN button pilot with one actual task-model preflight."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_button_reference_v388_r10 as driver
from our_mani.multitask.moving_button_native import MovingButtonBackend, verify_overlay
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset
from our_mani.multitask.evaluation import file_sha, atomic_json


def main():
    args = driver.parser().parse_args()
    if args.method == 'physmani':
        raise ValueError('this development ablation changes only our button control')
    prepared = driver.preflight(args)
    checked_subset(ROOT, prepared['files'], {str(Path(__file__)): file_sha(__file__)})
    overlay = verify_overlay(ROOT, prepared)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False,
            GPU_started=False, identity=prepared['identity'], model_preflight=overlay)))
        return
    record = driver.run_prepared(prepared, backend=MovingButtonBackend(args.task))
    atomic_json(prepared['out']/'moving_button_entrypoint.json', dict(complete=True,
        path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__),
        model_preflight=overlay, native_success=record['official_task_success'], formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True, native_success=record['official_task_success'],
        case=str(prepared['out']/'case.json'))))


if __name__ == '__main__':
    main()
