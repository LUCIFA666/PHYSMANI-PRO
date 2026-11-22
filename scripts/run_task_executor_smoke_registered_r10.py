"""One shared TRAIN entrypoint for the remaining eleven task adapters."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.task_executor_registry import driver, verify, SharedTaskBackend
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    from scripts.run_task_executor_smoke_joint16_v174_r10 import parser
    args = parser().parse_args()
    if args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('shared executor pilot accepts the two geometry methods only')
    module = driver(args.task)
    prepared = module.preflight(args)
    checked_subset(ROOT, prepared['files'], {str(Path(__file__)):file_sha(__file__)})
    overlay = verify(ROOT, prepared)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False,
            GPU_started=False, identity=prepared['identity'], shared_task_executor=overlay)))
        return
    record = module.run_prepared(prepared, backend=SharedTaskBackend(args.task))
    atomic_json(prepared['out']/'shared_task_executor_entrypoint.json', dict(complete=True,
        path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__),
        shared_task_executor=overlay,native_success=record['official_task_success'],formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],
        case=str(prepared['out']/'case.json'))))


if __name__ == '__main__':
    main()
