"""Resource-owner-compatible entrypoint for the common executor pilot."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_continuous_executor_smoke_r10 as pilot
from our_mani.multitask.evaluation import file_sha, atomic_json
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args = pilot.parser().parse_args()
    prepared = pilot.preflight(args)
    checked_subset(ROOT, prepared['files'], {str(Path(__file__)): file_sha(__file__)})
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False,
            GPU_started=False, identity=prepared['identity'],
            model_preflight=prepared['common_executor_preflight'],
            actual_entrypoint_sha256=file_sha(__file__))))
        return
    record = pilot.run_prepared(prepared)
    atomic_json(prepared['out']/'common_executor_bridge_entrypoint.json', dict(
        complete=True, path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__),
        delegated_entrypoint_sha256=file_sha(pilot.__file__), formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True, native_success=record['official_task_success'],
        case=str(prepared['out']/'case.json'))))


if __name__ == '__main__':
    main()
