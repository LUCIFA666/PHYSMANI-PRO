"""Shared TRAIN-only entrypoint; reuse frozen reset and rollout consumers."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.multitask.continuous_executor_native import (
    BUTTONS, TASKS, ContinuousExecutorBackend, model_preflight)
from our_mani.multitask.case_output_isolation_v238 import require_development_output
from our_mani.multitask.evaluation import file_sha, atomic_json
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def driver(task):
    if task not in TASKS:
        raise ValueError('unsupported common executor pilot task')
    if task in BUTTONS:
        from scripts import run_task_executor_smoke_button_reference_v388_r10 as module
    else:
        from scripts import run_task_executor_smoke_bin_physical_v239_r10 as module
    return module


def parser():
    return driver(BUTTONS[0]).parser()


def preflight(args):
    require_development_output(ROOT, args.out, task=args.task,
        raw=args.raw_episode_id, method=args.method)
    prepared = driver(args.task).preflight(args)
    checked_subset(ROOT, prepared['files'], {str(Path(__file__)): file_sha(__file__)})
    prepared['common_executor_preflight'] = (model_preflight(ROOT, prepared)
        if args.method != 'physmani' else dict(unchanged_author_baseline=True))
    return prepared


def run_prepared(prepared):
    method = prepared['args'].method
    backend = None if method == 'physmani' else ContinuousExecutorBackend(prepared['args'].task)
    record = driver(prepared['args'].task).run_prepared(prepared, backend=backend)
    atomic_json(prepared['out']/'common_executor_entrypoint.json', dict(
        complete=True, entrypoint=str(Path(__file__).relative_to(ROOT)),
        entrypoint_sha256=file_sha(__file__), delegated_consumer=driver(prepared['args'].task).__file__,
        shared_model_preflight=prepared['common_executor_preflight'],
        native_success=record['official_task_success'], formal_VAL_TEST=False))
    return record


def main():
    args = parser().parse_args()
    prepared = preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False,
            GPU_started=False, identity=prepared['identity'],
            model_preflight=prepared['common_executor_preflight'])))
    else:
        record = run_prepared(prepared)
        print(json.dumps(dict(complete=True, case=str(prepared['out']/'case.json'),
            native_success=record['official_task_success'], ticks=record['counted_clock']['ticks'])))


if __name__ == '__main__':
    main()
