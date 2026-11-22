"""Execute one declared task's complete TRAIN then CAL, with four total workers."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_ready_candidate_queue_r10 as queue
from our_mani.multitask.evaluation import atomic_json
from our_mani.multitask.ready_candidate_queue import pointer, require


def sequence(train_plan, cal_plan, journal, workers=4):
    require(1 <= workers <= 4, 'one task sequence permits at most four total CPU-native workers')
    train, train_config = queue.verify_plan(train_plan)
    cal, cal_config = queue.verify_plan(cal_plan)
    queue.locked(Path(__file__), queue.read(ROOT/train['frozen_inputs']['path'])['files'])
    require(train['stage'] == 'train' and cal['stage'] == 'cal'
            and train['config'] == cal['config'] and train['frozen_inputs'] == cal['frozen_inputs']
            and train_config == cal_config
            and len(train['per_task_original_parents']) == 1
            and set(train['per_task_original_parents']) == set(cal['per_task_original_parents']),
            'sequence requires one task, matching frozen TRAIN/CAL config and source lock')
    journal = queue.inside(journal)
    require(not journal.exists(), 'preserve prior sequence journal; no automatic failed-attempt retry')
    report = dict(version='r10_task_ready_TRAIN_then_CAL_sequence_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), task=next(iter(train['per_task_original_parents'])),
        actual_sequence_source=pointer(ROOT, Path(__file__)),
        process_id=os.getpid(), process_start_ticks=queue.process_start_ticks(os.getpid()),
        TRAIN_plan=pointer(ROOT, train_plan), CAL_plan=pointer(ROOT, cal_plan),
        maximum_simultaneous_native_cases=workers, TRAIN_completed=False, CAL_completed=False,
        formal_VAL_started=False, formal_TEST_started=False, automatic_retry=False,
        errors=[], stage='TRAIN', complete=False)
    atomic_json(journal, report)
    try:
        for stage, path in (('TRAIN', train_plan), ('CAL', cal_plan)):
            report['stage'] = stage; atomic_json(journal, report, overwrite=True)
            result = queue.run(path, workers)
            require(result.get('complete') is True, stage + ' incomplete: preserve stop/errors before next stage')
            report[stage + '_completed'] = True; report[stage + '_result'] = result
            atomic_json(journal, report, overwrite=True)
        report.update(stage='complete', complete=True)
    except BaseException as error:
        report.update(stage='incomplete', errors=[dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc())])
        raise
    finally:
        report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(journal, report, overwrite=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-plan', required=True); parser.add_argument('--cal-plan', required=True)
    parser.add_argument('--journal', required=True); parser.add_argument('--cpu-workers', type=int, default=4)
    args = parser.parse_args()
    sequence(ROOT/args.train_plan, ROOT/args.cal_plan, args.journal, args.cpu_workers)


if __name__ == '__main__': main()
