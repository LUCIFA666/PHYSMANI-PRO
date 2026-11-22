import fcntl
import json
from pathlib import Path
from .common import ROOT, PLAN, OUT, PREVIOUS_OUT, read, write, append, pointer, stopped, sources, verify, utc
from .owner import run_case
from ..multitask_executor_cycle_20261112_v1.pipeline import summarize


def status(stage, **kwargs):
    value = dict(stage=stage, updated_utc=utc(), **kwargs)
    temporary = OUT/'status.tmp'
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(OUT/'status.json')
    append(OUT/'events.jsonl', value)


def invalid_reset_only(folder, receipt):
    folder = Path(folder)
    failure = folder/'worker/failure.json'
    return (not receipt['complete'] and receipt['actual_wait_completed'] and not receipt['unreaped_descendants']
            and receipt['stop_reason'] is None and failure.is_file()
            and read(failure)['error_type'] == 'TaskEnvironmentError'
            and not (folder/'worker/initial.json').exists()
            and not (folder/'worker/execution.jsonl').exists()
            and not (folder/'worker/public_frame_provenance.jsonl').exists())


def execute():
    plan = read(PLAN)
    with (ROOT/'runtime/framework/multitask_executor_cycle_20261112_v7.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        stopped()
        frozen = sources()
        verify(frozen)
        if pointer(plan['checkpoint']['path']) != plan['checkpoint']:
            raise RuntimeError('Selected head checkpoint changed')
        write(OUT/'pipeline_request.json', dict(plan=pointer(PLAN), source_sha256=frozen, started_utc=utc(),
                                              no_training_restarted=True, previous_selected_model_reused=True))
        all_results, summaries = [], {}
        invalid_attempts = []
        try:
            for role in ('fit', 'selection', 'validation', 'test'):
                if role == 'test' and 'validation' not in summaries:
                    raise RuntimeError('Complete prospective validation before heldout test')
                results = []
                cases = [c for c in plan['cases'] if c['role'] == role]
                if role == 'fit':
                    # Exercise every task adapter before its second/third
                    # TRAIN replicate; fixed order, independent of outcomes.
                    order={task:i for i,task in enumerate(plan['tasks'])}
                    cases.sort(key=lambda c:(int(c['id'].rsplit('_',1)[1]),order[c['task']]))
                for case_index, case in enumerate(cases):
                    accepted = None
                    attempts = []
                    for attempt, seed in enumerate(case['cold_reset_seeds']):
                        stopped()
                        verify(frozen)
                        if pointer(plan['checkpoint']['path']) != plan['checkpoint']:
                            raise RuntimeError('Model changed after training selection')
                        status('Native_'+role, complete=False, cases_completed=len(all_results), expected_cases=84,
                               role_completed=case_index, role_cases=len(cases), current_case=case['id'],
                               task=case['task'], cold_reset_attempt=attempt, actual_seed=seed,
                               invalid_reset_attempts=len(invalid_attempts))
                        actual = dict(case, seed=seed, cold_reset_attempt=attempt,
                                      checkpoint=plan['checkpoint']['path'], checkpoint_sha256=plan['checkpoint']['sha256'])
                        case_path = OUT/'cases'/f'{case["id"]}_reset{attempt}.json'
                        write(case_path, actual)
                        imported = plan['import_first_failed_reset']
                        if imported['case_id'] == case['id'] and imported['attempt'] == attempt:
                            folder = Path(imported['owner'])
                            receipt = read(folder/'actual_exit.json')
                            value = None
                        else:
                            folder = OUT/'native'/f'{case["id"]}_reset{attempt}'
                            value, receipt = run_case(case_path, folder)
                        row = dict(case_id=case['id'], attempt=attempt, seed=seed, actual_exit=pointer(folder/'actual_exit.json'),
                                   imported_previous_TRAIN_reset=folder == Path(imported['owner']))
                        attempts.append(row)
                        if value is not None and receipt['complete']:
                            accepted = dict(value, case_id=case['id'], planned_seed=case['seed'],
                                            cold_reset_attempt=attempt, reset_attempts=attempts,
                                            actual_exit=row['actual_exit'])
                            break
                        if not invalid_reset_only(folder, receipt):
                            raise RuntimeError('Non-reset worker failure; no retry: '+str(folder))
                        invalid_attempts.append(row)
                        append(OUT/'invalid_reset_ledger.jsonl', row)
                    if accepted is None:
                        raise RuntimeError('All five fixed Native initialization attempts invalid: '+case['id'])
                    results.append(accepted)
                    all_results.append(accepted)
                    append(OUT/'case_ledger.jsonl', accepted)
                    print(dict(event='case_complete', role=role, task=case['task'], success=accepted['episode_success'],
                               seconds=accepted['actual_seconds'], completed=len(all_results)), flush=True)
                summaries[role] = summarize(results)
                write(OUT/(role+'_report.json'), summaries[role])
            verify(frozen)
            write(OUT/'report.json', dict(complete=True, finished_utc=utc(), tasks=plan['tasks'], cases_completed=84,
                                         role_summaries=summaries, training=read(PREVIOUS_OUT/'head_fit/fit_status.json'),
                                         selected_model=plan['checkpoint'], source_sha256=frozen,
                                         invalid_reset_attempts=invalid_attempts, all_valid_cases_actual_wait_completed=True,
                                         Native_at_most=1, GPU_fitting_Native_overlap=False, source_unchanged=True,
                                         original_task_success_only=True, task_time_limit_seconds=60,
                                         official_archived_VAL_TEST=False, new_generated_development_evaluation=True,
                                         TEST_tuning_or_model_selection=False, old_Cube_results_unchanged=True,
                                         default_runtime_changed=False, language_branch_trained=False))
            status('complete', complete=True, cases_completed=84, report=pointer(OUT/'report.json'))
        except BaseException as error:
            status('failed', complete=False, cases_completed=len(all_results), error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    execute()
