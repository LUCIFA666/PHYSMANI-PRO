"""One scoped, serial small TRAIN/VAL/TEST cycle; no historical job resumes."""
import fcntl
import os
import signal
import subprocess
import time

from .common import ROOT, OUT, PLAN, PYTHON, read, write, append, pointer, stopped, sources, verify, utc


def status(stage, **kwargs):
    import json
    value = dict(stage=stage, updated_utc=utc(), **kwargs)
    temporary = OUT / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(OUT / 'status.json')
    append(OUT / 'events.jsonl', value)


def fit_child():
    command = [PYTHON, '-B', '-m', 'our_mani.multitask_executor_cycle_20261112_v1.fit']
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    process = None
    started = time.monotonic()
    signals = []
    previous = {s: signal.signal(s, lambda n, f: signals.append(n)) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (OUT / 'fit_stdout.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                stopped()
                if signals or time.monotonic()-started > 3600:
                    raise InterruptedError('Head fit signal/timeout')
                time.sleep(.2)
            code = process.wait()
        write(OUT / 'fit_actual_exit.json', dict(actual_wait_completed=True, actual_process_returncode=code,
                                                complete=code == 0, elapsed_seconds=time.monotonic()-started))
        if code != 0:
            raise RuntimeError('Small TRAIN head fit failed; inspect fit_stdout.log')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def summarize(results):
    tasks = sorted({r['task'] for r in results})
    return dict(episodes=len(results), successes=sum(r['episode_success'] for r in results),
                per_task={task: dict(episodes=len(rows), successes=sum(r['episode_success'] for r in rows),
                                     failures=[r['stop_reason'] for r in rows if not r['episode_success']])
                          for task in tasks for rows in [[r for r in results if r['task'] == task]]},
                cases=results)


def execute():
    from .owner import run_case
    plan = read(PLAN)
    if plan['schema'] != 'fourteen_task_small_executor_cycle_20261112_v1':
        raise ValueError('Explicit new14-task small plan required')
    lock_path = ROOT / 'runtime/framework/multitask_executor_cycle_20261112_v1.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped()
        write(OUT / 'pipeline_request.json', dict(plan=pointer(PLAN), started_utc=utc(),
                                                  old_queues_resumed=False, Native_at_most=1,
                                                  GPU_fitting_Native_overlap=False))
        frozen = sources()
        write(OUT / 'pipeline_sources.json', frozen)
        all_results, summaries = [], {}
        try:
            status('TRAIN_head_fit', complete=False)
            fit_child()
            verify(frozen)
            selected = OUT / 'head_fit/selected.pt'
            selected_pointer = pointer(selected)
            write(OUT / 'frozen_model_before_Native.json', dict(complete=True, checkpoint=selected_pointer,
                                                               plan=pointer(PLAN), source_sha256=frozen,
                                                               frozen_before_native_TRAIN_VAL_TEST=True))
            cases_folder = OUT / 'cases'
            for role in ('fit', 'selection', 'validation', 'test'):
                if role == 'test' and 'validation' not in summaries:
                    raise RuntimeError('Complete prospective validation before new heldout test')
                results = []
                cases = [c for c in plan['cases'] if c['role'] == role]
                for index, case in enumerate(cases):
                    stopped()
                    verify(frozen)
                    if pointer(selected) != selected_pointer:
                        raise RuntimeError('Frozen model changed after head selection')
                    status('Native_' + role, complete=False, cases_completed=len(all_results),
                           expected_cases=84, role_completed=index, role_cases=len(cases),
                           current_case=case['id'], task=case['task'])
                    actual = dict(case, checkpoint=str(selected), checkpoint_sha256=selected_pointer['sha256'])
                    case_path = cases_folder / (case['id'] + '.json')
                    write(case_path, actual)
                    result = run_case(case_path, OUT / 'native' / case['id'], timeout=1800)
                    result.update(case_id=case['id'], actual_exit=pointer(OUT / 'native' / case['id'] / 'actual_exit.json'))
                    results.append(result)
                    all_results.append(result)
                    append(OUT / 'case_ledger.jsonl', result)
                    print(dict(event='case_complete', role=role, task=case['task'], success=result['episode_success'],
                               seconds=result['actual_seconds'], completed=len(all_results)), flush=True)
                summaries[role] = summarize(results)
                write(OUT / (role + '_report.json'), summaries[role])
            verify(frozen)
            report = dict(complete=True, finished_utc=utc(), tasks=plan['tasks'], cases_completed=84,
                          role_summaries=summaries, training=read(OUT/'head_fit/fit_status.json'),
                          frozen_checkpoint=selected_pointer, source_unchanged=True,
                          native_workers_actual_wait_completed=True, native_workers_closed=True,
                          original_task_success_only=True, task_time_limit_seconds=60,
                          official_archived_VAL_TEST=False, small_generated_development_evaluation=True,
                          no_TEST_tuning_or_checkpoint_selection=True, old_Cube_results_unchanged=True,
                          default_entry_or_weights_changed=False, language_branch_trained=False)
            write(OUT / 'report.json', report)
            status('complete', complete=True, cases_completed=84, report=pointer(OUT/'report.json'))
            return report
        except BaseException as error:
            status('failed', complete=False, cases_completed=len(all_results), error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    execute()
