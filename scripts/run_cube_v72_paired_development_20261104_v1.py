"""Serial actual-wait owner for four paired Cube development scenes."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cube_native_lifecycle',
    ROOT / 'scripts/run_large_network_training_continuation_20261104_v1.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def execute(path, digest):
    path = Path(path).resolve(strict=True)
    if base.sha(path) != digest:
        raise ValueError('External final owner plan SHA required')
    config = base.read(path)
    config['_config_path'], config['_config_sha256'] = str(path), digest
    base.verify_sources(config)
    worker_plan = base.checked_pointer(config['execution_plan'])
    freeze = base.checked_pointer(config['execution_source_freeze'])
    from our_mani.cube_v72_live_20261104_v1.contracts import validate_plan, check_freeze
    plan = validate_plan(worker_plan)
    check_freeze(worker_plan, freeze, config['execution_source_freeze']['sha256'])
    expected = [(profile, c['id']) for profile in plan['profiles_order'] for c in plan['cases']]
    if [(j['profile'], j['case_id']) for j in config['jobs']] != expected:
        raise ValueError('Exactly the predeclared baseline4 then new4 required')
    out = Path(config['output_root'])
    with base.driver_lock(ROOT / 'runtime/framework/cube_v72_paired_development_20261104_v1.lock'):
        base.assert_running(config)
        for job in config['jobs']:
            if Path(job['output']).exists():
                raise FileExistsError('No case overwrite or outcome-based replacement')
        out.mkdir(parents=True, exist_ok=False); (out / 'owners').mkdir()
        base.enable_subreaper()
        previous = {s: signal.signal(s, lambda n, f: base.SIGNAL_REQUEST.update(signal=n))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        base.write_new(out / 'request.json', dict(plan=base.pointer(path),
            native_cases=len(expected), formal_VAL_TEST=False, full_case_denominator_preserved=True,
            concurrent_Native_limit=1, GPU_training_already_complete=True))
        results = []
        status = dict(complete=False, cases_completed=0, formal_VAL_TEST=False)
        try:
            for job in config['jobs']:
                base.assert_running(config)
                result = base.run_stage(config, job['id'], [
                    'scripts/launch_cube_v72_live_20261104_v1.py',
                    '--plan', str(worker_plan), '--case-id', job['case_id'],
                    '--profile', job['profile'], '--source-freeze', str(freeze),
                    '--source-freeze-sha256', config['execution_source_freeze']['sha256'],
                    '--output', job['output'], '--execute'], native=True)
                if (result['actual_process_returncode'] != 0 or result['actual_wait_completed'] is not True
                        or result['source_unchanged'] is not True or result['unreaped_descendants']):
                    raise RuntimeError('Actual worker exit/wait/source required')
                case = Path(job['output'])
                worker = base.read(case / 'worker_exit.json')
                if worker.get('exit_code') != 0 or worker.get('native_closed') is not True:
                    raise RuntimeError('Native must close before the next case')
                record = dict(id=job['id'], profile=job['profile'], case_id=job['case_id'],
                    result=base.pointer(case / 'result.json'), actual_exit=base.pointer(out / 'owners' / job['id'] / 'actual_exit.json'))
                results.append(record); base.append(out / 'case_ledger.jsonl', record)
                status['cases_completed'] = len(results)
            base.verify_sources(config)
            status.update(complete=True, results=results, all_Native_closed=True,
                all_actual_waits_completed=True, source_unchanged=True)
            base.write_new(out / 'completion.json', status)
            return status
        except BaseException as error:
            status.update(error_type=type(error).__name__, error=str(error), results=results)
            base.write_new(out / 'failure.json', status)
            raise
        finally:
            base.write_new(out / 'terminal_status.json', dict(status, finished_utc=base.now()))
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.plan_sha256), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
