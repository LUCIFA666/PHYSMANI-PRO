"""Run one frozen Cube-revision phase, serially, with actual child waits."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cube_revision_lifecycle',
    ROOT / 'scripts/cube_revision_lifecycle_20261104_v2.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def execute(path, digest):
    path = Path(path).resolve(strict=True)
    if base.sha(path) != digest:
        raise ValueError('Externally frozen phase plan SHA required')
    config = base.read(path)
    if config.get('schema') != 'cube_revision_serial_phase_20261104_v2':
        raise ValueError('Explicit Cube correction phase required')
    config['_config_path'], config['_config_sha256'] = str(path), digest
    sources = base.verify_sources(config)
    jobs = config['jobs']
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Distinct, predeclared jobs required')
    out = Path(config['output_root'])
    lock = ROOT / 'runtime/framework' / (config['phase_id'] + '.lock')
    with base.driver_lock(lock):
        base.assert_running(config)
        for job in jobs:
            if job.get('output') and Path(job['output']).exists():
                raise FileExistsError('Fresh jobs only; preserve failures')
        out.mkdir(parents=True, exist_ok=False); (out / 'owners').mkdir()
        base.enable_subreaper()
        previous = {s: signal.signal(s, lambda n, f: base.SIGNAL_REQUEST.update(signal=n))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        status = dict(complete=False, phase=config['phase_id'], jobs_completed=0,
                      formal_VAL_TEST=False, no_automatic_retry=True)
        records = []
        base.write_new(out / 'request.json', dict(plan=base.pointer(path), source_sha256=sources,
            user_request='修正训练域、未来预测与候选交接并验证', job_ids=[j['id'] for j in jobs],
            Native_at_most=1, Native_GPU_training_overlap=False))
        try:
            for job in jobs:
                base.assert_running(config)
                terminal = base.run_stage(config, job['id'], job['argv'],
                    native=job.get('native', False), cpu=job.get('cpu', False))
                if (terminal['actual_process_returncode'] != 0 or not terminal['actual_wait_completed']
                        or not terminal['source_unchanged'] or terminal['unreaped_descendants']):
                    raise RuntimeError('Actual complete child wait and unchanged source required')
                for check in job.get('json_checks', []):
                    value = base.read(check['path'])
                    for key, expected in check.get('fields', {}).items():
                        if value.get(key) != expected or type(value.get(key)) is not type(expected):
                            raise ValueError('Child result field differs: ' + check['path'] + ':' + key)
                record = dict(id=job['id'], actual_exit=base.pointer(out/'owners'/job['id']/'actual_exit.json'))
                records.append(record); base.append(out/'job_ledger.jsonl', record)
                status['jobs_completed'] = len(records)
                print(json.dumps(dict(phase=config['phase_id'], completed=len(records), total=len(jobs),
                    last_job=job['id'])), flush=True)
            base.verify_sources(config)
            status.update(complete=True, records=records, all_actual_child_waits_completed=True,
                          source_unchanged=True)
            base.write_new(out / 'completion.json', status)
            return status
        except BaseException as error:
            status.update(error_type=type(error).__name__, error=str(error), records=records)
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
    result = execute(args.plan, args.plan_sha256)
    print(json.dumps({k: result[k] for k in ('complete','phase','jobs_completed')},ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
