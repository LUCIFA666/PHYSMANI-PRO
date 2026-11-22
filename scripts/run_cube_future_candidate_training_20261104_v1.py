"""One source-bound GPU fit, using the existing actual-wait lifecycle helper."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cube_candidate_lifecycle',
    ROOT / 'scripts/run_large_network_training_continuation_20261104_v1.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def execute(path, digest):
    path = Path(path).resolve(strict=True)
    if base.sha(path) != digest:
        raise ValueError('External training plan SHA required')
    config = base.read(path)
    config['_config_path'], config['_config_sha256'] = str(path), digest
    base.verify_sources(config)
    out = Path(config['output_root'])
    with base.driver_lock(ROOT / 'runtime/framework/cube_future_candidate_training_20261104_v1.lock'):
        base.assert_running(config)
        out.mkdir(parents=True, exist_ok=False)
        (out / 'owners').mkdir()
        base.enable_subreaper()
        previous = {s: signal.signal(s, lambda n, f: base.SIGNAL_REQUEST.update(signal=n))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        status = dict(complete=False, training_started=False, Native_started=False, TEST_read=False)
        base.write_new(out / 'request.json', dict(plan=base.pointer(path), user_request='先训练抓方块的任务看看效果',
            source_map=config['method_sources'], GPU_training_owns_exclusive_Native_GPU_locks=True))
        try:
            training_plan = base.checked_pointer(config['training_plan'])
            status['training_started'] = True
            result = base.run_stage(config, 'candidate_fit', [
                'scripts/train_cube_future_candidate_20261104_v1.py', '--execute',
                '--plan', str(training_plan), '--plan-sha256', config['training_plan']['sha256']])
            if (result['actual_process_returncode'] != 0 or result['actual_wait_completed'] is not True
                    or result['source_unchanged'] is not True or result['unreaped_descendants']):
                raise RuntimeError('Actual complete child wait/source required')
            fit = base.read(Path(config['fit_output']) / 'fit_status.json')
            if (fit.get('training_complete') is not True or fit.get('epochs_completed') != 200
                    or fit.get('actual_optimizer_steps') != 1400):
                raise ValueError('The entire fixed training budget must complete')
            selected = base.checked_pointer(fit['selected_checkpoint'])
            base.verify_sources(config)
            status.update(complete=True, selected_checkpoint=base.pointer(selected),
                fit_status=base.pointer(Path(config['fit_output']) / 'fit_status.json'),
                all_training_children_reaped=True, source_unchanged=True)
            base.write_new(out / 'completion.json', status)
            return status
        except BaseException as error:
            status.update(error_type=type(error).__name__, error=str(error))
            base.write_new(out / 'failure.json', status)
            raise
        finally:
            base.write_new(out / 'terminal_status.json', dict(status, finished_utc=base.now()))
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.plan_sha256), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
