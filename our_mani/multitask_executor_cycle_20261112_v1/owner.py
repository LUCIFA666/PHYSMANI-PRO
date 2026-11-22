"""Serial real-child wait/reap; no vanished-PID inference or old queue resume."""
import argparse
import importlib.util
import signal
import subprocess
import time
from pathlib import Path

from .common import ROOT, stopped, read, write, pointer


def run_case(case_path, output, *, timeout=1800):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('new_multitask_lifecycle', ROOT / 'scripts/cube_revision_lifecycle_20261104_v2.py')
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    lifecycle.enable_subreaper()
    stopped()
    argv = ['bash', str(ROOT / 'scripts/run_multitask_executor_cycle_native_20261112_v1.sh'),
            '--case', str(Path(case_path).resolve()), '--output', str(output / 'worker')]
    write(output / 'owner_request.json', dict(case=pointer(case_path), argv=argv,
                                              Native_at_most=1, actual_wait_required=True))
    process, tracked, reaped, stop_reason = None, {}, [], None
    signals = []
    previous = {s: signal.signal(s, lambda n, f: signals.append(n)) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    try:
        with (output / 'stdout.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stopping = None
            while process.poll() is None:
                tracked.update(lifecycle.descendants())
                try:
                    stopped()
                except InterruptedError:
                    stop_reason = 'scoped_STOP'
                if signals:
                    stop_reason = 'signal'
                if time.monotonic() - started > timeout:
                    stop_reason = 'wall_timeout'
                if stop_reason and stopping is None:
                    stopping = time.monotonic()
                if stopping is not None:
                    lifecycle.signal_known(tracked, signal.SIGKILL if time.monotonic()-stopping > 20 else signal.SIGTERM)
                time.sleep(.2)
            code = process.wait()
            tracked.update(lifecycle.descendants())
        remaining = tracked
        for requested in (None, signal.SIGTERM, signal.SIGKILL):
            if requested and remaining:
                lifecycle.signal_known(remaining, requested)
            for _ in range(40):
                remaining, adopted = lifecycle.reap_adopted(remaining, process.pid)
                reaped.extend(adopted)
                if not remaining:
                    break
                time.sleep(.05)
        path = output / 'worker/result.json'
        passed = code == 0 and not remaining and stop_reason is None and path.is_file()
        if passed:
            value = read(path)
            passed = value['complete'] and value['native_closed'] and value['source_unchanged']
        write(output / 'actual_exit.json', dict(complete=bool(passed), actual_process_returncode=code,
                                               actual_wait_completed=True, unreaped_descendants=remaining,
                                               adopted_reaped=reaped, stop_reason=stop_reason,
                                               elapsed_seconds=time.monotonic()-started))
        if not passed:
            raise RuntimeError('Native worker incomplete; inspect ' + str(output / 'stdout.log'))
        return value
    finally:
        if process is not None and process.poll() is None:
            tracked.update(lifecycle.descendants())
            lifecycle.signal_known(tracked, signal.SIGKILL)
            process.wait()
            for _ in range(40):
                tracked, _ = lifecycle.reap_adopted(tracked, process.pid)
                if not tracked:
                    break
                time.sleep(.05)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_case(args.case, args.output)
    print({k: result[k] for k in ('task', 'episode_success', 'actual_seconds', 'stop_reason')}, flush=True)


if __name__ == '__main__':
    main()
