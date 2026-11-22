"""Single new-protocol TRAIN owner with explicit noninterrupting drain support."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_native_resource_owner_v19_r10 import validate_request, interrupt_owned_tree
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair, process_visibility
from our_mani.multitask.evaluation import atomic_json, file_sha
BASE = ROOT/'runtime/framework/native_resource_owner_v748'


def utc():
    return datetime.now(timezone.utc).isoformat()


def execute(path, request, cancelled):
    leaf = BASE/'jobs'/request['job_id']
    leaf.mkdir(parents=True, exist_ok=False)
    started, process = time.monotonic(), None
    interrupted = timed_out = False
    try:
        validate_request(request)
        with (leaf/'stdout.log').open('xb') as log:
            process = subprocess.Popen(request['argv'], cwd=ROOT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            ticks = int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
            atomic_json(leaf/'launch.json', dict(created_utc=utc(), request=request,
                request_sha256=file_sha(path), launcher_pid=process.pid,
                launcher_start_ticks=ticks, parent_pid=os.getpid(),
                native_slots=1, GPU_slots=int(request['use_gpu'])))
            while process.poll() is None:
                timed_out = time.monotonic()-started > request['timeout_seconds']
                if cancelled.is_set() or timed_out:
                    interrupted = not timed_out
                    interrupt_owned_tree(process)
                    break
                cancelled.wait(.5)
            code = process.wait()
        return dict(complete=code == 0 and not interrupted and not timed_out,
            returncode=code, interrupted=interrupted, timed_out=timed_out,
            automatic_retry=False, wall_seconds=time.monotonic()-started,
            finished_utc=utc(), stdout_sha256=file_sha(leaf/'stdout.log'))
    except BaseException as exc:
        if process is not None and process.poll() is None:
            interrupt_owned_tree(process)
        return dict(complete=False, error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, finished_utc=utc())


def checked_request(path):
    """Validate before allocation; rejection is an isolated request outcome."""
    return validate_request(json.loads(path.read_text()))


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE/'launch.json').exists():
        raise FileExistsError('preserve existing owner; no implicit restart')
    for name in ('requests', 'jobs', 'rejected'):
        (BASE/name).mkdir(exist_ok=True)
    inventory = process_visibility()
    if (not inventory['host_visibility_available'] or inventory['live_project_native_processes']
            or inventory['same_uid_inventory_incomplete']):
        raise RuntimeError('native inventory incomplete or existing project native workers')
    cancelled = threading.Event()
    def cancel(*_):
        cancelled.set()
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    active, completed, seen, rejected = {}, {}, set(), {}
    draining = False
    with exclusive_pair():
        atomic_json(BASE/'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            source_sha256=file_sha(__file__), previous_owner='formal_VAL_v731_superseded_after_natural_child_completion',
            previous_requests_not_imported_or_retried=True, inventory=inventory,
            max_native_slots=2, max_GPU_slots=1, automatic_retry=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                draining = draining or (BASE/'DRAIN').exists()
                if (BASE/'STOP').exists() or (ROOT/'runtime/framework/multitask_r10_STOP').exists():
                    cancelled.set()
                for job, (future, request) in list(active.items()):
                    if future.done():
                        completed[job] = future.result()
                        atomic_json(BASE/'jobs'/job/'terminal.json', completed[job])
                        del active[job]
                if not cancelled.is_set() and not draining:
                    for path in sorted((BASE/'requests').glob('*.json')):
                        if path.name in seen or len(active) >= 2:
                            continue
                        try:
                            request = checked_request(path)
                            job = request['job_id']
                            if job in active or job in completed:
                                raise ValueError('duplicate owned job identity')
                        except Exception as exc:
                            seen.add(path.name)
                            rejected[path.name] = dict(error=repr(exc), rejected_utc=utc(),
                                native_started=False, automatic_retry=False)
                            atomic_json(BASE/'rejected'/path.name, rejected[path.name])
                            continue
                        if request['use_gpu'] and any(r['use_gpu'] for _, r in active.values()):
                            continue
                        seen.add(path.name)
                        active[job] = (pool.submit(execute, path, request, cancelled), request)
                atomic_json(BASE/'status.json', dict(updated_utc=utc(), active_jobs=list(active),
                    completed=completed, rejected=rejected, stop_requested=cancelled.is_set(), drain_requested=draining,
                    max_native_slots=2, allocated_native_slots=len(active),
                    allocated_GPU_slots=sum(r['use_gpu'] for _, r in active.values()),
                    automatic_retry=False), overwrite=True)
                if (cancelled.is_set() or draining) and not active:
                    break
                cancelled.wait(2.)
        atomic_json(BASE/'terminal.json', dict(complete=True, finished_utc=utc(), drained=draining,
            user_stop=cancelled.is_set(), completed=len(completed), rejected=len(rejected),
            native_children_interrupted_by_drain=False, automatic_retry=False))


if __name__ == '__main__':
    main()
