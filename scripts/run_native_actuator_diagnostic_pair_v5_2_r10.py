"""Fresh v5 controller attempt with a shared run lock and an X11 readiness gate."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import file_sha, verify_inputs
from scripts import run_native_actuator_diagnostic_pair_v5_r10 as original
from scripts import run_geometry_native_actuator_diagnostic_v5_r10 as worker

SETUP = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2_setup'
OUTPUT = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2'
WRAPPER = 'scripts/run_local_sim_display_checked_r10.sh'
LOCK = ROOT / 'runtime/framework/fixed_TRAIN104_geometry_native.lock'
original_command = original.command


@contextmanager
def exclusive_pair(path=LOCK):
    # flock is held by an open descriptor and is shared across PID namespaces.
    # A stale file is harmless: kernel ownership ends when the process exits.
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another fixed TRAIN104 geometry supervisor owns the run lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_visibility():
    comm = Path('/proc/1/comm').read_text().strip()
    host_view = comm in ('systemd', 'init')
    active = []
    incomplete = False
    own = {os.getpid()}
    parent = os.getppid()
    while parent > 0 and parent not in own:
        own.add(parent)
        try:
            fields = Path(f'/proc/{parent}/stat').read_text().rsplit(')', 1)[1].split()
            parent = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    prefixes = ('run_geometry_', 'run_native_actuator_diagnostic_pair_',
        'run_continuous_reference_pair_', 'collect_multitask_',
        'run_ready_candidate_', 'launch_multitask_', 'run_multitask_fresh_queue',
        'run_task_formal_', 'run_trained_task_', 'evaluate_multitask_case_',
        'run_PhysMani_task_')
    if host_view:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name) in own:
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                before = int(fields[19])
                tokens = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
                names = [Path(token).name for token in tokens if '\n' not in token]
                matched = [name for name in names if name.endswith('.py') and name.startswith(prefixes)]
                if matched and (entry / 'cwd').resolve() == ROOT:
                    after = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[19])
                    if before == after:
                        active.append(dict(pid=int(entry.name), start_ticks=before, scripts=matched))
            except PermissionError:
                try:
                    incomplete |= entry.stat().st_uid == os.getuid()
                except OSError:
                    pass
            except (OSError, ValueError, IndexError):
                pass
    return dict(pid_namespace=os.readlink('/proc/self/ns/pid'), pid1_comm=comm,
        host_visibility_available=host_view, live_project_native_processes=active,
        same_uid_inventory_incomplete=incomplete,
        note='Host view is inferred only for a systemd/init PID1; current isolated codex view is rejected')


def command(method):
    if method not in ('cv_geometry', 'context_geometry'):
        raise ValueError('Only the fixed CV/Context TRAIN diagnostic is allowed')
    # original.command reads original.SETUP/OUTPUT at call time.
    argv = original_command(method)
    argv[1] = WRAPPER
    return argv


def preflight():
    if OUTPUT.exists():
        raise RuntimeError('Preserve an existing native attempt; never rerun the same namespace')
    lock = json.loads((SETUP / 'frozen_inputs.json').read_text())
    files = lock['files']
    verify_inputs(ROOT, files)
    for path in (Path(__file__), ROOT / WRAPPER, ROOT / 'scripts/native_display_guard_r10.py'):
        if files.get(str(path.resolve())) != file_sha(path):
            raise RuntimeError('Actual guarded launch source is missing from the frozen lock')
    previous = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5/parent_summary.json'
    prior = json.loads(previous.read_text())
    if not prior['all_children_terminal']:
        raise RuntimeError('The preceding v5 pair has not reached its recorded terminal state')
    if (ROOT / 'runtime/framework/multitask_r10_STOP').exists():
        raise RuntimeError('The existing global STOP prevents a new diagnostic')
    original.SETUP, original.OUTPUT = SETUP, OUTPUT
    methods = []
    for method in ('cv_geometry', 'context_geometry'):
        argv = command(method)
        prepared = worker.preflight(worker.parser().parse_args(argv[3:]))
        methods.append(dict(method=method, argv=argv,
            model_preflight=prepared['model_preflight'], identity=prepared['identity']))
    return dict(schema='r10_guarded_v5_2_parent_preflight_v1',
        methods=methods, process_visibility=process_visibility(),
        previous_pair_terminal=True, frozen_input_count=len(files),
        original_v5_policy_unchanged=True, formal_metric=False, native_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check frozen CPU interfaces without native launch')
    args = parser.parse_args()
    result = preflight()
    if args.check:
        print(json.dumps(result))
        return
    if not result['process_visibility']['host_visibility_available']:
        raise RuntimeError('Host process visibility is unavailable; native pair remains unlaunched')
    if (result['process_visibility']['live_project_native_processes']
            or result['process_visibility']['same_uid_inventory_incomplete']):
        raise RuntimeError('Existing native jobs or incomplete process inventory prevent a new pair')
    with exclusive_pair():
        # Recheck the namespace and all files after acquiring the shared lock.
        checked = preflight()['process_visibility']
        if checked['live_project_native_processes'] or checked['same_uid_inventory_incomplete']:
            raise RuntimeError('Native inventory changed while acquiring the shared run lock')
        original.command = command
        original.main()


if __name__ == '__main__':
    main()
