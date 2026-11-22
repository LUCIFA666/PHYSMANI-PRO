"""One explicit, frozen TEST concurrency continuation; never retry a stage.

The owner902 holds the shared old-controller/native locks itself. This parent
holds only its new one-time controller lock and preserves all prior evidence.
STOP interrupts owned children. Infrastructure failures naturally drain902.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
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
SOURCE = 'scripts/continue_concurrent_TEST_v907_r10.py'
OWNER = 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
AUDITOR = 'scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py'
RUNTIME = 'runtime/framework/ordered_concurrent_TEST_v907'
OWNER_RUNTIME = 'runtime/framework/formal_TEST_concurrent_resume_v902'
OWNER_OUTPUT = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/test'
RESULTS = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/TEST_results.json'
ENVIRONMENT = None
FROZEN_CONTROL = None
CONTROL_BINDING = None
OWNED_OWNER = None
EVENT = threading.Event()
CHILDREN = []
STOP_PATHS = ('runtime/framework/multitask_r10_STOP', RUNTIME + '/STOP', OWNER_RUNTIME + '/STOP',
    'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP',
    'runtime/framework/formal_TEST_after_environment_resume_v753/STOP',
    'runtime/framework/ordered_resume_VAL_TEST_v775/STOP', 'runtime/framework/formal_VAL_resume_v772/STOP')


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def point(path):
    path = (ROOT / path).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'project control evidence required')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes()))


def atomic(path, data, *, overwrite=False):
    from our_mani.multitask.evaluation import atomic_json
    atomic_json(path, data, overwrite=overwrite)


def stopped():
    return EVENT.is_set() or any((ROOT / name).exists() for name in STOP_PATHS)


def guard():
    if stopped():
        raise InterruptedError('actual user STOP')


def pulse(stage, **fields):
    atomic(ROOT / RUNTIME / 'status.json', dict(updated_utc=utc(), stage=stage,
        frozen_environment=ENVIRONMENT, automatic_retry=False, **fields), overwrite=True)


def load_small_control(args):
    """Authenticate the controller entry before any child/source selection."""
    guard()
    lock_path, control_path = (ROOT / args.resume_lock, ROOT / args.resume_control)
    require(all(p.resolve(strict=True).is_relative_to(ROOT / 'artifacts/framework/multitask_r10')
                for p in (lock_path, control_path)), 'isolated frozen continuation control required')
    payload = lock_path.read_bytes()
    require(isinstance(args.resume_lock_sha256, str) and sha(payload) == args.resume_lock_sha256,
            'exact externally pinned continuation lock bytes required')
    lock = json.loads(payload)
    require(lock.get('schema') == 'r10_concurrent_resumed_formal_TEST_inputs_v902'
            and lock.get('approved_for_execution') is True, 'unactivated controller without a frozen902 input lock')
    files = lock.get('files')
    require(isinstance(files, dict) and files, 'explicit source/control input map required')
    def checked(path):
        canonical = Path(path).resolve(strict=True)
        require(canonical.is_relative_to(ROOT), 'controller source must remain in project')
        names = [str(canonical), str(canonical.relative_to(ROOT))]
        declared = [name for name in names if name in files]
        require(len(declared) == 1, 'exactly one source/control declaration required')
        data = canonical.read_bytes()
        require(sha(data) == files[declared[0]], 'controller source/control changed')
        return data
    control = json.loads(checked(control_path))
    require(control.get('schema') == 'r10_concurrent_resumed_formal_TEST_control_v902'
            and control.get('split') == 'test' and control.get('physical_policy_changed') is False
            and control.get('resume_controller') == point(SOURCE), 'actual explicit unchanged-policy907 controller required')
    for key, name in (('resume_controller', SOURCE), ('resume_owner', OWNER), ('resume_auditor', AUDITOR)):
        checked(ROOT / name)
        require(control.get(key) == point(name), 'frozen actual stage source differs: ' + key)
    return control, dict(control=point(control_path), lock=point(lock_path))


def owner_arguments(control, args):
    values = []
    for key, flag in (('suite', '--suite'), ('suite_lock', '--suite-lock'), ('source_inputs', '--source-inputs'),
                      ('validation_release', '--validation-release')):
        values += [flag, control[key]['path']]
    return [*values, '--runtime', OWNER_RUNTIME, '--out', OWNER_OUTPUT,
            '--resume-control', point(args.resume_control)['path'], '--resume-lock', point(args.resume_lock)['path'],
            '--resume-lock-sha256', args.resume_lock_sha256]


def auditor_arguments(args):
    return ['--resume-control', point(args.resume_control)['path'], '--resume-lock', point(args.resume_lock)['path'],
            '--resume-lock-sha256', args.resume_lock_sha256]


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def recheck_stage_inputs(source):
    require(FROZEN_CONTROL and CONTROL_BINDING, 'actual frozen controller binding required')
    key = 'resume_owner' if source == OWNER else 'resume_auditor' if source == AUDITOR else None
    require(key is not None and point(source) == FROZEN_CONTROL[key], 'actual stage entry source changed before execution')
    for name in ('resume_controller', 'resume_owner', 'resume_auditor'):
        require(point(FROZEN_CONTROL[name]['path']) == FROZEN_CONTROL[name], 'frozen controller/owner/auditor source changed')
    require(point(CONTROL_BINDING['control']['path']) == CONTROL_BINDING['control']
            and point(CONTROL_BINDING['lock']['path']) == CONTROL_BINDING['lock'],
            'frozen controller inputs changed before stage execution')


def launch(stage, source, arguments):
    global OWNED_OWNER
    guard()
    recheck_stage_inputs(source)
    require(ENVIRONMENT and os.readlink('/proc/self/ns/mnt') == ENVIRONMENT['current_mount_namespace'],
            'children must inherit the actually verified820 namespace')
    leaf = ROOT / RUNTIME / 'stages' / stage
    leaf.mkdir(parents=True, exist_ok=False)
    argv = [str(ROOT / '.conda-env/bin/python'), '-B', '-u', source, *arguments]
    out, err = (leaf / 'stdout.log').open('xb'), (leaf / 'stderr.log').open('xb')
    try:
        guard()
        child = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
    except BaseException:
        out.close(); err.close(); raise
    CHILDREN.append((child, out, err))
    start = ticks(child.pid)
    if source == OWNER:
        OWNED_OWNER = dict(pid=child.pid, start_ticks=start)
    atomic(leaf / 'launch.json', dict(created_utc=utc(), argv=argv, pid=child.pid, start_ticks=start,
        source_sha256=point(source)['sha256'], inherited_frozen_environment=ENVIRONMENT,
        current_mount_namespace=os.readlink('/proc/self/ns/mnt'), automatic_retry=False))
    return child, leaf


def failure_from_status(state):
    failed, error = state.get('failed'), state.get('owner_error')
    return dict(failed=failed or {}, owner_error=error) if failed or error else None


def request_drain(reason):
    path = ROOT / OWNER_RUNTIME / 'DRAIN'
    launch = ROOT / OWNER_RUNTIME / 'launch.json'
    if OWNED_OWNER is None or not launch.exists():
        return False
    record = json.loads(launch.read_bytes())
    if any(record.get(key) != value for key, value in OWNED_OWNER.items()):
        return False
    if not FROZEN_CONTROL or record.get('source_sha256') != FROZEN_CONTROL['resume_owner']['sha256']:
        return False
    if not path.exists():
        atomic(path, dict(created_utc=utc(), reason=reason, actual_controller=point(SOURCE)))
    return True


def require_final_results(result, control, binding):
    from our_mani.multitask.evaluation_roster16_v283 import TASKS, METHODS
    require(result.get('schema') == 'r10_composite_concurrent_resumed_formal_TEST_results_v906'
            and result.get('complete') is True and result.get('split') == 'test'
            and result.get('source_cases') == 640 and result.get('method_cases') == 1920
            and result.get('completed_method_cases') == 1920
            and result.get('methods') == list(METHODS) and isinstance(result.get('tasks'), list)
            and len(result['tasks']) == len(TASKS) and set(result['tasks']) == set(TASKS)
            and result.get('source_sha256') == control['resume_auditor']['sha256']
            and result.get('resume_control') == binding['control'] and result.get('resume_control_lock') == binding['lock']
            and result.get('policy_sha256') == control['policy_sha256']
            and result.get('source_input_lock') == control['source_inputs']
            and result.get('author_outcome_protocol') == control['author_outcome_protocol']
            and result.get('validation_release') == control['validation_release']
            and result.get('resource_qualification') == control['resource_qualification']
            and result.get('native_failures_retained') is True
            and result.get('Value_NGFF_FASTER_used') is False,
            'actual independently audited906 result must bind this exact complete continuation')


def run(stage, source, arguments, *, is_owner=False):
    child, leaf = launch(stage, source, arguments)
    detected = None
    while child.poll() is None:
        guard()
        state = None
        if is_owner and (ROOT / OWNER_RUNTIME / 'status.json').exists():
            state = json.loads((ROOT / OWNER_RUNTIME / 'status.json').read_bytes())
            detected = detected or failure_from_status(state)
            if detected:
                request_drain('concrete new infrastructure failure; preserve active cases and naturally drain')
        pulse(stage, child_pid=child.pid, infrastructure_failure_detected=detected,
              owner_error=None if state is None else state.get('owner_error'))
        EVENT.wait(2.)
    code = child.wait()
    atomic(leaf / 'terminal.json', dict(complete=code == 0, returncode=code, finished_utc=utc(), automatic_retry=False))
    require(code == 0 and detected is None, 'explicit one-time stage did not complete cleanly: ' + stage)
    if is_owner:
        terminal = json.loads((ROOT / OWNER_RUNTIME / 'terminal.json').read_bytes())
        require(terminal.get('complete') is True and terminal.get('completed') == 1920
                and terminal.get('failed') == 0 and terminal.get('stop_requested') is False,
                'full mixed-source1920 owner completion required')


@contextmanager
def controller_lock():
    with (ROOT / (RUNTIME + '.lock')).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    global ENVIRONMENT, FROZEN_CONTROL, CONTROL_BINDING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-control', required=True); parser.add_argument('--resume-lock', required=True)
    parser.add_argument('--resume-lock-sha256', required=True)
    args = parser.parse_args()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: EVENT.set())
    with controller_lock():
        control, binding = load_small_control(args)
        FROZEN_CONTROL, CONTROL_BINDING = control, binding
        require(not (ROOT / RUNTIME).exists(), 'preserve the one-time907 controller; no restart')
        require(not (ROOT / OWNER_RUNTIME).exists() and not (ROOT / OWNER_OUTPUT).exists(),
                'an existing902 namespace belongs to another attempt; never touch it')
        from scripts.run_environment_resumed_formal_VAL_owner_v823_r10 import require_frozen_environment
        ENVIRONMENT = require_frozen_environment(control)
        guard()
        (ROOT / RUNTIME).mkdir()
        atomic(ROOT / RUNTIME / 'launch.json', dict(created_utc=utc(), pid=os.getpid(), start_ticks=ticks(os.getpid()),
            argv=list(sys.argv), source_sha256=point(SOURCE)['sha256'], resume_control=binding['control'],
            resume_control_lock=binding['lock'], frozen_environment=ENVIRONMENT, automatic_retry=False))
        try:
            run('formal_TEST_1920', OWNER, owner_arguments(control, args), is_owner=True)
            run('independent_TEST_results', AUDITOR, auditor_arguments(args))
            result = json.loads((ROOT / RESULTS).read_bytes())
            require_final_results(result, control, binding)
            guard()
            atomic(ROOT / RUNTIME / 'terminal.json', dict(complete=True, finished_utc=utc(),
                TEST_results=point(RESULTS), automatic_retry=False))
            pulse('complete')
        except BaseException as error:
            user_stop = isinstance(error, InterruptedError) or stopped()
            if not user_stop:
                request_drain('concrete907 failure; no new902 cases, finish active owned cases')
                while any(child.poll() is None for child, _, _ in CHILDREN):
                    if stopped():
                        user_stop = True; break
                    request_drain('concrete907 failure; finish only the verified owned902 child')
                    pulse('draining_for_concrete_repair', error=repr(error)); EVENT.wait(2.)
            if user_stop:
                from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
                for child, _, _ in CHILDREN:
                    if child.poll() is None:
                        interrupt_owned_tree(child)
            atomic(ROOT / RUNTIME / 'failure.json', dict(complete=False, user_stop=user_stop,
                finished_utc=utc(), error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
            pulse('user_stopped' if user_stop else 'requires_concrete_repair', error=repr(error))
            raise
        finally:
            for child, out, err in CHILDREN:
                out.close(); err.close()


if __name__ == '__main__':
    main()
