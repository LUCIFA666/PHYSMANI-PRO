"""Candidate, unactivated five-tier TEST controller; no stage retries.

The owner1041 holds the shared historical-controller/native locks itself. This parent
holds only its new one-time controller lock and preserves all prior evidence.
STOP interrupts owned children. Infrastructure failures naturally drain1041.
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
SOURCE = 'scripts/continue_reboot_TEST_v1058_r10.py'
OWNER = 'scripts/run_reboot_TEST_owner_v1056_r10.py'
AUDITOR = 'scripts/audit_reboot_TEST_results_v1057_r10.py'
RUNTIME = 'runtime/framework/ordered_reboot_TEST_v1058'
OWNER_RUNTIME = 'runtime/framework/formal_TEST_reboot_resume_v1056'
OWNER_OUTPUT = 'artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056/test'
RESULTS = 'artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056/TEST_results.json'
ENVIRONMENT = None
FROZEN_CONTROL = None
CONTROL_BINDING = None
OWNED_OWNER = None
EVENT = threading.Event()
CHILDREN = []
STOP_PATHS = ('runtime/framework/ordered_composite_TEST_v1043/STOP','runtime/framework/formal_TEST_composite_resume_v1041/STOP','runtime/framework/ordered_composite_TEST_v1013/STOP',
    'runtime/framework/formal_TEST_composite_resume_v1011/STOP','runtime/framework/ordered_composite_TEST_v958/STOP',
    'runtime/framework/formal_TEST_composite_resume_v956/STOP', 'runtime/framework/multitask_r10_STOP', RUNTIME + '/STOP', OWNER_RUNTIME + '/STOP',
    'runtime/framework/ordered_concurrent_TEST_v907/STOP',
    'runtime/framework/formal_TEST_concurrent_resume_v902/STOP',
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
    path=Path(path)
    if path.exists() and not overwrite:raise FileExistsError('completed controller records cannot be silently overwritten: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fd=os.open(str(temporary),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    temporary.replace(path)
    target=Path(path).absolute()
    require(target.is_relative_to(ROOT),'only project controller evidence can be committed')
    while True:
        fd=os.open(str(target),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if target==ROOT:break
        target=target.parent


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
    setup='artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056_setup'
    require(control_path.resolve(strict=True)==ROOT/setup/'resume_control.json'
            and lock_path.resolve(strict=True)==ROOT/setup/'resume_inputs.json',
            'fixed fresh1040 control and input-lock paths required')
    require(all(p.resolve(strict=True).is_relative_to(ROOT / 'artifacts/framework/multitask_r10')
                for p in (lock_path, control_path)), 'isolated frozen continuation control required')
    payload = lock_path.read_bytes()
    require(isinstance(args.resume_lock_sha256, str) and sha(payload) == args.resume_lock_sha256,
            'exact externally pinned continuation lock bytes required')
    lock = json.loads(payload)
    require(lock.get('schema') == 'r10_six_tier_reboot_formal_TEST_resume_inputs_v1055'
            and lock.get('approved_for_execution') is True, 'unactivated controller without a frozen1040 input lock')
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
    require(lock.get('control')==point(control_path), 'input lock must bind the exact parsed1040 control')
    require(control.get('schema') == 'r10_six_tier_reboot_formal_TEST_resume_control_v1055'
            and control.get('split') == 'test' and control.get('physical_policy_changed') is False
            and control.get('resume_controller') == point(SOURCE)
            and control.get('result_auditor')==control.get('resume_auditor'),
            'actual explicit unchanged-policy1043 controller required')
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
    try:
        start = ticks(child.pid)
        if source == OWNER:
            OWNED_OWNER = dict(pid=child.pid, start_ticks=start)
        atomic(leaf / 'launch.json', dict(created_utc=utc(), argv=argv, pid=child.pid, start_ticks=start,
            source_sha256=point(source)['sha256'], inherited_frozen_environment=ENVIRONMENT,
            current_mount_namespace=os.readlink('/proc/self/ns/mnt'), automatic_retry=False))
    except BaseException:
        # This object is the Popen just returned above, never a PID supplied by
        # metadata. Reap it even if capturing its start ticks or receipt fails.
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
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
    require(result.get('schema') == 'r10_six_tier_reboot_formal_TEST_results_v1057'
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
            and result.get('resource_qualifier') == control['resource_qualifier']
            and result.get('resource_profile')==control['resource_profile']
            and type(control.get('resource_profile_changed')) is bool
            and result.get('resource_profile_changed') is control.get('resource_profile_changed')
            and result.get('resource_capacity_envelope')==control['resource_capacity_envelope']
            and result.get('resource_profile_within_capacity') is control.get('resource_profile_within_capacity') is True
            and result.get('resource_capacity_source')==control['resource_capacity_source']
            and result.get('previous_resource_profile')==control['previous_resource_profile']
            and result.get('previous_resource_qualification')==control['previous_resource_qualification']
            and result.get('execution_implementation')==control['execution_implementation']
            and result.get('independent_composite_audit') is True
            and result.get('infrastructure_recovery') is True and result.get('reboot_audit')==control['reboot_audit']
            and result.get('used_evidence_rechecked_at_close') is True
            and result.get('source_outcome_filtering') is False
            and result.get('motion_unknown_retained') is True
            and result.get('zero_tick_cases_retained') is True
            and result.get('motion_is_diagnostic_only') is True
            and result.get('automatic_retry') is False
            and result.get('native_failures_retained') is True
            and result.get('Value_NGFF_FASTER_used') is False,
            'actual independently audited1042 result must bind this exact complete continuation')
    owners=result.get('actual_stage_owners');tiers=result.get('evidence_tiers')
    runtimes={'runtime/framework/formal_TEST_after_environment_resume_v753',
        'runtime/framework/formal_TEST_concurrent_resume_v902',
        'runtime/framework/formal_TEST_composite_resume_v956','runtime/framework/formal_TEST_composite_resume_v1011',
        'runtime/framework/formal_TEST_composite_resume_v1041',OWNER_RUNTIME}
    require(isinstance(owners,list) and len(owners)==6 and {v.get('runtime') for v in owners}==runtimes
        and all(type(v.get('completed')) is int and v['completed']>=0 for v in owners)
        and sum(v['completed'] for v in owners)==1920
        and isinstance(tiers,list) and len(tiers)==1920 and len({v.get('job_id') for v in tiers})==1920
        and all(v.get('evidence_runtime') in runtimes for v in tiers)
        and all(sum(t.get('evidence_runtime')==v['runtime'] for t in tiers)==v['completed'] for v in owners)
        and isinstance(result.get('cases'),list) and len(result['cases'])==1920
        and isinstance(result.get('admission_driver_receipts'),list) and len(result['admission_driver_receipts'])==1920,
        'six actual physical tiers must partition exactly1920 independent case and751 receipts')


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
    guard()
    atomic(leaf / 'terminal.json', dict(complete=code == 0, returncode=code, finished_utc=utc(), automatic_retry=False))
    require(code == 0 and detected is None, 'explicit one-time stage did not complete cleanly: ' + stage)
    if is_owner:
        terminal = json.loads((ROOT / OWNER_RUNTIME / 'terminal.json').read_bytes())
        require(terminal.get('complete') is True and terminal.get('completed') == 1920
                and terminal.get('expected')==1920 and terminal.get('failed') == 0
                and terminal.get('stop_requested') is False and terminal.get('drain_requested') is False
                and terminal.get('drained') is False and terminal.get('CPU_not_launched')==0
                and terminal.get('CPU_ready_unexecuted')==[] and terminal.get('owner_error') is None
                and terminal.get('automatic_retry') is False,
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
        require(not (ROOT / RUNTIME).exists(), 'preserve the one-time1043 controller; no restart')
        require(not (ROOT / OWNER_RUNTIME).exists() and not (ROOT / OWNER_OUTPUT).exists(),
                'an existing1041 namespace belongs to another attempt; never touch it')
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
            payload = (ROOT / RESULTS).read_bytes()
            result = json.loads(payload)
            require_final_results(result, control, binding)
            result_pointer=dict(path=RESULTS,sha256=sha(payload))
            require(point(RESULTS)==result_pointer,'independent result changed while parsing')
            recheck_stage_inputs(AUDITOR)
            guard()
            atomic(ROOT / RUNTIME / 'terminal.json', dict(complete=True, finished_utc=utc(),
                TEST_results=result_pointer, automatic_retry=False))
            pulse('complete')
        except BaseException as error:
            user_stop = isinstance(error, InterruptedError) or stopped()
            if not user_stop:
                request_drain('concrete1043 failure; no new1041 cases, finish active owned cases')
                while any(child.poll() is None for child, _, _ in CHILDREN):
                    if stopped():
                        user_stop = True; break
                    request_drain('concrete1043 failure; finish only the verified owned956 child')
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
