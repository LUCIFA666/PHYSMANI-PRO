"""Inactive, stdlib-only observer for the already admitted TEST execution.

No config is generated and no case, simulator, GPU, source selection, or audit
is launched. A future operator supplies an immutable source config after the
TEST source lock exists. Before the exact824 VAL proof and actual827 TEST
stage are accepted, no TEST runtime or source-lock bytes are opened. Recorded
820 namespace receipts are observed as metadata; this observer never restores
an environment or probes its OS snapshot. Reporting does not replace the
independent754 audit. chat_ready is never a delivery ack.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo('Asia/Shanghai')
SCHEMA = 'r10_readonly_formal_TEST_report_source_v830'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
LABELS = dict(zip(METHODS, ('CV+几何', 'Context+几何', 'PhysMani')))
FAMILIES = (
    ('beat_the_rotating_buzz', '旋转电线穿环'),
    ('insert_onto_rotating_peg', '向旋转插杆套环'),
    ('moving_basketball_in_hoop', '向移动篮筐投球'),
    ('pick_moving_target_on_the_table', '抓取桌上移动方块'),
    ('push_moving_button', '按压移动按钮'),
    ('put_rubbish_in_moving_bin', '向移动垃圾桶投放垃圾'),
    ('place_cups_on_rotating_frame', '向旋转杯架挂杯子'),
    ('remove_cups_from_rotating_frame', '从旋转杯架取下杯子'),
)
TASK_LABELS = {name + suffix: label + '（' + speed + '）'
               for name, label in FAMILIES
               for suffix, speed in (('', '普通'), ('_high_speed', '高速'))}
REL_VAL_AUDITOR = 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py'
REL_CONTROLLER = 'scripts/continue_environment_resumed_VAL_TEST_v827_r10.py'
REL_OWNER = 'scripts/run_formal_suite16_owner_v753_r10.py'
REL_AUDITOR = 'scripts/audit_formal_suite16_results_v754_r10.py'
REL_SOURCE_PREPARER = 'scripts/prepare_formal_split_source_lock_v756_r10.py'
REL_RUNTIME = 'runtime/framework/formal_TEST_after_environment_resume_v753'
REL_CONTROL_RUNTIME = 'runtime/framework/ordered_env_resume_VAL_TEST_v827'
REL_VAL_RELEASE = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/VAL_release.json'
REL_SOURCE_INPUTS = 'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup/test_source_inputs.json'
REL_RESULTS = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/TEST_results.json'
REL_VAL_OWNER = 'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py'
REL_TEST_PREPARER = 'scripts/prepare_formal_TEST_after_environment_resume_v826_r10.py'
REL_RESUME_CONTROL = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_control.json'
REL_RESUME_LOCK = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_inputs.json'
REL_ENVIRONMENT_WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
REL_ENVIRONMENT_MANIFEST = ('artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/'
                            'hybrid_environment_manifest_v820.json')
REL_ENVIRONMENT_PROOF = ('artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/'
                         'cpu_namespace_proof_v820.json')
ENVIRONMENT_FIELDS = ('environment_wrapper', 'environment_manifest', 'environment_namespace_proof')
ENVIRONMENT_STABLE_FIELDS = ('schema', 'manifest', 'wrapper', 'source_locks', 'checked_OS_files',
    'current_OS_target_SHA_map_sha256', 'restored_four_target_SHA256', 'namespace_qualification',
    'qualified_host_mount_namespace', 'reference_mount_namespace')
RESTORED_TARGETS = frozenset(('/usr/bin/git', '/usr/bin/git-shell', '/usr/bin/scalar',
                            '/usr/lib/x86_64-linux-gnu/libXtst.so.6.1.0'))
TEST_STAGES = ('formal_TEST_1920', 'independent_TEST_results', 'complete')


class ObservationStopped(Exception):
    """An observation-only STOP; never signal a case or its owner."""


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def checked_sha(path, stop):
    if stop():
        raise ObservationStopped('STOP has priority over metadata hashing')
    return file_sha(path)


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def clock_text(stamp):
    return datetime.fromtimestamp(stamp, ZONE).strftime('%Y-%m-%d %H:%M:%S（北京时间）')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def start_ticks(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


@contextmanager
def instance_lock(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'single_instance.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('this TEST reporter state already has an owner') from error
        previous = read_json(state / 'launch.json', missing=True)
        pid, ticks = previous.get('pid'), previous.get('start_ticks')
        if pid != os.getpid() and type(ticks) is int and start_ticks(pid) == ticks:
            raise RuntimeError('previous TEST reporter PID/start_ticks is still live')
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def project_path(value, prefix):
    if not isinstance(value, str) or not value:
        raise ValueError('explicit project path required')
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / prefix).resolve()):
        raise ValueError('path must remain inside project ' + prefix)
    return path


def pointer_path(value, prefix):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('exact lowercase SHA256 evidence pointer required')
    return project_path(value['path'], prefix)


def pointer(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def checked_pointer(path, stop):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=checked_sha(path, stop))


def expected_protocol(stop=lambda: False):
    return dict(schema='r10_author_action_outcome_protocol_v741',
        implementation=checked_pointer(ROOT / 'our_mani/multitask/native_outcome_sampling_v741.py', stop),
        runtime_consumer=checked_pointer(ROOT / 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py', stop),
        original_runtime=checked_pointer(ROOT / 'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py', stop),
        failure_sampling='current_native_result_at_explicit_action_end',
        success_sampling='original_per_tick_success_latch', pose_action_max_ticks=10,
        public_hold_max_ticks=1, budget_partial_action_consumed=True,
        native_RML_parameters_unchanged=True, block_native_fail_conditions=0,
        block_extra_task_success_calls=0)


def load_source(path, stop=lambda: False):
    if stop():
        raise ObservationStopped('STOP has priority over loading observer configuration')
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    if (not isinstance(config, dict) or config.get('schema') != SCHEMA
            or config.get('protocol_status') != 'formal_TEST'):
        raise ValueError('explicit v830 formal_TEST source config required')
    for key, relative in (('runtime', REL_RUNTIME), ('controller', REL_CONTROL_RUNTIME),
                          ('test_results', REL_RESULTS)):
        prefix = 'runtime/framework' if key != 'test_results' else 'artifacts/framework/multitask_r10'
        if project_path(config.get(key), prefix) != (ROOT / relative).resolve():
            raise ValueError('exact unchanged TEST/controller/output namespace required: ' + key)
    for key, relative in (('validation_release', REL_VAL_RELEASE), ('test_source_inputs', REL_SOURCE_INPUTS),
                          ('resume_control', REL_RESUME_CONTROL), ('resume_control_lock', REL_RESUME_LOCK),
                          ('environment_wrapper', REL_ENVIRONMENT_WRAPPER),
                          ('environment_manifest', REL_ENVIRONMENT_MANIFEST),
                          ('environment_namespace_proof', REL_ENVIRONMENT_PROOF)):
        prefix = 'scripts' if key == 'environment_wrapper' else 'artifacts/framework/multitask_r10'
        if pointer_path(config.get(key), prefix) != (ROOT / relative).resolve():
            raise ValueError('exact admitted metadata pointer required: ' + key)
    policy = config.get('policy_sha256')
    if (not isinstance(policy, str) or len(policy) != 64
            or any(c not in '0123456789abcdef' for c in policy)):
        raise ValueError('explicit unchanged policy SHA256 required')
    if not same(config.get('author_outcome_protocol'), expected_protocol(stop)):
        raise ValueError('exact frozen author-outcome protocol required')
    source = dict(config)
    for key in ('runtime', 'controller', 'test_results'):
        source[key] = (ROOT / config[key]).resolve()
    # Deliberately do not stat, hash, or open TEST runtime/source-lock/results.
    return source, sha_bytes(raw)


def read_json(path, *, missing=False, stop=lambda: False, checksum=None):
    if stop():
        raise ObservationStopped('STOP has priority over reading evaluation status')
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        if missing:
            return {}
        raise
    if checksum is not None and sha_bytes(raw) != checksum:
        raise ValueError('frozen metadata SHA changed: ' + str(path))
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError('metadata must be a JSON object: ' + str(path))
    return result


def _require_shape(value, *, split, sources, cases, policy):
    if (value.get('complete') is not True or value.get('split') != split
            or any(type(value.get(k)) is not int or value[k] != n for k, n in
                   (('source_cases', sources), ('method_cases', cases), ('completed_method_cases', cases)))
            or value.get('policy_sha256') != policy or value.get('methods') != list(METHODS)
            or not isinstance(value.get('tasks'), list) or len(value['tasks']) != 16
            or set(value['tasks']) != set(TASK_LABELS) or value.get('TEST_used_for_fitting') is not False
            or value.get('motion_is_diagnostic_only') is not True
            or value.get('native_task_success_unchanged') is not True):
        raise ValueError('complete exact split/source/method/protocol aggregate required')


def _bindings(source, release):
    protocol = source['author_outcome_protocol']
    return dict(author_outcome_protocol=protocol, native_runtime=protocol['runtime_consumer'],
                original_native_runtime=protocol['original_runtime'],
                admission_driver=release['admission_driver'], admission_guard=release['admission_guard'])


def _require_bindings(value, expected):
    if value.get('execution_protocol_changed') is not True or any(
            not same(value.get(key), item) for key, item in expected.items()):
        raise ValueError('actual policy/runtime/driver/protocol binding changed')


def _hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def _recorded_environment(receipt, source, *, owner_pid=None, expected=None):
    """Check recorded metadata only; never call820's live namespace/OS probe."""
    if not isinstance(receipt, dict):
        raise ValueError('recorded qualified820 namespace receipt required')
    bindings = dict(wrapper=source['environment_wrapper'], manifest=source['environment_manifest'],
                    namespace_qualification=source['environment_namespace_proof'])
    current, reference = receipt.get('current_mount_namespace'), receipt.get('reference_mount_namespace')
    restored = receipt.get('restored_four_target_SHA256')
    if (receipt.get('schema') != 'r10_actual_frozen_OS_namespace_receipt_v820'
            or receipt.get('complete') is not True or type(receipt.get('checked_OS_files')) is not int
            or receipt['checked_OS_files'] != 6418
            or receipt.get('all_current_OS_targets_full_byte_SHA_verified') is not True
            or receipt.get('host_PID_network_namespace_preserved') is not True
            or receipt.get('proc_or_dev_replaced') is not False
            or receipt.get('task_parameters_or_strategy_changed') is not False
            or receipt.get('project_writable') is not True
            or any(not same(receipt.get(key), value) for key, value in bindings.items())
            or not isinstance(current, str) or not current or current != receipt.get('mount_namespace')
            or not isinstance(reference, str) or not reference or current == reference
            or receipt.get('qualified_host_mount_namespace') != reference
            or type(receipt.get('observer_pid')) is not int or receipt['observer_pid'] <= 0
            or receipt.get('pid') != receipt['observer_pid']
            or any(type(receipt.get(key)) is not int or receipt[key] <= 0 for key in
                   ('observer_start_ticks', 'outer_wrapper_pid', 'outer_wrapper_start_ticks'))
            or not _hex_sha(receipt.get('current_OS_target_SHA_map_sha256'))
            or not isinstance(restored, dict) or set(restored) != RESTORED_TARGETS
            or not all(_hex_sha(value) for value in restored.values())):
        raise ValueError('recorded qualified6418-file private namespace evidence is incomplete')
    if owner_pid is not None and (type(owner_pid) is not int or receipt['observer_pid'] != owner_pid):
        raise ValueError('recorded environment observer PID must match actual controller PID')
    if expected is not None and any(not same(receipt.get(key), expected.get(key))
                                    for key in ENVIRONMENT_STABLE_FIELDS):
        raise ValueError('recorded frozen environment binding changed')
    return receipt


def _resume_gate(source, release, stop):
    """Bind824's already completed VAL proof to the exact823/827/826 chain."""
    for field in ('resume_control', 'resume_control_lock', *ENVIRONMENT_FIELDS):
        value = source[field]
        if not same(release.get(field), value) or checked_pointer(ROOT / value['path'], stop) != value:
            raise ValueError('824 release/config metadata binding changed: ' + field)
    control_pointer = source['resume_control']
    control = read_json(ROOT / control_pointer['path'], checksum=control_pointer['sha256'], stop=stop)
    lock_pointer = source['resume_control_lock']
    lock = read_json(ROOT / lock_pointer['path'], checksum=lock_pointer['sha256'], stop=stop)
    frozen = lock.get('files')
    if (lock.get('schema') != 'r10_frozen_environment_formal_VAL_resume_inputs_v823'
            or lock.get('approved_for_execution') is not True or not isinstance(frozen, dict)):
        raise ValueError('actual823 approved resume input-lock metadata required')
    if (control.get('schema') != 'r10_frozen_environment_formal_VAL_resume_control_v823'
            or control.get('split') != 'validation' or type(control.get('expected_cases')) is not int
            or control['expected_cases'] != 945 or control.get('user_authorized_resume') is not True
            or control.get('policy_sha256') != source['policy_sha256']
            or not same(control.get('author_outcome_protocol'), source['author_outcome_protocol'])
            or control.get('resume_owner') != checked_pointer(ROOT / REL_VAL_OWNER, stop)
            or control.get('resume_auditor') != checked_pointer(ROOT / REL_VAL_AUDITOR, stop)
            or any(not same(control.get(field), source[field]) for field in ENVIRONMENT_FIELDS)
            or any(value.get('standing_authorization_reused') is not True
                   or value.get('new_user_event_invented') is not False
                   or value.get('physical_policy_changed') is not False for value in (control, release))):
        raise ValueError('same authorized823 environment-only continuation required')
    inputs = release.get('inputs_sha256')
    required = (REL_VAL_AUDITOR, REL_VAL_OWNER, REL_CONTROLLER, REL_TEST_PREPARER,
                *(source[field]['path'] for field in ('resume_control', 'resume_control_lock', *ENVIRONMENT_FIELDS)))
    if not isinstance(inputs, dict) or any(inputs.get(str((ROOT / path).resolve()))
            != checked_sha(ROOT / path, stop) for path in required):
        raise ValueError('824 release must bind actual resume, environment, and future TEST control metadata')
    for path in required:
        if path == lock_pointer['path']:
            continue  # A frozen map cannot contain its own byte SHA.
        absolute = str((ROOT / path).resolve())
        if frozen.get(absolute, frozen.get(path)) != inputs[absolute]:
            raise ValueError('actual823 input lock must freeze observation gate bindings')
    return _recorded_environment(release.get('actual_auditor_environment'), source)


def _gate(source, stop):
    val_pointer = source['validation_release']
    release = read_json(ROOT / val_pointer['path'], checksum=val_pointer['sha256'], stop=stop)
    _require_shape(release, split='validation', sources=315, cases=945, policy=source['policy_sha256'])
    if (release.get('schema') != 'r10_full16_common_author_RML_validation_release_v531'
            or release.get('source_sha256') != checked_sha(ROOT / REL_VAL_AUDITOR, stop)
            or release.get('base_result_auditor') != checked_pointer(ROOT / REL_AUDITOR, stop)
            or release.get('formal_test_permitted') is not True or release.get('TEST_used') is not False
            or not same(release.get('author_outcome_protocol'), source['author_outcome_protocol'])
            or not isinstance(release.get('cases'), list) or len(release['cases']) != 945
            or not isinstance(release.get('admission_driver_receipts'), list)
            or len(release['admission_driver_receipts']) != 945):
        raise ValueError('configuration-bound complete actual824 VAL release required before TEST status reads')
    expected = _bindings(source, release)
    _require_bindings(release, expected)
    environment = _resume_gate(source, release, stop)
    controller = read_json(source['controller'] / 'status.json', stop=stop)
    launch = read_json(source['controller'] / 'launch.json', stop=stop)
    if (launch.get('source_sha256') != checked_sha(ROOT / REL_CONTROLLER, stop)
            or launch.get('user_authorized_resume') is not True or controller.get('stage') not in TEST_STAGES
            or controller.get('automatic_retry') is not False
            or any(not same(launch.get(field), source[field]) for field in ('resume_control', 'resume_control_lock'))):
        raise ValueError('actual frozen827 must have entered TEST stage before TEST status reads')
    observed = _recorded_environment(launch.get('frozen_environment'), source,
                                    owner_pid=launch.get('pid'), expected=environment)
    if any(observed[key] != environment[key] for key in
           ('current_mount_namespace', 'outer_wrapper_pid', 'outer_wrapper_start_ticks')):
        raise ValueError('824 release auditor and827 controller must share the recorded namespace')
    if type(launch.get('start_ticks')) is not int or launch['start_ticks'] != observed['observer_start_ticks']:
        raise ValueError('actual827 PID/start_ticks must match recorded environment observer')
    _recorded_environment(controller.get('frozen_environment'), source, expected=observed)
    if not same(controller['frozen_environment'], observed):
        raise ValueError('827 status must preserve its actual launch namespace receipt')
    if stop():
        raise ObservationStopped('STOP has priority over controller identity observation')
    if controller['stage'] != 'complete' and (type(launch.get('start_ticks')) is not int
            or start_ticks(launch.get('pid')) != launch['start_ticks']):
        raise ValueError('active827 PID/start_ticks identity is not live')
    return release, expected, controller


def empty_data():
    return dict(expected=Counter(), completed=Counter(), success=Counter(), failed=0, cancelled=0,
                admitted=False, plan_valid=False, release_complete=False, status={}, controller={},
                errors=[], test_runtime_status_read=False, test_source_lock_metadata_read=False,
                test_aggregate_results_read=False)


def _option(argv, key):
    if not isinstance(argv, list) or argv.count(key) != 1:
        raise ValueError('one explicit plan argument required: ' + key)
    index = argv.index(key)
    if index + 1 >= len(argv):
        raise ValueError('plan argument has no value: ' + key)
    return argv[index + 1]


def _plan(source, plan, launch, expected, lock, stop=lambda: False):
    _require_bindings(launch, expected)
    if (launch.get('source_sha256') != checked_sha(ROOT / REL_OWNER, stop) or launch.get('split') != 'test'
            or launch.get('policy_sha256') != source['policy_sha256']
            or launch.get('split_source_lock_sha256') != source['test_source_inputs']['sha256']
            or launch.get('source_manifest', {}).get('sha256') != lock.get('source_manifest_sha256')
            or plan.get('split') != 'test' or type(plan.get('expected_cases')) is not int
            or plan['expected_cases'] != 1920 or plan.get('policy_sha256') != source['policy_sha256']):
        raise ValueError('actual frozen753 TEST launch/plan/source lock required')
    jobs, groups, by_id = plan.get('jobs'), Counter(), {}
    if not isinstance(jobs, list) or len(jobs) != 1920:
        raise ValueError('full1920 TEST plan required')
    for row in jobs:
        if (not isinstance(row, dict) or row.get('task') not in TASK_LABELS
                or row.get('method') not in METHODS or not isinstance(row.get('job_id'), str)
                or not row['job_id'] or row['job_id'] in by_id):
            raise ValueError('unknown or duplicated TEST plan job')
        if (_option(row.get('argv'), '--split') != 'test'
                or (ROOT / _option(row['argv'], '--source-inputs')).resolve()
                   != (ROOT / source['test_source_inputs']['path']).resolve()
                or (ROOT / _option(row['argv'], '--validation-release')).resolve()
                   != (ROOT / source['validation_release']['path']).resolve()):
            raise ValueError('non-TEST/mixed source-lock/release argv refused')
        if any(not same(row.get(k), v) for k, v in expected.items()):
            raise ValueError('TEST plan protocol bindings differ from the actual owner')
        by_id[row['job_id']] = row
        groups[row['task'], row['method']] += 1
    if len(groups) != 48 or any(groups[task, method] != 40 for task in TASK_LABELS for method in METHODS):
        raise ValueError('exact16 tasks x40 TEST sources x3 methods required')
    return by_id, groups


def _final(source, data, expected, terminal, stop):
    if (sum(data['completed'].values()) != 1920 or data['failed'] or data['cancelled'] or data['errors']
            or data['controller'].get('stage') != 'complete'):
        return False
    if (terminal.get('complete') is not True or terminal.get('split') != 'test'
            or type(terminal.get('completed')) is not int or terminal['completed'] != 1920
            or type(terminal.get('expected')) is not int or terminal['expected'] != 1920
            or type(terminal.get('failed')) is not int or terminal['failed'] != 0
            or terminal.get('policy_sha256') != source['policy_sha256']
            or not same(terminal.get('author_outcome_protocol'), source['author_outcome_protocol'])
            or terminal.get('execution_protocol_changed') is not True):
        raise ValueError('full1920 owner terminal is not a complete bound TEST stage')
    control_terminal = read_json(source['controller'] / 'terminal.json', stop=stop)
    if (control_terminal.get('complete') is not True or control_terminal.get('automatic_retry') is not False
            or control_terminal.get('TEST_results') != str(source['test_results'].relative_to(ROOT))):
        raise ValueError('actual827 complete terminal must bind the final TEST results path')
    result = read_json(source['test_results'], missing=True, stop=stop)
    if not result:
        return False
    data['test_aggregate_results_read'] = True
    _require_shape(result, split='test', sources=640, cases=1920, policy=source['policy_sha256'])
    _require_bindings(result, expected)
    terminal_path = source['runtime'] / 'terminal.json'
    terminal_pointer = dict(path=str(terminal_path.relative_to(ROOT)), sha256=checked_sha(terminal_path, stop))
    if (result.get('source_sha256') != checked_sha(ROOT / REL_AUDITOR, stop)
            or result.get('actual_stage_owner') != terminal_pointer
            or result.get('source_input_lock') != source['test_source_inputs']
            or not isinstance(result.get('cases'), list) or len(result['cases']) != 1920
            or not isinstance(result.get('admission_driver_receipts'), list)
            or len(result['admission_driver_receipts']) != 1920
            or result.get('source_outcome_filtering') is not False or result.get('native_failures_retained') is not True
            or result.get('Value_NGFF_FASTER_used') is not False):
        raise ValueError('actual754 TEST aggregate/source/protocol/owner binding required')
    inputs = result.get('inputs_sha256', {})
    for path in (ROOT / REL_AUDITOR, ROOT / REL_OWNER, ROOT / source['test_source_inputs']['path'],
                 source['runtime'] / 'launch.json', source['runtime'] / 'plan.json', terminal_path):
        if not isinstance(inputs, dict) or inputs.get(str(path.resolve())) != checked_sha(path, stop):
            raise ValueError('754 audit input does not bind actual status-source metadata')
    totals = {m: sum(data['success'][t, m] for t in TASK_LABELS) for m in METHODS}
    if not same(result.get('native_successes_by_method'), totals):
        raise ValueError('754 aggregate success totals differ from actual completed TEST records')
    rows = result.get('per_task')
    if not isinstance(rows, list) or len(rows) != 16 or {r.get('task') for r in rows if isinstance(r, dict)} != set(TASK_LABELS):
        raise ValueError('complete16-task independent TEST aggregate required')
    for row in rows:
        methods = row.get('methods', {})
        if set(methods) != set(METHODS):
            raise ValueError('complete three-method independent TEST aggregate required')
        for method in METHODS:
            value = methods[method]
            if (type(value.get('episodes')) is not int or value['episodes'] != 40
                    or type(value.get('successes')) is not int
                    or value['successes'] != data['success'][row['task'], method]):
                raise ValueError('independent task totals differ from actual TEST status')
    return True


def collect(source, stop=lambda: False):
    data = empty_data()
    try:
        release, expected, controller = _gate(source, stop)
        data.update(admitted=True, controller=controller)
        data['test_source_lock_metadata_read'] = True
        lock = read_json(ROOT / source['test_source_inputs']['path'],
                         checksum=source['test_source_inputs']['sha256'], stop=stop)
        if (lock.get('schema') != 'r10_full16_post_admission_split_source_lock_v1'
                or lock.get('split') != 'test' or type(lock.get('source_count')) is not int
                or lock['source_count'] != 640 or lock.get('policy_sha256') != source['policy_sha256']
                or lock.get('TEST_used_for_fitting') is not False
                or lock.get('source_content_used_for_policy_fitting') is not False
                or lock.get('complete_VAL_release_observed') is not True
                or lock.get('source_lock_preparer_sha256') != checked_sha(ROOT / REL_SOURCE_PREPARER, stop)):
            raise ValueError('exact actual756 post-VAL TEST source-lock metadata required')
        data['test_runtime_status_read'] = True
        launch = read_json(source['runtime'] / 'launch.json', stop=stop)
        plan = read_json(source['runtime'] / 'plan.json', stop=stop)
        status = read_json(source['runtime'] / 'status.json', stop=stop)
        data['status'] = status
        jobs, groups = _plan(source, plan, launch, expected, lock, stop)
        if (status.get('split') != 'test' or type(status.get('expected')) is not int or status['expected'] != 1920):
            raise ValueError('TEST status split/expected differs from actual TEST plan')
        data.update(plan_valid=True, expected=groups)
        records = status.get('completed')
        if not isinstance(records, dict):
            raise ValueError('TEST status.completed must be an object')
        for key, row in records.items():
            if (key not in jobs or not isinstance(row, dict) or row.get('complete') is not True
                    or type(row.get('native_success')) is not bool
                    or row.get('interrupted') is True or row.get('timed_out') is True):
                data['errors'].append('unclean/unknown completed record not counted: ' + str(key))
                continue
            group = jobs[key]['task'], jobs[key]['method']
            data['completed'][group] += 1
            data['success'][group] += int(row['native_success'])
        failed = status.get('failed', {})
        if not isinstance(failed, dict):
            raise ValueError('TEST status.failed must be an object')
        for key, row in failed.items():
            if key not in jobs or key in records:
                data['errors'].append('unknown/overlapping failure record: ' + str(key))
            if isinstance(row, dict) and any(row.get(k) is True for k in ('interrupted', 'cancelled', 'superseded')):
                data['cancelled'] += 1
            else:
                data['failed'] += 1
        terminal = read_json(source['runtime'] / 'terminal.json', missing=True, stop=stop)
        if terminal and terminal.get('complete') is False:
            data['errors'].append('TEST owner stopped or incomplete; partial progress only')
        failure = read_json(source['controller'] / 'failure.json', missing=True, stop=stop)
        if failure:
            data['errors'].append('controller failure recorded; not a final TEST comparison')
        data['release_complete'] = _final(source, data, expected, terminal, stop)
    except ObservationStopped:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        data['errors'].append(str(error))
    data['errors'] = sorted(set(data['errors']))
    return data


def parse_utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError('an explicit UTC deadline is required')
    return result.timestamp()


def initial_deadline(explicit, inherit_state, state, now, interval):
    if explicit:
        return parse_utc(explicit)
    for folder in (inherit_state, state):
        if folder is None:
            continue
        folder = Path(folder)
        for name in ('chat_ready.json', 'status.json'):
            value = read_json(folder / name, missing=True).get('next_update_utc')
            if value:
                return parse_utc(value)
        path = folder / 'history.jsonl'
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                try:
                    value = json.loads(line).get('next_update_utc')
                    if value:
                        return parse_utc(value)
                except (ValueError, AttributeError):
                    continue
    raise ValueError('supply explicit UTC deadline or a prior nonempty recorded deadline; never reset the schedule')


def advance_deadline(deadline, now, interval):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('positive finite interval required')
    return deadline + (int((now - deadline) // interval) + 1) * interval if deadline <= now else deadline


def observation_stop(state, event):
    return (event.is_set() or (ROOT / 'runtime/framework/multitask_r10_STOP').exists()
            or any((ROOT / runtime / 'STOP').exists() for runtime in
                   (REL_CONTROL_RUNTIME, REL_RUNTIME, 'runtime/framework/formal_VAL_env_resume_v823',
                    'runtime/framework/ordered_resume_VAL_TEST_v775', 'runtime/framework/formal_VAL_resume_v772'))
            or (Path(state) / 'STOP').exists())


def read_history(path):
    values = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict) and isinstance(value.get('summary'), str):
                    values.append(value)
            except ValueError:
                pass
    return values


def save_report(state, out, data, source, source_sha, interval, reason, history, *, stopping=False, next_due=None):
    now = time.time()
    finished = stopping or data['release_complete']
    if next_due is None and not finished:
        raise ValueError('reporting must retain an explicit scheduled deadline')
    next_update = None if finished else next_due
    methods = {m: dict(successes=sum(data['success'][t, m] for t in TASK_LABELS),
                       completed=sum(data['completed'][t, m] for t in TASK_LABELS)) for m in METHODS}
    total = sum(data['completed'].values())
    stamp = datetime.fromtimestamp(now, ZONE).strftime('%Y%m%dT%H%M%S.%f')
    snapshot = Path(state) / 'snapshots' / (stamp + '.txt')
    phase = ('完整1920条TEST已由实际754独立审计确认' if data['release_complete'] else
             '部分进度，尚未通过完整独立TEST审计；非最终比较结果')
    if not data['admitted']:
        phase = 'TEST状态读取门未通过，未读取TEST运行状态；不推算已完成数量'
    summary = clock_text(now) + f'｜TEST完成 {total}/1920｜infra {data["failed"]}｜' + '；'.join(
        f'{LABELS[m]} {v["successes"]}/{v["completed"]}' for m, v in methods.items()) + '｜' + reason
    rows = ['正式 TEST 结果更新', phase, '最近更新时间：' + clock_text(now),
            '实际状态时间（UTC）：' + str(data['status'].get('updated_utc', '尚未读取正式TEST状态')),
            '下一次更新时间：' + (clock_text(next_update) if next_update else '本观察器已结束'),
            '本次更新原因：' + reason, f'总体完成：{total}/1920；基础设施失败：{data["failed"]}；中断／取消：{data["cancelled"]}',
            f'已完成中的原任务未成功：{total - sum(v["successes"] for v in methods.values())}（不等于基础设施失败）',
            '只统计实际干净completed记录的native_success，不将未完成样本当成功或失败。',
            'TEST来源640×三组=1920；不混VAL或TRAIN。运动指标只作诊断，不改变native任务判成功。',
            '本报告只观察状态，不能替代完整独立审计；owner完成1920不等于754审计完成。',
            '只读已记录的824/827环境证明，不在宿主机重验或恢复冻结OS环境。',
            '控制器阶段：' + str(data['controller'].get('stage', '等待门控')), '']
    for method, value in methods.items():
        rows.append(f'{LABELS[method]}：成功{value["successes"]}／已完成{value["completed"]}／应评640')
    rows.extend(['', '任务 | CV成功/完成 | Context成功/完成 | PhysMani成功/完成 | 每组应评'])
    for task, label in TASK_LABELS.items():
        rows.append(label + ' | ' + ' | '.join(f'{data["success"][task, m]}/{data["completed"][task, m]}' for m in METHODS)
                    + (' | 40' if data['plan_valid'] else ' | 等待正式计划'))
    if data['errors']:
        rows.extend(['', '读取／口径提示：', *('- ' + x for x in data['errors'])])
    rows.extend(['', '历史摘要（仅TEST独立状态，不混VAL）：', *(x['summary'] for x in history), summary])
    flags = dict(TEST_runtime_status_read=data['test_runtime_status_read'],
                 TEST_source_lock_metadata_read=data['test_source_lock_metadata_read'],
                 TEST_aggregate_results_read=data['test_aggregate_results_read'],
                 TEST_manifest_read=False, TEST_raw_data_read=False, TEST_case_files_read=False,
                 native_started=False, GPU_started=False, recorded_environment_receipts_only=True,
                 live_frozen_OS_environment_probe_performed=False)
    item = dict(schema='r10_readonly_formal_TEST_snapshot_v830', updated_at=clock_text(now), timestamp=now,
                summary=summary, source_config_sha256=source_sha, completed=total, expected=1920, source_cases=640,
                infrastructure_failed=data['failed'], cancelled=data['cancelled'], methods=methods,
                independently_released=data['release_complete'], admitted=data['admitted'],
                source_status_updated_utc=data['status'].get('updated_utc'),
                snapshot=str(snapshot), next_update_utc=datetime.fromtimestamp(next_update, timezone.utc).isoformat()
                if next_update else None, reason=reason, **flags)
    atomic_text(snapshot, '\n'.join(rows) + '\n')
    with (Path(state) / 'history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
    history.append(item)
    atomic_text(out, '\n'.join(rows) + '\n')
    atomic_json(Path(state) / 'status.json', item)
    chat = (f'TEST更新（{clock_text(now)}）：完成{total}/1920，基础设施失败{data["failed"]}。\n'
            + '；'.join(f'{LABELS[m]} 成功/已完成 {v["successes"]}/{v["completed"]}' for m, v in methods.items())
            + '。\n' + phase + '\n' + reason + '\nTXT：' + str(out))
    atomic_json(Path(state) / 'chat_ready.json', dict(snapshot_id=stamp,
                created_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                next_update_utc=item['next_update_utc'], text=chat))
    # Only the root, after a real conversation message, may write chat_ack.json.
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-config', 'state', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--interval-seconds', type=float, default=1800)
    parser.add_argument('--initial-next-due-utc')
    parser.add_argument('--inherit-due-state')
    args = parser.parse_args()
    if args.interval_seconds != 1800:
        parser.error('the authorized reporting interval is1800 seconds')
    state = project_path(args.state, 'runtime/framework')
    config_path = project_path(args.source_config, 'runtime/framework')
    if (state != (ROOT / 'runtime/framework/formal_TEST_report_v830').resolve()
            or config_path != state / 'source.json'):
        parser.error('use the independent formal_TEST_report_v830 state and its source.json')
    out = (ROOT / args.out).resolve()
    if out != (ROOT / 'TEST_结果更新.txt').resolve() or out == config_path:
        parser.error('use the separate project TEST_结果更新.txt output')
    event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: event.set())
    inherited = project_path(args.inherit_due_state, 'runtime/framework') if args.inherit_due_state else None
    with instance_lock(state):
        # STOP wins even if a new observer never received its initial deadline.
        next_due = (time.time() if observation_stop(state, event) else
                    initial_deadline(args.initial_next_due_utc, inherited, state, time.time(), 1800))
        atomic_json(state / 'launch.json', dict(schema='r10_readonly_formal_TEST_observer_v830',
                    pid=os.getpid(), start_ticks=start_ticks(os.getpid()), source_sha256=file_sha(__file__),
                    source_config=str(config_path), out=str(out), native_started=False, GPU_started=False,
                    TEST_runtime_status_read=False, TEST_raw_data_read=False, TEST_manifest_read=False))
        history = read_history(state / 'history.jsonl')
        last_data = empty_data(); source = None; source_sha = None; first = True; old_signature = None
        while True:
            stop = observation_stop(state, event)
            if stop:
                data = deepcopy(last_data)
            else:
                try:
                    candidate, checksum = load_source(config_path, stop=lambda: observation_stop(state, event))
                    if source_sha is not None and checksum != source_sha:
                        raise ValueError('immutable observer source config changed; preserve this state')
                    source, source_sha = candidate, checksum
                    data = collect(source, stop=lambda: observation_stop(state, event))
                    if not data['admitted'] and last_data['admitted']:
                        previous = deepcopy(last_data)
                        previous['errors'] = data['errors']; previous['release_complete'] = False
                        data = previous
                except ObservationStopped:
                    stop = True; data = deepcopy(last_data)
                except (OSError, ValueError, TypeError) as error:
                    data = deepcopy(last_data); data['errors'] = [str(error)]; data['release_complete'] = False
            stop = stop or observation_stop(state, event)
            signature = sha_bytes(json.dumps(dict(errors=data['errors'], failed=data['failed'], admitted=data['admitted']), sort_keys=True).encode())
            changed = old_signature is not None and signature != old_signature
            old_signature = signature
            due = time.time() >= next_due
            if first or stop or data['release_complete'] or changed or due:
                reason = ('收到STOP／信号，仅停止状态观察并保留部分结果' if stop else
                          '实际754独立TEST审计及827终态已确认' if data['release_complete'] else
                          '观察器启动快照' if first else '读取／基础设施状态变化' if changed else '每30分钟定时更新')
                next_due = advance_deadline(next_due, time.time(), 1800)
                save_report(state, out, data, source, source_sha, 1800, reason, history, stopping=stop, next_due=next_due)
                first = False
            last_data = data
            if stop or data['release_complete']:
                atomic_json(state / 'terminal.json', dict(complete=True, observer_only=True,
                            final_TEST_independently_released=data['release_complete'], user_or_signal_stop=stop,
                            finished_utc=datetime.now(timezone.utc).isoformat(), native_started=False, GPU_started=False,
                            TEST_runtime_status_read=data['test_runtime_status_read'], TEST_raw_data_read=False,
                            TEST_manifest_read=False, TEST_case_files_read=False,
                            TEST_aggregate_results_read=data['test_aggregate_results_read']))
                return
            event.wait(min(10., max(.1, next_due - time.time())))


if __name__ == '__main__':
    main()
