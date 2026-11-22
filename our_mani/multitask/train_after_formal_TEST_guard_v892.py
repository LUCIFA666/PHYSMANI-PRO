"""Unactivated, stdlib-only permission boundary for a later TRAIN diagnostic.

Import and context-manager construction perform no I/O. There is no main,
worker import, simulator, preflight, repair, process launch or score selection.
An explicit future enabled=True scope must observe the genuine complete827 /
original754 TEST audit, acquire the existing shared locks and establish an
empty complete host process inventory. It retains the locks until the caller
has completed its own cleanup. Test callbacks cannot create a production permit.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

VERSION = 'r10_inert_post_complete_formal_TEST_TRAIN_guard_v892'
CONTROLLER = 'runtime/framework/ordered_env_resume_VAL_TEST_v827'
TEST_RUNTIME = 'runtime/framework/formal_TEST_after_environment_resume_v753'
ART = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823'
RESULT = ART + '/TEST_results.json'
VAL_RELEASE = ART + '/VAL_release.json'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup'
SUITE, SUITE_LOCK, SOURCE_LOCK = (SETUP + '/' + n for n in ('suite.json', 'frozen_inputs.json', 'test_source_inputs.json'))
OWNER_STAGE, AUDIT_STAGE = (CONTROLLER + '/stages/' + n for n in ('formal_TEST_1920', 'independent_TEST_results'))
VAL_OWNER_STAGE = CONTROLLER + '/stages/environment_resume_VAL_945'
CONTROLLER_SOURCE = 'scripts/continue_environment_resumed_VAL_TEST_v827_r10.py'
OWNER_SOURCE = 'scripts/run_formal_suite16_owner_v753_r10.py'
AUDITOR_SOURCE = 'scripts/audit_formal_suite16_results_v754_r10.py'
VAL_OWNER_SOURCE = 'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py'
SOURCE_SHA = {
    CONTROLLER_SOURCE: '6f0c13d410aba891df7ae51acfeacbe9714cfb9dabf87c5850763a3326f86e30',
    OWNER_SOURCE: 'ece7f0d57f4e533b22a62d35a4b5220deedb88dbfc44767b060fc889b30cb183',
    AUDITOR_SOURCE: '3b35c47796d0294ace8dbcdef7f5ca49fa4a227d8c3f2488e0864af08d599301',
    VAL_OWNER_SOURCE: 'd0cb699117fbd74f949d27b4a863249d4ab27a384f7a53756dc4da709a919ab4',
}
POLICY = 'eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
PROFILE = 'd9a36e5ba8a73123c5deb65229d7fa151089145dca4b55c68c1e9ad1dec4e562'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
TASKS = tuple(sorted((
    'beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed',
    'insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed',
    'moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed',
    'pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed',
    'place_cups_on_rotating_frame', 'place_cups_on_rotating_frame_high_speed',
    'push_moving_button', 'push_moving_button_high_speed',
    'put_rubbish_in_moving_bin', 'put_rubbish_in_moving_bin_high_speed',
    'remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')))
CONTROL = dict(path='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_control.json',
               sha256='435d684b415b26461005556f339f82c731a90ede873e8b64db2f6f6533d3fa36')
CONTROL_LOCK = dict(path='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_inputs.json',
                    sha256='45cc32204b28a3c3990d4d9f8a86488dd73d211e7b990a80e2152b9706227258')
ENVIRONMENT_POINTERS = dict(
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json', sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    wrapper=dict(path='scripts/run_frozen_system_environment_v820_r10.py', sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    namespace_qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json', sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))
PROTOCOL = dict(schema='r10_author_action_outcome_protocol_v741',
    implementation=dict(path='our_mani/multitask/native_outcome_sampling_v741.py', sha256='2289a93fcb9d8c27a700d66164da7e2ea8fbcc0fa4f0c775e986ab9f40a9cb12'),
    runtime_consumer=dict(path='scripts/run_task_executor_smoke_author_suite16_v744_r10.py', sha256='1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d'),
    original_runtime=dict(path='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py', sha256='e1ba1b1409762329d7624fc4e8e27cf2475860142d63de39e05f4fb08f58941f'),
    failure_sampling='current_native_result_at_explicit_action_end', success_sampling='original_per_tick_success_latch',
    pose_action_max_ticks=10, public_hold_max_ticks=1, budget_partial_action_consumed=True,
    native_RML_parameters_unchanged=True, block_native_fail_conditions=0, block_extra_task_success_calls=0)
DRIVER = dict(path='scripts/run_formal_case_author_outcome_v751_r10.py', sha256='06053736b8cb868c6783a4bb8a81c3f36698ef8137bf62f1e5f53e007ebe9d36')
GUARD = dict(path='our_mani/multitask/formal_suite_admission_v728.py', sha256='f325b48a9778e02aa9d6a7bb32593aa11c454903f87cddb59f1c0248b5395bea')
WORKER = dict(path='scripts/run_task_executor_smoke_author_outcome_v743_r10.py', sha256='714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174')
LOCKS = ('runtime/framework/ordered_resume_VAL_TEST_v775.lock',
         'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock',
         'runtime/framework/fixed_TRAIN104_geometry_native.lock')
STOP_PATHS = ('runtime/framework/multitask_r10_STOP',
    CONTROLLER + '/STOP', TEST_RUNTIME + '/STOP', 'runtime/framework/formal_VAL_env_resume_v823/STOP',
    'runtime/framework/ordered_resume_VAL_TEST_v775/STOP', 'runtime/framework/formal_VAL_resume_v772/STOP',
    'runtime/framework/train_after_formal_TEST_v892/STOP')
NATIVE_PREFIXES = ('run_geometry_', 'run_native_', 'run_continuous_reference_pair_', 'collect_multitask_',
    'run_ready_candidate_', 'launch_multitask_', 'run_multitask_fresh_queue', 'run_task_formal_', 'run_trained_task_',
    'evaluate_multitask_case_', 'run_PhysMani_task_', 'run_task_executor_smoke_', 'run_formal_case_',
    'run_formal_suite16_owner_', 'run_resumed_formal_VAL_owner_', 'run_environment_resumed_formal_VAL_owner_',
    'continue_resumed_VAL_TEST_', 'continue_environment_resumed_VAL_TEST_', 'continue_ordered_', 'continue_author_outcome_',
    'run_frozen_environment_TRAIN_canary_', 'diagnose_fixed_TRAIN_', 'diagnose_TRAIN_', 'run_TRAIN_')
_TOKEN = object()


def require(condition, message):
    if not condition:
        raise ValueError('post-TEST TRAIN guard: ' + message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def option(argv, flag):
    require(isinstance(argv, list) and argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'one explicit argv option required: ' + flag)
    return argv[argv.index(flag) + 1]


def _pointer(value, name=None):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'} and isinstance(value['path'], str)
            and valid_sha(value['sha256']) and (name is None or value['path'] == name), 'exact canonical evidence pointer required')


def _allowed(name):
    if name in {RESULT, *SOURCE_SHA, CONTROLLER + '/launch.json', CONTROLLER + '/terminal.json',
                TEST_RUNTIME + '/launch.json', TEST_RUNTIME + '/terminal.json', TEST_RUNTIME + '/plan.json'}:
        return True
    return any(name == stage + '/' + filename for stage in (OWNER_STAGE, AUDIT_STAGE, VAL_OWNER_STAGE)
               for filename in ('launch.json', 'terminal.json', 'stdout.log') if stage == AUDIT_STAGE or filename != 'stdout.log')


class _Metadata:
    def __init__(self, root, callback):
        self.root, self.callback, self.inputs = Path(root).resolve(), callback, {}

    def raw(self, name, expected=None):
        require(isinstance(name, str) and not Path(name).is_absolute() and str(PurePosixPath(name)) == name
                and '..' not in PurePosixPath(name).parts and _allowed(name), 'non-metadata read forbidden: ' + str(name))
        if self.callback is not None:
            raw = self.callback(name)
        else:
            path = (self.root / name).resolve(strict=True)
            require(path.is_relative_to(self.root) and str(path.relative_to(self.root)) == name, 'symlink or escaping metadata refused')
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            require(sig(before) == sig(after), 'completed metadata changed while reading')
        require(type(raw) is bytes, 'metadata reader must return bytes')
        checksum = sha(raw)
        require((expected is None or valid_sha(expected) and checksum == expected)
                and (name not in self.inputs or self.inputs[name] == checksum), 'metadata SHA changed: ' + name)
        self.inputs[name] = checksum
        return raw

    def json(self, name, expected=None):
        value = json.loads(self.raw(name, expected))
        require(isinstance(value, dict), 'JSON metadata object required')
        return value

    def pointer(self, name):
        return dict(path=name, sha256=self.inputs[name])


def _clean_stage(value):
    require(value.get('complete') is True and type(value.get('returncode')) is int and value['returncode'] == 0
            and value.get('automatic_retry') is False, 'clean original827 child-stage completion required')


def _pid(record):
    require(type(record.get('pid')) is int and record['pid'] > 0 and type(record.get('start_ticks')) is int
            and record['start_ticks'] > 0, 'actual PID/start identity required')
    return dict(pid=record['pid'], start_ticks=record['start_ticks'])


def _stage(value, source, argv, controller):
    _pid(value)
    require(value.get('argv') == argv and value.get('source_sha256') == SOURCE_SHA[source]
            and value.get('automatic_retry') is False
            and same(value.get('inherited_frozen_environment'), controller['frozen_environment'])
            and value.get('current_mount_namespace') == controller['frozen_environment']['current_mount_namespace'],
            'actual827 child stage argv/source/environment changed')


def _lookup(mapping, root, name):
    values = {mapping[key] for key in (name, str(root / name)) if key in mapping}
    require(len(values) == 1 and valid_sha(next(iter(values))), 'recorded evidence SHA missing or conflicting: ' + name)
    return next(iter(values))


def require_complete_TEST_metadata(root, expected_test_result_sha256, *, read_bytes=None):
    """Future explicit call: read completion evidence only, never cases or source data.

    A supplied read_bytes(name) callback is for pure CPU tests. The expected
    results SHA must be supplied externally, never taken from the result itself.
    """
    require(valid_sha(expected_test_result_sha256), 'externally bound final TEST_results SHA required')
    r = _Metadata(root, read_bytes)
    for source, expected in SOURCE_SHA.items():
        r.raw(source, expected)
    controller = r.json(CONTROLLER + '/launch.json')
    known = [_pid(controller)]
    require(controller.get('source_sha256') == SOURCE_SHA[CONTROLLER_SOURCE]
            and controller.get('resume_control') == CONTROL and controller.get('resume_control_lock') == CONTROL_LOCK,
            'actual frozen827 controller required')
    env = controller.get('frozen_environment', {})
    require(env.get('schema') == 'r10_actual_frozen_OS_namespace_receipt_v820' and env.get('complete') is True
            and env.get('all_current_OS_targets_full_byte_SHA_verified') is True and env.get('checked_OS_files') == 6418
            and env.get('host_PID_network_namespace_preserved') is True and env.get('proc_or_dev_replaced') is False
            and env.get('task_parameters_or_strategy_changed') is False
            and env.get('observer_pid') == controller['pid'] and env.get('pid') == controller['pid']
            and env.get('observer_start_ticks') == controller['start_ticks']
            and isinstance(env.get('current_mount_namespace'), str) and env['current_mount_namespace'] == env.get('mount_namespace')
            and env['current_mount_namespace'] != env.get('reference_mount_namespace')
            and all(env.get(k) == p for k, p in ENVIRONMENT_POINTERS.items()), 'recorded original820 controller environment required')
    final = r.json(CONTROLLER + '/terminal.json')
    require(final.get('complete') is True and final.get('automatic_retry') is False and final.get('TEST_results') == RESULT,
            'actual827 final completion required; old775 terminal is irrelevant')
    val_stage = r.json(VAL_OWNER_STAGE + '/launch.json')
    original_setup = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup'
    val_common = ['--suite', original_setup + '/suite.json', '--suite-lock', original_setup + '/frozen_inputs.json',
        '--source-inputs', original_setup + '/validation_source_inputs.json', '--split', 'validation',
        '--runtime', 'runtime/framework/formal_VAL_env_resume_v823', '--resume-control', CONTROL['path'], '--resume-lock', CONTROL_LOCK['path']]
    _stage(val_stage, VAL_OWNER_SOURCE, [str(r.root / '.conda-env/bin/python'), '-u', VAL_OWNER_SOURCE,
        *val_common, '--out', ART + '/validation'], controller)
    known.append(_pid(val_stage))
    common = ['--suite', SUITE, '--suite-lock', SUITE_LOCK, '--source-inputs', SOURCE_LOCK,
              '--split', 'test', '--validation-release', VAL_RELEASE, '--runtime', TEST_RUNTIME]
    stages = {}
    for stage, source, output in ((OWNER_STAGE, OWNER_SOURCE, ART + '/test'), (AUDIT_STAGE, AUDITOR_SOURCE, RESULT)):
        launch, terminal = r.json(stage + '/launch.json'), r.json(stage + '/terminal.json')
        _stage(launch, source, [str(r.root / '.conda-env/bin/python'), '-u', source, *common, '--out', output], controller)
        _clean_stage(terminal)
        stages[source] = launch
        known.append(_pid(launch))
    stdout = r.json(AUDIT_STAGE + '/stdout.log')
    require(stdout.get('complete') is True and stdout.get('split') == 'test' and stdout.get('cases') == 1920
            and stdout.get('TEST_permitted') is False and stdout.get('out') == RESULT,
            'actual754 completed TEST stdout required; TEST_permitted=False is normal')
    owner = r.json(TEST_RUNTIME + '/launch.json')
    owner_end = r.json(TEST_RUNTIME + '/terminal.json')
    require(_pid(owner) == _pid(stages[OWNER_SOURCE]) and owner.get('source_sha256') == SOURCE_SHA[OWNER_SOURCE]
            and owner.get('split') == 'test' and owner.get('policy_sha256') == POLICY
            and owner.get('automatic_retry') is False and owner.get('execution_protocol_changed') is True
            and same(owner.get('author_outcome_protocol'), PROTOCOL) and owner.get('admission_driver') == DRIVER
            and owner.get('admission_guard') == GUARD and owner.get('native_runtime') == PROTOCOL['runtime_consumer']
            and owner.get('original_native_runtime') == PROTOCOL['original_runtime']
            and valid_sha(owner.get('split_source_lock_sha256')), 'actual fresh original753 TEST owner required')
    require(owner_end.get('complete') is True and owner_end.get('split') == 'test'
            and owner_end.get('expected') == owner_end.get('completed') == 1920 and owner_end.get('failed') == 0
            and owner_end.get('stop_requested') is False and owner_end.get('automatic_retry') is False
            and owner_end.get('policy_sha256') == POLICY and owner_end.get('execution_protocol_changed') is True
            and same(owner_end.get('author_outcome_protocol'), PROTOCOL)
            and owner_end.get('CPU_ready_unexecuted') == [] and type(owner_end.get('CPU_not_launched')) is int
            and owner_end['CPU_not_launched'] == 0, 'actual complete1920/failed0 original753 terminal required')
    # No TEST_results byte is consumed until the real controller, independent
    # audit stage and owner have each proved their completed state above.
    result = r.json(RESULT, expected_test_result_sha256)
    require(result.get('complete') is True and result.get('split') == 'test'
            and result.get('source_cases') == 640 and result.get('method_cases') == 1920
            and result.get('completed_method_cases') == 1920 and result.get('methods') == list(METHODS)
            and result.get('tasks') == list(TASKS) and result.get('policy_sha256') == POLICY
            and result.get('common_RML_profile_sha256') == PROFILE, 'complete original640/1920 independent TEST audit required')
    for key, expected in dict(TEST_used_for_fitting=False, motion_is_diagnostic_only=True,
            motion_metrics_required_for_all_methods=True, native_task_success_unchanged=True,
            native_success_is_not_a_release_threshold=True, source_outcome_filtering=False,
            native_failures_retained=True, Value_NGFF_FASTER_used=False, execution_protocol_changed=True).items():
        require(result.get(key) is expected, 'native/diagnostic semantics changed: ' + key)
    source_pointer = result.get('source_input_lock')
    _pointer(source_pointer, SOURCE_LOCK)
    require(source_pointer['sha256'] == owner['split_source_lock_sha256']
            and result.get('source_sha256') == SOURCE_SHA[AUDITOR_SOURCE]
            and same(result.get('author_outcome_protocol'), PROTOCOL) and result.get('admission_driver') == DRIVER
            and result.get('admission_guard') == GUARD and result.get('native_runtime') == PROTOCOL['runtime_consumer']
            and result.get('original_native_runtime') == PROTOCOL['original_runtime']
            and result.get('actual_stage_owner') == r.pointer(TEST_RUNTIME + '/terminal.json'),
            'actual754 owner/policy/protocol/source-lock binding required')
    plan = r.json(TEST_RUNTIME + '/plan.json')
    rows = plan.get('jobs')
    require(plan.get('split') == 'test' and plan.get('expected_cases') == 1920 and plan.get('policy_sha256') == POLICY
            and plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False
            and isinstance(rows, list) and len(rows) == 1920, 'complete original fresh TEST plan required')
    cases, drivers = result.get('cases'), result.get('admission_driver_receipts')
    require(isinstance(cases, list) and len(cases) == 1920 and isinstance(drivers, list) and len(drivers) == 1920,
            '1920 real case/driver pointers required')
    keys, ids, case_paths, driver_paths, sources = set(), set(), set(), set(), {}
    for row, case, driver in zip(rows, cases, drivers):
        task, raw, method = row.get('task'), row.get('raw'), row.get('method')
        key = (task, raw, method)
        require(task in TASKS and isinstance(raw, str) and raw and '/' not in raw and '__policy_aug' not in raw
                and method in METHODS and key not in keys and row.get('job_id') == '_'.join(key)
                and row['job_id'] not in ids and valid_sha(row.get('source_row_sha256'))
                and not any(k in row for k in ('evidence_runtime', 'imported_completed', 'previous_row_sha256', 'previous_imported_completed')),
                'original unique raw/source/method grid without imports or augmentation required')
        source_key = (task, raw)
        require(source_key not in sources or sources[source_key] == row['source_row_sha256'], 'three methods source SHA mismatch')
        sources[source_key] = row['source_row_sha256']; keys.add(key); ids.add(row['job_id'])
        require(row.get('admission_driver') == DRIVER and row.get('admission_guard') == GUARD
                and row.get('case_worker') == WORKER and row.get('native_runtime') == PROTOCOL['runtime_consumer']
                and row.get('original_native_runtime') == PROTOCOL['original_runtime']
                and same(row.get('author_outcome_protocol'), PROTOCOL), 'actual751/743/744/664 plan bindings changed')
        argv = row.get('argv')
        require(argv[:3] == ['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh', DRIVER['path']],
                'original formal launcher required')
        output = ART + '/test/' + '/'.join(key)
        for flag, value in (('--split', 'test'), ('--task', task), ('--raw-episode-id', raw), ('--method', method),
                ('--suite', SUITE), ('--suite-lock', SUITE_LOCK), ('--source-inputs', SOURCE_LOCK),
                ('--validation-release', VAL_RELEASE), ('--out', output)):
            require(option(argv, flag) == value, 'original formal argv changed: ' + flag)
        _pointer(case, output + '/case.json'); _pointer(driver, output + '/admission_driver_receipt.json')
        require(case['path'] not in case_paths and driver['path'] not in driver_paths, 'duplicate case/driver pointer')
        case_paths.add(case['path']); driver_paths.add(driver['path'])
    require(len(sources) == 640 and keys == {(task, raw, method) for task, raw in sources for method in METHODS}
            and all(sum(t == task for t, _ in sources) == 40 for task in TASKS), 'exact640 originals x3 completed TEST bijection required')
    inputs = result.get('inputs_sha256', {})
    require(isinstance(inputs, dict), 'independent754 input-evidence SHA map required')
    require(_lookup(inputs, r.root, SOURCE_LOCK) == source_pointer['sha256'], 'actual754 source-lock SHA evidence differs')
    for name in (TEST_RUNTIME + '/launch.json', TEST_RUNTIME + '/terminal.json', TEST_RUNTIME + '/plan.json', OWNER_SOURCE, AUDITOR_SOURCE):
        require(_lookup(inputs, r.root, name) == r.inputs[name], 'actual754 evidence SHA mismatch: ' + name)
    return dict(schema=VERSION, complete=True, expected_cases=1920, policy_sha256=POLICY,
        TEST_results=r.pointer(RESULT), actual_controller=r.pointer(CONTROLLER + '/launch.json'),
        actual_controller_terminal=r.pointer(CONTROLLER + '/terminal.json'), actual_owner=r.pointer(TEST_RUNTIME + '/launch.json'),
        actual_owner_terminal=r.pointer(TEST_RUNTIME + '/terminal.json'), actual_audit_stage=r.pointer(AUDIT_STAGE + '/launch.json'),
        known_processes=known, inputs_sha256=r.inputs, source_case_bytes_read=False, scores_used_for_gate=False,
        motion6_used_for_gate=False, old775_terminal_required=False,
        namespace_evidence_scope='recorded827 stage inheritance only; original753 has no per-child live namespace receipts')


def _stop_guard(root):
    if any((Path(root) / name).exists() for name in STOP_PATHS):
        raise InterruptedError('global/formal/diagnostic STOP prevents a new TRAIN diagnostic')


@contextmanager
def _reserve(root):
    import fcntl
    with ExitStack() as stack:
        for name in LOCKS:
            path = Path(root) / name
            require(path.parent.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(), 'project lock path required')
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('formal/global native resource lock busy: ' + name) from error
        yield


def _is_alive(record):
    _pid(record)
    try:
        fields = (Path('/proc') / str(record['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] not in ('Z', 'X') and int(fields[19]) == record['start_ticks']


def _inventory(root):
    """Same-UID host inventory, excluding only this process and actual ancestors."""
    root, proc, uid = Path(root).resolve(), Path('/proc'), os.getuid()
    def stat(path):
        f = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return f[0], int(f[1]), int(f[19])
    own, parent = set(), os.getpid()
    while parent > 0 and parent not in own:
        own.add(parent); parent = stat(proc / str(parent))[1]
    comm = (proc / '1/comm').read_text().strip()
    active, unresolved = [], []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own: continue
        try:
            before = stat(entry)
            if before[0] in ('Z', 'X'): continue
            process_uid = entry.stat().st_uid
            try:
                tokens = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            except PermissionError:
                if process_uid == uid: unresolved.append(dict(pid=int(entry.name), reason='same-UID cmdline unreadable'))
                continue
            matches = [Path(x).name for x in tokens if '\n' not in x and '\r' not in x
                       and Path(x).name.endswith('.py') and Path(x).name.startswith(NATIVE_PREFIXES)]
            if not matches or process_uid != uid: continue
            cwd = Path(os.readlink(entry / 'cwd')).resolve()
            after = stat(entry)
            if before[2] == after[2] and after[0] not in ('Z', 'X') and cwd == root:
                active.append(dict(pid=int(entry.name), start_ticks=after[2], scripts=matches))
        except (FileNotFoundError, ProcessLookupError): continue
        except (OSError, ValueError, IndexError) as error:
            unresolved.append(dict(pid=int(entry.name), reason=repr(error)))
    return dict(host_visibility_available=comm in ('systemd', 'init'), same_uid_inventory_incomplete=bool(unresolved),
                live_project_native_processes=active, unresolved_processes=unresolved,
                excluded_self_and_ancestor_pids=sorted(own), host_namespace_links_claimed_readable=False)


class _Permit:
    def __init__(self, root, proof, inventory, stop_check, test_only):
        self.root, self.pid, self.proof, self.inventory = Path(root).resolve(), os.getpid(), proof, inventory
        self._token, self.active, self._stop_check, self.test_only = _TOKEN, True, stop_check, test_only


def require_permit(permit, root, *, allow_test_callbacks=False):
    require(isinstance(permit, _Permit) and permit._token is _TOKEN and permit.active
            and permit.pid == os.getpid() and permit.root == Path(root).resolve(), 'live owned resource permit required')
    require(not permit.test_only or allow_test_callbacks is True, 'CPU-injected callbacks cannot authorize native execution')
    permit._stop_check(permit.root)
    return permit.proof


@contextmanager
def reserved_resources_after_TEST(root, expected_test_result_sha256, *, enabled=False, read_bytes=None,
                                  stop_check=None, reserve=None, inventory=None, is_alive=None):
    """Future opt-in lease. Caller must retain this scope through native cleanup.

    All injected callbacks are test-only: require_permit rejects their permit
    unless its explicit allow_test_callbacks=True CPU-testing option is used.
    No execution entry is installed or activated by this module.
    """
    require(enabled is True, 'future explicit enabled=True opt-in required; guard remains inert')
    root = Path(root).resolve()
    test_only = any(x is not None for x in (read_bytes, stop_check, reserve, inventory, is_alive))
    stop, hold, scan, alive = stop_check or _stop_guard, reserve or _reserve, inventory or _inventory, is_alive or _is_alive
    stop(root)
    permit = None
    with hold(root):
        stop(root)
        proof = require_complete_TEST_metadata(root, expected_test_result_sha256, read_bytes=read_bytes)
        require(not any(alive(record) for record in proof['known_processes']), 'formal controller/owner/auditor still live')
        snapshot = scan(root)
        require(isinstance(snapshot, dict) and snapshot.get('host_visibility_available') is True
                and snapshot.get('same_uid_inventory_incomplete') is False
                and snapshot.get('live_project_native_processes') == [], 'complete empty host project-native inventory required')
        stop(root)
        permit = _Permit(root, proof, snapshot, stop, test_only)
        try:
            yield permit
        finally:
            permit.active = False
