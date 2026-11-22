"""Versioned read-only task audit with explicitly separate reporting dependencies.

This observes the already authorized 827/753 run; it cannot launch, retry,
admit or release a job. Only a task's exact original 40 sources x 3 completed
cases are read. Other TEST tasks may still be running. Raw reset/model/OS
contents and transitive frozen-input graphs are never opened or verified.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HELPER = 'scripts/summarize_formal_TEST_motion_v851_r10.py'
HELPER_SHA = 'd31fb9cad92f3346ccdd55290f72c853365719de5f883ca9ff33e3710d4839eb'
SUBSET_HELPER = 'scripts/audit_completed_VAL_task_v833_r10.py'
SUBSET_HELPER_SHA = '73c2c0ccfbff95ec758d596db717ea098342ff1530d5b6a926e436a983b0053e'


def load_helper():
    raw = (ROOT / HELPER).read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('audited TEST helper source SHA changed')
    module = types.ModuleType('audited_TEST_metadata_v851')
    module.__file__ = str(ROOT / HELPER)
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


T = load_helper()
require, digest, valid_sha, option = T.require, T.digest, T.valid_sha, T.option
METHODS, TASKS, POLICY, PROFILE = T.METHODS, T.TASKS, T.POLICY, T.PROFILE
RUNTIME, ART, CONTROL, LOCK = T.RUNTIME, T.ART, T.CONTROL, T.LOCK
SUITE, SUITE_LOCK, SOURCE_LOCK = T.SUITE, T.SUITE_LOCK, T.SOURCE_LOCK
MANIFEST, ORIGINAL_MANIFEST, ROUTES = T.MANIFEST, T.ORIGINAL_MANIFEST, T.ROUTES
VAL_RELEASE, CONTROLLER, CONTROLLER_SOURCE = T.VAL_RELEASE, T.CONTROLLER, T.CONTROLLER_SOURCE
OWNER, OWNER_STAGE = T.OWNER, T.OWNER_STAGE
VAL_AUDITOR = 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py'
VAL_AUDIT_STAGE = T.H.AUDIT_STAGE
VALIDATORS = ('our_mani/multitask/author_outcome_receipt_v749.py',
              'our_mani/multitask/common_rml_protocol_v530.py',
              'our_mani/multitask/evaluation.py')
REPORTING_DEPENDENCIES = {'our_mani/multitask/three_method_release16.py':
    'faf77babdd431be41c5765a34dfb1b08a18e7bf8ec1a70b6f26b78f7210477f3'}
ORIGINAL_SUBSET_AUDITOR = 'scripts/audit_completed_TEST_task_v887_r10.py'
ORIGINAL_SUBSET_AUDITOR_SHA = '74fb8285b278fb8b04a826f16bba65d15abb0e1333c82ce94be2a0e22d55c2ec'
PREVIOUS_OBSERVATION = 'runtime/framework/TEST_task_subset_audit_v887_observed_failure_v908.json'
PREVIOUS_OBSERVATION_SHA = '90c868628dd38a67c68c59d4067909565823e68c32a9319a5ccfa53e9cd34a70'
SOURCES = T.SOURCES | {VAL_AUDITOR, *VALIDATORS, *REPORTING_DEPENDENCIES, HELPER, SUBSET_HELPER, ORIGINAL_SUBSET_AUDITOR}
PINS = {CONTROL: T.PINS[CONTROL], LOCK: T.PINS[LOCK], HELPER: HELPER_SHA,
        T.HELPER: T.HELPER_SHA, SUBSET_HELPER: SUBSET_HELPER_SHA,
        VAL_RELEASE: '03b7ba8677e333512d5396411248784497ac72a82ffc68206865b68e2b0a577b',
        SOURCE_LOCK: '53ca9d2f8ee23c5245e9ebd2155d3e81bfbb17bd5fbc2ed4d4282975cfb79e4c',
        ORIGINAL_SUBSET_AUDITOR: ORIGINAL_SUBSET_AUDITOR_SHA,
        PREVIOUS_OBSERVATION: PREVIOUS_OBSERVATION_SHA, **REPORTING_DEPENDENCIES}
COLLISION_REASON = 'public_forecast_collision_on_RML_target_sweep'
BLOCKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
NAMESPACE_SCOPE = ('827 TEST-stage launch records inheritance of the controller820 proof; original753 '
                   'does not write per-child namespace receipts. No new live namespace measurement is claimed.')


def allowed(name):
    if name in {*PINS, *SOURCES, SUITE, SUITE_LOCK, MANIFEST, ORIGINAL_MANIFEST, ROUTES,
                CONTROLLER + '/launch.json', RUNTIME + '/launch.json', RUNTIME + '/plan.json',
                RUNTIME + '/status.json'}:
        return True
    if name == OWNER_STAGE + '/launch.json':
        return True
    if name.startswith(VAL_AUDIT_STAGE + '/') and name[len(VAL_AUDIT_STAGE) + 1:] in ('launch.json', 'terminal.json', 'stdout.log'):
        return True
    parts = name[len(RUNTIME) + 1:].split('/') if name.startswith(RUNTIME + '/') else []
    if len(parts) == 3 and parts[0] in ('CPU', 'native') and parts[2] in ('launch.json', 'terminal.json', 'CPU_preflight.json', 'stdout.log', 'stderr.log'):
        return bool(parts[1])
    parts = name[len(ART + '/test/'):].split('/') if name.startswith(ART + '/test/') else []
    return (len(parts) == 4 and parts[0] in TASKS and bool(parts[1]) and parts[2] in METHODS
            and parts[3] in ('case.json', 'admission_driver_receipt.json'))


class Reader:
    """Only declared evidence bytes; mutable status is embedded, never frozen by path."""
    lookup = T.Reader.lookup

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs, self.original, self.frozen, self.evidence = {}, {}, {}, {}

    def _path(self, name):
        require(isinstance(name, str) and not Path(name).is_absolute()
                and str(PurePosixPath(name)) == name and '..' not in PurePosixPath(name).parts
                and allowed(name), 'non-evidence or undeclared input refused before open: ' + str(name))
        path = (self.root / name).resolve(strict=True)
        require(path.is_relative_to(self.root) and str(path.relative_to(self.root)) == name,
                'canonical evidence required; symlink aliases refused before open')
        return path

    def _read(self, name, *, expected=None, frozen=False, original=False, record=True):
        path = self._path(name)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(signature(before) == signature(after), 'evidence changed while being read: ' + name)
        checksum = T.sha_bytes(raw)
        wanted = [x for x in (expected, PINS.get(name), self.lookup(self.frozen, name) if frozen else None,
                              self.lookup(self.original, name) if original else None) if x is not None]
        require(all(valid_sha(x) and x == checksum for x in wanted), 'input SHA changed: ' + name)
        if record:
            require(name not in self.inputs or self.inputs[name] == checksum, 'input changed during audit: ' + name)
            self.inputs[name] = checksum
        return raw

    def raw(self, name, **kwargs):
        return self._read(name, **kwargs)

    def json(self, name, **kwargs):
        return json.loads(self.raw(name, **kwargs))

    def pointer(self, name):
        return dict(path=name, sha256=self.inputs[name])

    def pointed(self, pointer, **kwargs):
        require_pointer(pointer)
        return self.json(pointer['path'], expected=pointer['sha256'], **kwargs)

    def snapshot(self, name):
        require(name == RUNTIME + '/status.json', 'only actual TEST status may be snapshotted')
        # The owner atomically replaces this pathname while other tasks run.
        # Read one file descriptor's bytes once; a before/after pathname-stat
        # comparison would wrongly reject a complete old snapshot on rename.
        raw = self._path(name).read_bytes()
        text = raw.decode('utf-8')
        return json.loads(text), dict(path=name, sha256=T.sha_bytes(raw)), text


def require_pointer(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and isinstance(value['path'], str) and valid_sha(value['sha256']), 'exact evidence path/SHA pointer required')


def verify_reporting_dependencies(reader):
    """Pin report-only reducers without inventing historical lock membership.

    A missing original scientific validator is still an error in the ordinary
    frozen/original read. Only this explicit, narrow dependency list may be
    absent from those older maps. Any declaration that is present must agree.
    """
    records = []
    for name, expected in REPORTING_DEPENDENCIES.items():
        membership = {}
        for key in ('original', 'frozen'):
            mapping = getattr(reader, key)
            require(isinstance(mapping, dict), 'explicit old lock metadata required')
            values = [mapping[path] for path in (name, str(reader.root/name)) if path in mapping]
            require(not values or all(valid_sha(value) and value == expected for value in values),
                    'existing lock declaration conflicts with report-only source: ' + name)
            membership[key] = bool(values)
        reader.raw(name, expected=expected)
        records.append(dict(path=name, sha256=expected, scope='supplementary_readonly_reporting_dependency',
            used_for='derive diagnostic motion_pass from already signed native case fields',
            original_physical_input_lock_member=membership['original'],
            original_TEST_suite_input_lock_member=membership['frozen'],
            independent_reporting_source_pin=True, old_input_lock_amended=False,
            proof_of_original_physical_freeze_membership_claimed=False))
    return records


def require_previous_observation(reader, path, checksum):
    require(path == PREVIOUS_OBSERVATION and checksum == PREVIOUS_OBSERVATION_SHA,
            'only the exact root-observed prior reporting failure may be referenced')
    record = reader.json(path, expected=checksum)
    require(record.get('schema') == 'r10_root_observed_readonly_TEST_subset_audit_failure_v908'
        and record.get('source_sha256') == ORIGINAL_SUBSET_AUDITOR_SHA
        and type(record.get('root_observed_actual_exit_code')) is int and record['root_observed_actual_exit_code'] == 1
        and record.get('historical_launch_PID_start_and_stderr_file_not_recorded') is True
        and record.get('old_launch_or_native_receipt_not_fabricated') is True
        and record.get('report_output_exists') is False and record.get('readonly_audit_not_a_native_attempt') is True
        and record.get('case_rerun_or_owner_restart_or_DRAIN') is False,
        'truthful observed reporter failure required; do not invent historical native/launch evidence')
    require(record.get('actual_command') == ['.conda-env/bin/python', '-B', ORIGINAL_SUBSET_AUDITOR,
        '--task', 'beat_the_rotating_buzz', '--out', RUNTIME+'/first_task_actual_TEST_results_v890.json'],
        'root observation must identify its real original887 command')
    return record


def require_stage_launch(stage, argv, source_sha, controller):
    require(stage.get('argv') == argv and stage.get('source_sha256') == source_sha
            and stage.get('automatic_retry') is False and type(stage.get('pid')) is int and stage['pid'] > 0
            and type(stage.get('start_ticks')) is int and stage['start_ticks'] > 0
            and stage.get('inherited_frozen_environment') == controller['frozen_environment']
            and stage.get('current_mount_namespace') == controller['frozen_environment']['current_mount_namespace'],
            'actual827 stage argv/PID/start/inherited environment required')


def require_prerequisites(reader):
    """Observe genuine VAL admission and launched TEST stage before TEST metadata."""
    val = reader.json(VAL_RELEASE, expected=PINS[VAL_RELEASE])
    T.H.require_release_header(val)
    control = reader.json(CONTROL)
    lock = reader.json(LOCK)
    require(lock.get('approved_for_execution') is True and control.get('policy_sha256') == POLICY,
            'actual frozen823 control graph required')
    reader.original = lock['files']
    for name in (CONTROLLER_SOURCE, VAL_AUDITOR, OWNER, T.AUDITOR, T.PREPARER, T.SOURCE_PREPARER):
        reader.raw(name, original=True)
    require(val.get('source_sha256') == reader.inputs[VAL_AUDITOR] == control['resume_auditor']['sha256']
            and control['resume_auditor']['path'] == VAL_AUDITOR
            and val.get('resume_control') == reader.pointer(CONTROL)
            and val.get('resume_control_lock') == reader.pointer(LOCK)
            and val.get('author_outcome_protocol') == control.get('author_outcome_protocol'),
            'actual824 complete VAL release/control/protocol prerequisite required')
    controller = reader.json(CONTROLLER + '/launch.json')
    require(controller.get('source_sha256') == reader.inputs[CONTROLLER_SOURCE]
            and controller.get('resume_control') == reader.pointer(CONTROL)
            and controller.get('resume_control_lock') == reader.pointer(LOCK)
            and type(controller.get('pid')) is int and controller['pid'] > 0
            and type(controller.get('start_ticks')) is int and controller['start_ticks'] > 0,
            'actual827 controller source/control/PID required')
    T.H.require_environment(controller.get('frozen_environment', {}), control, observer=controller['pid'])
    require(controller['frozen_environment'].get('observer_start_ticks') == controller['start_ticks'],
            'recorded controller environment start identity differs')
    stage = reader.json(VAL_AUDIT_STAGE + '/launch.json')
    end = reader.json(VAL_AUDIT_STAGE + '/terminal.json')
    common = ['--suite', control['suite']['path'], '--suite-lock', control['suite_lock']['path'],
              '--source-inputs', control['source_inputs']['path'], '--split', 'validation',
              '--runtime', ART.replace('artifacts/framework/multitask_r10/', 'runtime/framework/'),
              '--resume-control', CONTROL, '--resume-lock', LOCK]
    require_stage_launch(stage, [str(reader.root / '.conda-env/bin/python'), '-u', VAL_AUDITOR, *common, '--out', VAL_RELEASE],
                         reader.inputs[VAL_AUDITOR], controller)
    T.H.require_clean_terminal(end, controller_stage=True)
    proof = val.get('actual_auditor_environment', {})
    T.H.require_environment(proof, control, observer=stage['pid'])
    require(proof.get('observer_start_ticks') == stage['start_ticks']
            and proof.get('current_mount_namespace') == controller['frozen_environment']['current_mount_namespace'],
            'actual824 auditor PID/start/inherited environment differs')
    stdout = reader.json(VAL_AUDIT_STAGE + '/stdout.log')
    require(stdout.get('complete') is True and stdout.get('split') == 'validation' and stdout.get('cases') == 945
            and stdout.get('TEST_permitted') is True and stdout.get('out') == VAL_RELEASE
            and stdout.get('imported') == 591 and stdout.get('new_native') == 354,
            'actual824 clean VAL audit stdout required')
    test_stage = reader.json(OWNER_STAGE + '/launch.json')
    common = ['--suite', SUITE, '--suite-lock', SUITE_LOCK, '--source-inputs', SOURCE_LOCK, '--split', 'test',
              '--validation-release', VAL_RELEASE, '--runtime', RUNTIME]
    require_stage_launch(test_stage, [str(reader.root / '.conda-env/bin/python'), '-u', OWNER, *common, '--out', ART + '/test'],
                         reader.inputs[OWNER], controller)
    owner = reader.json(RUNTIME + '/launch.json')
    protocol = control['author_outcome_protocol']
    require(owner.get('source_sha256') == reader.inputs[OWNER] and owner.get('split') == 'test'
            and owner.get('pid') == test_stage['pid'] and owner.get('start_ticks') == test_stage['start_ticks']
            and owner.get('policy_sha256') == POLICY and owner.get('automatic_retry') is False
            and owner.get('execution_protocol_changed') is True and owner.get('author_outcome_protocol') == protocol
            and owner.get('native_runtime') == protocol['runtime_consumer']
            and owner.get('original_native_runtime') == protocol['original_runtime'],
            'actual original fresh753 TEST owner source/PID/protocol required')
    return control, controller, owner, test_stage, reader.pointer(VAL_RELEASE)


def select_complete_task(plan, status, task, sources):
    require(task in TASKS and plan.get('split') == status.get('split') == 'test'
            and plan.get('expected_cases') == status.get('expected') == 1920
            and status.get('automatic_retry') is False, 'original TEST-only1920 plan/status required')
    raws = {raw for t, raw in sources if t == task}
    require(len(raws) == 40, 'exact40 original task sources required; no padding')
    rows = [r for r in plan.get('jobs', []) if r.get('task') == task]
    expected = {(raw, method) for raw in raws for method in METHODS}
    require(len(rows) == 120 and {(r.get('raw'), r.get('method')) for r in rows} == expected
            and len({r.get('job_id') for r in rows}) == 120,
            'task must retain exact40 same original sources x3 methods')
    completed, failed = status.get('completed'), status.get('failed')
    require(isinstance(completed, dict) and isinstance(failed, dict), 'actual TEST completed/failed maps required')
    for row in rows:
        name = row['job_id']
        require(name in completed and name not in failed, 'task is pending or contains infrastructure failure: ' + name)
        result = completed[name]
        T.H.require_clean_terminal(result)
        out = option(row['argv'], '--out')
        for field, filename in (('case', 'case.json'), ('admission_driver_receipt', 'admission_driver_receipt.json')):
            require_pointer(result.get(field))
            require(result[field]['path'] == out + '/' + filename, 'status pointer does not match original TEST output')
    return sorted(rows, key=lambda r: (r['raw'], METHODS.index(r['method'])))


def require_CPU_evidence(cpu, stdout, row, expected, source_sha, *, policy_sha=POLICY):
    identity = cpu.get('identity', {})
    domain = cpu.get('component_domain', {})
    require(cpu == stdout and cpu.get('complete') is True and cpu.get('native_started') is False
            and cpu.get('GPU_started') is False and cpu.get('actual_split') == 'test'
            and cpu.get('execution_protocol_changed') is True and all(cpu.get(k) == v for k, v in expected.items())
            and domain.get('purpose') == 'evaluation' and domain.get('split') == 'test'
            and domain.get('task') == row['task'] and domain.get('method') == row['method']
            and domain.get('raw_episode_id') == row['raw'] and domain.get('policy_sha256') == policy_sha,
            'actual CPU-only TEST receipt/stdout/evaluation bindings required')
    require(identity.get('version') == 'r10_fresh_case_identity_v1'
            and identity.get('split') == 'test' and identity.get('task') == row['task'] and identity.get('method') == row['method']
            and identity.get('raw_episode_id') == identity.get('source_raw_episode_id') == row['raw']
            and identity.get('parent_raw_episode_id') is None and type(identity.get('replica')) is int and identity['replica'] == 0
            and identity.get('source_row_sha256') == row['source_row_sha256']
            and identity.get('config_sha256') == row['configuration_sha256']
            and identity.get('frozen_inputs_sha256') == row['input_lock_sha256']
            and identity.get('source_input_lock_sha256') == source_sha and identity.get('demo_only') is False
            and identity.get('exploration_seed') is None
            and identity.get('sha256') == digest({k: v for k, v in identity.items() if k != 'sha256'}),
            'original unaugmented TEST CPU identity/source/hash required')
    return identity


def require_child_launch(child, row, identity, owner, *, check):
    expected = {k: row[k] for k in T.H.BINDINGS}
    require(child.get('argv') == row['argv'] + (['--check'] if check else []) and child.get('check_only') is check
            and child.get('owner_pid') == owner['pid'] and type(child.get('launcher_pid')) is int and child['launcher_pid'] > 0
            and type(child.get('launcher_start_ticks')) is int and child['launcher_start_ticks'] > 0
            and child.get('native_slots') == (0 if check else 1)
            and child.get('GPU_slots') == (0 if check else int(row['method'] == 'physmani'))
            and child.get('execution_protocol_changed') is True
            and child.get('expected_identity_sha256') == (None if check else identity['sha256'])
            and all(child.get(k) == v for k, v in expected.items()),
            'actual753 child argv/identity/PID/resources/protocol required')


def require_driver_evidence(driver, row, case_pointer):
    require(driver.get('schema') == 'r10_explicit_author_outcome_formal_driver_v751'
            and driver.get('complete') is True and driver.get('actual_split') == 'test'
            and driver.get('execution_protocol_changed') is True and driver.get('case') == case_pointer
            and all(driver.get(k) == row[k] for k in T.H.BINDINGS),
            'actual751 external driver/case SHA/protocol required')


def require_case(case, row, source_sha, protocol, source_lock_sha):
    # These read-only validators never admit sources or inspect any source files.
    from our_mani.multitask.evaluation import validate_completed_case
    from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case
    from our_mani.multitask.common_rml_protocol_v530 import require_attestation
    T.require_case(case, row, source_sha, protocol, source_lock_sha)
    validate_completed_case(case, case['identity'])
    require(case.get('component_domain', {}).get('policy_sha256') == POLICY, 'actual TEST domain policy differs')
    require_sampling_case(case, protocol, block=row['task'] in BLOCKS)
    require_attestation(case.get('common_author_RML_parameters', {}), PROFILE)


def classify_case(case):
    # Same mutually exclusive diagnostic semantics as reviewed833: native fatal wins over budget.
    success, fatal = case.get('official_task_success'), case.get('native_terminal_failure')
    require(type(success) is bool and type(fatal) is bool and not (success and fatal), 'exclusive native boolean outcomes required')
    if success: return 'native_success'
    if fatal: return 'native_terminal_failure'
    if isinstance(case.get('policy_failure'), str) and case['policy_failure'].strip(): return 'explicit_policy_failure'
    clock = case.get('counted_clock', {})
    if (type(clock.get('ticks')) is int and clock['ticks'] == 200 and clock.get('dt') == .05
            and type(clock.get('seconds')) in (int, float) and math.isfinite(clock['seconds']) and abs(clock['seconds'] - 10.) <= 1e-6):
        return 'budget_exhausted_no_native_terminal'
    return 'unknown'


def geometry_diagnostics(case, method):
    require(method in METHODS, 'one of the three methods required')
    actor = case.get('actor_metrics') or {}
    intents = actor.get('geometry_intents') or {}
    phase = intents.get('current_phase')
    phase = phase if isinstance(phase, str) and phase else None
    executor = actor.get('physmani_geometry_executor') or {}
    events = executor.get('generated_target_admissions')
    if not isinstance(events, list):
        return dict(final_geometry_phase=phase, public_forecast_collision_rejections=None,
                    recorded_rejection_reasons=None, admission_events_recorded=False)
    reasons, valid = Counter(), True
    for event in events:
        if not isinstance(event, dict) or type(event.get('accepted')) is not bool:
            valid = False; continue
        if event['accepted'] is not False: continue
        proof = event.get('proof')
        reason = proof.get('reason') if isinstance(proof, dict) else None
        if not isinstance(reason, str) or not reason:
            valid = False; continue
        reasons[reason] += 1
    return dict(final_geometry_phase=phase, public_forecast_collision_rejections=reasons[COLLISION_REASON] if valid else None,
                recorded_rejection_reasons=dict(reasons) if valid else None, admission_events_recorded=valid)


def statistics(records):
    result = {}
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        counts = Counter(r['classification'] for r in group)
        instrumented = [r for r in group if r['admission_events_recorded']]
        reasons = Counter()
        for row in instrumented: reasons.update(row['recorded_rejection_reasons'])
        result[method] = dict(cases=len(group), **{k: counts[k] for k in ('native_success', 'native_terminal_failure', 'budget_exhausted_no_native_terminal', 'explicit_policy_failure', 'unknown')},
            infrastructure_errors=0, cleanup_incomplete=0, measured_motion_diagnostic_false=sum(r['motion_pass'] is False for r in group),
            final_geometry_phase=dict(Counter(r['final_geometry_phase'] or 'unknown' for r in group)),
            public_forecast_collision_rejections=sum(r['public_forecast_collision_rejections'] for r in instrumented) if len(instrumented) == len(group) else None,
            public_forecast_collision_cases=sum(r['public_forecast_collision_rejections'] > 0 for r in instrumented) if len(instrumented) == len(group) else None,
            target_admission_metrics_cases=len(instrumented), target_admission_metrics_missing_cases=len(group) - len(instrumented),
            all_recorded_rejection_reasons=dict(reasons) if len(instrumented) == len(group) else None)
    return result


def audit_task(task, *, reader=None):
    require(task in TASKS, 'only an original TEST task is supported')
    reader = Reader(ROOT) if reader is None else reader
    control, controller, owner, stage, val_pointer = require_prerequisites(reader)
    # From this point only post-VAL frozen TEST metadata and the selected task's receipts are consumed.
    suite = reader.json(SUITE)
    lock = reader.json(SUITE_LOCK)
    reader.frozen = lock['files']
    require(lock.get('approved_for_execution') is True and lock.get('actual_formal_split') == 'test'
            and lock.get('policy_sha256') == POLICY and lock.get('model_or_policy_parameters_changed') is False
            and lock.get('original_945_VAL_release') == val_pointer and reader.lookup(reader.frozen, SUITE) == reader.inputs[SUITE],
            'actual826 frozen post-VAL TEST suite required')
    require(suite.get('validation_release_path') == VAL_RELEASE and suite.get('source_manifests', {}).get('test', {}).get('path') == MANIFEST,
            'TEST suite must bind actual VAL prerequisite and source metadata')
    protocol = suite['author_outcome_protocol']
    require(protocol == control['author_outcome_protocol']
            and suite.get('formal_result_auditor') == dict(path=T.AUDITOR, sha256=reader.inputs[T.AUDITOR])
            and suite.get('formal_case_driver') == owner.get('admission_driver')
            and suite.get('formal_admission_guard') == owner.get('admission_guard'), 'unchanged frozen TEST driver/auditor/protocol required')
    require(reader.lookup(reader.frozen, VAL_RELEASE) == val_pointer['sha256'], 'TEST lock does not freeze actual824 VAL release')
    for p in (suite['formal_case_driver'], suite['formal_admission_guard'], protocol['implementation'], protocol['runtime_consumer'], protocol['original_runtime']):
        reader.raw(p['path'], expected=p['sha256'], frozen=True, original=True)
    for name in (OWNER, T.AUDITOR, T.PREPARER, T.SOURCE_PREPARER, *VALIDATORS):
        reader.raw(name, frozen=True, original=True)
    reporting_dependencies = verify_reporting_dependencies(reader)
    # Review helpers are supplementary source evidence, not an amendment to the frozen execution graph.
    for name in (HELPER, T.HELPER, SUBSET_HELPER, ORIGINAL_SUBSET_AUDITOR): reader.raw(name, expected=PINS[name])
    manifest = reader.pointed(suite['source_manifests']['test'], frozen=True)
    require(manifest.get('original_source', {}).get('path') == ORIGINAL_MANIFEST, 'exact original TEST roster pointer required')
    original = reader.pointed(manifest['original_source'], frozen=True)
    source_lock = reader.json(SOURCE_LOCK, expected=PINS[SOURCE_LOCK])
    require(source_lock.get('source_manifest_sha256') == reader.inputs[MANIFEST], 'actual756 TEST source/manifest binding differs')
    sources = T.source_grid(manifest, original, source_lock, val_pointer, reader.inputs[T.SOURCE_PREPARER])
    routes = reader.pointed(suite['routes'], frozen=True, original=True)
    require(routes.get('author_outcome_protocol') == protocol, 'original physical routes outcome protocol differs')
    plan = reader.json(RUNTIME + '/plan.json')
    T.require_plan(plan, sources, routes, protocol)
    require(all('previous_imported_completed' not in row for row in plan['jobs']),
            'original fresh753 TEST plan cannot contain resume provenance')
    require(owner.get('suite_lock_sha256') == reader.inputs[SUITE_LOCK]
            and owner.get('split_source_lock_sha256') == reader.inputs[SOURCE_LOCK]
            and owner.get('source_manifest') == suite['source_manifests']['test'], 'actual753 source/manifest/locks differ')
    status, status_pointer, status_text = reader.snapshot(RUNTIME + '/status.json')
    rows = select_complete_task(plan, status, task, sources)
    from our_mani.multitask.three_method_release16 import measured_motion
    records = []
    verified_workers = set()
    for row in rows:
        require(row['admission_driver'] == suite['formal_case_driver'] and row['admission_guard'] == suite['formal_admission_guard'],
                'actual plan driver/guard differs')
        expected = {key: row[key] for key in T.H.BINDINGS}
        worker = row['case_worker']
        if worker['path'] not in verified_workers:
            reader.raw(worker['path'], expected=worker['sha256'], frozen=True, original=True); verified_workers.add(worker['path'])
        cpu_leaf, native_leaf = (RUNTIME + '/' + kind + '/' + row['job_id'] for kind in ('CPU', 'native'))
        cpu = reader.json(cpu_leaf + '/CPU_preflight.json')
        identity = require_CPU_evidence(cpu, reader.json(cpu_leaf + '/stdout.log'), row, expected, reader.inputs[SOURCE_LOCK])
        launches = {}
        for leaf, check in ((cpu_leaf, True), (native_leaf, False)):
            child = reader.json(leaf + '/launch.json')
            end = reader.json(leaf + '/terminal.json')
            require_child_launch(child, row, identity, owner, check=check)
            T.H.require_clean_terminal(end)
            for filename in ('stdout.log', 'stderr.log'): reader.raw(leaf + '/' + filename)
            launches[check] = end
        accepted = status['completed'][row['job_id']]
        case_pointer, driver_pointer = accepted['case'], accepted['admission_driver_receipt']
        case = reader.pointed(case_pointer)
        driver = reader.pointed(driver_pointer)
        require(launches[False].get('case') == case_pointer and launches[False].get('admission_driver_receipt') == driver_pointer,
                'actual native terminal/case/driver SHA differs from accepted status snapshot')
        require_driver_evidence(driver, row, case_pointer)
        require(cpu.get('identity') == case.get('identity') and cpu.get('component_domain') == case.get('component_domain'),
                'actual CPU/native case identity/domain differs')
        require_case(case, row, sources[(task, row['raw'])], protocol, reader.inputs[SOURCE_LOCK])
        motion_pass = measured_motion(case)
        require(type(motion_pass) is bool and accepted.get('native_success') is case['official_task_success']
                and accepted.get('motion_pass') is motion_pass, 'actual status/native success or motion diagnostic differs')
        records.append(dict(job_id=row['job_id'], source_raw_episode_id=row['raw'], source_row_sha256=row['source_row_sha256'],
            method=row['method'], case=case_pointer, external_driver_receipt=driver_pointer, evidence_runtime=RUNTIME,
            imported_completed=False, actual_owner_pid=owner['pid'], actual_owner_launch=reader.pointer(RUNTIME + '/launch.json'),
            CPU_receipt=reader.pointer(cpu_leaf + '/CPU_preflight.json'), CPU_launch=reader.pointer(cpu_leaf + '/launch.json'),
            native_launch=reader.pointer(native_leaf + '/launch.json'), native_terminal=reader.pointer(native_leaf + '/terminal.json'),
            actual_argv=row['argv'], official_task_success=case['official_task_success'], native_terminal_failure=case['native_terminal_failure'],
            counted_ticks=case['counted_clock']['ticks'], classification=classify_case(case), motion_pass=motion_pass,
            cleanup_completed=True, infrastructure_error=None, **geometry_diagnostics(case, row['method'])))
    require(len(records) == 120 and len({r['job_id'] for r in records}) == 120, 'exact complete120 task receipts required')
    return dict(schema='r10_complete_task_actual_TEST_readonly_results_v910', created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True, read_only_analysis=True, task=task, split='test', source_cases=40, method_cases=120, methods=list(METHODS),
        source_raw_episode_ids=sorted({r['source_raw_episode_id'] for r in records}), same_exact_original_sources_for_all_methods=True,
        unique_job_ids=120, statistics=statistics(records), case_evidence=records, missing_or_invalid_fields=[],
        unknown_diagnostic_fields=[dict(job_id=r['job_id'], fields=[k for k in ('final_geometry_phase', 'public_forecast_collision_rejections', 'recorded_rejection_reasons') if r[k] is None])
                                   for r in records if any(r[k] is None for k in ('final_geometry_phase', 'public_forecast_collision_rejections', 'recorded_rejection_reasons'))],
        status_snapshot_utc=status.get('updated_utc'), status_snapshot_sha256=status_pointer['sha256'], status_snapshot_source_path=status_pointer['path'],
        status_snapshot_json_utf8=status_text, status_snapshot_mutable_source_is_frozen_input=False,
        status_snapshot_total_completed=len(status['completed']), status_snapshot_total_infrastructure_failed=len(status['failed']),
        status_snapshot_target_completed_records={r['job_id']: status['completed'][r['job_id']] for r in rows},
        validation_release=val_pointer, resume_control=reader.pointer(CONTROL), resume_control_lock=reader.pointer(LOCK),
        actual_TEST_stage_launch=reader.pointer(OWNER_STAGE + '/launch.json'), actual_VAL_audit_stage_launch=reader.pointer(VAL_AUDIT_STAGE + '/launch.json'),
        actual_stage_launch=reader.pointer(RUNTIME + '/launch.json'), actual_plan=reader.pointer(RUNTIME + '/plan.json'),
        source_input_lock=reader.pointer(SOURCE_LOCK), policy_sha256=POLICY, author_outcome_protocol=protocol,
        evidence_runtime_provenance_counts={RUNTIME:120}, namespace_evidence_scope=NAMESPACE_SCOPE,
        per_child_live_namespace_receipts_claimed=False, classification_rule='native success,then native fatal,then explicit policy failure,then200ticks with no terminal;otherwise unknown',
        collision_count_definition=dict(field='actor_metrics.physmani_geometry_executor.generated_target_admissions', accepted=False,
            exact_reason=COLLISION_REASON, unit='recorded admission rejections,not physical collisions or pairs',
            baseline_uninstrumented_is_unknown_not_zero=True, case_wide_rejection_counts_not_attributed_to_final_phase=True),
        named_collision_pair='not recorded;no pair or root cause inferred', geometry_phase_field='actor_metrics.geometry_intents.current_phase',
        full1920_TEST_release_granted=False, full1920_TEST_results_audit_performed=False,
        scope='one complete original TEST task subset only;not a full1920 results audit or release',
        native_false_retained=True, motion_false_retained=True, success_rate_or_acceleration_filter_used=False,
        raw_reset_PKL_image_HDF5_contents_opened=False, transitive_TRAIN_model_OS_inputs_walked=False,
        TEST_used_for_fitting=False, jobs_or_native_GPU_started=False, execution_or_configuration_modified=False,
        live_environment_probe_performed=False, audited_metadata_helper=dict(path=HELPER, sha256=HELPER_SHA),
        audited_subset_semantics=dict(path=SUBSET_HELPER, sha256=SUBSET_HELPER_SHA),
        reporting_dependencies=reporting_dependencies, original_physical_locks_amended=False,
        previous_subset_auditor=dict(path=ORIGINAL_SUBSET_AUDITOR, sha256=ORIGINAL_SUBSET_AUDITOR_SHA),
        inputs_sha256=reader.inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--previous-attempt-record', default=PREVIOUS_OBSERVATION)
    parser.add_argument('--previous-attempt-sha256', default=PREVIOUS_OBSERVATION_SHA)
    args = parser.parse_args(argv)
    name = PurePosixPath(args.out)
    require(not name.is_absolute() and str(name) == args.out and '..' not in name.parts,
            'canonical relative report path required')
    out = ROOT / args.out
    require(out.parent.resolve() == (ROOT / RUNTIME).resolve() and out.suffix == '.json'
            and out.name not in ('launch.json', 'plan.json', 'status.json', 'terminal.json')
            and not out.exists() and not out.is_symlink(), 'fresh TEST runtime report required; never overwrite')
    reader = Reader(ROOT)
    observation = require_previous_observation(reader, args.previous_attempt_record, args.previous_attempt_sha256)
    report = audit_task(args.task, reader=reader)
    report['previous_readonly_reporter_failure'] = dict(observation_record=reader.pointer(PREVIOUS_OBSERVATION),
        observation_source=observation['observation_source'], recorded_utc=observation['recorded_utc'],
        root_observed_actual_exit_code=1, actual_error=observation['actual_error'],
        recorded_time_is_not_a_fabricated_process_finish_time=True,
        historical_PID_launch_stderr_disk_evidence_claimed=False, native_attempt_or_infrastructure_failure_claimed=False,
        original_task='beat_the_rotating_buzz', original_output=observation['actual_command'][-1],
        same_task_and_output=(args.task == 'beat_the_rotating_buzz' and args.out == observation['actual_command'][-1]))
    report['source_sha256'] = T.sha_bytes(Path(__file__).read_bytes())
    report['inputs_sha256'][str(Path(__file__).relative_to(ROOT))] = report['source_sha256']
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, task=args.task, cases=120, out=args.out,
                         full1920_TEST_release_granted=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
