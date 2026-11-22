"""CPU-only synthetic910 checks; no actual TEST metadata or native jobs are read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask.protocol import MultitaskProtocolError


PROJECT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = module_at('completed_TEST_task_v910_subject', PROJECT / 'scripts/audit_completed_TEST_task_v910_r10.py')
fakes851 = module_at('completed_TEST_task_v910_851_fakes', PROJECT / 'tests/test_formal_TEST_motion_v851.py')
fakes749 = module_at('completed_TEST_task_v910_749_fakes', PROJECT / 'tests/test_author_outcome_receipt_v749.py')
T = report.T


def source(name):
    """Synthetic storage receipts, never a claim about real artifact bytes."""
    return dict(path=name, sha256=report.PINS.get(name, T.PINS.get(name,
        T.sha_bytes(('CPU-fake-source:' + name).encode()))))


def checksum(data, name):
    if isinstance(data[name], bytes):
        return T.sha_bytes(data[name])
    return report.PINS.get(name, T.PINS.get(name, T.digest(data[name])))


def pointer(data, name):
    return dict(path=name, sha256=checksum(data, name))


def case_fixture(row, protocol, source_lock_sha):
    """Combine saved-window fakes with a fully valid zero-action749 transcript."""
    base = fakes851.test_case(row, protocol, source_lock_sha)
    native = fakes749.fixture(block=row['task'] in fakes749.BLOCKS,
                              method=row['method'], ticks=0)
    clock, alignment = deepcopy(base['counted_clock']), deepcopy(base['alignment_receipt'])
    base.update(native)
    base['counted_clock'] = clock
    alignment.update(native['alignment_receipt'])
    base.update(task=row['task'], split='test', method=row['method'],
        raw_episode_id=row['raw'], parent_raw_episode_id=None,
        source_raw_episode_id=row['raw'], author_outcome_protocol=deepcopy(protocol),
        process_id=12001, case_isolation='fresh_simulator_process_per_episode',
        recorded_initial_conditions=dict(metric_only=True, policy_use_allowed=False),
        alignment_receipt=alignment, original_source_motion_replayed=None)
    identity = base['identity']
    identity.update(version='r10_fresh_case_identity_v1', exploration_seed=None, demo_only=False)
    identity.pop('sha256', None)
    identity['sha256'] = T.digest(identity)
    base['common_author_RML_parameters']['schema'] = 'r10_observed_common_author_RML_parameters_v530'
    base['component_domain']['policy_sha256'] = T.POLICY
    return base


def memory_fixture():
    """A1920-row plan with only one120-case task stored; all other cases absent."""
    old = fakes851.memory_fixture(None)
    data = old.data
    task = T.TASKS[0]
    protocol = data[T.CONTROL]['author_outcome_protocol']
    protocol.update(schema='r10_author_action_outcome_protocol_v741', native_RML_parameters_unchanged=True)
    for key in ('implementation', 'runtime_consumer', 'original_runtime'):
        protocol[key] = source(protocol[key]['path'])
    control = data[T.CONTROL]
    control.update(schema='r10_frozen_environment_formal_VAL_resume_control_v823',
        expected_cases=945, physical_policy_changed=False, completed_results_selected_by_success=False,
        resume_auditor=source(report.VAL_AUDITOR), user_authorized_resume=True,
        standing_authorization_reused=True, new_user_event_invented=False, split='validation',
        suite=dict(path=T.H.SUITE, sha256='1' * 64),
        suite_lock=dict(path=T.H.SUITE_LOCK, sha256='2' * 64),
        source_inputs=dict(path=T.H.SOURCE_LOCK, sha256='3' * 64))
    original = data[T.LOCK]
    original.update(schema='r10_frozen_environment_formal_VAL_resume_inputs_v823')
    original_files = original['files']
    for name in set(T.SOURCES) | set(report.SOURCES) | {report.VAL_AUDITOR}:
        original_files[str(PROJECT / name)] = source(name)['sha256']

    val = data[T.VAL_RELEASE]
    val.update(source_sha256=source(report.VAL_AUDITOR)['sha256'],
        resume_control=pointer(data, T.CONTROL), resume_control_lock=pointer(data, T.LOCK),
        author_outcome_protocol=protocol,
        base_result_auditor=source(T.AUDITOR),
        admission_driver=source('scripts/run_formal_case_author_outcome_v751_r10.py'),
        admission_guard=source('our_mani/multitask/formal_suite_admission_v728.py'))
    val['inputs_sha256'] = {str(PROJECT / name): source(name)['sha256']
                           for name in set(T.SOURCES) | set(report.SOURCES)}
    val['inputs_sha256'].update({str(PROJECT / name): checksum(data, name)
                                for name in (T.CONTROL, T.LOCK)})
    val_pointer = pointer(data, T.VAL_RELEASE)

    manifest, original_manifest, source_lock = fakes851.source_fixture(
        val_pointer, source(T.SOURCE_PREPARER)['sha256'])
    data[T.ORIGINAL_MANIFEST] = original_manifest
    manifest['original_source'] = pointer(data, T.ORIGINAL_MANIFEST)
    data[T.MANIFEST] = manifest
    source_lock['source_manifest_sha256'] = checksum(data, T.MANIFEST)
    data[T.SOURCE_LOCK] = source_lock
    sources = T.source_grid(manifest, original_manifest, source_lock, val_pointer,
                            source(T.SOURCE_PREPARER)['sha256'])

    driver = source('scripts/run_formal_case_author_outcome_v751_r10.py')
    guard = source('our_mani/multitask/formal_suite_admission_v728.py')
    worker = source('scripts/run_task_executor_smoke_author_outcome_v743_r10.py')
    routes = data[T.ROUTES]
    routes['author_outcome_protocol'] = protocol
    for route in routes['routes']:
        route.update(worker=worker, consumer=protocol['runtime_consumer'])
    original_files[str(PROJECT / T.ROUTES)] = checksum(data, T.ROUTES)
    suite = data[T.SUITE]
    suite.update(validation_release_path=T.VAL_RELEASE,
        source_manifests={'test': pointer(data, T.MANIFEST)}, routes=pointer(data, T.ROUTES),
        author_outcome_protocol=protocol, formal_case_driver=driver, formal_admission_guard=guard,
        formal_result_auditor=source(T.AUDITOR),
        resumed_validation_auditor=source(report.VAL_AUDITOR),
        environment_resumed_validation_auditor=source(report.VAL_AUDITOR),
        resumed_validation_control=pointer(data, T.CONTROL),
        resumed_validation_control_lock=pointer(data, T.LOCK))
    frozen = dict(original_files)
    for name in (T.SUITE, T.MANIFEST, T.ORIGINAL_MANIFEST, T.VAL_RELEASE):
        frozen[str(PROJECT / name)] = checksum(data, name)
    data[T.SUITE_LOCK].update(files=frozen, original_945_VAL_release=val_pointer)

    controller = data[T.CONTROLLER + '/launch.json']
    controller.update(source_sha256=source(T.CONTROLLER_SOURCE)['sha256'], start_ticks=401,
        resume_control=pointer(data, T.CONTROL), resume_control_lock=pointer(data, T.LOCK))
    env = controller['frozen_environment']
    stage = report.VAL_AUDIT_STAGE
    val_argv = [str(PROJECT / '.conda-env/bin/python'), '-u', report.VAL_AUDITOR,
        '--suite', T.H.SUITE, '--suite-lock', T.H.SUITE_LOCK,
        '--source-inputs', T.H.SOURCE_LOCK, '--split', 'validation', '--runtime', T.H.CURRENT,
        '--resume-control', T.CONTROL, '--resume-lock', T.LOCK, '--out', T.VAL_RELEASE]
    data[stage + '/launch.json'] = dict(argv=val_argv, pid=800, start_ticks=801,
        source_sha256=source(report.VAL_AUDITOR)['sha256'], automatic_retry=False,
        inherited_frozen_environment=deepcopy(env), current_mount_namespace=env['current_mount_namespace'])
    data[stage + '/terminal.json'] = dict(complete=True, returncode=0, automatic_retry=False)
    data[stage + '/stdout.log'] = dict(complete=True, split='validation', cases=945,
        imported=591, new_native=354, TEST_permitted=True, out=T.VAL_RELEASE)
    val['actual_auditor_environment'] = fakes851.environment(800, control)
    test_stage = data[T.OWNER_STAGE + '/launch.json']
    test_stage.update(source_sha256=source(T.OWNER)['sha256'], inherited_frozen_environment=deepcopy(env))
    owner = data[T.RUNTIME + '/launch.json']
    owner.update(source_sha256=source(T.OWNER)['sha256'], suite_lock_sha256=checksum(data, T.SUITE_LOCK),
        split_source_lock_sha256=checksum(data, T.SOURCE_LOCK), source_manifest=pointer(data, T.MANIFEST),
        author_outcome_protocol=protocol, admission_driver=driver, admission_guard=guard,
        native_runtime=protocol['runtime_consumer'], original_native_runtime=protocol['original_runtime'],
        max_native_slots=2, max_GPU_slots=1, max_CPU_preflight_slots=2)

    # No future full TEST completion/result receipt exists in this fixture.
    for name in list(data):
        if name in (T.RESULT, T.RUNTIME + '/terminal.json', T.CONTROLLER + '/terminal.json') \
                or name.startswith(T.AUDIT_STAGE + '/') or name == T.OWNER_STAGE + '/terminal.json' \
                or name.startswith(T.ART + '/test/') or name.startswith(T.RUNTIME + '/CPU/') \
                or name.startswith(T.RUNTIME + '/native/'):
            del data[name]
    clean = dict(complete=True, returncode=0, timed_out=False, interrupted=False, automatic_retry=False)
    completed = {}
    jobs = data[T.RUNTIME + '/plan.json']['jobs']
    for row in jobs:
        row.update(source_row_sha256=sources[(row['task'], row['raw'])], admission_driver=driver,
            admission_guard=guard, case_worker=worker, native_runtime=protocol['runtime_consumer'],
            original_native_runtime=protocol['original_runtime'], author_outcome_protocol=protocol,
            worker_sha256=driver['sha256'])
        if row['task'] != task:
            continue
        case = case_fixture(row, protocol, checksum(data, T.SOURCE_LOCK))
        out = T.option(row['argv'], '--out')
        data[out + '/case.json'] = case
        case_pointer = pointer(data, out + '/case.json')
        bindings = {key: row[key] for key in T.H.BINDINGS}
        data[out + '/admission_driver_receipt.json'] = dict(schema='r10_explicit_author_outcome_formal_driver_v751',
            complete=True, actual_split='test', execution_protocol_changed=True, case=case_pointer, **bindings)
        driver_pointer = pointer(data, out + '/admission_driver_receipt.json')
        for kind in ('CPU', 'native'):
            check = kind == 'CPU'
            leaf = T.RUNTIME + '/' + kind + '/' + row['job_id']
            data[leaf + '/launch.json'] = dict(argv=row['argv'] + (['--check'] if check else []),
                launcher_pid=900, launcher_start_ticks=901, owner_pid=owner['pid'], check_only=check,
                native_slots=0 if check else 1, GPU_slots=0 if check else int(row['method'] == 'physmani'),
                execution_protocol_changed=True,
                expected_identity_sha256=None if check else case['identity']['sha256'], **bindings)
            terminal = dict(clean, **({} if check else dict(case=case_pointer,
                                                          admission_driver_receipt=driver_pointer)))
            data[leaf + '/terminal.json'] = terminal
            data[leaf + '/stderr.log'] = b''
            if check:
                cpu = dict(complete=True, native_started=False, GPU_started=False, actual_split='test',
                    execution_protocol_changed=True, identity=deepcopy(case['identity']),
                    component_domain=deepcopy(case['component_domain']), **bindings)
                data[leaf + '/CPU_preflight.json'] = cpu
                data[leaf + '/stdout.log'] = deepcopy(cpu)
            else:
                data[leaf + '/stdout.log'] = b'CPU-only fake native stdout\n'
                completed[row['job_id']] = dict(terminal, native_success=False, motion_pass=False)
    data[T.RUNTIME + '/status.json'] = dict(updated_utc='CPU-fake-snapshot', split='test', expected=1920,
        CPU_launched=120, checking=[], CPU_ready=0, active=[], completed=completed, failed={}, ready_jobs=[],
        drain_requested=False, native_slots=0, GPU_slots=0, stop_requested=False, automatic_retry=False)
    return MemoryReader(data), task, sources


class MemoryReader:
    """Synthetic metadata store; opening absent raw or other-task evidence fails."""
    root = PROJECT
    lookup = T.Reader.lookup

    def __init__(self, data):
        self.data = data
        self.inputs, self.evidence, self.original, self.frozen = {}, {}, {}, {}
        self.opened = []
        self.snapshot_reads = 0
        self.mutate_after_snapshot = False

    def raw(self, name, **kwargs):
        self.opened.append(name)
        if name.endswith(('/stdout.log', '/stderr.log')):
            value = self.data[name]
            raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True,
                separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
            actual = T.sha_bytes(raw)
            self._check(name, actual, kwargs)
            self.inputs[name] = actual
            return raw
        if not (name.startswith('scripts/') or name.startswith('our_mani/')) or not name.endswith('.py'):
            raise AssertionError('unexpected non-source raw read: ' + name)
        actual = source(name)['sha256']
        self._check(name, actual, kwargs)
        self.inputs[name] = actual
        return ('CPU-fake-source:' + name).encode()

    def _check(self, name, actual, kwargs):
        if kwargs.get('expected') is not None:
            T.require(kwargs['expected'] == actual, 'synthetic explicit SHA mismatch: ' + name)
        for key in ('evidence', 'original', 'frozen'):
            if kwargs.get(key):
                T.require(self.lookup(getattr(self, key), name) == actual, 'synthetic ' + key + ' SHA mismatch: ' + name)

    def json(self, name, **kwargs):
        self.opened.append(name)
        if self.mutate_after_snapshot and self.snapshot_reads and '/CPU/' in name:
            self.data[T.RUNTIME + '/status.json']['completed'].clear()
        actual = checksum(self.data, name)
        self._check(name, actual, kwargs)
        self.inputs[name] = actual
        return deepcopy(self.data[name])

    def pointer(self, name):
        return dict(path=name, sha256=self.inputs[name])

    def pointed(self, value, **kwargs):
        T.require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'explicit fake pointer required')
        return self.json(value['path'], expected=value['sha256'], **kwargs)

    def snapshot(self, name):
        self.opened.append(name)
        self.snapshot_reads += 1
        text = json.dumps(self.data[name], ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'
        return json.loads(text), dict(path=name, sha256=T.sha_bytes(text.encode())), text


class CompletedTESTTaskAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader, cls.task, cls.sources = memory_fixture()
        cls.base_data = reader.data

    def fixture(self):
        return MemoryReader(deepcopy(self.base_data))

    def first_row_case(self, reader):
        row = next(r for r in reader.data[T.RUNTIME + '/plan.json']['jobs'] if r['task'] == self.task)
        return row, reader.data[T.option(row['argv'], '--out') + '/case.json']

    def assert_no_TEST_metadata_opened(self, reader):
        for name in (T.SUITE, T.SUITE_LOCK, T.SOURCE_LOCK, T.MANIFEST, T.ORIGINAL_MANIFEST,
                     T.ROUTES, T.RUNTIME + '/plan.json', T.RUNTIME + '/status.json'):
            self.assertNotIn(name, reader.opened)

    def test_invalid_VAL_refused_before_any_TEST_metadata(self):
        for change in ({'complete': False}, {'formal_test_permitted': False}, {'completed_method_cases': 944},
                       {'source_sha256': 'f' * 64}):
            with self.subTest(change=change):
                reader = self.fixture()
                reader.data[T.VAL_RELEASE].update(change)
                with self.assertRaises(ValueError):
                    report.audit_task(self.task, reader=reader)
                self.assert_no_TEST_metadata_opened(reader)

    def test_wrong_TEST_stage_namespace_or_owner_identity_refused_before_metadata(self):
        for name, change in ((T.OWNER_STAGE + '/launch.json', {'current_mount_namespace': 'mnt:[wrong]'}),
                             (T.OWNER_STAGE + '/launch.json', {'argv': ['unrelated']}),
                             (T.RUNTIME + '/launch.json', {'start_ticks': 999}),
                             (T.RUNTIME + '/launch.json', {'source_sha256': 'f' * 64})):
            with self.subTest(name=name, change=change):
                reader = self.fixture()
                reader.data[name].update(change)
                with self.assertRaises(ValueError):
                    report.audit_task(self.task, reader=reader)
                self.assert_no_TEST_metadata_opened(reader)

    def test_VAL_audit_stage_must_be_complete_and_bind_actual_auditor(self):
        for suffix, change in (('terminal.json', {'complete': False}),
                               ('stdout.log', {'cases': 944}),
                               ('launch.json', {'pid': 801})):
            with self.subTest(suffix=suffix):
                reader = self.fixture()
                reader.data[report.VAL_AUDIT_STAGE + '/' + suffix].update(change)
                with self.assertRaises(ValueError):
                    report.audit_task(self.task, reader=reader)
                self.assert_no_TEST_metadata_opened(reader)

    def test_exact120_native_false_motion_false_zero_ticks_remain_without_full_finish(self):
        reader = self.fixture()
        with patch('subprocess.Popen', side_effect=AssertionError('audit cannot launch a job')):
            value = report.audit_task(self.task, reader=reader)
        self.assertEqual(value['source_cases'], 40)
        self.assertEqual(value['method_cases'], 120)
        self.assertEqual(len(value['case_evidence']), 120)
        self.assertTrue(all(r['official_task_success'] is False and r['counted_ticks'] == 0
                            for r in value['case_evidence']))
        self.assertTrue(all(r['motion_pass'] is False for r in value['case_evidence']))
        self.assertEqual({m: value['statistics'][m]['cases'] for m in T.METHODS}, {m: 40 for m in T.METHODS})
        self.assertEqual({m: value['statistics'][m]['measured_motion_diagnostic_false'] for m in T.METHODS},
                         {m: 40 for m in T.METHODS})
        self.assertEqual(reader.snapshot_reads, 1)
        self.assertNotIn(T.RESULT, reader.opened)
        self.assertNotIn(T.RUNTIME + '/terminal.json', reader.opened)
        self.assertNotIn(T.CONTROLLER + '/terminal.json', reader.opened)
        cases = [name for name in reader.opened if name.endswith('/case.json')]
        self.assertEqual(len(cases), 120)
        self.assertTrue(all('/test/' + self.task + '/' in name for name in cases))
        self.assertFalse(any(name.startswith('/UNOPENED') or name.endswith(('.pkl', '.hdf5', '.pt')) for name in reader.opened))

    def test_status_snapshot_is_single_version_not_mutable_input(self):
        reader = self.fixture()
        reader.mutate_after_snapshot = True
        value = report.audit_task(self.task, reader=reader)
        self.assertEqual(reader.snapshot_reads, 1)
        self.assertEqual(reader.data[T.RUNTIME + '/status.json']['completed'], {})
        self.assertEqual(value['status_snapshot_total_completed'], 120)
        text = value['status_snapshot_json_utf8']
        self.assertEqual(len(json.loads(text)['completed']), 120)
        self.assertEqual(value['status_snapshot_sha256'], T.sha_bytes(text.encode()))
        self.assertNotIn(T.RUNTIME + '/status.json', value['inputs_sha256'])
        self.assertNotIn(str(PROJECT / T.RUNTIME / 'status.json'), value['inputs_sha256'])

    def test_real_status_snapshot_keeps_bytes_when_owner_atomically_replaces_path(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            status = root / T.RUNTIME / 'status.json'
            status.parent.mkdir(parents=True)
            old_text = '{"updated_utc":"old","completed":{}}\n'
            status.write_text(old_text)
            real_read_bytes = Path.read_bytes

            def read_then_owner_replaces(path):
                raw = real_read_bytes(path)
                if path == status:
                    replacement = status.with_suffix('.owner-tmp')
                    replacement.write_text('{"updated_utc":"new","completed":{"future":{}}}\n')
                    replacement.replace(status)
                return raw

            reader = report.Reader(root)
            with patch.object(Path, 'read_bytes', read_then_owner_replaces):
                value, receipt, text = reader.snapshot(T.RUNTIME + '/status.json')
            self.assertEqual(value['updated_utc'], 'old')
            self.assertEqual(text, old_text)
            self.assertEqual(receipt['sha256'], T.sha_bytes(old_text.encode()))
            self.assertNotIn(T.RUNTIME + '/status.json', reader.inputs)

    def test_audit_rejects_unbound_pointer_stdout_terminal_and_saved_outcome(self):
        for fault in ('accepted_case_SHA', 'native_case_SHA', 'CPU_stdout', 'native_timeout', 'motion_status'):
            reader = self.fixture()
            row, _ = self.first_row_case(reader)
            key = row['job_id']
            if fault == 'accepted_case_SHA':
                accepted = reader.data[T.RUNTIME + '/status.json']['completed'][key]
                accepted['case'] = dict(accepted['case'], sha256='f' * 64)
            elif fault == 'native_case_SHA':
                terminal = reader.data[T.RUNTIME + '/native/' + key + '/terminal.json']
                terminal['case'] = dict(terminal['case'], sha256='f' * 64)
            elif fault == 'CPU_stdout':
                reader.data[T.RUNTIME + '/CPU/' + key + '/stdout.log']['actual_split'] = 'validation'
            elif fault == 'native_timeout':
                reader.data[T.RUNTIME + '/native/' + key + '/terminal.json']['timed_out'] = True
            else:
                reader.data[T.RUNTIME + '/status.json']['completed'][key]['motion_pass'] = True
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                report.audit_task(self.task, reader=reader)

    def test_fresh_TEST_plan_rejects_even_false_VAL_import_metadata(self):
        for field in ('evidence_runtime', 'imported_completed', 'previous_row_sha256', 'previous_imported_completed'):
            reader = self.fixture()
            reader.data[T.RUNTIME + '/plan.json']['jobs'][0][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.audit_task(self.task, reader=reader)

    def test_task_selection_rejects_wrong_split_or_incomplete120(self):
        data = self.fixture().data
        plan, status = data[T.RUNTIME + '/plan.json'], data[T.RUNTIME + '/status.json']
        self.assertEqual(len(report.select_complete_task(plan, status, self.task, self.sources)), 120)
        for mutate in (lambda p, s: p.update(split='validation'),
                       lambda p, s: s.update(expected=945),
                       lambda p, s: s['completed'].pop(next(iter(s['completed']))),
                       lambda p, s: s['failed'].update({next(iter(s['completed'])): {'complete': False}})):
            p, s = deepcopy(plan), deepcopy(status)
            mutate(p, s)
            with self.assertRaises(ValueError):
                report.select_complete_task(p, s, self.task, self.sources)

    def test_duplicate_plan_or_source_mismatch_is_not_a_complete_task(self):
        data = self.fixture().data
        plan, status = data[T.RUNTIME + '/plan.json'], data[T.RUNTIME + '/status.json']
        changed = deepcopy(plan)
        changed['jobs'][1] = deepcopy(changed['jobs'][0])
        with self.assertRaises(ValueError):
            report.select_complete_task(changed, status, self.task, self.sources)
        changed = deepcopy(self.sources)
        del changed[next(k for k in changed if k[0] == self.task)]
        with self.assertRaises(ValueError):
            report.select_complete_task(plan, status, self.task, changed)
        changed_plan = deepcopy(plan)
        changed_plan['jobs'][0]['source_row_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            T.require_plan(changed_plan, self.sources, data[T.ROUTES], data[T.SUITE]['author_outcome_protocol'])

    def test_TEST_source_lock_count_and_alias_collision_are_rejected(self):
        data = self.fixture().data
        args = (data[T.MANIFEST], data[T.ORIGINAL_MANIFEST], data[T.SOURCE_LOCK],
                pointer(data, T.VAL_RELEASE), source(T.SOURCE_PREPARER)['sha256'])
        self.assertEqual(len(T.source_grid(*args)), 640)
        invalid = deepcopy(args)
        invalid[2]['source_count'] = 639
        with self.assertRaises(ValueError):
            T.source_grid(*invalid)
        invalid = deepcopy(args)
        invalid[0]['episodes'][1]['official_episode_index'] = 0
        with self.assertRaises(ValueError):
            T.source_grid(*invalid)

    def test_CPU_identity_original_raw_and_self_digest_are_required(self):
        reader = self.fixture()
        row, case = self.first_row_case(reader)
        leaf = T.RUNTIME + '/CPU/' + row['job_id']
        cpu = reader.data[leaf + '/CPU_preflight.json']
        expected = {key: row[key] for key in T.H.BINDINGS}
        identity = report.require_CPU_evidence(cpu, deepcopy(cpu), row, expected,
            checksum(reader.data, T.SOURCE_LOCK))
        self.assertEqual(identity, case['identity'])
        for change in ({'raw_episode_id': row['raw'] + '__policy_aug_r0'},
                       {'parent_raw_episode_id': row['raw']}, {'replica': True},
                       {'exploration_seed': 1}, {'sha256': 'f' * 64}, {'version': 'wrong-version'}):
            bad = deepcopy(cpu)
            bad['identity'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.require_CPU_evidence(bad, deepcopy(bad), row, expected,
                    checksum(reader.data, T.SOURCE_LOCK))
        bad = deepcopy(cpu)
        bad['component_domain']['raw_episode_id'] = 'different-source'
        with self.assertRaises(ValueError):
            report.require_CPU_evidence(bad, deepcopy(bad), row, expected,
                checksum(reader.data, T.SOURCE_LOCK))

    def test751_driver_requires_exact_case_pointer_SHA_and_protocol(self):
        reader = self.fixture()
        row, _ = self.first_row_case(reader)
        out = T.option(row['argv'], '--out')
        case_pointer = pointer(reader.data, out + '/case.json')
        driver = reader.data[out + '/admission_driver_receipt.json']
        report.require_driver_evidence(driver, row, case_pointer)
        for change in ('case_SHA', 'protocol', 'split'):
            bad = deepcopy(driver)
            if change == 'case_SHA':
                bad['case']['sha256'] = 'f' * 64
            elif change == 'protocol':
                bad['author_outcome_protocol']['implementation']['sha256'] = 'f' * 64
            else:
                bad['actual_split'] = 'validation'
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.require_driver_evidence(bad, row, case_pointer)

    def test_child_checks_original753_fields_without_invented_namespace(self):
        reader = self.fixture()
        row, case = self.first_row_case(reader)
        owner = reader.data[T.RUNTIME + '/launch.json']
        for kind in ('CPU', 'native'):
            check = kind == 'CPU'
            child = reader.data[T.RUNTIME + '/' + kind + '/' + row['job_id'] + '/launch.json']
            self.assertNotIn('inherited_frozen_environment', child)
            report.require_child_launch(child, row, case['identity'], owner, check=check)
            for change in ({'owner_pid': owner['pid'] + 1}, {'launcher_start_ticks': 0},
                           {'expected_identity_sha256': 'f' * 64}, {'GPU_slots': 2}):
                bad = deepcopy(child)
                bad.update(change)
                with self.subTest(kind=kind, change=change), self.assertRaises(ValueError):
                    report.require_child_launch(bad, row, case['identity'], owner, check=check)

    def test_case_requires_cleanup_fresh_identity530_and749_sampling(self):
        reader = self.fixture()
        row, case = self.first_row_case(reader)
        args = (row, row['source_row_sha256'], row['author_outcome_protocol'], checksum(reader.data, T.SOURCE_LOCK))
        report.require_case(case, *args)
        mutations = (
            lambda c: c.update(cleanup_completed=False),
            lambda c: c.update(process_id=0),
            lambda c: c.update(case_isolation='shared_process'),
            lambda c: c['recorded_initial_conditions'].update(policy_use_allowed=True),
            lambda c: c['common_author_RML_parameters'].update(schema='wrong530schema'),
            lambda c: c['native_outcome_sampling'].update(aborted=True),
            lambda c: c['author_outcome_protocol']['implementation'].update(sha256='f' * 64),
        )
        for i, mutate in enumerate(mutations):
            bad = deepcopy(case)
            mutate(bad)
            with self.subTest(mutation=i), self.assertRaises((ValueError, MultitaskProtocolError)):
                report.require_case(bad, *args)

    def test_both_registered_blocks_use_original_no_failure_sampler_for_all_methods(self):
        reader = self.fixture()
        rows = reader.data[T.RUNTIME + '/plan.json']['jobs']
        source_lock_sha = checksum(reader.data, T.SOURCE_LOCK)
        for task in report.BLOCKS:
            for method in T.METHODS:
                row = next(r for r in rows if r['task'] == task and r['method'] == method)
                case = case_fixture(row, row['author_outcome_protocol'], source_lock_sha)
                args = (row, row['source_row_sha256'], row['author_outcome_protocol'], source_lock_sha)
                with self.subTest(task=task, method=method):
                    report.require_case(case, *args)
                    self.assertEqual(case['native_outcome_sampling']['schema'], fakes749.BLOCK_SCHEMA)
                    self.assertIs(case['native_terminal_failure'], False)
                    case['native_outcome_sampling']['extra_task_success_calls'] = 1
                    fakes749.sync(case)
                    with self.assertRaisesRegex(ValueError, 'zero native fail conditions and extra success calls'):
                        report.require_case(case, *args)

    def test_real_reader_refuses_raw_models_and_unrelated_sources_before_open(self):
        with tempfile.TemporaryDirectory() as name:
            reader = report.Reader(Path(name))
            for path in ('/UNOPENED_TEST/a/first.pkl', 'artifacts/framework/protocol/train_manifest.json',
                         'artifacts/models/weights.pt', 'images/test.png', 'unrelated.py'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    reader.raw(path)

    def test_mock_CLI_rejects_scheduler_filenames_before_audit_even_if_absent(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch.object(report, 'ROOT', root), patch.object(report, 'audit_task',
                    side_effect=AssertionError('scheduler output guard must run before audit')) as audit:
                for filename in ('launch.json', 'plan.json', 'status.json', 'terminal.json'):
                    output = T.RUNTIME + '/' + filename
                    with self.subTest(filename=filename), self.assertRaises(ValueError):
                        report.main(['--task', self.task, '--out', output])
                    self.assertFalse((root / output).exists())
                audit.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_report_only_helper_may_be_absent_from_both_old_locks_without_claiming_membership(self):
        reader = self.fixture()
        helper = 'our_mani/multitask/three_method_release16.py'
        for lock in (T.LOCK, T.SUITE_LOCK):
            files = reader.data[lock]['files']
            for name in (helper, str(PROJECT/helper)):
                files.pop(name, None)
        # The synthetic owner must describe this synthetic original lock's
        # actual bytes, just as the real owner describes its already absent key.
        reader.data[T.RUNTIME+'/launch.json']['suite_lock_sha256'] = checksum(reader.data, T.SUITE_LOCK)
        result = report.audit_task(self.task, reader=reader)
        self.assertEqual(result['schema'], 'r10_complete_task_actual_TEST_readonly_results_v910')
        self.assertEqual(result['method_cases'], 120)
        self.assertEqual(len(result['reporting_dependencies']), 1)
        dependency = result['reporting_dependencies'][0]
        self.assertEqual(dependency['sha256'], report.REPORTING_DEPENDENCIES[helper])
        self.assertFalse(dependency['original_physical_input_lock_member'])
        self.assertFalse(dependency['original_TEST_suite_input_lock_member'])
        self.assertFalse(dependency['proof_of_original_physical_freeze_membership_claimed'])
        self.assertFalse(result['original_physical_locks_amended'])
        self.assertTrue(dependency['independent_reporting_source_pin'])
        self.assertTrue(all(row['official_task_success'] is False and row['counted_ticks'] == 0
                            for row in result['case_evidence']))

    def test_existing_reporting_declaration_cannot_disagree_with_independent_pin(self):
        helper = 'our_mani/multitask/three_method_release16.py'
        for lock in (T.LOCK, T.SUITE_LOCK):
            reader = self.fixture()
            reader.data[lock]['files'][str(PROJECT/helper)] = 'f'*64
            with self.subTest(lock=lock), self.assertRaisesRegex(ValueError, 'conflicts with report-only'):
                report.audit_task(self.task, reader=reader)

    def test_missing_original_scientific_validator_never_uses_reporting_fallback(self):
        for missing in report.VALIDATORS:
            reader = self.fixture()
            for lock in (T.LOCK, T.SUITE_LOCK):
                for key in (missing, str(PROJECT/missing)):
                    reader.data[lock]['files'].pop(key, None)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                report.audit_task(self.task, reader=reader)
            self.assertFalse(any('/CPU/' in name or name.endswith('/case.json') for name in reader.opened))

    def test_reporting_dependency_independent_byte_pin_is_mandatory(self):
        helper = 'our_mani/multitask/three_method_release16.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root/helper
            target.parent.mkdir(parents=True)
            target.write_bytes(b'corrupted supplementary report code')
            reader = report.Reader(root)
            with self.assertRaisesRegex(ValueError, 'input SHA changed'):
                report.verify_reporting_dependencies(reader)
        reader = self.fixture()
        with patch.dict(report.PINS, {helper: 'e'*64}), self.assertRaises(ValueError):
            report.audit_task(self.task, reader=reader)

    def test_prior_failure_record_is_explicit_observation_not_native_evidence(self):
        record = dict(schema='r10_root_observed_readonly_TEST_subset_audit_failure_v908',
            source_sha256=report.ORIGINAL_SUBSET_AUDITOR_SHA, root_observed_actual_exit_code=1,
            historical_launch_PID_start_and_stderr_file_not_recorded=True,
            old_launch_or_native_receipt_not_fabricated=True, report_output_exists=False,
            readonly_audit_not_a_native_attempt=True, case_rerun_or_owner_restart_or_DRAIN=False,
            actual_command=['.conda-env/bin/python','-B',report.ORIGINAL_SUBSET_AUDITOR,
                '--task','beat_the_rotating_buzz','--out',T.RUNTIME+'/first_task_actual_TEST_results_v890.json'])
        reader = self.fixture()
        reader.data[report.PREVIOUS_OBSERVATION] = record
        self.assertEqual(report.require_previous_observation(reader, report.PREVIOUS_OBSERVATION,
                         report.PREVIOUS_OBSERVATION_SHA), record)
        before = len(reader.opened)
        with self.assertRaisesRegex(ValueError, 'exact root-observed'):
            report.require_previous_observation(reader, 'artifacts/models/weights.pt', 'f'*64)
        self.assertEqual(len(reader.opened), before)
        reader.data[report.PREVIOUS_OBSERVATION]['historical_launch_PID_start_and_stderr_file_not_recorded'] = False
        with self.assertRaisesRegex(ValueError, 'truthful observed'):
            report.require_previous_observation(reader, report.PREVIOUS_OBSERVATION, report.PREVIOUS_OBSERVATION_SHA)


if __name__ == '__main__':
    unittest.main()
