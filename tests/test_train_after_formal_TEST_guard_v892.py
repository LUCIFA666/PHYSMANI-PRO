"""Pure memory fakes only: no real TEST metadata, /proc, kernel locks or jobs."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import train_after_formal_TEST_guard_v892 as guard


ROOT = Path('/synthetic/our-mani-v892')


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prohibit_production_helpers(test):
    """A regression may fail a fake test, but must never reach real OS probes."""
    denied = Mock(side_effect=AssertionError('CPU fake reached a production OS helper'))
    for name in ('_stop_guard', '_reserve', '_inventory', '_is_alive'):
        scope = patch.object(guard, name, denied)
        scope.start()
        test.addCleanup(scope.stop)


class MemoryReader:
    """A strict metadata namespace; even an attempted extra read fails."""
    def __init__(self, values):
        self.values = {str(name): raw for name, raw in values.items()}
        self.reads = []

    def __call__(self, name):
        name = str(name)
        self.reads.append(name)
        if name not in self.values:
            raise AssertionError('unexpected metadata read: ' + name)
        return self.values[name]


class Resources:
    """In-memory reservation and visibility callbacks, never an OS probe."""
    def __init__(self):
        self.events = []
        self.stop_at = None
        self.stop_calls = 0
        self.busy = False
        self.held = False
        self.live = False
        self.visibility = dict(host_visibility_available=True,
            live_project_native_processes=[], same_uid_inventory_incomplete=False)

    def stop(self, root):
        self.events.append('stop')
        self.stop_calls += 1
        if self.stop_at == self.stop_calls:
            raise InterruptedError('synthetic STOP')

    @contextmanager
    def reserve(self, root):
        self.events.append('reserve_enter')
        if self.busy:
            raise RuntimeError('synthetic lock busy')
        self.held = True
        try:
            yield
        finally:
            self.held = False
            self.events.append('reserve_exit')

    def inventory(self, root):
        self.events.append('inventory')
        return dict(self.visibility)

    def is_alive(self, record):
        self.events.append('is_alive')
        return self.live

    def kwargs(self):
        return dict(enabled=True, stop_check=self.stop, reserve=self.reserve,
            inventory=self.inventory, is_alive=self.is_alive)


class MetadataFixture:
    """Complete1920 metadata with deliberately zero successes and failed motion.

    Code pins are replaced by synthetic hashes for this memory-only fixture;
    the guard still hashes all four fake source byte strings independently.
    No manifest, case, raw data, historical775 terminal or source lock exists.
    """
    def __init__(self):
        self.sources = {name: ('CPU fake pinned source: ' + name).encode()
                        for name in guard.SOURCE_SHA}
        self.source_sha = {name: sha(raw) for name, raw in self.sources.items()}
        self.controller_source = next(name for name in self.sources if 'v827' in name)
        self.owner_source = next(name for name in self.sources if 'v753' in name)
        self.auditor_source = next(name for name in self.sources if 'v754' in name)
        self.owner_stage = guard.CONTROLLER + '/stages/formal_TEST_1920'
        self.audit_stage = guard.CONTROLLER + '/stages/independent_TEST_results'
        self.owner_launch = guard.TEST_RUNTIME + '/launch.json'
        self.owner_terminal = guard.TEST_RUNTIME + '/terminal.json'
        self.owner_plan = guard.TEST_RUNTIME + '/plan.json'
        self.source_lock_pointer = dict(path=guard.SOURCE_LOCK, sha256='d' * 64)
        self.data = dict(self.sources)
        environment = dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',
            complete=True, all_current_OS_targets_full_byte_SHA_verified=True,
            checked_OS_files=6418, host_PID_network_namespace_preserved=True,
            proc_or_dev_replaced=False, task_parameters_or_strategy_changed=False,
            observer_pid=410, pid=410, observer_start_ticks=411,
            current_mount_namespace='CPU-fake-private-mount',
            mount_namespace='CPU-fake-private-mount', reference_mount_namespace='CPU-fake-host-mount')
        environment.update(deepcopy(guard.ENVIRONMENT_POINTERS))
        self.data[guard.CONTROLLER + '/launch.json'] = dict(pid=410, start_ticks=411,
            source_sha256=self.source_sha[self.controller_source],
            resume_control=deepcopy(guard.CONTROL), resume_control_lock=deepcopy(guard.CONTROL_LOCK),
            frozen_environment=deepcopy(environment), user_authorized_resume=True)
        self.data[guard.CONTROLLER + '/terminal.json'] = dict(complete=True,
            finished_utc='CPU-fake-finished', TEST_results=guard.RESULT, automatic_retry=False)
        original_setup = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup'
        self.data[guard.VAL_OWNER_STAGE + '/launch.json'] = dict(pid=450, start_ticks=451,
            argv=[str(ROOT / '.conda-env/bin/python'), '-u', guard.VAL_OWNER_SOURCE,
                '--suite', original_setup + '/suite.json',
                '--suite-lock', original_setup + '/frozen_inputs.json',
                '--source-inputs', original_setup + '/validation_source_inputs.json',
                '--split', 'validation', '--runtime', 'runtime/framework/formal_VAL_env_resume_v823',
                '--resume-control', guard.CONTROL['path'], '--resume-lock', guard.CONTROL_LOCK['path'],
                '--out', guard.ART + '/validation'],
            source_sha256=self.source_sha[guard.VAL_OWNER_SOURCE], automatic_retry=False,
            inherited_frozen_environment=deepcopy(environment),
            current_mount_namespace=environment['current_mount_namespace'])
        common = ['--suite', guard.SUITE, '--suite-lock', guard.SUITE_LOCK,
            '--source-inputs', guard.SOURCE_LOCK, '--split', 'test',
            '--validation-release', guard.VAL_RELEASE, '--runtime', guard.TEST_RUNTIME]
        for stage, source, pid, out in (
                (self.owner_stage, self.owner_source, 510, guard.ART + '/test'),
                (self.audit_stage, self.auditor_source, 610, guard.RESULT)):
            self.data[stage + '/launch.json'] = dict(pid=pid, start_ticks=pid + 1,
                argv=[str(ROOT / '.conda-env/bin/python'), '-u', source, *common, '--out', out],
                source_sha256=self.source_sha[source], automatic_retry=False,
                inherited_frozen_environment=deepcopy(environment),
                current_mount_namespace=environment['current_mount_namespace'])
            self.data[stage + '/terminal.json'] = dict(complete=True, returncode=0,
                finished_utc='CPU-fake-finished', automatic_retry=False)
        self.data[self.audit_stage + '/stdout.log'] = dict(complete=True, split='test',
            cases=1920, native_successes_by_method={m: 0 for m in guard.METHODS},
            motion_is_diagnostic_only=True, out=guard.RESULT, TEST_permitted=False)
        self.data[self.owner_launch] = dict(pid=510, start_ticks=511, split='test',
            source_sha256=self.source_sha[self.owner_source], policy_sha256=guard.POLICY,
            split_source_lock_sha256=self.source_lock_pointer['sha256'],
            suite_lock_sha256='2' * 64, source_manifest=dict(path='forbidden_TEST_manifest.json', sha256='3' * 64),
            author_outcome_protocol=deepcopy(guard.PROTOCOL), execution_protocol_changed=True,
            admission_driver=deepcopy(guard.DRIVER), admission_guard=deepcopy(guard.GUARD),
            native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),
            original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']),
            automatic_retry=False)
        self.data[self.owner_terminal] = dict(complete=True, split='test', expected=1920,
            completed=1920, failed=0, policy_sha256=guard.POLICY, automatic_retry=False,
            drain_requested=False, drained=False, stop_requested=False,
            CPU_ready_unexecuted=[], CPU_not_launched=0, execution_protocol_changed=True,
            author_outcome_protocol=deepcopy(guard.PROTOCOL), native_successes=0,
            formal_release_requires_independent_audit=True)
        jobs, cases, drivers = [], [], []
        for task in guard.TASKS:
            for raw_index in range(40):
                raw = 'CPU_fake_' + str(raw_index)
                for method in guard.METHODS:
                    out = guard.ART + '/test/' + task + '/' + raw + '/' + method
                    jobs.append(dict(task=task, raw=raw, method=method,
                        job_id=task + '_' + raw + '_' + method,
                        source_row_sha256=sha((task + '/' + raw).encode()),
                        admission_driver=deepcopy(guard.DRIVER), admission_guard=deepcopy(guard.GUARD),
                        case_worker=deepcopy(guard.WORKER),
                        native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),
                        original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']),
                        author_outcome_protocol=deepcopy(guard.PROTOCOL),
                        argv=['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani'
                              else 'scripts/run_local_sim.sh', guard.DRIVER['path'],
                              '--split', 'test', '--task', task, '--raw-episode-id', raw,
                              '--method', method, '--suite', guard.SUITE,
                              '--suite-lock', guard.SUITE_LOCK, '--source-inputs', guard.SOURCE_LOCK,
                              '--validation-release', guard.VAL_RELEASE, '--out', out]))
                    cases.append(dict(path=out + '/case.json', sha256='4' * 64))
                    drivers.append(dict(path=out + '/admission_driver_receipt.json', sha256='5' * 64))
        self.data[self.owner_plan] = dict(split='test', jobs=jobs, expected_cases=1920,
            policy_sha256=guard.POLICY, source_outcome_filtering=False, automatic_retry=False)
        self.data[guard.RESULT] = dict(complete=True, split='test', source_cases=640,
            method_cases=1920, completed_method_cases=1920, methods=list(guard.METHODS),
            tasks=list(guard.TASKS), policy_sha256=guard.POLICY,
            common_RML_profile_sha256=guard.PROFILE, motion_is_diagnostic_only=True,
            motion_metrics_required_for_all_methods=True, native_task_success_unchanged=True,
            native_success_is_not_a_release_threshold=True, baseline_algorithm_unchanged=True,
            ours_measured_motion_passed=False, TEST_used_for_fitting=False,
            native_successes_by_method={m: 0 for m in guard.METHODS},
            measured_motion_violations_by_method={m: 640 for m in guard.METHODS},
            source_sha256=self.source_sha[self.auditor_source],
            source_input_lock=deepcopy(self.source_lock_pointer),
            cases=cases, admission_driver_receipts=drivers,
            admission_driver=deepcopy(guard.DRIVER), admission_guard=deepcopy(guard.GUARD),
            native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),
            original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']),
            author_outcome_protocol=deepcopy(guard.PROTOCOL),
            execution_protocol_changed=True, source_outcome_filtering=False,
            native_failures_retained=True, Value_NGFF_FASTER_used=False,
            one_fixed_original_checkpoint_not_top3_selected=True)
        self.bind_inputs()

    def raw(self, name):
        value = self.data[name]
        return value if isinstance(value, bytes) else encode(value)

    def pointer(self, name):
        return dict(path=name, sha256=sha(self.raw(name)))

    def bind_inputs(self):
        result = self.data[guard.RESULT]
        result['actual_stage_owner'] = self.pointer(self.owner_terminal)
        names = (*self.sources, self.owner_launch, self.owner_terminal, self.owner_plan)
        result['inputs_sha256'] = {str(ROOT / name): sha(self.raw(name)) for name in names}
        result['inputs_sha256'][str(ROOT / guard.SOURCE_LOCK)] = result['source_input_lock']['sha256']

    def reader(self):
        return MemoryReader({name: self.raw(name) for name in self.data})

    def result_sha(self):
        return sha(self.raw(guard.RESULT))

    def check(self, reader=None):
        return guard.require_complete_TEST_metadata(ROOT, self.result_sha(),
            read_bytes=reader or self.reader())


class DisabledTests(unittest.TestCase):
    def setUp(self):
        prohibit_production_helpers(self)

    def test_default_disabled_scope_rejects_without_any_io_or_callback(self):
        denied = Mock(side_effect=AssertionError('disabled scope performed work'))
        with self.assertRaises((ValueError, RuntimeError)):
            with guard.reserved_resources_after_TEST(ROOT, '0' * 64,
                    read_bytes=denied, stop_check=denied, reserve=denied,
                    inventory=denied, is_alive=denied):
                self.fail('disabled scope yielded a permit')
        denied.assert_not_called()

    def test_enabled_context_construction_is_also_inert_until_entry(self):
        denied = Mock(side_effect=AssertionError('context construction performed work'))
        scope = guard.reserved_resources_after_TEST(ROOT, '0' * 64, enabled=True,
            read_bytes=denied, stop_check=denied, reserve=denied,
            inventory=denied, is_alive=denied)
        self.assertIsNotNone(scope)
        denied.assert_not_called()

    def test_forged_permit_cannot_authorize_a_future_prepare(self):
        prepare = Mock()
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            guard.require_permit(object(), ROOT)
            prepare()
        prepare.assert_not_called()

    def test_import_executes_no_metadata_process_or_lock_operation(self):
        # Read and compile code before the no-I/O scope; no artifact is opened.
        source = Path(guard.__file__)
        code = compile(source.read_bytes(), str(source), 'exec')
        name = 'CPU_only_inert_guard_v892_import'
        module = ModuleType(name)
        module.__file__ = str(source)
        denied = Mock(side_effect=AssertionError('import performed I/O or process work'))
        with patch.dict(sys.modules, {name: module}), \
             patch('builtins.open', denied), \
             patch.object(Path, 'read_bytes', denied), \
             patch.object(Path, 'read_text', denied), \
             patch.object(Path, 'open', denied), \
             patch.object(Path, 'stat', denied), \
             patch.object(Path, 'exists', denied), \
             patch.object(Path, 'resolve', denied), \
             patch.object(os, 'listdir', denied), \
             patch.object(os, 'scandir', denied), \
             patch.object(os, 'readlink', denied), \
             patch.object(os, 'kill', denied), \
             patch.object(fcntl, 'flock', denied), \
             patch.object(subprocess, 'Popen', denied):
            exec(code, module.__dict__)
        denied.assert_not_called()


class MetadataTests(unittest.TestCase):
    def setUp(self):
        prohibit_production_helpers(self)
        self.fixture = MetadataFixture()
        pins = patch.dict(guard.SOURCE_SHA, self.fixture.source_sha, clear=True)
        pins.start()
        self.addCleanup(pins.stop)

    def reject(self, mutate, *, bind=True):
        mutate(self.fixture)
        if bind:
            self.fixture.bind_inputs()
        reader = self.fixture.reader()
        prepare = Mock()
        with self.assertRaises((ValueError, RuntimeError, KeyError, TypeError)):
            self.fixture.check(reader)
            prepare()
        prepare.assert_not_called()
        self.assertTrue(set(reader.reads) <= set(self.fixture.data))

    def test_all1920_native_false_and_motion_false_still_pass_metadata_gate(self):
        reader = self.fixture.reader()
        proof = self.fixture.check(reader)
        self.assertEqual(set(reader.reads), set(self.fixture.data))
        self.assertEqual(len(proof['known_processes']), 4)
        self.assertTrue(proof['inputs_sha256'])
        self.assertNotIn('native_successes_by_method', proof)
        self.assertNotIn('ours_measured_motion_passed', proof)
        self.assertFalse(any('v775' in name or 'case.json' in name or 'manifest' in name
                             for name in reader.reads))

    def test_wrong_result_sha_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            guard.require_complete_TEST_metadata(ROOT, '0' * 64, read_bytes=self.fixture.reader())

    def test_incomplete_original_completion_proofs_reject_before_first_RESULT_read(self):
        original = deepcopy(self.fixture.data)
        for name, key, value in [
                (guard.CONTROLLER + '/terminal.json', 'complete', False),
                (self.fixture.owner_stage + '/terminal.json', 'complete', False),
                (self.fixture.audit_stage + '/terminal.json', 'returncode', 1),
                (self.fixture.owner_terminal, 'completed', 1919)]:
            with self.subTest(name=name, key=key):
                self.fixture.data = deepcopy(original)
                self.fixture.data[name][key] = value
                self.fixture.bind_inputs()
                reader = self.fixture.reader()
                def completion_only(path):
                    if path == guard.RESULT:
                        raise AssertionError('incomplete phase read TEST_results')
                    return reader(path)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.fixture.check(completion_only)
                self.assertNotIn(guard.RESULT, reader.reads)

    def test_missing_or_malformed_external_result_sha_rejected_before_any_read(self):
        for value in (None, '', '0' * 63, 'x' * 64):
            with self.subTest(value=value):
                denied = Mock(side_effect=AssertionError('invalid SHA performed read'))
                with self.assertRaises((ValueError, RuntimeError)):
                    guard.require_complete_TEST_metadata(ROOT, value, read_bytes=denied)
                denied.assert_not_called()

    def test_result_counts_and_full_roster_are_required(self):
        fields = dict(complete=False, split='validation', source_cases=639,
            method_cases=1919, completed_method_cases=1919,
            methods=list(guard.METHODS)[:-1], tasks=list(guard.TASKS)[:-1])
        original = deepcopy(self.fixture.data)
        for key, value in fields.items():
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[guard.RESULT].__setitem__(key, value))

    def test_result_frozen_policy_profile_and_flags_are_required(self):
        fields = dict(policy_sha256='9' * 64, common_RML_profile_sha256='9' * 64,
            motion_is_diagnostic_only=False, native_task_success_unchanged=False,
            TEST_used_for_fitting=True, native_failures_retained=False,
            Value_NGFF_FASTER_used=True, source_outcome_filtering=True)
        original = deepcopy(self.fixture.data)
        for key, value in fields.items():
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[guard.RESULT].__setitem__(key, value))

    def test_result_case_count_and_owner_pointer_are_required_without_reading_cases(self):
        self.reject(lambda f: f.data[guard.RESULT]['cases'].pop())
        self.fixture = MetadataFixture()
        self.reject(lambda f: f.data[guard.RESULT].__setitem__('actual_stage_owner',
            dict(path='forbidden_other_owner/terminal.json', sha256='9' * 64)), bind=False)

    def test_changed_pinned_source_bytes_are_rejected(self):
        name = next(iter(self.fixture.sources))
        self.reject(lambda f: f.data.__setitem__(name, b'changed source bytes'))

    def test_result_source_sha_and_source_lock_pointer_are_required(self):
        self.reject(lambda f: f.data[guard.RESULT].__setitem__('source_sha256', '9' * 64))
        self.fixture = MetadataFixture()
        self.reject(lambda f: f.data[guard.RESULT].__setitem__('source_input_lock',
            dict(path='forbidden_other_source_inputs.json', sha256='9' * 64)))

    def test_source_lock_sha_is_bound_across_receipts_without_reading_its_content(self):
        pointer = dict(path=guard.SOURCE_LOCK, sha256='e' * 64)
        self.fixture.data[guard.RESULT]['source_input_lock'] = pointer
        self.fixture.data[self.fixture.owner_launch]['split_source_lock_sha256'] = pointer['sha256']
        self.fixture.bind_inputs()
        reader = self.fixture.reader()
        self.fixture.check(reader)
        self.assertNotIn(guard.SOURCE_LOCK, reader.reads)
        self.fixture.data[self.fixture.owner_launch]['split_source_lock_sha256'] = 'f' * 64
        self.fixture.bind_inputs()
        with self.assertRaises((ValueError, RuntimeError)):
            self.fixture.check()

    def test_owner_terminal_full_grid_and_no_pending_work_are_required(self):
        fields = dict(complete=False, split='validation', expected=1919, completed=1919,
            failed=1, stop_requested=True,
            CPU_ready_unexecuted=['pending'], CPU_not_launched=1)
        original = deepcopy(self.fixture.data)
        for key, value in fields.items():
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[f.owner_terminal].__setitem__(key, value))
        self.fixture.data = deepcopy(original)
        self.reject(lambda f: f.data[f.owner_terminal].__setitem__('CPU_not_launched', False))

    def test_late_DRAIN_after_all1920_are_complete_does_not_become_a_STOP(self):
        self.fixture.data[self.fixture.owner_terminal]['drain_requested'] = True
        self.fixture.data[self.fixture.owner_terminal]['drained'] = True
        self.fixture.bind_inputs()
        self.fixture.check()

    def test_controller_completion_and_exact_result_pointer_are_required(self):
        name = guard.CONTROLLER + '/terminal.json'
        self.reject(lambda f: f.data[name].__setitem__('complete', False))
        self.fixture = MetadataFixture()
        self.reject(lambda f: f.data[name].__setitem__('TEST_results', 'forbidden_other_results.json'))

    def test_controller_control_pointers_and_source_sha_are_required(self):
        name = guard.CONTROLLER + '/launch.json'
        for key, value in [('source_sha256', '9' * 64),
                ('resume_control', dict(path='wrong_control.json', sha256='9' * 64)),
                ('resume_control_lock', dict(path='wrong_lock.json', sha256='9' * 64))]:
            with self.subTest(key=key):
                self.fixture = MetadataFixture()
                self.reject(lambda f: f.data[name].__setitem__(key, value))

    def test_owner_and_auditor_stage_completion_source_argv_and_pid_are_bound(self):
        original = deepcopy(self.fixture.data)
        mutations = [
            (self.fixture.owner_stage + '/terminal.json', 'returncode', 1),
            (self.fixture.audit_stage + '/terminal.json', 'complete', False),
            (self.fixture.owner_stage + '/launch.json', 'pid', 900),
            (self.fixture.audit_stage + '/launch.json', 'start_ticks', 0),
            (self.fixture.audit_stage + '/launch.json', 'source_sha256', '9' * 64),
            (self.fixture.audit_stage + '/launch.json', 'argv', ['wrong.py']),
            (guard.VAL_OWNER_STAGE + '/launch.json', 'source_sha256', '9' * 64),
            (guard.VAL_OWNER_STAGE + '/launch.json', 'pid', 0),
            (guard.VAL_OWNER_STAGE + '/launch.json', 'argv', ['wrong-val.py']),
        ]
        for name, key, value in mutations:
            with self.subTest(name=name, key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[name].__setitem__(key, value))

    def test_auditor_stdout_cannot_claim_wrong_grid_or_output(self):
        name = self.fixture.audit_stage + '/stdout.log'
        original = deepcopy(self.fixture.data)
        for key, value in [('split', 'validation'), ('cases', 1919),
                ('out', 'wrong_results.json'), ('TEST_permitted', True)]:
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[name].__setitem__(key, value))

    def test_missing_input_sha_binding_and_changed_plan_are_rejected(self):
        self.reject(lambda f: f.data[guard.RESULT]['inputs_sha256'].pop(str(ROOT / f.owner_plan)), bind=False)
        self.fixture = MetadataFixture()
        self.reject(lambda f: f.data[f.owner_plan].__setitem__('expected_cases', 1919))

    def test_plan_duplicate_import_augmented_raw_and_source_mismatch_are_rejected(self):
        original = deepcopy(self.fixture.data)
        mutations = [
            lambda f: f.data[f.owner_plan]['jobs'].__setitem__(1,
                deepcopy(f.data[f.owner_plan]['jobs'][0])),
            lambda f: f.data[f.owner_plan]['jobs'][0].__setitem__('evidence_runtime', 'old-runtime'),
            lambda f: f.data[f.owner_plan]['jobs'][0].__setitem__('raw', '0__policy_aug_r0'),
            lambda f: f.data[f.owner_plan]['jobs'][1].__setitem__('source_row_sha256', '9' * 64),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.fixture.data = deepcopy(original)
                self.reject(mutation)

    def test_ordered_case_and_driver_pointers_cannot_be_substituted(self):
        original = deepcopy(self.fixture.data)
        for key in ('cases', 'admission_driver_receipts'):
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[guard.RESULT][key].__setitem__(0,
                    deepcopy(f.data[guard.RESULT][key][1])))

    def test_conflicting_relative_and_absolute_input_sha_is_rejected(self):
        self.reject(lambda f: f.data[guard.RESULT]['inputs_sha256'].__setitem__(
            f.owner_plan, '9' * 64), bind=False)

    def test_recorded_environment_cannot_be_incomplete_or_for_another_observer(self):
        name = guard.CONTROLLER + '/launch.json'
        original = deepcopy(self.fixture.data)
        for key, value in [('complete', False), ('checked_OS_files', 6417),
                ('observer_pid', 999), ('observer_start_ticks', 999),
                ('proc_or_dev_replaced', True), ('task_parameters_or_strategy_changed', True)]:
            with self.subTest(key=key):
                self.fixture.data = deepcopy(original)
                self.reject(lambda f: f.data[name]['frozen_environment'].__setitem__(key, value))


class ReservationTests(unittest.TestCase):
    setUp = MetadataTests.setUp

    def scope(self, resources, reader=None):
        return guard.reserved_resources_after_TEST(ROOT, self.fixture.result_sha(),
            read_bytes=reader or self.fixture.reader(), **resources.kwargs())

    def test_callback_permit_is_valid_only_in_own_live_scope_and_is_not_production_authority(self):
        resources = Resources()
        with self.scope(resources) as permit:
            self.assertTrue(resources.held)
            guard.require_permit(permit, ROOT, allow_test_callbacks=True)
            with self.assertRaises((ValueError, RuntimeError)):
                guard.require_permit(permit, ROOT)
            with self.assertRaises((ValueError, RuntimeError)):
                guard.require_permit(permit, ROOT / 'other', allow_test_callbacks=True)
        self.assertFalse(resources.held)
        with self.assertRaises((ValueError, RuntimeError)):
            guard.require_permit(permit, ROOT, allow_test_callbacks=True)

    def test_busy_reservation_cannot_call_prepare_or_inventory(self):
        resources = Resources()
        resources.busy = True
        prepare = Mock()
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            with self.scope(resources):
                prepare()
        prepare.assert_not_called()
        self.assertNotIn('inventory', resources.events)
        self.assertFalse(resources.held)

    def test_STOP_before_and_after_reservation_prevents_prepare_and_releases_resources(self):
        for stop_at in (1, 2):
            with self.subTest(stop_at=stop_at):
                resources = Resources()
                resources.stop_at = stop_at
                prepare = Mock()
                with self.assertRaises(InterruptedError):
                    with self.scope(resources):
                        prepare()
                prepare.assert_not_called()
                self.assertFalse(resources.held)

    def test_live_known_pid_start_prevents_future_prepare(self):
        for live_pid in (410, 450, 510, 610):
            with self.subTest(live_pid=live_pid):
                resources = Resources()
                resources.is_alive = Mock(side_effect=lambda record: record['pid'] == live_pid)
                prepare = Mock()
                with self.assertRaises((ValueError, RuntimeError)):
                    with self.scope(resources):
                        prepare()
                prepare.assert_not_called()
                self.assertFalse(resources.held)

    def test_incomplete_visibility_or_live_native_inventory_releases_reservation(self):
        for key, value in [('host_visibility_available', False),
                ('live_project_native_processes', [dict(pid=999, start_ticks=1000)]),
                ('same_uid_inventory_incomplete', ['CPU fake unreadable process'])]:
            with self.subTest(key=key):
                resources = Resources()
                resources.visibility[key] = value
                prepare = Mock()
                with self.assertRaises((ValueError, RuntimeError)):
                    with self.scope(resources):
                        prepare()
                prepare.assert_not_called()
                self.assertFalse(resources.held)

    def test_require_permit_checks_STOP_again_immediately_before_prepare(self):
        resources = Resources()
        prepare = Mock()
        with self.scope(resources) as permit:
            resources.stop_at = resources.stop_calls + 1
            with self.assertRaises(InterruptedError):
                guard.require_permit(permit, ROOT, allow_test_callbacks=True)
                prepare()
            prepare.assert_not_called()
        self.assertFalse(resources.held)

    def test_permit_cannot_cross_process_identity_or_use_a_forged_token(self):
        resources = Resources()
        with self.scope(resources) as permit:
            with patch.object(guard.os, 'getpid', return_value=permit.pid + 1):
                with self.assertRaises((ValueError, RuntimeError)):
                    guard.require_permit(permit, ROOT, allow_test_callbacks=True)
            permit._token = object()
            with self.assertRaises((ValueError, RuntimeError)):
                guard.require_permit(permit, ROOT, allow_test_callbacks=True)

    def test_metadata_failure_releases_lease_and_never_reaches_inventory_or_prepare(self):
        resources = Resources()
        self.fixture.data[guard.RESULT]['complete'] = False
        prepare = Mock()
        with self.assertRaises((ValueError, RuntimeError)):
            with self.scope(resources):
                prepare()
        prepare.assert_not_called()
        self.assertFalse(resources.held)
        self.assertNotIn('inventory', resources.events)
        self.assertNotIn('is_alive', resources.events)

    def test_inventory_exception_is_preserved_and_releases_lease(self):
        resources = Resources()
        error = RuntimeError('CPU-fake inventory read denied')
        resources.inventory = Mock(side_effect=error)
        with self.assertRaises(RuntimeError) as caught:
            with self.scope(resources):
                self.fail('inventory failure yielded permit')
        self.assertIs(caught.exception, error)
        self.assertFalse(resources.held)

    def test_body_exception_identity_is_preserved_and_permit_is_revoked(self):
        resources = Resources()
        error = RuntimeError('CPU-fake original body exception')
        with self.assertRaises(RuntimeError) as caught:
            with self.scope(resources) as permit:
                raise error
        self.assertIs(caught.exception, error)
        self.assertFalse(resources.held)
        with self.assertRaises((ValueError, RuntimeError)):
            guard.require_permit(permit, ROOT, allow_test_callbacks=True)


if __name__ == '__main__':
    unittest.main()
