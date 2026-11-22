"""CPU-only contracts for the TEST observer, using a temporary fake project."""
import importlib.util
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'report_formal_TEST_halfhour_v818_r10.py'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalTESTReportContracts(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('isolated_v818_report', SCRIPT)
        self.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.report)
        self.temporary = tempfile.TemporaryDirectory(prefix='fake-v818-report-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report.ROOT = self.root
        self.state = self.root / 'runtime/framework/fake_TEST_report'
        self.state.mkdir(parents=True)
        self.out = self.root / 'fake_TEST_report.txt'
        self.addCleanup(mock.patch.stopall)
        self.ticks = mock.patch.object(self.report, 'start_ticks', return_value=777).start()

    def fixture(self):
        """Create only fake aggregate metadata, without any case or source data."""
        for relative in (self.report.REL_VAL_AUDITOR, self.report.REL_CONTROLLER,
                         self.report.REL_OWNER, self.report.REL_AUDITOR,
                         self.report.REL_SOURCE_PREPARER,
                         'our_mani/multitask/native_outcome_sampling_v741.py',
                         'scripts/run_task_executor_smoke_author_suite16_v744_r10.py',
                         'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py',
                         'scripts/fake_admission_driver.py', 'scripts/fake_admission_guard.py'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# inert fake source: ' + relative + '\n', encoding='utf-8')
        self.runtime = self.root / self.report.REL_RUNTIME
        self.control = self.root / self.report.REL_CONTROL_RUNTIME
        self.lock_path = self.root / self.report.REL_SOURCE_INPUTS
        self.results_path = self.root / self.report.REL_RESULTS
        self.policy = 'a' * 64
        self.protocol = self.report.expected_protocol()
        self.bindings = dict(author_outcome_protocol=self.protocol,
                             native_runtime=self.protocol['runtime_consumer'],
                             original_native_runtime=self.protocol['original_runtime'],
                             admission_driver=self.report.pointer(self.root / 'scripts/fake_admission_driver.py'),
                             admission_guard=self.report.pointer(self.root / 'scripts/fake_admission_guard.py'))
        self.release = dict(self.aggregate_shape('validation', 315, 945),
                            schema='r10_full16_common_author_RML_validation_release_v531',
                            source_sha256=digest(self.root / self.report.REL_VAL_AUDITOR),
                            base_result_auditor=self.report.pointer(self.root / self.report.REL_AUDITOR),
                            formal_test_permitted=True, TEST_used=False,
                            execution_protocol_changed=True, **self.bindings)
        write_json(self.root / self.report.REL_VAL_RELEASE, self.release)
        self.lock = dict(schema='r10_full16_post_admission_split_source_lock_v1', split='test',
                         source_count=640, policy_sha256=self.policy, TEST_used_for_fitting=False,
                         source_content_used_for_policy_fitting=False,
                         complete_VAL_release_observed=True,
                         source_lock_preparer_sha256=digest(self.root / self.report.REL_SOURCE_PREPARER),
                         source_manifest_sha256='b' * 64)
        write_json(self.lock_path, self.lock)
        self.config = dict(schema=self.report.SCHEMA, protocol_status='formal_TEST',
                           runtime=self.report.REL_RUNTIME, controller=self.report.REL_CONTROL_RUNTIME,
                           test_results=self.report.REL_RESULTS,
                           validation_release=self.report.pointer(self.root / self.report.REL_VAL_RELEASE),
                           test_source_inputs=self.report.pointer(self.lock_path),
                           policy_sha256=self.policy, author_outcome_protocol=self.protocol)
        self.config_path = self.state / 'fake_source.json'
        self.refresh_source()
        write_json(self.control / 'launch.json', dict(
            source_sha256=digest(self.root / self.report.REL_CONTROLLER),
            user_authorized_resume=True, pid=424242, start_ticks=777))
        write_json(self.control / 'status.json', dict(stage='formal_TEST_1920'))
        self.owner_launch = dict(source_sha256=digest(self.root / self.report.REL_OWNER), split='test',
                                 policy_sha256=self.policy,
                                 split_source_lock_sha256=self.config['test_source_inputs']['sha256'],
                                 source_manifest=dict(sha256=self.lock['source_manifest_sha256']),
                                 execution_protocol_changed=True, **self.bindings)
        write_json(self.runtime / 'launch.json', self.owner_launch)
        self.jobs = [dict(job_id=f'{task}--{method}--{index:02d}', task=task, method=method,
                          argv=['--split', 'test', '--source-inputs', self.report.REL_SOURCE_INPUTS,
                                '--validation-release', self.report.REL_VAL_RELEASE],
                          execution_protocol_changed=True, **self.bindings)
                     for task in self.report.TASK_LABELS for method in self.report.METHODS
                     for index in range(40)]
        self.plan = dict(split='test', expected_cases=1920, policy_sha256=self.policy, jobs=self.jobs)
        write_json(self.runtime / 'plan.json', self.plan)
        self.status = dict(split='test', expected=1920, completed={}, failed={})
        write_json(self.runtime / 'status.json', self.status)

    def aggregate_shape(self, split, sources, cases):
        return dict(complete=True, split=split, source_cases=sources, method_cases=cases,
                    completed_method_cases=cases, policy_sha256=self.policy,
                    methods=list(self.report.METHODS), tasks=list(self.report.TASK_LABELS),
                    TEST_used_for_fitting=False, motion_is_diagnostic_only=True,
                    native_task_success_unchanged=True,
                    cases=[dict(fake_case=i) for i in range(cases)],
                    admission_driver_receipts=[dict(fake_receipt=i) for i in range(cases)])

    def refresh_source(self):
        write_json(self.config_path, self.config)
        self.source, self.source_sha = self.report.load_source(self.config_path)

    def rewrite_release(self):
        write_json(self.root / self.report.REL_VAL_RELEASE, self.release)
        self.config['validation_release'] = self.report.pointer(self.root / self.report.REL_VAL_RELEASE)
        self.refresh_source()

    def complete_owner(self):
        self.status['completed'] = {
            row['job_id']: dict(complete=True, native_success=index % 3 == 0)
            for index, row in enumerate(self.jobs)}
        write_json(self.runtime / 'status.json', self.status)
        self.owner_terminal = dict(complete=True, split='test', completed=1920, expected=1920,
                                   failed=0, policy_sha256=self.policy,
                                   author_outcome_protocol=self.protocol, execution_protocol_changed=True)
        write_json(self.runtime / 'terminal.json', self.owner_terminal)
        write_json(self.control / 'status.json', dict(stage='complete'))
        write_json(self.control / 'terminal.json', dict(
            complete=True, automatic_retry=False, TEST_results=self.report.REL_RESULTS))

    def independent_result(self):
        totals = {method: 0 for method in self.report.METHODS}
        per_task = []
        for task in self.report.TASK_LABELS:
            methods = {}
            for method in self.report.METHODS:
                successes = sum(self.status['completed'][row['job_id']]['native_success']
                                for row in self.jobs if row['task'] == task and row['method'] == method)
                methods[method] = dict(episodes=40, successes=successes)
                totals[method] += successes
            per_task.append(dict(task=task, methods=methods))
        paths = (self.root / self.report.REL_AUDITOR, self.root / self.report.REL_OWNER,
                 self.lock_path, self.runtime / 'launch.json', self.runtime / 'plan.json',
                 self.runtime / 'terminal.json')
        self.result = dict(self.aggregate_shape('test', 640, 1920),
                           source_sha256=digest(self.root / self.report.REL_AUDITOR),
                           actual_stage_owner=self.report.pointer(self.runtime / 'terminal.json'),
                           source_input_lock=self.config['test_source_inputs'],
                           source_outcome_filtering=False, native_failures_retained=True,
                           Value_NGFF_FASTER_used=False, inputs_sha256={str(path.resolve()): digest(path) for path in paths},
                           native_successes_by_method=totals, per_task=per_task,
                           execution_protocol_changed=True, **self.bindings)
        write_json(self.results_path, self.result)

    @contextmanager
    def forbid_content_reads(self, *paths):
        """A rejected gate must never even try to open TEST evidence."""
        protected = tuple(Path(path).resolve() for path in paths)
        real_open = Path.open

        def checked_open(path, mode='r', *args, **kwargs):
            candidate = Path(path).resolve()
            if ('r' in mode or '+' in mode) and any(
                    candidate == entry or candidate.is_relative_to(entry)
                    for entry in protected):
                raise AssertionError('gate attempted forbidden TEST read: ' + str(candidate))
            return real_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, 'open', checked_open):
            yield

    def test_explicit_deadline_preserves_utc_instant(self):
        actual = self.report.initial_deadline(
            '2026-10-21T04:00:00+00:00', None, self.state, 0, 1800)
        self.assertEqual(actual, 1792555200)

    def test_deadline_rejects_missing_or_non_utc_instant(self):
        for value in ('2026-10-21T04:00:00', '2026-10-21T12:00:00+08:00'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.report.initial_deadline(value, None, self.state, 0, 1800)
        with self.assertRaises(ValueError):
            self.report.initial_deadline(None, None, self.state, 0, 1800)

    def test_deadline_falls_back_to_recorded_history_after_final_status(self):
        previous = self.root / 'runtime/framework/fake_previous_report'
        write_json(previous / 'chat_ready.json', dict(next_update_utc=None))
        write_json(previous / 'status.json', dict(next_update_utc=None))
        (previous / 'history.jsonl').write_text(
            json.dumps(dict(next_update_utc='2026-10-21T04:00:00+00:00')) + '\n'
            + json.dumps(dict(next_update_utc=None)) + '\nmalformed\n', encoding='utf-8')
        self.assertEqual(self.report.initial_deadline(None, previous, self.state, 999, 1800), 1792555200)

    def test_inherited_deadline_precedes_local_deadline(self):
        previous = self.root / 'runtime/framework/fake_previous_report'
        write_json(previous / 'status.json', dict(next_update_utc='2026-10-21T04:00:00+00:00'))
        write_json(self.state / 'chat_ready.json', dict(next_update_utc='2026-10-21T05:00:00+00:00'))
        self.assertEqual(self.report.initial_deadline(None, previous, self.state, 999, 1800), 1792555200)

    def test_overdue_deadline_retains_original_phase(self):
        self.assertEqual(self.report.advance_deadline(100, 3701, 1800), 5500)
        self.assertEqual(self.report.advance_deadline(5500, 3701, 1800), 5500)

    def test_signal_stop_is_observed(self):
        event = threading.Event()
        self.assertFalse(self.report.observation_stop(self.state, event))
        event.set()
        self.assertTrue(self.report.observation_stop(self.state, event))

    def test_private_global_and_controller_stops_are_observed(self):
        event = threading.Event()
        for path in (self.state / 'STOP', self.root / 'runtime/framework/multitask_r10_STOP',
                     self.root / self.report.REL_CONTROL_RUNTIME / 'STOP'):
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                self.assertTrue(self.report.observation_stop(self.state, event))
                path.unlink()

    def test_stop_has_priority_over_even_the_val_gate_read(self):
        self.fixture()
        with self.forbid_content_reads(self.root / self.report.REL_VAL_RELEASE, self.runtime, self.lock_path):
            with self.assertRaises(self.report.ObservationStopped):
                self.report.collect(self.source, stop=lambda: True)

    def test_same_state_lock_rejects_second_holder(self):
        with self.report.instance_lock(self.state):
            with self.assertRaises((SystemExit, RuntimeError, BlockingIOError)):
                with self.report.instance_lock(self.state):
                    self.fail('a second observer acquired the same state')

    def test_live_pid_identity_rejects_unlocked_state(self):
        write_json(self.state / 'launch.json', dict(pid=424242, start_ticks=777))
        with self.assertRaises(RuntimeError):
            with self.report.instance_lock(self.state):
                self.fail('live prior PID/start_ticks was ignored')

    def test_reused_pid_with_changed_start_ticks_allows_lock(self):
        write_json(self.state / 'launch.json', dict(pid=424242, start_ticks=776))
        with self.report.instance_lock(self.state):
            pass

    def test_source_loading_does_not_open_test_evidence(self):
        self.fixture()
        with self.forbid_content_reads(self.runtime, self.lock_path, self.results_path):
            source, checksum = self.report.load_source(self.config_path)
        self.assertEqual(checksum, self.source_sha)
        self.assertEqual(source['runtime'], self.runtime)

    def test_source_rejects_a_different_namespace(self):
        self.fixture()
        self.config['runtime'] = 'runtime/framework/fake_other_TEST'
        write_json(self.config_path, self.config)
        with self.assertRaises(ValueError):
            self.report.load_source(self.config_path)

    def test_failed_val_release_gate_never_reads_test_evidence(self):
        self.fixture()
        original = deepcopy(self.release)
        changes = (('complete', False), ('source_cases', 314), ('method_cases', 944),
                   ('source_sha256', '0' * 64), ('TEST_used', True),
                   ('policy_sha256', '0' * 64), ('cases', []), ('admission_driver_receipts', []))
        for key, value in changes:
            with self.subTest(field=key):
                self.release = deepcopy(original)
                self.release[key] = value
                self.rewrite_release()
                with self.forbid_content_reads(self.runtime, self.lock_path, self.results_path):
                    data = self.report.collect(self.source)
                self.assertFalse(data['admitted'])
                self.assertFalse(data['test_runtime_status_read'])
                self.assertFalse(data['test_source_lock_metadata_read'])
                self.assertFalse(data['release_complete'])
                self.assertTrue(data['errors'])

    def test_stale_val_release_pointer_never_reads_test_evidence(self):
        self.fixture()
        self.release['unbound_mutation'] = True
        write_json(self.root / self.report.REL_VAL_RELEASE, self.release)
        with self.forbid_content_reads(self.runtime, self.lock_path, self.results_path):
            data = self.report.collect(self.source)
        self.assertFalse(data['admitted'])
        self.assertTrue(data['errors'])

    def test_controller_not_in_test_stage_cannot_open_test_evidence(self):
        self.fixture()
        for stage in ('formal_VAL_945', 'repair_acceptance'):
            with self.subTest(stage=stage):
                write_json(self.control / 'status.json', dict(stage=stage))
                with self.forbid_content_reads(self.runtime, self.lock_path, self.results_path):
                    data = self.report.collect(self.source)
                self.assertFalse(data['admitted'])
                self.assertTrue(data['errors'])

    def test_wrong_controller_source_or_pid_cannot_open_test_evidence(self):
        self.fixture()
        for field, value in (('source_sha256', '0' * 64), ('user_authorized_resume', False),
                             ('start_ticks', 776)):
            with self.subTest(field=field):
                launch = dict(source_sha256=digest(self.root / self.report.REL_CONTROLLER),
                              user_authorized_resume=True, pid=424242, start_ticks=777)
                launch[field] = value
                write_json(self.control / 'launch.json', launch)
                with self.forbid_content_reads(self.runtime, self.lock_path, self.results_path):
                    data = self.report.collect(self.source)
                self.assertFalse(data['admitted'])
                self.assertTrue(data['errors'])

    def test_admitted_empty_plan_counts_no_successes(self):
        self.fixture()
        data = self.report.collect(self.source)
        self.assertTrue(data['admitted'])
        self.assertTrue(data['plan_valid'])
        self.assertTrue(data['test_runtime_status_read'])
        self.assertEqual(sum(data['expected'].values()), 1920)
        self.assertEqual(sum(data['completed'].values()), 0)
        self.assertEqual(sum(data['success'].values()), 0)
        self.assertEqual(data['errors'], [])

    def test_changed_source_lock_never_opens_runtime_or_result(self):
        self.fixture()
        self.lock['unbound_mutation'] = True
        write_json(self.lock_path, self.lock)
        with self.forbid_content_reads(self.runtime, self.results_path):
            data = self.report.collect(self.source)
        self.assertTrue(data['admitted'])
        self.assertFalse(data['test_runtime_status_read'])
        self.assertFalse(data['release_complete'])
        self.assertTrue(data['errors'])

    def test_stop_between_val_read_and_code_hash_preempts_further_reads(self):
        self.fixture()
        checks = iter((False, True))
        with self.forbid_content_reads(self.root / self.report.REL_VAL_AUDITOR,
                                       self.runtime, self.lock_path, self.results_path):
            with self.assertRaises(self.report.ObservationStopped):
                self.report.collect(self.source, stop=lambda: next(checks, True))

    def test_false_success_partial_unknown_and_interrupted_are_distinct(self):
        self.fixture()
        keys = [row['job_id'] for row in self.jobs[:8]]
        self.status['completed'] = {
            keys[0]: dict(complete=True, native_success=False),
            keys[1]: dict(complete=True, native_success=True),
            keys[2]: dict(complete=False, native_success=True),
            keys[3]: dict(complete=True, native_success=1),
            keys[4]: dict(complete=True, native_success=True, interrupted=True),
            keys[5]: dict(complete=True, native_success=True, timed_out=True),
            'not_in_plan': dict(complete=True, native_success=True)}
        self.status['failed'] = {keys[6]: dict(cancelled=True), keys[7]: dict(infrastructure_error=True)}
        write_json(self.runtime / 'status.json', self.status)
        data = self.report.collect(self.source)
        self.assertEqual(sum(data['completed'].values()), 2)
        self.assertEqual(sum(data['success'].values()), 1)
        self.assertEqual(data['cancelled'], 1)
        self.assertEqual(data['failed'], 1)
        self.assertEqual(len(data['errors']), 5)
        self.assertFalse(data['release_complete'])

    def test_non_test_plan_cannot_contribute_completed_records(self):
        self.fixture()
        self.plan['jobs'][0]['argv'][1] = 'validation'
        write_json(self.runtime / 'plan.json', self.plan)
        self.status['completed'][self.jobs[0]['job_id']] = dict(complete=True, native_success=True)
        write_json(self.runtime / 'status.json', self.status)
        data = self.report.collect(self.source)
        self.assertFalse(data['plan_valid'])
        self.assertEqual(sum(data['completed'].values()), 0)
        self.assertTrue(data['errors'])

    def test_incomplete_group_distribution_cannot_contribute_completed_records(self):
        self.fixture()
        self.plan['jobs'][0]['method'] = self.report.METHODS[1]
        write_json(self.runtime / 'plan.json', self.plan)
        self.status['completed'][self.jobs[0]['job_id']] = dict(complete=True, native_success=True)
        write_json(self.runtime / 'status.json', self.status)
        data = self.report.collect(self.source)
        self.assertFalse(data['plan_valid'])
        self.assertEqual(sum(data['completed'].values()), 0)
        self.assertTrue(data['errors'])

    def test_full_owner_without_754_result_is_not_final(self):
        self.fixture()
        self.complete_owner()
        data = self.report.collect(self.source)
        self.assertEqual(sum(data['completed'].values()), 1920)
        self.assertFalse(data['release_complete'])

    def test_exact_independent_754_result_is_final(self):
        self.fixture()
        self.complete_owner()
        self.independent_result()
        data = self.report.collect(self.source)
        self.assertEqual(data['errors'], [])
        self.assertTrue(data['release_complete'])

    def test_mismatched_independent_result_never_marks_final(self):
        self.fixture()
        self.complete_owner()
        self.independent_result()
        original = deepcopy(self.result)
        mutations = (('source_sha256', '0' * 64), ('inputs_sha256', {}),
                     ('source_outcome_filtering', True), ('per_task', []),
                     ('native_successes_by_method', {method: 640 for method in self.report.METHODS}))
        for field, value in mutations:
            with self.subTest(field=field):
                result = deepcopy(original)
                result[field] = value
                write_json(self.results_path, result)
                data = self.report.collect(self.source)
                self.assertFalse(data['release_complete'])
                self.assertTrue(data['errors'])

    def test_reporting_retains_deadline_and_never_writes_ack(self):
        data = self.report.empty_data()
        history = []
        with mock.patch.object(self.report.time, 'time', return_value=1792554000):
            item = self.report.save_report(self.state, self.out, data, None, None, 1800,
                                           'fake observation', history, next_due=1792555200)
        self.assertEqual(item['next_update_utc'], '2026-10-21T04:00:00+00:00')
        self.assertTrue((self.state / 'chat_ready.json').exists())
        self.assertFalse((self.state / 'chat_ack.json').exists())
        self.assertFalse(item['TEST_runtime_status_read'])
        self.assertFalse(item['TEST_raw_data_read'])
        self.assertFalse(item['native_started'])
        self.assertEqual(history, [item])

    def test_stopping_report_preserves_partial_counts_and_existing_ack(self):
        data = self.report.empty_data()
        data['admitted'] = data['plan_valid'] = True
        group = next(iter(self.report.TASK_LABELS)), self.report.METHODS[0]
        data['completed'][group] = 2
        data['success'][group] = 1
        ack = self.state / 'chat_ack.json'
        ack.write_bytes(b'{"snapshot_id":"previous-real-message"}\n')
        before = ack.read_bytes()
        item = self.report.save_report(self.state, self.out, data, None, None, 1800,
                                       'fake STOP', [], stopping=True, next_due=1792555200)
        self.assertEqual(item['completed'], 2)
        self.assertFalse(item['independently_released'])
        self.assertIsNone(item['next_update_utc'])
        self.assertEqual(ack.read_bytes(), before)
        self.assertIn('非最终比较结果', self.out.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
