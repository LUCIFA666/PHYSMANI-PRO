"""Metadata-only recovery continuation boundaries; Popen/native are mocked.

The underlying registered plan/raw-result validators have their own complete
CPU/native integration tests. These tests isolate the watcher: all54 validation
calls must finish before the exact fixed5 command is launched, with no retries.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import continue_registered_recovery_r17_after_v16 as continuation


class RegisteredRecoveryContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=continuation.ROOT/'runtime')
        self.root = Path(self.temp.name)
        self.base = self.root/'artifacts/framework/multitask_r10'
        self.setup16 = self.base/'TRAIN_recovery_v16_known_failures_setup'
        self.setup17 = self.base/'TRAIN_recovery_v17_additional_known_failures_setup'
        self.out16 = self.base/'TRAIN_recovery_v16_known_failures'
        self.out17 = self.base/'TRAIN_recovery_v17_additional_known_failures'
        self.ids16 = ['old'+format(i,'013d') for i in range(54)]
        self.ids17 = ['new'+format(i,'013d') for i in range(5)]
        self.cases = [dict(case_id=cid, task='fixture_task', split='train') for cid in self.ids16+self.ids17]
        self.source = self.root/'scripts/continue_registered_recovery_r17_after_v16.py'
        self.put(self.source, {'CPU_fixture_source_bytes_only':True})
        for setup in (self.setup16, self.setup17):
            self.put(setup/'plan.json', {'cases':self.cases})
            self.put(setup/'frozen/execution_registration.json', {'CPU_fixture_registry_only':True})
        self.command = [str(self.root/'.conda-env/bin/python'), '-B',
            str(self.root/'scripts/collect_multitask_recovery_r10.py'), '--check',
            '--plan', str(self.setup17/'plan.json'),
            '--execution-registration', str(self.setup17/'frozen/execution_registration.json'),
            '--output', str(self.out17)]
        for cid in self.ids17: self.command += ['--queue-case-id', cid]
        self.put(self.setup16/'preflight.json', dict(check_returncode=0,
            check_stdout=dict(fixed_failed_cases=self.ids16),
            plan=self.pointer(self.setup16/'plan.json'),
            execution_registration=self.pointer(self.setup16/'frozen/execution_registration.json')))
        self.preflight17 = dict(check_returncode=0, check_command=self.command,
            check_stdout=dict(fixed_failed_cases=self.ids17), fixed_known_failed_cases=5,
            plan=self.pointer(self.setup17/'plan.json'),
            execution_registration=self.pointer(self.setup17/'frozen/execution_registration.json'),
            collection=str(self.out17.relative_to(self.root)))
        self.put(self.setup17/'preflight.json', self.preflight17)
        self.report16 = self.report(self.ids16)
        self.put(self.out16/'queue_report.json', self.report16)
        self.validated = []; self.verify_calls = []; self.launches = []
        self.report17 = self.report(self.ids17)

    def tearDown(self): self.temp.cleanup()

    def put(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    def pointer(self, path):
        return dict(path=str(path.relative_to(self.root)), sha256=continuation.sha(path))

    def report(self, ids):
        return dict(selected_case_ids=list(ids), selected_complete=True, expected=2800,
            receipts=[dict(case_id=cid,status='completed') for cid in ids],
            workers=2, automatic_retries=False, recovery_component_only=True)

    def verify(self, plan, registration, *, output, case_ids=None):
        actual = self.ids16 if Path(plan).parent == self.setup16 else self.ids17
        case_ids = actual if case_ids is None else case_ids
        if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= set(actual):
            raise ValueError('mock of independently tested fixed unique subset validator')
        self.verify_calls.append((Path(plan),Path(registration),Path(output),list(case_ids)))
        return dict(plan={'cases':self.cases}, allowed=list(actual))

    def validate(self, path, context, case):
        self.validated.append(case['case_id'])
        return dict(complete=True)

    def popen(self, command, **kwargs):
        # This checks call ordering at the exact boundary that could launch
        # native. Every predecessor raw completion must already have passed.
        self.assertEqual(self.validated, self.ids16)
        self.launches.append((list(command),kwargs))
        self.put(self.out17/'queue_report.json', self.report17)
        return SimpleNamespace(pid=5678, wait=lambda:0)

    def run_watch(self, *, identity=None, validate=None, sleep=None):
        with patch.object(continuation,'ROOT',self.root), patch.object(continuation,'BASE',self.base), \
             patch.object(continuation,'__file__',str(self.source)), \
             patch.object(continuation.runner.collection,'ROOT',self.root), \
             patch.object(continuation.runner.collection,'BASE',self.base), \
             patch.object(continuation,'process_identity',side_effect=identity or [None]), \
             patch.object(continuation.time,'sleep',side_effect=sleep) as sleeping, \
             patch.object(continuation.runner,'verify_execution',side_effect=self.verify), \
             patch.object(continuation.runner,'validate_recovery_result',side_effect=validate or self.validate), \
             patch.object(continuation.subprocess,'Popen',side_effect=self.popen) as process:
            result = continuation.run(1234,'original_start')
        return result, sleeping, process

    def rejected_before_native(self, *, historical=False, **options):
        with self.assertRaises(Exception): self.run_watch(**options)
        self.assertEqual(self.launches, [])
        self.assertFalse((self.setup17/'registered_queue.log').exists())
        path = self.setup17/'continuation_status.json'
        if path.exists() and not historical:
            self.assertEqual(json.loads(path.read_text())['stage'], 'pending_infrastructure_review_no_automatic_retry')

    def test_missing_pid_and_incomplete_or_subset_queue_never_launch(self):
        self.report16['selected_complete'] = False
        self.put(self.out16/'queue_report.json',self.report16)
        self.rejected_before_native(identity=[None])

    def test_subset_receipts_cannot_pass_even_if_selected_complete_true(self):
        self.report16['receipts'].pop()
        self.put(self.out16/'queue_report.json',self.report16)
        self.rejected_before_native()

    def test_wrong_selected_roster_or_duplicate_report_is_rejected(self):
        self.report16['selected_case_ids'][-1] = 'unregistered_source'
        self.put(self.out16/'queue_report.json',self.report16)
        self.rejected_before_native()

    def test_duplicate_selected_id_is_not_an_exact54_queue_report(self):
        self.report16['selected_case_ids'].append(self.ids16[0])
        self.put(self.out16/'queue_report.json',self.report16)
        self.rejected_before_native()

    def test_all54_raw_validations_finish_before_exact_fixed5_two_worker_command(self):
        result, sleeping, process = self.run_watch(identity=[dict(starttime='original_start',state='S'),None])
        self.assertEqual(result,0); sleeping.assert_called_once_with(10)
        process.assert_called_once()
        expected = list(self.command); expected[expected.index('--check')] = '--run-queue'
        self.assertEqual(self.launches[0][0], expected+['--workers','2'])
        self.assertEqual(self.launches[0][1]['cwd'],self.root)
        self.assertEqual(self.verify_calls[0][3],self.ids16)
        self.assertEqual(self.verify_calls[1][3],self.ids17)
        status = json.loads((self.setup17/'continuation_status.json').read_text())
        self.assertEqual(status['stage'],'v17_fixed5_completed')
        self.assertFalse(status['automatic_retries'])

    def test_one_invalid_raw_predecessor_stops_before_Popen(self):
        def invalid(path, context, case):
            if len(self.validated)==27: raise ValueError('actual raw parent/SHA verification failed')
            return self.validate(path,context,case)
        self.rejected_before_native(validate=invalid)
        self.assertEqual(len(self.validated),27)

    def test_registered_inputs_changing_while_waiting_do_not_launch(self):
        def mutate(_): self.put(self.setup17/'plan.json',{'cases':self.cases,'tamper':True})
        self.rejected_before_native(identity=[dict(starttime='original_start',state='S'),None],sleep=mutate)
        self.assertEqual(self.verify_calls,[])

    def test_v17_prior_attempt_is_preserved_without_retry(self):
        (self.out17/'infrastructure_attempts'/self.ids17[3]/'0').mkdir(parents=True)
        self.rejected_before_native()
        self.assertEqual(self.validated,self.ids16)

    def test_existing_continuation_or_launch_log_is_never_overwritten(self):
        self.put(self.setup17/'continuation_status.json',{'stage':'historical'})
        self.rejected_before_native(historical=True)
        self.assertEqual(json.loads((self.setup17/'continuation_status.json').read_text()),{'stage':'historical'})

    def test_preflight_argv_wrong_plan_or_unknown_native_option_is_not_executed(self):
        command = list(self.command); command[command.index('--plan')+1] = str(self.setup16/'plan.json')
        self.preflight17['check_command'] = command
        self.put(self.setup17/'preflight.json',self.preflight17)
        self.rejected_before_native()

    def test_preflight_argv_may_not_add_hidden_worker_or_native_parameter(self):
        self.preflight17['check_command'] = self.command+['--case-timeout','1']
        self.put(self.setup17/'preflight.json',self.preflight17)
        self.rejected_before_native()

    def test_report_claiming_complete_with_failed_or_duplicate_v16_receipt_is_rejected(self):
        self.report16['receipts'][-1] = dict(case_id=self.ids16[0],status='infrastructure_error')
        self.put(self.out16/'queue_report.json',self.report16)
        self.rejected_before_native()

    def test_exit_zero_with_subset_v17_report_cannot_claim_fixed5_completion(self):
        self.report17 = self.report(self.ids17[:-1])
        result, _, process = self.run_watch()
        self.assertEqual(result,2); process.assert_called_once()
        status = json.loads((self.setup17/'continuation_status.json').read_text())
        self.assertEqual(status['stage'],'v17_pending_infrastructure_review')

    def test_duplicate_selected_v17_id_cannot_claim_exact5_completion(self):
        self.report17['selected_case_ids'].append(self.ids17[0])
        result, _, process = self.run_watch()
        self.assertEqual(result,2); process.assert_called_once()

    def test_exit_zero_and_complete_report_still_require_final_five_raw_results(self):
        def invalid(path, context, case):
            if case['case_id'] == self.ids17[-1]:
                raise ValueError('final actual raw5 validation failed')
            return self.validate(path,context,case)
        with self.assertRaisesRegex(ValueError,'raw5 validation failed'):
            self.run_watch(validate=invalid)
        self.assertEqual(len(self.launches),1)
        self.assertEqual(self.validated,self.ids16+self.ids17[:-1])
        status = json.loads((self.setup17/'continuation_status.json').read_text())
        self.assertEqual(status['stage'],'pending_infrastructure_review_no_automatic_retry')
        self.assertFalse(status['automatic_retries'])

    def test_missing_pid_reader_does_not_spawn_any_process(self):
        self.assertIsNone(continuation.process_identity(999999999))
        value = continuation.process_identity(os.getpid())
        self.assertIsInstance(value['starttime'],str)
        self.assertIn(value['state'], ('R','S','D','I'))


if __name__=='__main__': unittest.main()
