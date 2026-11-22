"""Independent CPU checks of candidate metadata, commit and no-retry rules.

The known-v1-gap test records a reproduced missing-marker acceptance, and is
not a passing qualification of that candidate. No native subprocess is made.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import requery_acquisition as audit
from scripts import collect_TRAIN_high_cup_requery_r10 as worker
from tests import test_multitask_requery_acquisition as original_tests

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/'artifacts/framework/multitask_r10/TRAIN_high_cup_requery_all100_candidate_v1_setup/execution_registration_candidate.json'


@unittest.skipUnless(audit.REGISTRATION.endswith('_v1'), 'historical v1 peer; preserved v1 source archive owns its counterexamples')
class ActualCandidateRequeryPeerTests(unittest.TestCase):
    def test_actual_candidate_all100_scope_and_draft_cannot_collect(self):
        context = audit.verify_registration(ROOT, worker.pointer(REGISTRY), live=True)
        self.assertIs(context['registry']['execution_ready'], False)
        self.assertEqual(len(context['expected_cases']), 100)
        self.assertEqual(len(context['current']['sources']), 77)
        self.assertNotIn('recovery_input', context['plan']['protocol'])
        for case in context['expected_cases'].values():
            old = next(row for row in context['primary']['cases'] if row['task'] == case['task']
                       and row['parent_source_group'] == case['parent_source_group'] and row['teacher_used'])
            changed = {key for key in set(old)|set(case) if old.get(key) != case.get(key)}
            self.assertEqual(changed, {'case_id','raw_episode_id','replica','behavior_replica'})
            self.assertIs(case['context_primary_view'], False)
            self.assertEqual(case['maximum_collection_ticks'], 200)
        first = next(iter(context['expected_cases']))
        with patch.object(worker.collection, 'collect_native_case', side_effect=AssertionError('native must remain unreachable')) as native:
            with self.assertRaisesRegex(ValueError, 'not execution ready'):
                worker.collect_case(context['current']['path'], first, context['current']['collection'],
                                    ROOT/'CPU_not_read_guard.json', REGISTRY)
            native.assert_not_called()

    def test_v1_known_missing_fresh_marker_is_accepted_and_is_a_reported_gap(self):
        ledger, scope = original_tests.RequeryRawLedgerTests().fixture()
        second = ledger['commands'][1]
        for key in ('fresh_query_after_failed_command_id','same_episode_after_counted_wait','actual_planning_requery_index'):
            second.pop(key)
        scope['actual_requeries'] = 0
        # Reproduction, deliberately not phrased as a safety-positive test.
        proof = audit.validate_counted_commands(ledger, scope, ticks=10)
        self.assertEqual((proof['raw_commands'],proof['failed_commands'],proof['actual_requeries']), (2,2,0))

    def test_v1_known_wait_outside_actual_clock_is_accepted_and_is_a_reported_gap(self):
        ledger, scope = original_tests.RequeryRawLedgerTests().fixture()
        for row in ledger['commands']:
            for key in ('generated_at','planning_requery_wait_started_at','planning_requery_wait_completed_at',
                        'planning_requery_last_actual_feedback_at','cutoff_at'):
                if key in row: row[key] += 100.
        proof = audit.validate_counted_commands(ledger, scope, ticks=10)
        self.assertEqual(proof['actual_requeries'], 1)


@unittest.skipUnless(audit.REGISTRATION.endswith('_v1'), 'historical v1 peer; current v2 has a separate source-bound audit suite')
class RequeryCommitAndDispatchPeerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='requery_peer_CPU_', dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup); self.directory = Path(self.temp.name)
        self.collection = self.directory/'collection'; self.plan_path = self.directory/'CPU_plan.json'
        self.plan_path.write_text('{}')
        self.cases = {str(i): dict(case_id=str(i),split='train',task=audit.TASK) for i in range(3)}
        self.context = dict(current=dict(path=self.plan_path, collection=self.collection,
            registration=dict(plan=dict(path=str(self.plan_path.relative_to(ROOT)),sha256='CPU_only'))),
            expected_cases=self.cases, files={}, registry_pointer=dict(path='CPU_fixture',sha256='CPU_only'),
            plan=dict(plan_sha256='CPU_only'), registry=dict(execution=dict(plan=dict(path='CPU_plan',sha256='CPU_only'))))
        self.partial = dict(complete=False,acquisition_projection=dict(completed=0,entries=[],case_receipts=[]))

    def test_uncommitted_canonical_cannot_count_complete_or_satisfy_required_audit(self):
        canonical = self.collection/'cases/train'/audit.TASK/'0/case.json'
        canonical.parent.mkdir(parents=True); canonical.write_text('{}')
        with patch.object(audit, 'verify_registration', return_value=self.context):
            proof = audit.audit_collection(ROOT, dict(path='CPU_fixture',sha256='CPU_only'))
            self.assertIs(proof['complete'], False)
            self.assertEqual(proof['acquisition_projection']['completed'], 0)
            self.assertEqual(proof['acquisition_projection']['missing_case_ids'], ['0','1','2'])
            with self.assertRaisesRegex(ValueError,'all100 committed'):
                audit.audit_collection(ROOT, dict(path='CPU_fixture',sha256='CPU_only'), require_complete=True)

    def test_extra_hidden_attempt_is_rejected_even_without_canonical_or_parent(self):
        (self.collection/'infrastructure_attempts/0/1').mkdir(parents=True)
        with patch.object(audit,'verify_registration',return_value=self.context), self.assertRaisesRegex(ValueError,'hidden infrastructure attempt'):
            audit.audit_collection(ROOT,dict(path='CPU_fixture',sha256='CPU_only'))

    def invoke_queue(self, process):
        with patch.object(audit,'verify_registration',return_value=self.context), \
             patch.object(audit,'audit_collection',return_value=self.partial), \
             patch.object(worker.collection,'require_r09_release',return_value={'CPU_fixture':True}), \
             patch.object(worker.subprocess,'Popen',return_value=process) as popen, \
             patch.object(worker.os,'killpg') as kill:
            report = worker.run_queue(self.plan_path,self.collection,ROOT/'CPU_guard.json',REGISTRY,
                                      case_ids=['0','1','2'],workers=1,case_timeout=1)
            return report,popen.call_count,list(kill.call_args_list)

    def test_first_infrastructure_failure_stops_dispatch_and_cannot_auto_resume(self):
        report, calls, kills = self.invoke_queue(SimpleNamespace(pid=123456,wait=lambda **kw:1))
        self.assertEqual(calls,1); self.assertEqual(kills,[])
        self.assertIs(report['complete'],False)
        self.assertIs(report['stopped_dispatch_after_first_infrastructure_error'],True)
        parent=json.loads((self.collection/'infrastructure_attempts/0/0/parent_receipt.json').read_text())
        self.assertEqual(parent['status'],'infrastructure_error')
        self.assertIs(parent['policy_failure_label_assigned'],False)
        self.assertIs(parent['automatic_retries'],False)
        with patch.object(audit,'verify_registration',return_value=self.context), \
             patch.object(worker.collection,'require_r09_release',return_value={}), \
             patch.object(worker.subprocess,'Popen',side_effect=AssertionError('no redispatch')) as popen:
            with self.assertRaisesRegex(ValueError,'no automatic retry'):
                worker.run_queue(self.plan_path,self.collection,ROOT/'CPU_guard.json',REGISTRY)
            popen.assert_not_called()

    def test_native_wall_timeout_keeps_infra_and_terminates_exact_group_once(self):
        process=SimpleNamespace(pid=123457,wait=unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired('CPU_mock',1),0]))
        report,calls,kills=self.invoke_queue(process)
        self.assertEqual(calls,1); self.assertEqual(len(kills),1)
        self.assertEqual(kills[0].args[0],123457)
        parent=json.loads((self.collection/'infrastructure_attempts/0/0/parent_receipt.json').read_text())
        self.assertEqual(parent['status'],'infrastructure_error')
        self.assertIs(parent['policy_failure_label_assigned'],False)
        self.assertIn('timeout',parent['detail'])
        self.assertIs(report['complete'],False)


if __name__ == '__main__': unittest.main()
