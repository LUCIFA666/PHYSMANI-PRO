"""Consumer integration after the separately tested strict lineage audit.

The audit is mocked at its boundary here to exercise mixed actual-plan SHAs
and prove that existing native clock and causal-view checks still execute.
The acquisition-lineage tests independently exercise real source/infra chains.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from our_mani.multitask.acquisition_manifest import COMPOSITE_VERSION, load_acquisition_manifest
from our_mani.multitask.artifact_fitting import load_train_cases
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.training_view import load_acquisitions
from tests import test_multitask_training_view as fixture_module


class CompositeConsumerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.TrainingViewTests(methodName='runTest')
        self.fixture.setUp()
        self.root = self.fixture.root
        self.acquisitions = [self.fixture.acquisition('train', 'source0', 0, 30, behavior=behavior)
            for behavior in ('TRAIN_teacher_committed_waypoint', 'TRAIN_hold_motion_identification')]
        self.replace_record(1, lambda record: record.update(plan_sha256='b'*64))
        ordinary = self.fixture.manifest('ordinary', self.acquisitions)
        self.projection = json.loads(ordinary.read_text())
        self.projection.update(version=COMPOSITE_VERSION, plan_sha256=None,
            single_plan_SHA_claimed=False, projection_rewrites_original_receipts=False,
            actual_plan_sha256_values=['a'*64, 'b'*64])
        mapping = {item['case_id']: digest for item, digest in
                   zip(self.projection['case_receipts'], ('a'*64, 'b'*64))}
        self.projection['per_case_actual_plan_sha256'] = mapping
        for item in self.projection['case_receipts']:
            item['actual_plan_sha256'] = mapping[item['case_id']]
        self.path = self.root/'composite.json'
        self.path.write_text(json.dumps(dict(schema=COMPOSITE_VERSION)))

    def tearDown(self):
        self.fixture.tearDown()

    def replace_record(self, index, update):
        entry, item, _ = self.acquisitions[index]
        path = self.root/item['path']
        record = json.loads(path.read_text())
        update(record)
        parent = record['parent_process_receipt']
        child = self.root/parent['child_case_path']
        child.write_text(json.dumps({k: v for k, v in record.items() if k != 'parent_process_receipt'}))
        parent['child_case_sha256'] = file_sha(child)
        path.write_text(json.dumps(record))
        item['sha256'] = file_sha(path)
        return item

    def proof(self):
        return dict(audit_passed=True, complete=True,
            acquisition_projection=deepcopy(self.projection),
            actual_plan_sha256_values=deepcopy(self.projection['actual_plan_sha256_values']),
            per_case_actual_plan_sha256=deepcopy(self.projection['per_case_actual_plan_sha256']),
            verified_files_sha256={}, declared_protocol_transitions={'b'*64: ['explicit fixture transition']})

    def audited(self, proof=None):
        return patch('our_mani.multitask.acquisition_lineage.audit_acquisition_lineage',
                     return_value=self.proof() if proof is None else proof)

    def test_context_view_retains_both_actual_plans_but_only_hold_is_primary(self):
        with self.audited() as audit:
            selected, ledger, lineage = load_acquisitions(self.root, [self.path], 'train', {})
        audit.assert_called_once_with(self.root, self.path, require_complete=True)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['behavior'], 'TRAIN_hold_motion_identification')
        self.assertFalse(ledger[0]['included_in_training_view'])
        self.assertTrue(lineage[0]['historical_composite'])
        self.assertIsNone(lineage[0]['plan_sha256'])
        self.assertEqual(lineage[0]['actual_plan_sha256_values'], ['a'*64, 'b'*64])

    def test_artifact_fitting_accepts_actual_shas_without_rewriting_receipts(self):
        original = [(self.root/item['path']).read_bytes() for _, item, _ in self.acquisitions]
        with self.audited():
            manifest, cases = load_train_cases(self.root, self.path)
        self.assertIsNone(manifest['plan_sha256'])
        self.assertEqual([case['record']['plan_sha256'] for case in cases], ['a'*64, 'b'*64])
        self.assertEqual(original, [(self.root/item['path']).read_bytes() for _, item, _ in self.acquisitions])

    def test_incomplete_audit_cannot_enter_either_consumer(self):
        proof = self.proof()
        proof['complete'] = False
        for consumer in (lambda: load_train_cases(self.root, self.path),
                         lambda: load_acquisitions(self.root, [self.path], 'train', {})):
            with self.audited(proof), self.assertRaisesRegex(ValueError, 'complete verified lineage'):
                consumer()

    def test_per_case_plan_mismatch_is_rejected_after_valid_raw_hashes(self):
        item = self.replace_record(1, lambda record: record.update(plan_sha256='c'*64))
        self.projection['case_receipts'][1]['sha256'] = item['sha256']
        for consumer in (lambda: load_train_cases(self.root, self.path),
                         lambda: load_acquisitions(self.root, [self.path], 'train', {})):
            with self.audited(), self.assertRaisesRegex(ValueError, 'incomplete|wrongly supervised'):
                consumer()

    def test_native_clock_check_still_runs_for_a_composite(self):
        item = self.replace_record(1, lambda record: record['alignment'].update(actual_task_seconds=1.5))
        self.projection['case_receipts'][1]['sha256'] = item['sha256']
        for consumer in (lambda: load_train_cases(self.root, self.path),
                         lambda: load_acquisitions(self.root, [self.path], 'train', {})):
            with self.audited(), self.assertRaisesRegex(ValueError, 'clock'):
                consumer()

    def test_projected_receipt_cannot_claim_another_actual_plan(self):
        proof = self.proof()
        proof['acquisition_projection']['case_receipts'][1]['actual_plan_sha256'] = 'c'*64
        with self.audited(proof), self.assertRaisesRegex(ValueError, 'per-case plan identity'):
            load_acquisition_manifest(self.root, self.path)

    def test_changed_current_launcher_does_not_replace_its_verified_archive(self):
        archived = self.root/'archived_launch.sh'
        archived.write_bytes(self.fixture.launcher.read_bytes())
        proof = self.proof()
        proof['verified_files_sha256'][str(archived)] = file_sha(archived)
        self.fixture.launcher.write_text('# new runtime; historical source remains archived\n')
        with self.audited(proof):
            files = {}
            selected, ledger, _ = load_acquisitions(self.root, [self.path], 'train', files)
            _, cases = load_train_cases(self.root, self.path)
        self.assertEqual((len(selected), len(ledger), len(cases)), (1, 2, 2))
        self.assertEqual(files[str(archived)], file_sha(archived))
        self.assertNotIn(str(self.fixture.launcher), files)
        ordinary = self.fixture.manifest('ordinary_teacher', [self.acquisitions[0]])
        with self.assertRaisesRegex(ValueError, 'launcher source changed'):
            load_train_cases(self.root, ordinary)


if __name__ == '__main__':
    unittest.main()
