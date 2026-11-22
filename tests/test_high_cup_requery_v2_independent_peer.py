"""Independent v2 CPU boundary replay; fixtures are not native rollouts."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.multitask import requery_acquisition as audit
from scripts import collect_TRAIN_high_cup_requery_r10 as worker
from tests import test_multitask_requery_acquisition as old_fixtures
from tests import test_multitask_requery_acquisition_v2 as v2_fixtures

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/'artifacts/framework/multitask_r10/TRAIN_high_cup_requery_all100_candidate_v2_setup/execution_registration_candidate.json'


class IndependentCountedRequeryV2Tests(unittest.TestCase):
    def test_actual_fixed_all100_candidate_cannot_collect_before_ready(self):
        context = audit.verify_registration(ROOT, audit.lineage.pointer(ROOT, REGISTRY), live=True)
        self.assertFalse(context['registry']['execution_ready'])
        self.assertEqual(len(context['expected_cases']), 100)
        self.assertEqual(len(context['current']['sources']), 79)
        self.assertEqual(next(iter(context['expected_cases'])), '5bad1717485347d1')
        for case in context['expected_cases'].values():
            self.assertEqual(case['replica'], 1)
            self.assertEqual(case['behavior_replica'], 1)
            self.assertFalse(case['context_primary_view'])
            self.assertTrue(case['teacher_used'])
            self.assertEqual(case['maximum_collection_ticks'], 200)
        with patch.object(worker.collection, 'collect_native_case', side_effect=AssertionError('native forbidden')) as native:
            with self.assertRaisesRegex(ValueError, 'not execution ready'):
                worker.collect_case(context['current']['path'], '5bad1717485347d1', context['current']['collection'],
                    ROOT/'CPU_peer_must_not_read_native_guard.json', REGISTRY)
            native.assert_not_called()

    def test_preserved_v1_missing_marker_counterexample_now_rejected(self):
        ledger, scope = old_fixtures.RequeryRawLedgerTests().fixture()
        for name in ('fresh_query_after_failed_command_id', 'same_episode_after_counted_wait',
                'actual_planning_requery_index'):
            ledger['commands'][1].pop(name)
        scope['actual_requeries'] = 0
        with self.assertRaisesRegex(ValueError, 'all fresh-query markers'):
            audit.validate_counted_commands(ledger, scope, ticks=10)

    def test_preserved_v1_outside_clock_counterexample_now_rejected(self):
        ledger, scope = old_fixtures.RequeryRawLedgerTests().fixture()
        for command in ledger['commands']:
            for name in ('generated_at', 'planning_requery_wait_started_at', 'planning_requery_wait_completed_at',
                    'planning_requery_last_actual_feedback_at', 'cutoff_at'):
                if name in command:
                    command[name] += 100.
        with self.assertRaisesRegex(ValueError, 'within the counted clock'):
            audit.validate_counted_commands(ledger, scope, ticks=10)

    def test_saved_NPZ_reconstructs_ten_distinct_feedback_rows_and_changed_file_rejected(self):
        ledger, scope, arrays = v2_fixtures.actual_CPU_follower_rows()
        with tempfile.TemporaryDirectory(prefix='high_cup_v2_peer_', dir=ROOT/'artifacts/framework/multitask_r10') as temp:
            path = Path(temp)/'CPU_feedback.npz'
            np.savez(path, **arrays)
            pointer = audit.lineage.pointer(ROOT, path)
            checked = audit.lineage.checked_pointer(ROOT, pointer, {})
            with np.load(checked, allow_pickle=False) as stored:
                saved = {name: stored[name].copy() for name in stored.files}
            proof = audit.validate_counted_commands(ledger, scope, ticks=12, arrays=saved)
            self.assertTrue(proof['actual_feedback_arrays_verified'])
            self.assertEqual(proof['reconstructed_requery_waits'][0]['wait_action_frames'], list(range(10)))
            self.assertEqual(proof['reconstructed_requery_waits'][0]['next_issue_frame'], 10)
            saved['actuator_feedback_timestamp'][4] = .2
            with self.assertRaisesRegex(ValueError, 'distinct consecutive50ms'):
                audit.validate_counted_commands(ledger, scope, ticks=12, arrays=saved)
            np.savez(path, **saved)
            with self.assertRaises(ValueError):
                audit.lineage.checked_pointer(ROOT, pointer, {})

    def test_terminal_frame_cannot_add_command_or_execution_completion(self):
        for name, value in (('teacher_command_id', 1), ('teacher_command_is_new', True),
                ('teacher_execution_valid', True), ('teacher_action_valid', True), ('teacher_command_completed', True)):
            ledger, scope, arrays = v2_fixtures.actual_CPU_follower_rows()
            arrays[name][-1] = value
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'terminal cache row'):
                audit.validate_counted_commands(ledger, scope, ticks=12, arrays=arrays)

    def test_failed_command_has_neither_positive_phase_event_nor_positive_actions(self):
        ledger, scope, arrays = v2_fixtures.actual_CPU_follower_rows()
        ledger['phase_completion_events'] = [dict(origin_command_id=0, phase='transport', completed=True)]
        with self.assertRaisesRegex(ValueError, 'positive phase completion'):
            audit.validate_counted_commands(ledger, scope, ticks=12, arrays=arrays)
        ledger['phase_completion_events'] = []
        arrays['teacher_action_valid'][3] = True
        with self.assertRaisesRegex(ValueError, 'promoted'):
            audit.validate_counted_commands(ledger, scope, ticks=12, arrays=arrays)

    def test_actual_feedback_cannot_be_optional_for_an_executed_wait(self):
        ledger, scope, arrays = v2_fixtures.actual_CPU_follower_rows()
        arrays.pop('actuator_feedback_timestamp')
        with self.assertRaisesRegex(ValueError, 'arrays missing'):
            audit.validate_counted_commands(ledger, scope, ticks=12, arrays=arrays)

    def test_typed_manifest_preserves_exact_projection_and_actual_plan_identity(self):
        fixture = v2_fixtures.TypedManifestV2Tests()
        fixture.setUp()
        try:
            with patch.object(audit, 'audit_collection', return_value=deepcopy(fixture.proof)) as raw, \
                    patch.object(audit, 'verify_registration', return_value=fixture.context):
                proof = audit.audit_manifest(fixture.root, fixture.path)
            raw.assert_called_once_with(fixture.root, fixture.projection['execution_registration'], require_complete=True)
            self.assertEqual(proof['acquisition_projection'], fixture.projection)
            changed = deepcopy(fixture.projection)
            changed['actual_plan'] = dict(path='CPU_foreign_plan.json', sha256='foreign')
            fixture.path.write_text(json.dumps(changed))
            broken = deepcopy(fixture.proof)
            broken['acquisition_projection'] = changed
            with patch.object(audit, 'audit_collection', return_value=broken), \
                    patch.object(audit, 'verify_registration', return_value=fixture.context), \
                    self.assertRaisesRegex(ValueError, 'actual plan'):
                audit.audit_manifest(fixture.root, fixture.path)
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
