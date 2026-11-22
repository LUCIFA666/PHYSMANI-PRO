import json
from pathlib import Path
import tempfile
import unittest

from our_mani.multitask_executor_cycle_20261112_v2.common import PLAN, read
from our_mani.multitask_executor_cycle_20261112_v2.pipeline import invalid_reset_only


class ColdResetBoundary(unittest.TestCase):
    def test_every_potential_seed_fixed_before_VAL_TEST_and_unique(self):
        plan = read(PLAN)
        seeds = [s for c in plan['cases'] for s in c['cold_reset_seeds']]
        self.assertEqual(len(plan['cases']), 84)
        self.assertEqual(len(seeds), 420)
        self.assertEqual(len(set(seeds)), 420)
        self.assertFalse(plan['VAL_TEST_outcomes_observed_before_retry_rule'])
        self.assertFalse(plan['valid_policy_failure_or_NativeFalse_retry_allowed'])

    def test_native_reset_error_only_before_any_policy_or_action_can_retry(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'worker').mkdir()
            path = root/'worker/failure.json'
            path.write_text(json.dumps({'error_type': 'TaskEnvironmentError'}))
            receipt = dict(complete=False, actual_wait_completed=True, unreaped_descendants={}, stop_reason=None)
            self.assertTrue(invalid_reset_only(root, receipt))
            for marker in ('initial.json', 'execution.jsonl', 'public_frame_provenance.jsonl'):
                p = root/'worker'/marker
                p.write_text('{}')
                self.assertFalse(invalid_reset_only(root, receipt))
                p.unlink()
            path.write_text(json.dumps({'error_type': 'RuntimeError'}))
            self.assertFalse(invalid_reset_only(root, receipt))

    def test_policy_failure_or_unwaited_or_scoped_stop_does_not_retry(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'worker').mkdir()
            (root/'worker/failure.json').write_text(json.dumps({'error_type': 'TaskEnvironmentError'}))
            for receipt in (dict(complete=True, actual_wait_completed=True, unreaped_descendants={}, stop_reason=None),
                            dict(complete=False, actual_wait_completed=False, unreaped_descendants={}, stop_reason=None),
                            dict(complete=False, actual_wait_completed=True, unreaped_descendants={'1': {}}, stop_reason=None),
                            dict(complete=False, actual_wait_completed=True, unreaped_descendants={}, stop_reason='scoped_STOP')):
                self.assertFalse(invalid_reset_only(root, receipt))


if __name__ == '__main__':
    unittest.main()
