from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.multitask.block_full_budget_v176 import finish_counted_budget


class Boundary:
    def __init__(self, outcome=None):
        self.ledger = NS(ticks=197, dt=.05)
        self.holds = 0
        self.outcome = outcome or (lambda tick: (np.bool_(False), np.bool_(False)))
        self.task = NS(success=lambda: self.outcome(self.ledger.ticks))
    def hold_tick(self):
        self.holds += 1
        self.ledger.ticks += 1


class Tests(unittest.TestCase):
    def test_failed_policy_consumes_only_remaining_counted_ticks_without_retry(self):
        b = Boundary()
        r = finish_counted_budget(b, dict(official_task_success=False, failure='policy_failed'))
        self.assertEqual((b.holds, b.ledger.ticks), (3, 200))
        self.assertEqual(r['failure'], 'policy_failed')
        self.assertEqual(r['block_full_budget_v176']['policy_failed_retries'], 0)

    def test_original_numpy_bool_native_success_ends_at_actual_success_tick(self):
        b = Boundary(lambda tick: (np.bool_(tick == 199), np.bool_(tick == 199)))
        r = finish_counted_budget(b, dict(official_task_success=False, failure='policy_failed'))
        self.assertTrue(r['official_task_success'])
        self.assertEqual((b.holds, b.ledger.ticks), (2, 199))
        self.assertIsNone(r['failure'])
        self.assertEqual(r['block_full_budget_v176']['original_component_failure'], 'policy_failed')

    def test_native_terminal_failure_or_existing_success_never_advances_again(self):
        b = Boundary(lambda tick: (False, True))
        r = finish_counted_budget(b, dict(official_task_success=False, failure='policy_failed'))
        self.assertTrue(r['native_terminal_failure'])
        self.assertEqual(b.holds, 1)
        b = Boundary()
        finish_counted_budget(b, dict(official_task_success=True, failure=None))
        self.assertEqual(b.holds, 0)

    def test_failed_reset_is_infrastructure_error_and_cannot_be_filled_with_holds(self):
        b = Boundary()
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            finish_counted_budget(b, dict(official_task_success=False, failure='development_reset_ik_failed'))
        self.assertEqual(b.holds, 0)


if __name__ == '__main__': unittest.main()
