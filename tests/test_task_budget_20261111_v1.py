"""Every task gets a physical-time budget; mismatched limits cannot hide."""
import unittest
from our_mani.highspeed_control_20261111_v1.task_budget import TaskBudget


class TaskBudgetContracts(unittest.TestCase):
    def test_common_default_is_16seconds_320_physical_ticks(self):
        for task in ['ordinary','high_speed','unrelated_task']:
            b=TaskBudget.from_config(dict(task=task));self.assertEqual(b.seconds,16.);self.assertEqual(b.ticks,320)
            self.assertEqual(b.receipt()['success_sampling_period_seconds'],.05)
    def test_explicit_override_and_original_comparison_are_distinct(self):
        self.assertEqual(TaskBudget.from_config(dict(task_time_limit_seconds=20.)).ticks,400)
        self.assertEqual(TaskBudget.from_config(dict(budget_ticks=160)).seconds,8.)
    def test_conflicting_or_fractional_limits_are_rejected(self):
        for config in [dict(task_time_limit_seconds=16.,budget_ticks=160),dict(task_time_limit_seconds=.075),dict(task_time_limit_seconds=True),dict(budget_ticks=3.2)]:
            with self.assertRaises(ValueError):TaskBudget.from_config(config)


if __name__=='__main__':unittest.main()
