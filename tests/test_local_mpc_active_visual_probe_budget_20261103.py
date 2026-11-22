"""Fixed-budget timing ablation preserves evidence and never renews a deadline."""
import unittest

from our_mani.local_mpc_control_20261103.active_visual_attachment import (
    ActiveVisualAttachment, ActiveVisualConfig)
from test_local_mpc_active_visual_attachment_20261103 import sample, h


class ProbeBudgetTests(unittest.TestCase):
    def admitted(self):
        estimator = ActiveVisualAttachment(ActiveVisualConfig(probe_seconds=1.2))
        obj, robot = sample(0.)
        estimator.observe(obj, robot, close_requested=False)
        obj, robot = sample(.05)
        estimator.observe(obj, robot, close_requested=True)
        return estimator, estimator.begin_probe(obj, robot, h(.05))

    def test_only_predeclared_fixed_budgets_are_allowed(self):
        self.assertEqual(ActiveVisualConfig().probe_seconds, .8)
        for duration in (0., .9, 1., 1.3):
            with self.assertRaises(ValueError):
                ActiveVisualConfig(probe_seconds=duration)
        with self.assertRaises(ValueError):
            ActiveVisualConfig(probe_seconds=1.2, minimum_TCP_vertical_m=.009)

    def test_slow_actual_comotion_can_confirm_after_original_budget(self):
        estimator, spec = self.admitted()
        self.assertAlmostEqual(spec['deadline'], 1.25)
        for step in range(1, 20):
            obj, robot = sample(.05 + .05 * step, dz=.0105 * step / 19)
            result = estimator.observe(obj, robot, close_requested=True)
            again = estimator.begin_probe(obj, robot, h(robot.timestamp))
            if again is not None:
                self.assertEqual(again['deadline'], spec['deadline'])
            if step <= 17:
                self.assertNotEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(result['fixed_probe_budget_protocol'], 'development_fixed_12s_timing_ablation')
        self.assertGreater(result['diagnostic']['TCP_signed_vertical_m'], .010)

    def test_stationary_object_still_fails_and_deadline_never_rolls(self):
        estimator, spec = self.admitted()
        for step in range(1, 9):
            obj, robot = sample(.05 + .05 * step, dz=.012 * step / 8, object_dz=0.)
            result = estimator.observe(obj, robot, close_requested=True)
            if result['probe_failed']:
                break
        self.assertTrue(result['probe_failed'])
        self.assertIn('fixed_r0', result['reason'])
        self.assertNotEqual(result['mode'], 'ATTACHED_PUBLIC')
        self.assertEqual(estimator.spec['deadline'], spec['deadline'])


if __name__ == '__main__':
    unittest.main()
