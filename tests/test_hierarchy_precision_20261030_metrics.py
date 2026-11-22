import json
import unittest

import numpy as np
import torch

from our_mani.hierarchy_precision_20261030.precision import (
    position_loss_m, precision_report, validate_split_usage,
)


class PhysicalPrecisionTests(unittest.TestCase):
    def ledger(self, count, *, split="validation", task="pick", component="candidate"):
        return [dict(split=split, task=task, head=component, sample_id=str(i)) for i in range(count)]

    def observations(self, ledger, distances):
        return [{**row, "pred_xyz": [float(distance), 0., 0.], "target_xyz": [0., 0., 0.]}
                for row, distance in zip(ledger, distances)]

    def metrics(self, report, split="validation", task="pick", component="candidate"):
        return report["by_split"][split]["tasks"][task][component]

    def test_physical_metre_loss_and_finite_zero_error_gradient(self):
        predicted = torch.tensor([[.03, .04, 0.], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(predicted)
        distances = position_loss_m(predicted, target, split="train", reduction="none")
        torch.testing.assert_close(distances, torch.tensor([.05, 0.], dtype=torch.float64))
        loss = position_loss_m(predicted, target, split="train")
        self.assertAlmostEqual(float(loss.detach()), .025)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        np.testing.assert_allclose(position_loss_m([[.03, .04, 0.]], [[0., 0., 0.]], split="train", reduction="none"), [.05])
        with self.assertRaises(ValueError):
            position_loss_m([[float("nan"), 0., 0.]], [[0., 0., 0.]], split="train")
        with self.assertRaises(ValueError):
            position_loss_m(np.empty((0, 3)), np.empty((0, 3)), split="train")

    def test_single_outlier_fails_all_samples_gate_even_when_p95_passes(self):
        ledger = self.ledger(100)
        report = precision_report(self.observations(ledger, [.01] * 99 + [.051]), expected_samples=ledger)
        metrics = self.metrics(report)
        self.assertLess(metrics["p95_m"], .05)
        self.assertLess(metrics["mean_m"], .03)
        self.assertAlmostEqual(metrics["max_m"], .051)
        self.assertEqual(metrics["fraction_le_5cm"], .99)
        self.assertFalse(metrics["hard_gate_passed"])
        self.assertTrue(metrics["mean_le_3cm"])
        self.assertFalse(metrics["precision_goal_met"])

    def test_exact_threshold_boundaries_and_distinct_mean_target(self):
        ledger = self.ledger(2)
        report = precision_report(self.observations(ledger, [.03, .05]), expected_samples=ledger)
        metrics = self.metrics(report)
        self.assertEqual(metrics["fraction_le_3cm"], .5)
        self.assertEqual(metrics["fraction_le_5cm"], 1.)
        self.assertTrue(metrics["hard_gate_passed"])
        self.assertFalse(metrics["mean_le_3cm"])
        self.assertAlmostEqual(metrics["p50_m"], .04)
        report = precision_report(self.observations(ledger, [.03, .03]), expected_samples=ledger)
        self.assertTrue(self.metrics(report)["precision_goal_met"])

    def test_missing_and_nonfinite_rows_are_not_filtered_from_denominator(self):
        ledger = self.ledger(3)
        rows = self.observations(ledger, [.01, .02])
        rows[1]["pred_xyz"] = [float("inf"), 0., 0.]
        report = precision_report(rows, expected_samples=ledger)
        metrics = self.metrics(report)
        self.assertEqual(metrics["expected_samples"], 3)
        self.assertEqual(metrics["valid_samples"], 1)
        self.assertEqual(metrics["invalid_or_missing_samples"], 2)
        self.assertEqual(metrics["fraction_le_5cm"], 1 / 3)
        for key in ("mean_m", "p50_m", "p95_m", "max_m"):
            self.assertIsNone(metrics[key])
        self.assertFalse(metrics["hard_gate_passed"])
        self.assertEqual({row["reason"] for row in report["invalid_samples"]}, {"missing_sample", "nonfinite_pred_xyz"})
        json.dumps(report, allow_nan=False)

    def test_task_component_and_split_statistics_never_mix(self):
        groups = [self.ledger(1), self.ledger(1, component="TCP"),
                  self.ledger(1, task="insert"), self.ledger(1, split="train"),
                  self.ledger(1, split="test")]
        expected = sum(groups, [])
        rows = sum([self.observations(group, [distance]) for group, distance in zip(groups, [.01, .06, .02, .001, .04])], [])
        report = precision_report(rows, expected_samples=expected)
        self.assertAlmostEqual(self.metrics(report)["mean_m"], .01)
        self.assertAlmostEqual(self.metrics(report, component="tcp")["mean_m"], .06)
        self.assertAlmostEqual(self.metrics(report, task="insert")["mean_m"], .02)
        self.assertAlmostEqual(self.metrics(report, split="train")["mean_m"], .001)
        self.assertAlmostEqual(self.metrics(report, split="test")["mean_m"], .04)
        self.assertFalse(report["by_split"]["validation"]["hard_gate_passed"])
        self.assertTrue(report["by_split"]["train"]["hard_gate_passed"])

    def test_test_is_rejected_for_training_or_selection_before_position_access(self):
        class UnreadablePositions(dict):
            def __getitem__(self, key):
                if key in ("pred_xyz", "target_xyz"):
                    raise AssertionError("TEST positions were read for model selection")
                return super().__getitem__(key)
        ledger = self.ledger(1, split="test")
        with self.assertRaises(ValueError):
            precision_report([UnreadablePositions(ledger[0])], expected_samples=ledger, usage="selection")
        with self.assertRaises(ValueError):
            position_loss_m([[0., 0., 0.]], [[0., 0., 0.]], split="TEST", usage="training")
        with self.assertRaises(ValueError):
            position_loss_m([[0., 0., 0.]], [[0., 0., 0.]], split="test", usage="selection")
        with self.assertRaises(ValueError):
            validate_split_usage("validation", "training")

    def test_duplicates_extra_samples_and_empty_ledgers_cannot_pass(self):
        ledger = self.ledger(1)
        rows = self.observations(ledger, [.01])
        duplicate = precision_report(rows + rows, expected_samples=ledger)
        self.assertFalse(self.metrics(duplicate)["hard_gate_passed"])
        self.assertIsNone(self.metrics(duplicate)["mean_m"])
        extra_row = {**rows[0], "sample_id": "unplanned"}
        extra = precision_report(rows + [extra_row], expected_samples=ledger)
        self.assertFalse(self.metrics(extra)["hard_gate_passed"])
        self.assertIsNone(self.metrics(extra)["mean_m"])
        self.assertEqual(len(extra["unplanned_samples"]), 1)
        with self.assertRaises(ValueError):
            precision_report([], expected_samples=[])
        with self.assertRaises(ValueError):
            precision_report(rows, expected_samples=ledger + ledger)


if __name__ == "__main__":
    unittest.main()
