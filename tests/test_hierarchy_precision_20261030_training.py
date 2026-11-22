import copy
import unittest

import torch

from our_mani.hierarchy_precision_20261030.precision import precision_report
from our_mani.hierarchy_precision_20261030.training import metric_position_objective, selection_key, weighted_mean


class TrainingContracts(unittest.TestCase):
    def test_physical_thresholds_penalize_distant_samples_without_clipping(self):
        prediction = torch.tensor([[.02, 0, 0], [.04, 0, 0], [.08, 0, 0]], requires_grad=True)
        loss = metric_position_objective(prediction, torch.zeros_like(prediction))
        self.assertLess(float(loss[0]), float(loss[1]))
        self.assertGreater(float(loss[2]), 10 * float(loss[1]))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[:, 0] > 0).all())
        self.assertAlmostEqual(float(prediction.detach()[2, 0]), .08, places=6)

    def test_one_outlier_changes_selection_even_with_low_average(self):
        ledger = [dict(sample_id=f"{task}:{head}:{i}", split="validation", task=task, head=head)
                  for task in ("a", "b", "c", "d") for head in ("candidate", "tcp") for i in range(10)]
        observations = [{**row, "pred_xyz": [.01, 0, 0], "target_xyz": [0, 0, 0]} for row in ledger]
        report = precision_report(observations, expected_samples=ledger, usage="selection")
        reference = selection_key(report)
        observations[0]["pred_xyz"] = [.051, 0, 0]
        changed = precision_report(observations, expected_samples=ledger, usage="selection")
        self.assertGreater(selection_key(changed), reference)
        self.assertFalse(changed["by_split"]["validation"]["hard_gate_passed"])

    def test_test_or_incomplete_cohort_cannot_select(self):
        ledger = [dict(sample_id=f"{task}:{head}", split="test", task=task, head=head)
                  for task in ("a", "b", "c", "d") for head in ("candidate", "tcp")]
        observations = [{**row, "pred_xyz": [0, 0, 0], "target_xyz": [0, 0, 0]} for row in ledger]
        test = precision_report(observations, expected_samples=ledger)
        with self.assertRaises(ValueError):
            selection_key(test)
        for row in ledger + observations:
            row["split"] = "validation"
        incomplete = precision_report(observations[:-1], expected_samples=ledger, usage="selection")
        with self.assertRaises(ValueError):
            selection_key(incomplete)

    def test_invalid_weights_do_not_remove_difficult_samples(self):
        with self.assertRaises(ValueError):
            weighted_mean(torch.tensor([1., 20.]), torch.tensor([1., 0.]))
        self.assertAlmostEqual(float(weighted_mean(torch.tensor([1., 20.]), torch.ones(2))), 10.5)


if __name__ == "__main__":
    unittest.main()
