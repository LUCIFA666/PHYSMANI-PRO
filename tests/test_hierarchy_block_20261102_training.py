import copy
import unittest

import torch

from our_mani.hierarchy_block_20261102.training import (
    candidate_loss, skill_loss, position_penalty, qualified, selection_key,
)
from our_mani.hierarchy_precision_20261030.precision import precision_report


def report(split="validation", error=.01):
    ledger = [dict(sample_id=f"{task}:{head}", split=split, task=task, head=head)
              for task in ("normal", "high") for head in ("candidate", "tcp")]
    rows = [{**entry, "pred_xyz": [error, 0, 0], "target_xyz": [0, 0, 0]} for entry in ledger]
    value = precision_report(rows, expected_samples=ledger)
    value["sequence_precision"] = copy.deepcopy(value)
    value["predicted_goal_conditioned_precision"] = copy.deepcopy(value)
    value["source_coverage"] = dict(complete=True)
    return value


class BlockTrainingTests(unittest.TestCase):
    def test_centimetre_penalty_has_no_output_clipping(self):
        distance = torch.tensor([.02, .04, .08], requires_grad=True)
        value = position_penalty(distance)
        self.assertGreater(float(value.detach()[2]), float(value.detach()[1]))
        value.sum().backward()
        self.assertTrue(torch.isfinite(distance.grad).all())
        self.assertTrue((distance.grad > 0).all())
        self.assertGreater(float(distance.grad[2]), float(distance.grad[1]))

    def test_sequence_tail_is_not_extra_supervision(self):
        target = torch.zeros(1, 4, 8)
        target[..., 6] = 1.
        labels = dict(action_sequence=target, action_mask=torch.tensor([[True, False, False, False]]))
        prediction = dict(positions=torch.zeros(1, 4, 3), rotation_matrices=torch.eye(3).repeat(1, 4, 1, 1),
                          gripper_logits=torch.tensor([[[5., -5.]] * 4]))
        original = skill_loss(prediction, labels, torch.ones(1), [1., .5, .3, .2])
        target[:, 1:, :3] = 100.
        target[:, 1:, 7] = 1.
        changed = skill_loss(prediction, labels, torch.ones(1), [1., .5, .3, .2])
        self.assertEqual(float(original), float(changed))

    def test_candidate_mixture_allows_differentiable_proposals(self):
        positions = torch.tensor([[[.01, 0., 0.], [.06, 0., 0.]]], requires_grad=True)
        output = dict(positions=positions, rotation_matrices=torch.eye(3).repeat(1, 2, 1, 1),
                      times=torch.tensor([[.3, .5]], requires_grad=True), logits=torch.zeros(1, 2, requires_grad=True))
        labels = dict(handoff_position=torch.zeros(1, 3), handoff_quaternion=torch.tensor([[0., 0., 0., 1.]]),
                      handoff_remaining_seconds=torch.tensor([.3]))
        value = candidate_loss(output, labels, torch.ones(1))
        value.backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        self.assertTrue(torch.isfinite(output["logits"].grad).all())

    def test_gate_and_selection_include_future_sequence_and_source_coverage(self):
        original = report()
        self.assertTrue(qualified(original, "validation"))
        key = selection_key(original)
        changed = report()
        changed["sequence_precision"] = report(error=.051)
        self.assertFalse(qualified(changed, "validation"))
        self.assertGreater(selection_key(changed), key)
        changed = report()
        changed["source_coverage"]["complete"] = False
        self.assertFalse(qualified(changed, "validation"))
        with self.assertRaises(ValueError):
            selection_key(report(split="test"))


if __name__ == "__main__":
    unittest.main()
