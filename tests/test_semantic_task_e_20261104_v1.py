import inspect
import unittest

import torch

from our_mani.semantic_task_e_20261104_v1.model import (
    OPERATIONS, SemanticTaskEncoderV1, operation_loss, semantic_label,
)


class SemanticTaskTests(unittest.TestCase):
    def test_speed_is_not_an_operation_label(self):
        for index, name in enumerate(OPERATIONS):
            self.assertEqual(semantic_label(name), index)
            self.assertEqual(semantic_label(name + '_high_speed'), index)
        with self.assertRaises(ValueError):
            semantic_label('place_cups_on_rotating_frame/cup2')

    def test_public_signature_and_no_old_policy(self):
        self.assertEqual(set(inspect.signature(SemanticTaskEncoderV1.forward).parameters),
                         {'self', 'rgbd', 'robot_history', 'lengths'})
        model = SemanticTaskEncoderV1()
        self.assertFalse(any('skill_embeddings' in key or key.startswith('classifier.')
                             for key in model.state_dict()))
        self.assertFalse(model.spec()['all16_closed_loop_ready'])

    def test_real_feature_path_simnorm_and_padded_future_ignored(self):
        torch.manual_seed(3)
        model = SemanticTaskEncoderV1().eval()
        rgbd = torch.randn(1, 2, 4, 4, 64, 64)
        robot = torch.randn(1, 2, 15)
        lengths = torch.tensor([1])
        with torch.no_grad():
            first = model(rgbd=rgbd, robot_history=robot, lengths=lengths)
            rgbd[:, 1] = float('nan'); robot[:, 1] = float('nan')
            changed = model(rgbd=rgbd, robot_history=robot, lengths=lengths)
        self.assertTrue(torch.equal(first['task_e'], changed['task_e']))
        self.assertTrue(torch.equal(first['operation_logits'], changed['operation_logits']))
        self.assertTrue(torch.allclose(first['task_e'].reshape(1, 2, 8).sum(-1), torch.ones(1, 2)))
        self.assertTrue(torch.isfinite(operation_loss(first, torch.tensor([7]))))

    def test_valid_nonfinite_and_bad_lengths_rejected(self):
        model = SemanticTaskEncoderV1()
        x = torch.zeros(1, 1, 4, 4, 64, 64)
        robot = torch.zeros(1, 1, 15)
        with self.assertRaises(ValueError):
            model(rgbd=x, robot_history=robot, lengths=torch.tensor([0]))
        x[0, 0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(rgbd=x, robot_history=robot, lengths=torch.tensor([1]))


if __name__ == '__main__':
    unittest.main()
