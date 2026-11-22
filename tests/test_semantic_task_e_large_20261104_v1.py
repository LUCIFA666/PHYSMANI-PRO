import unittest

import torch

from our_mani.semantic_task_e_large_20261104_v1.model import (
    EncoderConfig, LargeSemanticTaskEncoderV1, parameter_count,
)


class LargeSemanticTests(unittest.TestCase):
    def test_exact_capacity_counts_without_allocating_weights(self):
        for cfg, count in (
            (EncoderConfig(width=512, layers=8, heads=8), 30454040),
            (EncoderConfig(width=768, layers=12, heads=12), 90760472),
            (EncoderConfig(), 207847704),
        ):
            self.assertEqual(parameter_count(cfg), count)

    def test_small_real_path_prefix_and_camera_set(self):
        torch.manual_seed(9)
        cfg = EncoderConfig(width=32, layers=1, heads=4, cnn_base=4,
                            maximum_history=2, dropout=0.)
        model = LargeSemanticTaskEncoderV1(cfg).eval()
        x = torch.randn(1, 2, 4, 4, 64, 64)
        robot = torch.randn(1, 2, 15); robot[:, :, -1] = 0
        lengths = torch.tensor([1])
        with torch.no_grad():
            before = model(rgbd=x, robot_history=robot, lengths=lengths)
            x[:, 1] = float('nan'); robot[:, 1] = float('nan')
            after = model(rgbd=x, robot_history=robot, lengths=lengths)
            permuted = model(rgbd=x[:, :, [3, 1, 0, 2]], robot_history=robot, lengths=lengths)
        self.assertTrue(torch.equal(before['task_e'], after['task_e']))
        self.assertTrue(torch.allclose(before['task_e'], permuted['task_e'], atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(before['task_e'].reshape(1, 2, 8).sum(-1), torch.ones(1, 2)))

    def test_privileged_gripper_channel_and_invalid_shapes_rejected(self):
        cfg = EncoderConfig(width=32, layers=1, heads=4, cnn_base=4)
        model = LargeSemanticTaskEncoderV1(cfg)
        robot = torch.zeros(1, 1, 15); robot[:, :, -1] = 1
        with self.assertRaises(ValueError):
            model(rgbd=torch.zeros(1, 1, 4, 4, 64, 64),
                  robot_history=robot, lengths=torch.ones(1, dtype=torch.long))
        with self.assertRaises(ValueError):
            EncoderConfig(width=128, heads=3)

    def test_real_BF16_autocast_prefix_has_FP32_normalized_outputs(self):
        cfg = EncoderConfig(width=32, layers=1, heads=4, cnn_base=4,
                            maximum_history=2, dropout=0.)
        model = LargeSemanticTaskEncoderV1(cfg).train()
        x=torch.randn(1,2,4,4,64,64)
        robot=torch.zeros(1,2,15)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=model(rgbd=x,robot_history=robot,lengths=torch.tensor([1]))
            loss=torch.nn.functional.cross_entropy(result['operation_logits'],torch.tensor([0]))
        loss.backward()
        self.assertEqual(result['task_e'].dtype,torch.float32)
        self.assertEqual(result['operation_probabilities'].dtype,torch.float32)
        self.assertTrue(torch.allclose(result['task_e'].reshape(1,2,8).sum(-1),torch.ones(1,2),atol=1e-6))


if __name__ == '__main__':
    unittest.main()
