"""Pure CPU loss/gradient checks: no optimizer, fit, data capture or weights."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

import torch

from our_mani.near_goal_depth_refiner_20261103.model import binary_segmentation_loss
from our_mani.near_goal_depth_refiner_20261103.training import _loss_batch
from our_mani.near_goal_depth_refiner_20261103_v5 import losses


torch.set_num_threads(2)


class UnknownAbstentionV5LossTests(unittest.TestCase):
    def test_no_UNKNOWN_reduces_to_old_value_and_every_parameter_gradient_exactly(self):
        torch.manual_seed(17)
        model = torch.nn.Conv2d(9, 2, 1)
        clone = deepcopy(model)
        x = torch.randn(2, 4, 9, 64, 64)
        x[:, :, 8] = (x[:, :, 8] > 0).float()
        batch = dict(features=x, labels=torch.randint(2, (2, 4, 64, 64)))
        before = _loss_batch(model, batch, 'cpu')
        after = losses.loss_batch(clone, batch, 'cpu')
        self.assertTrue(torch.equal(before, after))
        gradients_a = torch.autograd.grad(before, tuple(model.parameters()))
        gradients_b = torch.autograd.grad(after, tuple(clone.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(gradients_a, gradients_b)))

    def test_real_UNKNOWN_has_restoring_gradient_padding_and_uniform_have_zero_gradient(self):
        logits = torch.tensor([[[[8., 8., 0.]], [[-8., -8., 0.]]]], requires_grad=True)
        labels = torch.full((1, 1, 3), 255)
        real = torch.tensor([[[True, False, True]]])
        terms = losses.per_view_components(logits, labels, real)
        self.assertTrue(torch.equal(terms['known'], torch.zeros(1)))
        self.assertGreater(float(terms['abstention'][0].detach()), 0)
        gradient = torch.autograd.grad(terms['total'].sum(), logits)[0]
        self.assertGreater(float(gradient[0, 0, 0, 0].detach()), .24)
        self.assertLess(float(gradient[0, 1, 0, 0].detach()), -.24)
        self.assertTrue(torch.equal(gradient[:, :, :, 1:], torch.zeros_like(gradient[:, :, :, 1:])))

    def test_per_view_normalization_is_independent_of_UNKNOWN_pixel_count(self):
        logits = torch.empty(2, 2, 10, 10)
        logits[:, 0] = 4.; logits[:, 1] = -4.
        labels = torch.full((2, 10, 10), 255)
        real = torch.zeros_like(labels, dtype=torch.bool); real[0, 0, 0] = True; real[1] = True
        values = losses.unknown_abstention_loss(logits, labels, real)
        self.assertTrue(torch.allclose(values[0], values[1], rtol=0, atol=1e-6))

    def test_all_UNKNOWN_saturated_logits_finite_and_not_flat_gradient(self):
        logits = torch.tensor([[[[1000.]], [[-1000.]]]], requires_grad=True)
        labels = torch.full((1, 1, 1), 255); real = torch.ones_like(labels, dtype=torch.bool)
        terms = losses.per_view_components(logits, labels, real)
        self.assertTrue(torch.isfinite(terms['total']).all())
        gradient = torch.autograd.grad(terms['total'].sum(), logits)[0]
        torch.testing.assert_close(gradient.flatten(), torch.tensor([.5, -.5]), rtol=0, atol=0)
        uniform = losses.per_view_components(torch.zeros(1, 2, 1, 1), labels, real)
        self.assertEqual(float(uniform['total'][0]), 0.)

    def test_known_term_and_gradient_are_original_even_when_UNKNOWN_is_present(self):
        torch.manual_seed(21)
        logits = torch.randn(2, 2, 3, 3, requires_grad=True)
        labels = torch.randint(2, (2, 3, 3)); labels[:, 1, 1] = 255
        original = binary_segmentation_loss(logits, labels)
        terms = losses.per_view_components(logits, labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertTrue(torch.equal(terms['known'], original))
        a = torch.autograd.grad(original.sum(), logits, retain_graph=True)[0]
        b = torch.autograd.grad(terms['known'].sum(), logits)[0]
        self.assertTrue(torch.equal(a, b))

    def test_features8_controls_UNKNOWN_and_only_one_forward_is_used(self):
        class Fixed(torch.nn.Module):
            calls = 0
            def forward(self, x):
                self.calls += 1
                return torch.stack((x[:, 0], -x[:, 0]), 1)
        x = torch.zeros(1, 4, 9, 64, 64); x[:, :, 0] = 8.
        y = torch.full((1, 4, 64, 64), 255)
        model = Fixed()
        self.assertEqual(float(losses.loss_batch(model, dict(features=x, labels=y), 'cpu')), 0.)
        self.assertEqual(model.calls, 1)
        x[:, :, 8] = 1.
        self.assertGreater(float(losses.loss_batch(model, dict(features=x, labels=y), 'cpu')), 7.)
        self.assertEqual(model.calls, 2)
        x[:, :, 8] = .5
        with self.assertRaises(ValueError): losses.loss_batch(model, dict(features=x, labels=y), 'cpu')
        self.assertEqual(model.calls, 2)

    def test_protocol_and_original_loss_function_identity(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_unknown_abstention_v5_protocol_v1/protocol.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), losses.PROTOCOL_SHA256)
        self.assertIs(losses.binary_segmentation_loss, binary_segmentation_loss)
        self.assertEqual(losses.UNKNOWN_WEIGHT, 1.)
        self.assertEqual(losses.REAL_CROP_FEATURE_INDEX, 8)


if __name__ == '__main__':
    unittest.main()
