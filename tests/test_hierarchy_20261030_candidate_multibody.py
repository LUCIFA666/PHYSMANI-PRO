import unittest
import torch

from our_mani.hierarchy_20261030.candidate_multibody import (
    MultiBodyCandidateTransformer, candidate_imitation_loss,
)


class CandidateMultiBodyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        self.model = MultiBodyCandidateTransformer([0, 0, .8], [.5, .6, .5],
                                                   role_names=("box", "frame"), model_dim=32)
        self.trajectory = torch.randn(2, 2, 23, 20)
        self.trajectory[..., -1] = torch.arange(23)*.05
        self.mask = torch.ones(2, 2, 23, dtype=torch.bool)
        self.skill = torch.randn(2, 64, requires_grad=True)
        self.robot, self.scene = torch.randn(2, 15), torch.randn(2, 128)

    def test_real_forecasts_with_equal_times_across_bodies_and_gradient(self):
        output = self.model(self.trajectory, self.mask, self.skill, self.robot, self.scene)
        loss, outside = candidate_imitation_loss(self.model, output, torch.tensor([[0., 0., .8], [.1, .1, .9]]))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(self.skill.grad).all())
        self.assertGreater(float(self.skill.grad.abs().sum()), 0)
        self.assertFalse(bool(outside.any()))
        self.assertEqual(tuple(output["goal_xyz"].shape), (2, 3))

    def test_unknown_body_values_cannot_affect_prediction(self):
        self.model.eval()
        self.mask[:, 1] = False
        a = self.model(self.trajectory, self.mask, self.skill, self.robot, self.scene)
        other = self.trajectory.clone()
        other[:, 1] = float("nan")
        b = self.model(other, self.mask, self.skill, self.robot, self.scene)
        torch.testing.assert_close(a["point_logits"], b["point_logits"], rtol=0, atol=0)

    def test_no_fake_forecast_and_bad_clock_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.trajectory, self.mask & False, self.skill, self.robot, self.scene)
        self.trajectory[0, 1, -1, -1] += .01
        with self.assertRaises(ValueError):
            self.model(self.trajectory, self.mask, self.skill, self.robot, self.scene)


if __name__ == "__main__":
    unittest.main()
