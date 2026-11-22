"""Language/action gradient and causal physical-contract regression checks."""
import unittest

import torch
from torch.nn import functional as F

from our_mani.candidate_transformer_cube_20261104_v1.model import CandidateConfig
from our_mani.cube_matched_candidate_20261104_v1.model import CubeMatchedCandidateTransformer
from our_mani.cube_matched_candidate_20261104_v1.losses import candidate_loss
from our_mani.language_conditioned_cube_20261112_v1.model import InstructionConditionedCubeCandidate
from our_mani.language_conditioned_cube_20261112_v1.losses import instruction_candidate_loss
from our_mani.language_conditioned_cube_20261112_v1.text import validate_instructions


def public_features(batch=2):
    future = torch.zeros(batch, 17, 16)
    future[:, :, 0] = torch.arange(17) / 16.
    future[:, :, 4:10] = torch.tensor([1., 0., 0., 0., 1., 0.])
    robot = torch.zeros(batch, 25)
    robot[:, -2:] = 1.
    return dict(future_tokens=future, robot=robot,
                geometry=torch.tensor([[.05, .05, .05, .9]]).expand(batch, 4),
                context_c=torch.full((batch, 16), .125))


class LanguageContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(112)
        self.config = CandidateConfig(d_model=32, num_layers=1, num_heads=4, dropout=0.)
        self.model = InstructionConditionedCubeCandidate(self.config).eval()
        # A fitted candidate has nonzero heads. Use nonzero synthetic weights
        # here; the separate real-weight verifier checks the actual checkpoint.
        with torch.no_grad():
            for head in (self.model.candidate.position_head, self.model.candidate.rotation_head,
                         self.model.candidate.time_head):
                head.weight.normal_(std=.03)
        self.text = F.normalize(torch.randn(2, 1024), dim=-1)

    def test_exact_baseline_warm_start_before_any_language_fit(self):
        baseline = CubeMatchedCandidateTransformer(self.config).eval()
        baseline.load_state_dict(self.model.candidate.state_dict(), strict=True)
        features = public_features()
        old = baseline(**features, task_e=torch.full((2, 16), .125))
        new = self.model(**features, instruction_features=self.text)
        for name, value in old.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, new[name]), name)
            else:
                self.assertEqual(value, new[name])
        self.assertTrue(torch.equal(new['task_e'], torch.full((2, 16), .125)))

    def test_action_loss_reaches_language_adapter_and_not_input_ids(self):
        output = self.model(**public_features(), instruction_features=self.text)
        loss = candidate_loss(output, torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1),
                              torch.full((2,), .3), torch.ones(2, dtype=torch.bool))
        loss['per_example'].sum().backward()
        grad = self.model.language.to_task_e.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)

    def test_text_change_can_change_actions_without_changing_motion_context(self):
        with torch.no_grad():
            self.model.language.to_task_e.weight.normal_(std=.05)
        features = public_features(1)
        before = {k: v.clone() for k, v in features.items()}
        first = self.model(**features, instruction_features=self.text[:1])
        second = self.model(**features, instruction_features=self.text[1:])
        self.assertFalse(torch.equal(first['task_e'], second['task_e']))
        self.assertFalse(torch.allclose(first['positions'], second['positions'], atol=1e-7, rtol=0))
        for name in before:
            self.assertTrue(torch.equal(before[name], features[name]), name)
        self.assertTrue(torch.equal(first['proposal_logits'], torch.zeros(1, 8)))
        self.assertTrue(torch.allclose(first['task_e'].reshape(1, 2, 8).sum(-1), torch.ones(1, 2)))

    def test_physical_motion_still_affects_candidate_with_same_instruction(self):
        features = public_features(1)
        first = self.model(**features, instruction_features=self.text[:1])
        features['future_tokens'][:, 1:, 1] = .04
        features['future_tokens'][:, :, 10] = .2
        second = self.model(**features, instruction_features=self.text[:1])
        self.assertTrue(torch.equal(first['task_e'], second['task_e']))
        self.assertFalse(torch.equal(first['positions'], second['positions']))

    def test_original_causal_clock_and_simnorm_guards_remain(self):
        for name in ('future_tokens', 'context_c'):
            features = public_features()
            features[name].zero_()
            with self.assertRaises(ValueError):
                self.model(**features, instruction_features=self.text)

    def test_no_old_task_e_or_label_channels_in_forward(self):
        for extra in ('task_e', 'operation_labels', 'speed_labels', 'Native_target_position'):
            with self.assertRaises(TypeError):
                self.model(**public_features(), instruction_features=self.text, **{extra: torch.zeros(2)})

    def test_missing_wrong_batch_and_unnormalized_language_rejected(self):
        for value in (torch.zeros(2, 1024), self.text[:1], self.text.double(),
                      torch.full((2, 1024), float('nan'))):
            with self.assertRaises((ValueError, TypeError)):
                self.model(**public_features(), instruction_features=value)
        for value in ('instruction', [], [''], ['  '], ['x' * 1025], [None]):
            with self.assertRaises(ValueError):
                validate_instructions(value)

    def test_auxiliary_labels_are_offline_and_physical_time_mask_preserved(self):
        output = self.model(**public_features(), instruction_features=self.text)
        args = (output, torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1),
                torch.tensor([.3, 3.]), torch.tensor([True, False]))
        base = candidate_loss(*args)
        only_physics = instruction_candidate_loss(*args, operation_weight=0., speed_weight=0.)
        self.assertTrue(torch.equal(base['per_example'], only_physics['per_example']))
        output['times'].retain_grad()
        total = instruction_candidate_loss(*args, operation_labels=torch.tensor([3, 3]),
                                           speed_labels=torch.tensor([0, 1]))
        total['per_example'].sum().backward()
        self.assertTrue(torch.equal(output['times'].grad[1], torch.zeros(8)))
        self.assertGreater(self.model.language.speed_head.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(total['candidate_success_probability_trained'])
        with self.assertRaises(ValueError):
            instruction_candidate_loss(*args)

    def test_scope_does_not_promote_text_recognition_to_sixteen_controllers(self):
        spec = self.model.spec()
        self.assertTrue(spec['physical_motion_context_c_separate'])
        self.assertTrue(spec['language_conditioned_action_training_required'])
        self.assertFalse(spec['all16_closed_loop_ready'])
        self.assertFalse(spec['production_runtime_compatible'])
        self.assertEqual(len(spec['auxiliary_operation_labels']), 8)


if __name__ == '__main__':
    unittest.main()
