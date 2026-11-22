"""Synthetic CPU contracts; no optimizer step, real teacher forward or Native."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from our_mani.multitask.context_se3 import SE3ContextNetwork
from our_mani.context_gap_dropout_20261103_v1.data import Window, dropout_indices, features
from our_mani.context_gap_dropout_20261103_v1.training import loss_components, tensor_digest, PREFIXES, check


def window(n=20):
    times = np.arange(n, dtype=float)*.05
    position = np.c_[times*.2, times*0, times*0+.8]
    return Window(('pick_moving_target_on_the_table', 'synthetic'), 'fit', n-1, times, position,
        np.tile(np.eye(3), (n, 1, 1)), np.tile([.2, 0, 0], (n, 1)), np.zeros((n, 3)),
        np.ones(n), np.zeros(n, bool), np.zeros((17, 3)), np.tile(np.eye(3), (17, 1, 1)), np.zeros(17, bool))


class PreparationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_drop_retains_real_current_and_gap_domain(self):
        for n in (5, 6, 12, 20):
            w = window(n)
            for mode in ('isolated', 'burst'):
                for seed in range(12):
                    keep, trace = dropout_indices(w, np.random.default_rng(seed), mode)
                    self.assertGreaterEqual(len(keep), 5)
                    self.assertEqual(keep[-1], n-1)
                    self.assertLessEqual(n-len(keep), 4)
                    self.assertLessEqual(np.diff(w.times[keep]).max(), .25+1e-8)
                    self.assertTrue(np.all(np.diff(keep) > 0))
                    if mode == 'burst' and trace['removed']:
                        self.assertTrue(np.all(np.diff(trace['removed_indices']) == 1))

    def test_real_delta_feature_and_causal_velocity(self):
        w, model = window(), SE3ContextNetwork()
        keep = np.array([0, 1, 4, 6, 10, 14, 18, 19])
        data = features(w, model, keep, recompute_twist=True)
        np.testing.assert_allclose(data['history'][:len(keep), 19], np.r_[.05, np.diff(w.times[keep])]/.05)
        np.testing.assert_allclose(data['state'][9:12], [.2, 0, 0], atol=1e-6)
        self.assertTrue(np.all(data['history'][len(keep):] == 0))
        w.future_positions[:] = np.nan
        np.testing.assert_array_equal(features(w, model, keep, recompute_twist=True)['history'], data['history'])

    def test_selected_input_must_be_real_ordered_and_current(self):
        w, model = window(), SE3ContextNetwork()
        for keep in ([0, 1, 2, 3], [0, 1, 2, 3, 18], [0, 1, 1, 2, 19], [0, 1, 2, 3, 19]):
            with self.assertRaises(ValueError):
                features(w, model, np.array(keep), recompute_twist=True)

    def test_only_context_parameters_receive_gradients_teacher_frozen(self):
        torch.manual_seed(1018)
        teacher = SE3ContextNetwork().eval().requires_grad_(False)
        student = deepcopy(teacher)
        for name, p in student.named_parameters():
            p.requires_grad_(name.startswith(PREFIXES))
        w = window()
        dense = features(w, teacher)
        keep, _ = dropout_indices(w, np.random.default_rng(7), 'burst')
        dropped = features(w, teacher, keep, recompute_twist=True)
        def batch(x):
            return torch.from_numpy(x['history'][None]), torch.tensor([x['length']]), torch.from_numpy(x['state'][None])
        before = tensor_digest(student.state_dict())
        result = loss_components(teacher, student, batch(dense), batch(dropped), 'cpu')
        self.assertEqual(float(result['dense_replay_context']), 0.)
        self.assertTrue(torch.isfinite(result['total']).all())
        result['total'].mean().backward()
        self.assertTrue(any(p.grad is not None for name, p in student.named_parameters() if name.startswith(PREFIXES)))
        self.assertTrue(all(p.grad is None for name, p in student.named_parameters() if not name.startswith(PREFIXES)))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(before, tensor_digest(student.state_dict()))

    def test_default_check_constructs_no_optimizer_or_forward(self):
        teacher = SE3ContextNetwork().eval().requires_grad_(False)
        student = deepcopy(teacher)
        fixture = (dict(rows=[dict(role='fit'), dict(role='selection')]), [window()], [], teacher, student, teacher.spec, {})
        with patch('our_mani.context_gap_dropout_20261103_v1.training.prepare', return_value=fixture), \
             patch('our_mani.context_gap_dropout_20261103_v1.training.verify_identity'), \
             patch.object(torch.optim, 'AdamW', side_effect=AssertionError('must not construct')), \
             patch.object(SE3ContextNetwork, 'forward', side_effect=AssertionError('must not forward')), \
             patch.object(SE3ContextNetwork, 'encode_context', side_effect=AssertionError('must not encode')):
            report = check('unused')
        self.assertFalse(report['GPU_used'])
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertEqual(report['warmup_or_model_forward_calls'], 0)


if __name__ == '__main__':
    unittest.main()
