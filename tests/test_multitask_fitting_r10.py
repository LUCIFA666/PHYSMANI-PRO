"""Check parent weighting, unobservable labels and learned causal context."""
import copy
import unittest

import numpy as np
import torch

from our_mani.multitask.context_se3 import SE3ContextNetwork, history_inputs
from our_mani.multitask.fitting import batch_tensors, balanced_parent_indices, loss_per_window


def accelerating_example(acceleration, model, source):
    t = np.arange(-4, 1) * .05
    rows = [dict(timestamp=float(k + .2), position=np.array([.5 * acceleration * k * k, 0., .8]),
                 rotation=np.eye(3), velocity=np.array([acceleration * k, 0., 0.]),
                 angular_velocity=np.zeros(3), confidence=1., mode=0) for k in t]
    history, length, state, _ = history_inputs(rows, model)
    future = np.arange(7) * .05
    p = np.c_[.5 * acceleration * future ** 2, np.zeros(7), np.full(7, .8)]
    mask = np.ones(7, dtype=bool); mask[0] = False
    return dict(history=history[0].numpy(), length=int(length[0]), state=state[0].numpy(),
        targets=dict(positions=p.astype('f'), rotations=np.broadcast_to(np.eye(3), (7, 3, 3)).copy().astype('f'),
            linear_velocities=np.c_[acceleration * future, np.zeros((7, 2))].astype('f'),
            angular_velocities=np.zeros((7, 3), dtype='f')),
        mask=mask, rotation_mask=mask.copy(), rotation_observable=True,
        source_key=('train', 'synthetic', source))


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_replica_count_does_not_change_parent_optimization_weight(self):
        samples = [dict(source_key=('train', 'task', 'parent_a')) for _ in range(100)]
        samples += [dict(source_key=('train', 'task', 'parent_b'))]
        ii = balanced_parent_indices(samples, 7, np.random.default_rng(3))
        self.assertEqual(sum(samples[i]['source_key'][-1] == 'parent_a' for i in ii), 7)
        self.assertEqual(sum(samples[i]['source_key'][-1] == 'parent_b' for i in ii), 7)
        with self.assertRaises(ValueError):
            balanced_parent_indices([dict(source_key=('validation', 'task', 'source'))], 1, np.random.default_rng(3))

    def test_unobservable_and_invalid_targets_cannot_poison_gradients(self):
        model = SE3ContextNetwork()
        original = accelerating_example(1., model, 'a')
        original['rotation_mask'][:] = False
        original['rotation_observable'] = False
        original['mask'][-2:] = False
        changed = copy.deepcopy(original)
        changed['targets']['rotations'][:] = np.nan
        changed['targets']['angular_velocities'][:] = np.nan
        changed['targets']['positions'][-2:] = np.nan
        a, b = [batch_tensors([x], [0], 'cpu') for x in (original, changed)]
        torch.testing.assert_close(loss_per_window(model, a, 6), loss_per_window(model, b, 6))
        loss_per_window(model, b, 6).mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_same_current_state_can_learn_opposite_dynamics_from_past(self):
        torch.manual_seed(321)
        model = SE3ContextNetwork(hidden=32, latent_dim=32, context_dim=16)
        samples = [accelerating_example(a, model, str(a)) for a in (-3., 3.)]
        data = batch_tensors(samples, [0, 1], 'cpu')
        torch.testing.assert_close(data['state'][0], data['state'][1])
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        before = float(loss_per_window(model, data, 6).mean().detach())
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_per_window(model, data, 6).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        after = float(loss_per_window(model, data, 6).mean().detach())
        prediction = model(data['history'], data['length'], data['state'], steps=6)
        self.assertLess(after, before * .5)
        self.assertLess(float(prediction['positions'][0, -1, 0].detach()), -.03)
        self.assertGreater(float(prediction['positions'][1, -1, 0].detach()), .03)


if __name__ == '__main__':
    unittest.main()
