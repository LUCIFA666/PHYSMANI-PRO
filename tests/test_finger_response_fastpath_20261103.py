"""CPU computation contracts; synthetic fixtures are not task evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from our_mani.local_mpc_p2_20261103.kinematics import SCHEMA
from our_mani.local_mpc_p2_20261103_v2.servo import JointServoResidual
from our_mani.local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition
from our_mani.local_mpc_control_20261103.finger_response_fastpath_20261103 import FrozenFingerOnlyResponse


def original():
    pose = np.eye(4); pose[:3, 3] = [.4, .1, .9]
    snapshot = dict(schema=SCHEMA, twist_order='v_omega', q_ref=[0.] * 7,
                    space_screws_vw=np.tile([0., 0., 0., 0., 0., 1.], (7, 1)).tolist(),
                    tcp_reference_matrix=pose.tolist())
    model = JointServoResidual(snapshot, residual_q_bound=[.02] * 7, residual_dq_bound=[.3] * 7)
    # Exercise all learned outputs instead of the zero initialization alone.
    with torch.no_grad():
        model.network[-1].weight.normal_(0., .03)
        model.network[-1].bias.normal_(0., .05)
    return TorchServoTransition(model)


def inputs(n=19):
    rng = np.random.default_rng(20261103)
    x = dict(q=rng.uniform(-1., 1., (n, 7)), dq=rng.uniform(-.5, .5, (n, 7)),
             finger_open_amounts=rng.uniform(0., 1., (n, 2)))
    refs = dict(q_ref=x['q'] + rng.uniform(-.1, .1, (n, 7)),
                dq_ref=rng.uniform(-1., 1., (n, 7)), ddq_ref=rng.uniform(-8., 8., (n, 7)),
                gripper_target=rng.integers(0, 2, n).astype(float))
    return x, refs, np.arange(n) % 2 == 0


class FingerFastpathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(121)
        self.old = original(); self.fast = FrozenFingerOnlyResponse(self.old)
        self.x, self.refs, self.anchor = inputs()

    def compare(self, x=None, refs=None, anchor=None, latent=None):
        x = self.x if x is None else x; refs = self.refs if refs is None else refs
        anchor = self.anchor if anchor is None else anchor
        old = self.old.predict_execution(x, refs, latent, .05, held_anchor=anchor)
        new = self.fast.predict_execution(x, refs, latent, .05, held_anchor=anchor)
        for key in ('finger_open_amounts', 'feasible'):
            np.testing.assert_array_equal(old[key], new[key])
        return old, new

    def test_exact_fingers_and_shared_unchanged_parameter_identity(self):
        before = {key: value.clone() for key, value in self.old.model.state_dict().items()}
        old, new = self.compare()
        self.assertIs(self.fast.model, self.old.model)
        self.assertIs(self.fast.original, self.old)
        self.assertNotIn('q', new); self.assertNotIn('dq', new)
        for key, value in self.old.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        self.assertFalse(np.array_equal(old['finger_open_amounts'][:, 0], old['finger_open_amounts'][:, 1]))

    def test_fk_is_not_called_in_fastpath(self):
        with patch.object(self.old.model.fk, 'forward', side_effect=AssertionError('unused FK')):
            self.fast.predict_execution(self.x, self.refs, None, .05, held_anchor=self.anchor)
            with self.assertRaisesRegex(AssertionError, 'unused FK'):
                self.old.predict_execution(self.x, self.refs, None, .05, held_anchor=self.anchor)

    def test_two_finger_endpoints_targets_and_held_are_preserved(self):
        x, refs, anchor = inputs(4)
        x['finger_open_amounts'] = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        self.compare(x, refs, anchor)

    def test_latent_and_inputs_remain_independent_and_unmodified(self):
        latent = {'a': np.array([1., 2.]), 'history': [{'b': 3}]}
        x0, r0, l0 = deepcopy(self.x), deepcopy(self.refs), deepcopy(latent)
        old, new = self.compare(latent=latent)
        np.testing.assert_array_equal(old['latent']['a'], new['latent']['a'])
        new['latent']['a'][0] = 99.; new['latent']['history'][0]['b'] = 99
        np.testing.assert_array_equal(latent['a'], l0['a'])
        self.assertEqual(latent['history'], l0['history'])
        for key in self.x: np.testing.assert_array_equal(self.x[key], x0[key])
        for key in self.refs: np.testing.assert_array_equal(self.refs[key], r0[key])

    def test_nan_unused_joint_output_is_rejected_by_both(self):
        for index in (0, 7):
            with self.subTest(index=index):
                with torch.no_grad(): self.old.model.network[-1].bias[index] = float('nan')
                for adapter in (self.old, self.fast):
                    with self.assertRaises(ValueError):
                        adapter.predict_execution(self.x, self.refs, None, .05, held_anchor=self.anchor)
                with torch.no_grad(): self.old.model.network[-1].bias[index] = 0.

    def test_infinite_raw_joint_residual_saturates_and_is_not_over_rejected(self):
        with torch.no_grad():
            self.old.model.network[-1].bias[0] = float('inf')
            self.old.model.network[-1].bias[7] = float('-inf')
        self.compare()

    def test_invalid_inputs_have_matching_rejection(self):
        invalid = []
        for key in ('q', 'dq', 'finger_open_amounts'):
            x = deepcopy(self.x); x[key].flat[0] = np.nan
            invalid.append((x, self.refs, self.anchor, .05))
        x = deepcopy(self.x); x['finger_open_amounts'][0, 0] = 1.001
        invalid.append((x, self.refs, self.anchor, .05))
        refs = deepcopy(self.refs); refs['gripper_target'][0] = .5
        invalid.append((self.x, refs, self.anchor, .05))
        invalid.append((self.x, self.refs, self.anchor.astype(float), .05))
        invalid.append((self.x, self.refs, self.anchor, .1))
        refs = deepcopy(self.refs); refs['q_ref'][0, 0] = np.finfo(np.float64).max
        invalid.append((self.x, refs, self.anchor, .05))
        for index, (x, refs, anchor, dt) in enumerate(invalid):
            for adapter in (self.old, self.fast):
                with self.subTest(index=index, adapter=type(adapter).__name__):
                    with self.assertRaises(ValueError):
                        adapter.predict_execution(x, refs, None, dt, held_anchor=anchor)

    def test_exact_type_and_eval_contract(self):
        class Other(TorchServoTransition): pass
        with self.assertRaises(TypeError): FrozenFingerOnlyResponse(Other(self.old.model))
        self.old.model.train()
        with self.assertRaises(ValueError): FrozenFingerOnlyResponse(self.old)

    def test_parent_and_fk_forward_hooks_are_explicitly_unsupported(self):
        for module in (self.old.model, self.old.model.fk):
            handle = module.register_forward_hook(lambda *args: None)
            try:
                with self.assertRaisesRegex(ValueError, 'hook-free'):
                    FrozenFingerOnlyResponse(self.old)
            finally:
                handle.remove()


if __name__ == '__main__': unittest.main()
