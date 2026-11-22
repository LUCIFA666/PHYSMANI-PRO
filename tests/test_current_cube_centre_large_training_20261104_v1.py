"""Pure loss/source/metadata tests. No NN model, GPU, Native or actual dataset."""
from contextlib import nullcontext
from pathlib import Path
import ast
import tempfile
import json
import unittest
from unittest.mock import patch
import torch

from our_mani.current_cube_rgbd_centre_large_20261104_v1 import training as t
from our_mani.current_cube_rgbd_centre_large_20261104_v1 import evaluation as e
from our_mani.current_cube_rgbd_centre_large_20261104_v1.config import fixed_config


class PureContracts(unittest.TestCase):
    def synthetic(self, unknown=False):
        torch.manual_seed(12)
        centre = torch.randn(16, 3).requires_grad_()
        logit = torch.randn(16).requires_grad_()
        score = torch.full((16, 3), .04, requires_grad=True)
        presence = torch.tensor([-1, -1, 1, -1, 0, 0, 1, 1, -1, 1, 0, -1, 1, 0, 1, 0])
        if unknown: presence.fill_(-1)
        valid = presence == 1
        target = torch.zeros(16, 3)
        target[~valid] = torch.nan  # Invalid labels cannot leak through either loss.
        return centre, logit, score, target, valid, presence

    def gradients(self, micro, unknown=False):
        c, logit, score, target, valid, presence = self.synthetic(unknown)
        nc, nk = int(valid.sum()), int((presence >= 0).sum())
        total = 0.
        for start in range(0, 16, micro):
            sl = slice(start, start + micro)
            loss = t.loss_from_outputs(dict(centre_world=c[sl], measurement_availability_logit=logit[sl],
                uncertainty_score_m=score[sl]), dict(centre_target_world=target[sl],
                    centre_valid=valid[sl], presence_target=presence[sl]))
            weighted = t.weighted_micro_loss(loss, nc, nk)
            weighted.backward(); total += float(weighted.detach())
        return total, [v.grad for v in (c, logit, score)]

    def test_micro2x8_equals_whole16_with_unequal_valid_counts(self):
        whole, whole_grads = self.gradients(16)
        accumulated, grads = self.gradients(2)
        self.assertAlmostEqual(whole, accumulated, places=4)
        for a, b in zip(whole_grads, grads):
            self.assertTrue(torch.allclose(a, b, atol=2e-5, rtol=2e-6))

    def test_all_unknown_finite_zero_loss_and_gradient(self):
        loss, grads = self.gradients(2, True)
        self.assertEqual(loss, 0.)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(int(torch.count_nonzero(grad)), 0)

    def test_fixed_budget_and_default_no_model(self):
        c = fixed_config()
        self.assertEqual((c['epochs'], c['planned_optimizer_steps'], c['updates_per_epoch']), (40, 520, 13))
        self.assertEqual(c['micro_batch_size'] * c['gradient_accumulation'], c['logical_batch_size'])
        self.assertEqual((c['head_only_epochs'], c['encoder_and_head_epochs']), (5, 35))
        with patch.object(t, '_seed_and_model', side_effect=AssertionError('model must not be constructed')):
            value = t.check()
        self.assertFalse(value['model_constructed'])
        self.assertFalse(value['optimizer_constructed'])
        self.assertEqual(value['model_forward_calls'], 0)

    def test_incomplete40_preflight_refuses_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'preflight'
            with patch.object(t, 'exclusive_resources', return_value=nullcontext({})), \
                    patch.object(t, '_datasets', side_effect=ValueError('complete40 required')), \
                    patch.object(t, '_seed_and_model', side_effect=AssertionError('no model')):
                with self.assertRaisesRegex(ValueError, 'complete40'):
                    t.gpu_preflight('no manifest', 'no initializer', 'no SHA', output)
            self.assertFalse(output.exists())

    def test_freeze_validation_precedes_any_target_dataset(self):
        with patch.object(e, 'validate_freeze', side_effect=ValueError('bad freeze')), \
                patch.object(e, 'DualCropCubeDataset', side_effect=AssertionError('must not read labels')):
            with self.assertRaisesRegex(ValueError, 'bad freeze'):
                e.score('missing prediction directory')

    def test_passed_flag_cannot_bypass_memory_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / 'manifest'; encoder = root / 'encoder'
            manifest.write_bytes(b'synthetic manifest'); encoder.write_bytes(b'synthetic initializer')
            config = fixed_config()
            report = dict(version=t.VERSION, passed=True, actual_optimizer_steps=1, model_saved=False,
                updated_weights_discarded=True, encoder_trainable=True, config=config,
                manifest=t.pointer(manifest), initializer=t.pointer(encoder), source_sha256={},
                peak_reserved_bytes=90, total_memory_bytes=100, free_after_bytes=2**32,
                resources=dict(dual_Native_locks_held=True, training_lock_held=True),
                environment=dict(device='cuda:0', autocast='bfloat16', loss='float32'))
            path = root / 'bad_preflight.json'; path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'memory gates'):
                t._check_preflight(path, t.sha256(path), manifest, encoder, config, {})

    def test_new_version_and_same_reserved_global_claim(self):
        self.assertNotEqual(t.VERSION, t.legacy.VERSION)
        source = Path(e.__file__).read_text()
        self.assertIn("data.path.parent / 'reserved_centre_evaluation_consumption.request.json'", source)
        self.assertIn('role == \'reserved_fresh_centre_VAL\'', source)
        self.assertEqual(e.ROLES['selection'], (16, 208))
        self.assertEqual(e.ROLES['reserved_fresh_centre_VAL'], (8, 104))

    def test_one_step_after_all_microbatches_and_single_discarded_preflight_step(self):
        tree = ast.parse(Path(t.__file__).read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        micro = ast.unparse(funcs['logical_batch_backward'])
        self.assertNotIn('optimizer.step', micro)
        self.assertNotIn('/ 8', micro)
        preflight = ast.unparse(funcs['gpu_preflight'])
        self.assertEqual(preflight.count('optimizer.step()'), 1)
        fit = ast.unparse(funcs['fit'])
        self.assertEqual(fit.count('optimizer.step()'), 1)  # one call site within13logical batches
        self.assertIn('logical_batch_backward', fit)
        self.assertIn('clip_grad_norm_', fit)


if __name__ == '__main__': unittest.main()
