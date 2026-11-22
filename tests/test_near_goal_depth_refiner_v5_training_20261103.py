"""Fit-only v5 contracts: temporary metadata, no real data, optimizer or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.near_goal_depth_refiner_20261103_v5 import training as t


class V5TrainingContracts(unittest.TestCase):
    def test_fixed_training_source_map_ignores_future_qualification_module(self):
        with tempfile.TemporaryDirectory() as td:
            package = Path(td)
            for name in t.TRAINING_FILES:
                (package/name).write_bytes((Path(t.__file__).parent/name).read_bytes())
            with patch.object(t, '__file__', str(package/'training.py')):
                before = t.implementation_source_map()
                (package/'runtime.py').write_text('future independent qualification code\n')
                after = t.implementation_source_map()
                self.assertEqual(before, after)
                self.assertNotIn(str(package/'runtime.py'), after)

    def test_calibration_shot_is_rejected_before_underlying_getitem(self):
        class Source:
            records = [dict(role=role, parent_id='p', scene_id='p/'+role)
                       for role in ('fit', 'selection', 'calib')]
            def __init__(self): self.calls = []
            def __len__(self): return 3
            def __getitem__(self, index): self.calls.append(index); return index
        source = Source(); view = t.FitSelectionAccessView(source)
        self.assertEqual((view[0], view[1]), (0, 1))
        with self.assertRaisesRegex(ValueError, 'calibration shots'): view[2]
        self.assertEqual(source.calls, [0, 1])
        self.assertEqual([x['role'] for x in view.access], ['fit', 'selection'])

    def test_bad_prior_gate_precedes_model_optimizer_cache_or_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'must_not_exist'
            with patch.object(t, 'implementation_source_map', return_value={}), \
                 patch.object(t, 'load_full32_dataset', return_value=object()), \
                 patch.object(t, 'validate_prior_oracle', side_effect=ValueError('prior SHA failed')), \
                 patch.object(t.CachedRoleDataset, 'from_dataset') as cache, \
                 patch.object(t, 'TinyDepthInfluenceUNet') as model, \
                 patch.object(t.torch.optim, 'AdamW') as optimizer, \
                 patch.object(t.torch.cuda, 'manual_seed_all') as cuda:
                with self.assertRaises(ValueError):
                    t.train_selected('not_opened', output, oracle_evidence='not_opened',
                        oracle_evidence_sha256='0'*64, device='cuda:0')
                cache.assert_not_called(); model.assert_not_called()
                optimizer.assert_not_called(); cuda.assert_not_called()
                self.assertFalse(output.exists())

    def test_gate_evidence_is_hashed_not_decoded_or_used_as_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest = root/'manifest.json'; manifest.write_text('{}')
            oracle = root/'oracle_before_fit.json'; oracle.write_bytes(b'opaque gate bytes, intentionally not JSON')
            dataset = type('Source', (), {'path': manifest, 'bound_pointers': {}})()
            status = dict(complete=True, training_complete=True, completed_epochs=160,
                actual_optimizer_steps=1920, best_checkpoint_sha256=t.PRIOR_CHECKPOINT_SHA256,
                identity=dict(manifest_sha256=t.sha256(manifest), bound_data_sha256={}, code_sha256={}),
                geometry_source_sha256={})
            status_path = root/'status.json'; status_path.write_text(json.dumps(status))
            with patch.object(t, 'PRIOR_V4', root), \
                 patch.object(t, 'PRIOR_STATUS_SHA256', t.sha256(status_path)), \
                 patch.object(t, 'PRIOR_ORACLE_SHA256', t.sha256(oracle)), \
                 patch.object(t, 'v4_source_map', return_value={}), \
                 patch.object(t, 'checked_geometry_sources', return_value={}):
                evidence = t.validate_prior_oracle(dataset, oracle, t.sha256(oracle))
                self.assertFalse(evidence['old_oracle_or_calibration_JSON_decoded'])
                self.assertFalse(evidence['old_model_weights_loaded'])
                with self.assertRaises(ValueError): t.validate_prior_oracle(dataset, oracle, 'wrong')

    def test_explicit_seed_steps_pending_stage_and_no_default_fit(self):
        config = t.fixed_config()
        self.assertEqual((config['seed'], config['epochs'], config['actual_optimizer_steps_required']),
                         (261103700, 160, 1920))
        self.assertFalse(config['calibration_performed'])
        self.assertEqual(config['unknown_weight'], 1.)
        cli_path = t.ROOT/'scripts/train_near_goal_depth_refiner_20261103_v5.py'
        spec = importlib.util.spec_from_file_location('v5_cli_test', cli_path)
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        with patch.object(cli, 'train_selected') as fit:
            self.assertEqual(cli.main([]), 0)
            fit.assert_not_called()


if __name__ == '__main__': unittest.main()
