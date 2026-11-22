"""New v4 contracts only: synthetic CPU, no dataset/calibration read or fit."""
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from our_mani.near_goal_depth_refiner_20261103 import model as frozen_model
from our_mani.near_goal_depth_refiner_20261103 import training as frozen_training
from our_mani.near_goal_depth_refiner_20261103_v2 import runtime as old_runtime
from our_mani.near_goal_depth_refiner_20261103_v4 import calibration, evaluation, runtime, training
from our_mani.near_goal_depth_refiner_20261103_v4 import cache


class V4Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_learning_loop_cache_and_loss_are_unchanged(self):
        root = training.ROOT
        self.assertEqual((root/'our_mani/near_goal_depth_refiner_20261103_v2/cache.py').read_bytes(),
                         Path(cache.__file__).read_bytes())
        self.assertIs(training.TinyDepthInfluenceUNet, frozen_model.TinyDepthInfluenceUNet)
        self.assertIs(training._collate, frozen_training._collate)
        self.assertIs(training._loss_batch, frozen_training._loss_batch)
        def epoch_loop(path):
            tree = ast.parse(Path(path).read_text())
            function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train_capture')
            return next(x for x in ast.walk(function) if isinstance(x, ast.For)
                        and isinstance(x.target, ast.Name) and x.target.id == 'epoch')
        old = root/'our_mani/near_goal_depth_refiner_20261103_v2/training.py'
        self.assertEqual(ast.dump(epoch_loop(old)), ast.dump(epoch_loop(training.__file__)))
        self.assertEqual((training.EPOCHS, training.OPTIMIZER_STEPS, training.SEED), (160, 1920, 261103700))

    def test_exact_geometry_map_and_cli_are_bound(self):
        sources = training.implementation_source_map()
        geometry = training.checked_geometry_sources()
        self.assertEqual(len(geometry), 10)
        self.assertTrue(all(sources[p] == digest for p, digest in geometry.items()))
        self.assertIn(str(training.ROOT/'scripts/train_near_goal_depth_refiner_20261103_v4.py'), sources)
        missing = dict(geometry); missing.pop(next(iter(missing)))
        with patch.object(training, 'geometry_source_map', return_value=missing):
            with self.assertRaises(ValueError): training.checked_geometry_sources()
        with patch.object(training, 'GEOMETRY_VERSION', 'foreign_geometry'):
            with self.assertRaises(ValueError): training.checked_geometry_sources()

    def test_oracle_failure_precedes_cache_model_optimizer_and_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'failed'
            with patch.object(training, 'load_full32_dataset', return_value=object()), \
                 patch.object(training, 'source_identity', return_value={'code_sha256': {}}), \
                 patch.object(training, 'oracle_diagnostic', return_value={'ready_for_segmentation_fit': False}), \
                 patch.object(training.CachedRoleDataset, 'from_dataset') as cached, \
                 patch.object(training, 'TinyDepthInfluenceUNet') as model, \
                 patch.object(torch.optim, 'AdamW') as optimizer, \
                 patch.object(torch.cuda, 'manual_seed_all') as cuda:
                with self.assertRaisesRegex(ValueError, 'oracle gate'):
                    training.train_capture('synthetic_not_opened', output, device='cuda:0')
                cached.assert_not_called(); model.assert_not_called()
                optimizer.assert_not_called(); cuda.assert_not_called()
                import json
                self.assertEqual(json.loads((output/'status.json').read_text())['actual_optimizer_steps'], 0)

    def test_v3_and_v4_versions_are_mutually_rejected_before_NN(self):
        with tempfile.TemporaryDirectory() as td:
            for payload_version, reader, model_module in (
                (training.VERSION, old_runtime.CurrentCubeDepthRefinerV2, old_runtime),
                (old_runtime.VERSION, runtime.CurrentCubeDepthRefinerV4, runtime)):
                path = Path(td)/'synthetic_version_only.pt'
                torch.save({'version': payload_version, 'metadata': {}}, path)
                with patch.object(model_module, 'TinyDepthInfluenceUNet') as model:
                    with self.assertRaises(ValueError): reader(path, training.sha256(path))
                    model.assert_not_called()

    def test_runtime_live_geometry_guard_and_unqualified_no_measurement(self):
        refiner = object.__new__(runtime.CurrentCubeDepthRefinerV4)
        refiner.qualified = True
        refiner.model = frozen_model.TinyDepthInfluenceUNet().cpu().eval()
        refiner.calibration_environment = calibration.calibration_environment()
        refiner.metadata = {'geometry_source_sha256': training.checked_geometry_sources()}
        with patch.object(runtime, 'checked_geometry_sources', return_value={}), \
             patch.object(runtime, 'make_near_goal_crops') as crop:
            with self.assertRaisesRegex(ValueError, 'geometry source'):
                refiner.predict({})
            crop.assert_not_called()
        refiner.qualified = False
        refiner.checkpoint_sha256 = '0'*64
        probability = np.full((4, 2, 64, 64), .5, np.float32)
        with patch.object(runtime, 'make_near_goal_crops', return_value={'features': object()}), \
             patch.object(runtime, 'predict_probabilities', return_value=probability), \
             patch.object(runtime, 'probability_to_observation') as geometry:
            result = refiner.predict({})
            self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
            geometry.assert_not_called()

    def test_calibration_fixed20_report_guard_before_threshold_choice(self):
        features = np.zeros((4, 9, 64, 64), np.float32)
        samples = [dict(parent_id=p, scene_id=p+'/'+shot, role='calib', features=features)
                   for p in calibration.PARTITION_IDS[28:] for shot in calibration.SHOT_IDS]
        selected_model = torch.nn.Linear(1, 1).cpu().eval()
        wrong = dict(geometry_version=calibration.GEOMETRY_VERSION,
            geometry_source_sha256=training.checked_geometry_sources(),
            baseline_eligibility_protocol=evaluation.BASELINE_ELIGIBILITY_PROTOCOL,
            positions={'oracle_geometry_eligible': 19})
        with patch.object(calibration, 'predict_probabilities', return_value=np.zeros((4, 2, 64, 64))), \
             patch.object(calibration, 'evaluate_probabilities', return_value=wrong), \
             patch.object(calibration, 'choose_calibrated_threshold') as choose:
            with self.assertRaisesRegex(ValueError, 'fixedold20'):
                calibration.calibrate_cpu_selected_model(selected_model, samples)
            choose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
