"""Schedule160 guard tests, no optimizer steps/Native/real capture reads."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from our_mani.near_goal_depth_refiner_20261103 import data as v1data
from our_mani.near_goal_depth_refiner_20261103 import model as v1model
from our_mani.near_goal_depth_refiner_20261103 import training as v1training
from our_mani.near_goal_depth_refiner_20261103_v2 import training as training
from our_mani.near_goal_depth_refiner_20261103_v2 import runtime
from our_mani.near_goal_depth_refiner_20261103_v2.cache import CachedRoleDataset, canonical_sha
from our_mani.near_goal_depth_refiner_20261103_v2.calibration import calibration_environment
from tests.test_near_goal_depth_refiner_v2_independent_20261103 import SyntheticFullSource
from tests.test_near_goal_depth_refiner_20261103 import public


def header():
    return dict(scope='full32', dataset_split='TRAIN', TEST_read=False, training_complete=True,
        epochs_completed=160, actual_optimizer_steps=1920, selected_epoch=1, model_qualified=False,
        schedule_protocol_sha256=training.PROTOCOL_SHA256, label_protocol=v1data.LABEL_PROTOCOL,
        quality_protocol=training.QUALITY_PROTOCOL, completion=False, current_depth_influence_only=True,
        complete_visible_instance_claimed=False, geometry_erosions=1, calibration_cached=False,
        TCP_GRU_GT_input=False, absolute_worldXYZ_input=False,
        feature_names=list(v1data.FEATURE_NAMES), class_names=list(v1data.CLASS_NAMES),
        model_config=dict(input_channels=9, classes=2, base_channels=8),
        calibration_parent_ids=list(v1data.PARTITION_IDS[28:]),
        calibration_device='cpu', calibration_num_threads=2,
        calibration_addendum_sha256=training.CPU_CALIBRATION_ADDENDUM_SHA256,
        identity=dict(config=training.fixed_config(), code_sha256={}))


class V2TrainingGuardTests(unittest.TestCase):
    def test_exact_frozen_schedule_and_sources_no_model_needed(self):
        self.assertEqual(training.EPOCHS, 160)
        self.assertEqual(training.OPTIMIZER_STEPS, 160*(192//16))
        self.assertEqual(training.fixed_config()['threshold_grid'], [.5, .7, .9, .95, .99])
        sources = training.implementation_source_map()
        self.assertEqual(sources[str(training.PROTOCOL)], training.PROTOCOL_SHA256)
        self.assertTrue(any(name.endswith('/runtime.py') for name in sources))
        self.assertEqual(training._collate, v1training._collate)
        self.assertEqual(training._loss_batch, v1training._loss_batch)

    def test_pilot_rejected_before_oracle_cache_model_or_optimizer(self):
        pilot = SimpleNamespace(scope='pilot_smoke', __len__=lambda self: 64)
        class Pilot:
            scope = 'pilot_smoke'
            def __len__(self): return 64
        with patch.object(training, 'DepthInfluenceDataset', return_value=Pilot()), \
                patch.object(training, 'TinyDepthInfluenceUNet', side_effect=AssertionError('no model')), \
                patch('torch.optim.AdamW', side_effect=AssertionError('no optimizer')):
            with self.assertRaisesRegex(ValueError, 'full32 only'):
                training.load_full32_dataset('not_read')

    def test_oracle_failure_writes_zero_steps_before_cache_or_optimizer(self):
        with tempfile.TemporaryDirectory(prefix='synthetic160_oracle_') as td:
            out = Path(td)/'unqualified'
            with patch.object(training, 'load_full32_dataset', return_value=object()), \
                    patch.object(training, 'source_identity', return_value=dict(code_sha256={})), \
                    patch.object(training, 'oracle_diagnostic', return_value=dict(ready_for_segmentation_fit=False)), \
                    patch.object(training.CachedRoleDataset, 'from_dataset', side_effect=AssertionError('no cachefit')), \
                    patch.object(training, 'TinyDepthInfluenceUNet', side_effect=AssertionError('no model')), \
                    patch('torch.optim.AdamW', side_effect=AssertionError('no optimizer')):
                with self.assertRaisesRegex(ValueError, 'oracle gate failed'):
                    training.train_capture('not_read', out)
            status = json.loads((out/'status.json').read_text())
            self.assertFalse(status['training_started'])
            self.assertEqual(status['actual_optimizer_steps'], 0)
            self.assertFalse((out/'best.pt').exists())

    def test_source_change_fails_end_verification_without_any_optimizer(self):
        with tempfile.TemporaryDirectory(prefix='synthetic160_source_') as td:
            source = SyntheticFullSource(td)
            source.capture_source_identity = dict(synthetic=True)
            identity = training.source_identity(source)
            training._verify_identity(source, identity)
            source.raw.write_text('changed original input after caching')
            with self.assertRaises(ValueError):
                training._verify_identity(source, identity)

    def test_runtime_versions_and_current_implementation_map_failclosed(self):
        with tempfile.TemporaryDirectory(prefix='synthetic160_version_') as td:
            path = Path(td)/'not_a_trained_model.pt'
            torch.save(dict(version=training.VERSION, metadata=header()), path)
            digest = v1data.sha256(path)
            with patch.object(v1model, 'TinyDepthInfluenceUNet', side_effect=AssertionError('no old model')):
                with self.assertRaises(ValueError):
                    v1model.CurrentCubeDepthRefiner(path, digest)
            with patch.object(runtime, 'TinyDepthInfluenceUNet', side_effect=AssertionError('no new model')):
                with self.assertRaisesRegex(ValueError, 'implementation source SHA map'):
                    runtime.CurrentCubeDepthRefinerV2(path, digest, allow_unqualified_diagnostic=True)
            torch.save(dict(version=v1training.VERSION, metadata={}), path)
            with self.assertRaisesRegex(ValueError, 'schedule160/1920'):
                runtime.CurrentCubeDepthRefinerV2(path, v1data.sha256(path))

    def test_default_CLI_never_constructs_model_or_optimizer(self):
        path = Path(__file__).resolve().parents[1]/'scripts/train_near_goal_depth_refiner_20261103_v2.py'
        spec = importlib.util.spec_from_file_location('schedule160_cli_unit', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class EmptyCheck:
            def __iter__(self): return iter([])
            def revalidate(self): pass
        with patch.object(cli, 'load_full32_dataset', return_value=EmptyCheck()), \
                patch.object(cli, 'train_capture', side_effect=AssertionError('explicit fit only')), \
                patch('torch.optim.AdamW', side_effect=AssertionError('no optimizer in check')):
            self.assertEqual(cli.main(['--manifest', 'not_opened']), 0)

    def test_synthetic_completed_unqualified_schema_roundtrip_never_measures(self):
        # This is deliberately unqualified random-state test data, not evidence
        # of actual160epoch training. Everything is temporary and deleted.
        with tempfile.TemporaryDirectory(prefix='synthetic160_unqualified_') as td:
            base = Path(td)
            source = SyntheticFullSource(td)
            source.capture_source_identity = dict(synthetic_fixture=True)
            identity = training.source_identity(source)
            fit = CachedRoleDataset.from_dataset(source, 'fit')
            selection = CachedRoleDataset.from_dataset(source, 'selection')
            cache = training._cache_report(fit, selection)
            def ledger(parents, roles):
                return [dict(parent_id=p, scene_id=p+'/'+sid, role=role)
                        for p, role in zip(parents, roles) for sid in v1data.SHOT_IDS]
            cal_records = ledger(v1data.PARTITION_IDS[28:], ['calib']*4)
            reports = []
            for threshold in training.THRESHOLD_GRID:
                reports.append(dict(threshold=threshold,
                    pixels=dict(predicted_positive_count=0, conservative_precision_lowerbound=None,
                        known_precision=None, U=0),
                    positions=dict(oracle_geometry_eligible=1, oracle_eligible_coverage=0.,
                        qualified_visible_centre_error=dict(p95_m=None), owner_unobserved_falseavailable=0,
                        negatives={k: dict(samples=4, falseavailable=0) for k in ('sphere_only', 'cube_far')}),
                    records=cal_records, per_parent_pixels={p: {} for p in v1data.PARTITION_IDS[28:]}))
            history = [dict(epoch=i, actual_optimizer_steps=i*12, fit_loss=2.-i/1000,
                            selection_loss=2.-i/1000) for i in range(1, 161)]
            execution = dict(calibration_device='cpu', calibration_num_threads=2,
                model_eval=True, model_dtype='float32', forward_calls=32, samples=32,
                parent_count=4, threshold_grid=list(training.THRESHOLD_GRID),
                GPU_CPU_selection_performed=False, **calibration_environment())
            evidence = dict(calibration=dict(selection=training.choose_calibrated_threshold(reports), reports=reports,
                    calibration_device='cpu', calibration_num_threads=2, execution=execution),
                oracle_before_fit=dict(ready_for_segmentation_fit=True,
                    records=ledger(v1data.PARTITION_IDS, ['fit']*24+['selection']*4+['calib']*4)),
                cache_before=cache, cache_after=cache,
                source_verification=dict(complete=True, raw_data_and_sources_verified_before_after=True,
                    cache_content_unchanged=True, calibration_cached=False, actual_optimizer_steps=1920,
                    calibration_device='cpu', calibration_num_threads=2, calibration_environment=execution,
                    original_identity_sha256=canonical_sha(identity), source_bindings_sha256=canonical_sha(source.bound_pointers)),
                history=history)
            pointers = {}
            for name, value in evidence.items():
                training.write_new(base/(name+'.json'), value)
                pointers[name] = dict(path=name+'.json', sha256=v1data.sha256(base/(name+'.json')))
            meta = header()
            meta.update(identity=identity, selected_epoch=160, selected_threshold=None, evidence=pointers,
                        calibration_environment=execution)
            path = base/'synthetic_unqualified_random_state.pt'
            torch.save(dict(version=training.VERSION, metadata=meta,
                            model_state_dict=v1model.TinyDepthInfluenceUNet().state_dict()), path)
            digest = v1data.sha256(path)
            training.write_new(base/'status.json', dict(version=training.VERSION, complete=True, training_started=True,
                training_complete=True, completed_epochs=160, actual_optimizer_steps=1920,
                calibration_device='cpu', calibration_num_threads=2,
                calibration_addendum_sha256=training.CPU_CALIBRATION_ADDENDUM_SHA256,
                selected_epoch=160, scope='full32', identity=identity, model_qualified=False, best_checkpoint_sha256=digest))
            with self.assertRaisesRegex(ValueError, 'Unqualified'):
                runtime.CurrentCubeDepthRefinerV2(path, digest)
            diagnostic = runtime.CurrentCubeDepthRefinerV2(path, digest, allow_unqualified_diagnostic=True)
            result = diagnostic.predict(public())
            self.assertFalse(result['refiner_available'])
            self.assertIsNone(result['centre'])
            self.assertFalse((result.get('measurement') or {}).get('refiner_available', False))


if __name__ == '__main__':
    unittest.main()
