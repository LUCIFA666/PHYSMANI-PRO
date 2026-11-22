"""Independent synthetic CPU tests; never fit or use real capture/TEST data."""
from pathlib import Path
from contextlib import redirect_stdout
import hashlib
from copy import deepcopy
import importlib.util
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from our_mani.near_goal_depth_refiner_20261103 import data as frozen_data
from our_mani.near_goal_depth_refiner_20261103 import training as frozen_training
from our_mani.near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from our_mani.near_goal_depth_refiner_20261103_v2.cache import CachedRoleDataset
from our_mani.near_goal_depth_refiner_20261103_v2 import training as schedule160
from our_mani.near_goal_depth_refiner_20261103_v2 import runtime as runtime160
from tests.test_near_goal_depth_refiner_independent_20261103 import public

torch.set_num_threads(2)


class SyntheticFullSource:
    """Only public feature arrays/targets; extra privileged fields are poison."""
    scope = 'full32'

    def __init__(self, directory):
        self.path = Path(directory)/'manifest.json'; self.path.write_text('synthetic source only')
        self.raw = Path(directory)/'raw_source'; self.raw.write_text('fixed synthetic source')
        self.bound_pointers = {str(x): frozen_data.sha256(x) for x in (self.path, self.raw)}
        self.records = []
        for i, parent in enumerate(frozen_data.PARTITION_IDS):
            role = 'fit' if i < 24 else ('selection' if i < 28 else 'calib')
            self.records.extend(dict(parent_id=parent, scene_id=parent+'/'+shot, role=role)
                for shot in frozen_data.SHOT_IDS)
        self.accesses = []; self.mutate_at = None
        self.base = np.zeros((4, 9, 64, 64), np.float32)
        self.semantic = np.zeros((4, 64, 64), np.uint8)
        self.semantic[:, 24:40, 24:40] = 1
        self.semantic[:, :3] = 255
        self.semantic[0] = 255  # retain an entirely UNKNOWN view in every shot

    def __len__(self): return 256

    def __getitem__(self, index):
        self.accesses.append(index)
        if self.mutate_at == index: self.raw.write_text('source changed during feature extraction')
        features = self.base.copy(); features[0, 0, 0, 0] = index/256
        return dict(self.records[index], features=features, semantic=self.semantic.copy(),
            cube_center_world=np.full(3, np.nan), Native_held=object(), TCP_future=object())

    def revalidate(self):
        if any(frozen_data.sha256(path) != digest for path, digest in self.bound_pointers.items()):
            raise ValueError('synthetic original source hash changed')


class IndependentCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='synthetic_cache_v2_')
        cls.source = SyntheticFullSource(cls.temp.name)
        cls.fit = CachedRoleDataset.from_dataset(cls.source, 'fit')
        cls.selection = CachedRoleDataset.from_dataset(cls.source, 'selection')

    @classmethod
    def tearDownClass(cls):
        del cls.fit, cls.selection, cls.source
        cls.temp.cleanup()

    def test_only_exact_features_and_targets_cached_calibration_never_read(self):
        self.assertEqual(len(self.fit), 192); self.assertEqual(len(self.selection), 32)
        self.assertEqual(self.source.accesses, list(range(224)))
        self.assertEqual(set(self.fit[0]), {'features', 'semantic'})
        np.testing.assert_array_equal(self.fit[10]['features'], self.source[10]['features'])
        np.testing.assert_array_equal(self.fit[10]['semantic'], self.source[10]['semantic'])
        a = self.fit.attestation()
        self.assertEqual(a['features']['shape'], [192, 4, 9, 64, 64])
        self.assertEqual(a['semantic']['shape'], [192, 4, 64, 64])
        self.assertFalse(a['calibration_cached'])
        self.assertFalse(a['privileged_metadata_or_robot_or_CF_cached_as_features'])
        self.assertTrue((self.fit[0]['semantic'] == 255).any())
        self.assertTrue(np.all(self.fit[0]['semantic'][0] == 255))
        with self.assertRaisesRegex(ValueError, 'Calibration'):
            CachedRoleDataset.from_dataset(self.source, 'calib')

    def test_wrong_scope_or_record_order_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_invalid_cache_') as directory:
            source = SyntheticFullSource(directory); source.scope = 'pilot_smoke'
            with self.assertRaises(ValueError): CachedRoleDataset.from_dataset(source, 'fit')
            self.assertFalse(source.accesses)
            source.scope = 'full32'; source.records[0], source.records[1] = source.records[1], source.records[0]
            with self.assertRaises(ValueError): CachedRoleDataset.from_dataset(source, 'fit')
            self.assertFalse(source.accesses)

    def test_cache_readonly_and_copy_attestation_cannot_hide_content_mutation(self):
        with self.assertRaises(ValueError): self.fit[0]['features'][0, 0, 0, 0] = 9.
        clone = self.fit.attestation(); clone['features']['shape'][0] = 1
        self.fit.revalidate()
        value = self.fit._features[0, 0, 0, 0, 0].copy()
        try:
            self.fit._features.setflags(write=True); self.fit._features[0, 0, 0, 0, 0] = 9.
            with self.assertRaisesRegex(ValueError, 'Cached role'):
                self.fit.revalidate()
        finally:
            self.fit._features[0, 0, 0, 0, 0] = value
            self.fit._features.setflags(write=False)
        self.fit.revalidate()

    def test_original_source_change_during_cache_prevents_a_completed_cache(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_mutation_cache_') as directory:
            source = SyntheticFullSource(directory); source.mutate_at = 191
            with self.assertRaisesRegex(ValueError, 'original source hash'):
                CachedRoleDataset.from_dataset(source, 'fit')

    def test_runtime_cache_attestation_keeps_exact_order_roles_and_source_binding(self):
        attestation = schedule160._cache_report(self.fit, self.selection)
        identity = dict(manifest_sha256=frozen_data.sha256(self.source.path),
                        bound_data_sha256=deepcopy(self.source.bound_pointers))
        runtime160._cache_evidence(attestation, deepcopy(attestation), identity)
        changed = deepcopy(attestation); changed['calibration_cached'] = True
        with self.assertRaises(ValueError): runtime160._cache_evidence(changed, changed, identity)
        other_source = deepcopy(identity); other_source['manifest_sha256'] = '0'*64
        with self.assertRaises(ValueError): runtime160._cache_evidence(attestation, attestation, other_source)

    def test_same_sample_order_seeded_shuffle_batch_values_loss_and_gradients(self):
        source = self.source
        class RawFit:
            def __len__(self): return 192
            def __getitem__(self, index): return source[index]
        loaders = [DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(261103700), collate_fn=frozen_training._collate)
            for dataset in (RawFit(), self.fit)]
        for _ in range(2):
            # Fully exhaust each sampler. zip would leave the second sampler's
            # final RNG bookkeeping unconsumed when the first one terminates.
            batches = [[(hashlib.sha256(b['features'].numpy().tobytes()).hexdigest(),
                         hashlib.sha256(b['labels'].numpy().tobytes()).hexdigest())
                        for b in loader] for loader in loaders]
            self.assertEqual(batches[0], batches[1])
        raw = frozen_training._collate([source[1], source[2]])
        cached = frozen_training._collate([self.fit[1], self.fit[2]])
        torch.manual_seed(17); network = TinyDepthInfluenceUNet().eval()
        loss_raw = frozen_training._loss_batch(network, raw, 'cpu')
        loss_cache = frozen_training._loss_batch(network, cached, 'cpu')
        self.assertTrue(torch.equal(loss_raw, loss_cache))
        first = torch.autograd.grad(loss_raw, tuple(network.parameters()))
        second = torch.autograd.grad(loss_cache, tuple(network.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))


class IndependentScheduleAndRuntimeTests(unittest.TestCase):
    def test_default_CLI_is_check_only_never_train_or_optimizer(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('independent_cli160',
            root/'scripts/train_near_goal_depth_refiner_20261103_v2.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        class CheckSource:
            revalidated = False
            def __iter__(self):
                for _ in range(256): yield dict(features=np.zeros((4,9,64,64),np.float32))
            def revalidate(self): self.revalidated = True
        source = CheckSource(); output = io.StringIO()
        with patch.object(cli, 'load_full32_dataset', return_value=source), \
             patch.object(cli, 'train_capture', side_effect=AssertionError('train forbidden')), \
             patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')), \
             redirect_stdout(output):
            self.assertEqual(cli.main(['--manifest','synthetic_only.json']), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report['training_started']); self.assertTrue(source.revalidated)
        self.assertEqual(report['required_actual_optimizer_steps'], 1920)

    def test_exact1920_schedule_and_frozen_loss_functions_reused(self):
        self.assertEqual(schedule160.EPOCHS, 160)
        self.assertEqual(schedule160.OPTIMIZER_STEPS, (192//16)*160)
        self.assertIs(schedule160._collate, frozen_training._collate)
        self.assertIs(schedule160._loss_batch, frozen_training._loss_batch)
        self.assertEqual(schedule160.fixed_config()['calibration_in_training_cache'], False)

    def test_pilot_or_old_capture_scope_cannot_enter_v2_training(self):
        class WrongDataset:
            scope = 'pilot_smoke'
            def __len__(self): return 256
        with patch.object(schedule160, 'DepthInfluenceDataset', return_value=WrongDataset()):
            with self.assertRaisesRegex(ValueError, 'full32 only'):
                schedule160.load_full32_dataset('synthetic manifest not opened')
        class OldCapture:
            scope = 'full32'
            capture_plan = dict(schema='P3_near_goal_depth_owner_pilot64_plan_20261103_v2', scope='pilot64')
            def __len__(self): return 256
        with patch.object(schedule160, 'DepthInfluenceDataset', return_value=OldCapture()):
            with self.assertRaisesRegex(ValueError, 'new full32'):
                schedule160.load_full32_dataset('synthetic manifest not opened')

    def test_failed_oracle_precedes_cache_model_and_optimizer(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_oracle160_') as directory:
            source = SyntheticFullSource(directory)
            identity = dict(code_sha256={}, synthetic_unit_only=True)
            out = Path(directory)/'blocked'
            with patch.object(schedule160, 'load_full32_dataset', return_value=source), \
                 patch.object(schedule160, 'source_identity', return_value=identity), \
                 patch.object(schedule160, 'oracle_diagnostic', return_value=dict(ready_for_segmentation_fit=False)), \
                 patch.object(schedule160.CachedRoleDataset, 'from_dataset', side_effect=AssertionError('cache forbidden')), \
                 patch.object(schedule160, 'TinyDepthInfluenceUNet', side_effect=AssertionError('model forbidden')), \
                 patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')):
                with self.assertRaisesRegex(ValueError, 'Original oracle gate failed'):
                    schedule160.train_capture('synthetic unused manifest', out)
            status = json.loads((out/'status.json').read_text())
            self.assertFalse(status['training_started']); self.assertEqual(status['actual_optimizer_steps'], 0)
            self.assertFalse((out/'best.pt').exists())

    def test_original_source_cannot_be_changed_or_rebound_at_final_verification(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_final_source_') as directory:
            source = SyntheticFullSource(directory)
            identity = dict(bound_data_sha256=deepcopy(source.bound_pointers), code_sha256={})
            source.raw.write_text('changed after the cache was built')
            with self.assertRaisesRegex(ValueError, 'original source hash'):
                schedule160._verify_identity(source, identity)
            source.bound_pointers[str(source.raw)] = frozen_data.sha256(source.raw)
            with self.assertRaisesRegex(ValueError, 'bindings changed'):
                schedule160._verify_identity(source, identity)

    def test_runtime_new_source_map_missing_changed_or_foreign_rejects_before_NN(self):
        current = {'synthetic_current_v2_runtime.py': 'a'*64}
        with tempfile.TemporaryDirectory(prefix='synthetic_source160_') as directory:
            path = Path(directory)/'placeholder.pt'; path.write_bytes(b'synthetic only, never trained')
            digest = frozen_data.sha256(path)
            for sources in ({}, {**current, 'foreign.py':'b'*64},
                            {'synthetic_current_v2_runtime.py':'b'*64}):
                meta = dict(scope='full32', dataset_split='TRAIN', TEST_read=False,
                    training_complete=True, epochs_completed=160, actual_optimizer_steps=1920,
                    schedule_protocol_sha256=schedule160.PROTOCOL_SHA256,
                    label_protocol=runtime160.LABEL_PROTOCOL, quality_protocol=runtime160.QUALITY_PROTOCOL,
                    completion=False, current_depth_influence_only=True, complete_visible_instance_claimed=False,
                    geometry_erosions=1, calibration_cached=False, TCP_GRU_GT_input=False,
                    absolute_worldXYZ_input=False, feature_names=list(runtime160.FEATURE_NAMES),
                    class_names=list(runtime160.CLASS_NAMES), model_config=dict(input_channels=9, classes=2, base_channels=8),
                    selected_epoch=160, model_qualified=False, calibration_parent_ids=list(runtime160.PARTITION_IDS[28:]),
                    identity=dict(config=schedule160.fixed_config(), code_sha256=sources, synthetic_fixture_not_training=True))
                payload = dict(version=schedule160.VERSION, metadata=meta)
                with patch.object(torch, 'load', return_value=payload), \
                     patch.object(runtime160, 'implementation_source_map', return_value=current) as expected, \
                     patch.object(runtime160, 'TinyDepthInfluenceUNet', side_effect=AssertionError('NN forbidden')):
                    with self.assertRaisesRegex(ValueError, 'implementation source SHA map'):
                        runtime160.CurrentCubeDepthRefinerV2(path, digest)
                    expected.assert_called_once()

    def test_40_and160_runtime_versions_reject_each_other_before_NN(self):
        from our_mani.near_goal_depth_refiner_20261103 import model as runtime40
        with tempfile.TemporaryDirectory(prefix='synthetic_versions_') as directory:
            path = Path(directory)/'placeholder.pt'; path.write_bytes(b'synthetic unit fixture, no trained weights')
            digest = frozen_data.sha256(path)
            with patch.object(torch, 'load', return_value=dict(version=frozen_training.VERSION)), \
                 patch.object(runtime160, 'TinyDepthInfluenceUNet', side_effect=AssertionError('NN forbidden')):
                with self.assertRaisesRegex(ValueError, 'schedule160'):
                    runtime160.CurrentCubeDepthRefinerV2(path, digest)
            with patch.object(torch, 'load', return_value=dict(version=schedule160.VERSION)), \
                 patch.object(runtime40, 'TinyDepthInfluenceUNet', side_effect=AssertionError('NN forbidden')):
                with self.assertRaisesRegex(ValueError, 'Completed exact'):
                    runtime40.CurrentCubeDepthRefiner(path, digest)

    def test_unqualified160_diagnostic_cannot_construct_measurement(self):
        obj = runtime160.CurrentCubeDepthRefinerV2.__new__(runtime160.CurrentCubeDepthRefinerV2)
        obj.qualified, obj.model, obj.checkpoint_sha256 = False, object(), 'synthetic_only'
        probs = np.zeros((4, 2, 64, 64), np.float32); probs[:, 1] = 1.
        with patch.object(runtime160, 'predict_probabilities', return_value=probs), \
             patch.object(runtime160, 'probability_to_observation', side_effect=AssertionError('measurement forbidden')):
            result = obj.predict(public())
        self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
        self.assertFalse(result['completion'])


if __name__ == '__main__': unittest.main()
