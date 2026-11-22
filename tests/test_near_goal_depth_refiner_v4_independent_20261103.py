"""Independent v4 CPU contracts; synthetic only, no fit or real calib reads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from our_mani import near_goal_cube_geometry_20261103_v1 as geometry
from our_mani.near_goal_depth_refiner_20261103 import model as frozen_model
from our_mani.near_goal_depth_refiner_20261103 import training as frozen_training
from our_mani.near_goal_depth_refiner_20261103.data import make_near_goal_crops, PARTITION_IDS, SHOT_IDS
from our_mani.near_goal_depth_refiner_20261103_v4 import training, evaluation, calibration, runtime
from tests.test_near_goal_depth_refiner_independent_20261103 import public
from tests.test_near_goal_depth_refiner_v2_independent_20261103 import SyntheticFullSource


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class SyntheticOracleCohort:
    """Only aggregation logic is tested; measurement stubs never read images."""
    scope = 'full32'

    def __init__(self):
        owner = np.ones((4, 2, 2), np.uint8)
        empty = np.zeros_like(owner)
        self.samples = []
        ordinal = 0
        for i, parent in enumerate(PARTITION_IDS):
            role = 'fit' if i < 24 else ('selection' if i < 28 else 'calib')
            for j, shot in enumerate(SHOT_IDS):
                near = 2 <= j < 7
                self.samples.append(dict(parent_id=parent, scene_id=parent+'/'+shot,
                    role=role, condition='near_goal' if near else ('sphere_only' if j == 0 else 'cube_far'),
                    inputs=dict(unit_stub_near=near, unit_stub_positive_ordinal=ordinal if near else -1),
                    crops={}, semantic=owner if near else empty, cube_center_world=np.zeros(3)))
                ordinal += int(near)

    def __iter__(self): return iter(self.samples)


def stub_measurement(available):
    return dict(refiner_available=bool(available), reasons=[] if available else ['synthetic_no_measurement'],
                centre=[.001, 0., 0.] if available else None)


class IndependentGeometryAndV4Contracts(unittest.TestCase):
    def test_schedule_network_loss_and_new_geometry_dispatch_are_explicit(self):
        self.assertEqual(training.EPOCHS, 160)
        self.assertEqual(training.OPTIMIZER_STEPS, 1920)
        self.assertIs(training.TinyDepthInfluenceUNet, frozen_model.TinyDepthInfluenceUNet)
        self.assertIs(training._collate, frozen_training._collate)
        self.assertIs(training._loss_batch, frozen_training._loss_batch)
        self.assertIs(training.oracle_diagnostic, evaluation.oracle_diagnostic)
        self.assertIs(calibration.evaluate_probabilities, evaluation.evaluate_probabilities)
        self.assertIs(runtime.probability_to_observation, evaluation.probability_to_observation)
        self.assertEqual(calibration.CALIBRATION_NUM_THREADS, 2)

    def test_source_map_covers_new_package_cli_and_exact_three_lower_copies(self):
        package = ROOT/'our_mani/near_goal_cube_geometry_20261103_v1'
        lower_hashes = {
            '75108de1859e4efb527beaec644134c02e70735718c91aa325da3366a3c5a6b8',
            '5488046b3e045179ba2651631604ed2bc862f4b3b332cb8a952b248a91189eca',
            '5c949ee82da9a308312c372f556269a1d577e2197eb62f24fa7d2cbfdb19077d'}
        local = {hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')}
        self.assertTrue(lower_hashes <= local)
        geometric = geometry.geometry_source_map()
        implementation = training.implementation_source_map()
        for path, digest in geometric.items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            self.assertEqual(implementation[path], digest)
        for path in (ROOT/'our_mani/near_goal_depth_refiner_20261103_v4').glob('*.py'):
            self.assertIn(str(path), implementation)
        self.assertIn(str(ROOT/'scripts/train_near_goal_depth_refiner_20261103_v4.py'), implementation)

    def test_current_public_empty_measurement_never_accepts_privileged_input(self):
        inputs = public()
        crops = make_near_goal_crops(inputs)
        raw = np.zeros((4, 64, 64), bool)
        result = geometry.measure_current_cube(inputs, crops, raw)
        self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
        for key in ('cube_pose_world', 'Native_held', 'future_object_position', 'TCP'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                geometry.measure_current_cube(dict(inputs, **{key: np.zeros(7)}), crops, raw)

    def test_one_erosion_and_raw_crop_boundary_are_not_changed(self):
        inputs = public(); crops = make_near_goal_crops(inputs)
        raw = np.zeros((4, 64, 64), bool); raw[:2, 30:33, 30:33] = True
        result = geometry.measure_current_cube(inputs, crops, raw)
        self.assertEqual(result['measured_points'], 2)
        self.assertFalse(result['refiner_available'])
        raw[:] = False; raw[:, 0:16, 20:40] = True
        result = geometry.measure_current_cube(inputs, crops, raw)
        self.assertFalse(result['refiner_available'])
        self.assertIn('raw_Cube_mask_touches_crop_boundary_before_erosion', result['reasons'])

    def test_empty_or_failed_oracle_precedes_cache_model_and_optimizer(self):
        for oracle in (dict(ready_for_segmentation_fit=False),
                       dict(ready_for_segmentation_fit=False, aggregate=dict(oracle_geometry_eligible=0))):
            with self.subTest(oracle=oracle), tempfile.TemporaryDirectory(prefix='synthetic_v4_oracle_') as directory:
                source = SyntheticFullSource(directory)
                out = Path(directory)/'blocked'
                with patch.object(training, 'load_full32_dataset', return_value=source), \
                     patch.object(training, 'source_identity', return_value=dict(code_sha256={})), \
                     patch.object(training, 'oracle_diagnostic', return_value=oracle), \
                     patch.object(training.CachedRoleDataset, 'from_dataset', side_effect=AssertionError('cache forbidden')), \
                     patch.object(training, 'TinyDepthInfluenceUNet', side_effect=AssertionError('model forbidden')), \
                     patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')):
                    with self.assertRaises(ValueError):
                        training.train_capture('synthetic_unused_manifest', out)
                status = json.loads((out/'status.json').read_text())
                self.assertFalse(status['training_started'])
                self.assertEqual(status['actual_optimizer_steps'], 0)
                self.assertFalse((out/'best.pt').exists())

    def test_fixed_old160_denominator_retains_new_geometry_unknowns(self):
        source = SyntheticOracleCohort()
        for new_available in (145, 143, 0):
            with self.subTest(new_available=new_available), \
                 patch.object(evaluation, 'legacy_make_measurement',
                    side_effect=lambda inp, crops, raw: stub_measurement(inp['unit_stub_near'])), \
                 patch.object(evaluation, 'measure_current_cube',
                    side_effect=lambda inp, crops, raw: stub_measurement(inp['unit_stub_near'] and inp['unit_stub_positive_ordinal'] < new_available)), \
                 patch.object(evaluation, 'geometry_source_map', return_value={}):
                report = evaluation.oracle_diagnostic(source)
            aggregate = report['aggregate']
            self.assertEqual(aggregate['samples'], 256)
            self.assertEqual(aggregate['oracle_geometry_eligible'], 160)
            self.assertEqual(aggregate['oracle_eligible_refiner_available'], new_available)
            self.assertEqual(aggregate['oracle_eligible_coverage'], new_available/160)
            self.assertEqual(aggregate['qualified_visible_centre_error']['count'], new_available)
            self.assertEqual(report['ready_for_segmentation_fit'], new_available >= 144)
            self.assertEqual([report['by_role'][k]['oracle_geometry_eligible'] for k in ('fit', 'selection', 'calib')], [120, 20, 20])
            self.assertEqual(sum(v['samples'] for v in aggregate['negatives'].values()), 96)
            self.assertTrue(all(v['falseavailable'] == 0 for v in aggregate['negatives'].values()))
            if not new_available:
                self.assertIsNone(aggregate['qualified_visible_centre_error']['p95_m'])

    def test_redefined159_baseline_cannot_make_full32_oracle_pass(self):
        with patch.object(evaluation, 'legacy_make_measurement',
                side_effect=lambda inp, crops, raw: stub_measurement(inp['unit_stub_near'] and inp['unit_stub_positive_ordinal'] != 159)), \
             patch.object(evaluation, 'measure_current_cube',
                side_effect=lambda inp, crops, raw: stub_measurement(inp['unit_stub_near'])), \
             patch.object(evaluation, 'geometry_source_map', return_value={}):
            report = evaluation.oracle_diagnostic(SyntheticOracleCohort())
        self.assertEqual(report['aggregate']['oracle_geometry_eligible'], 159)
        self.assertFalse(report['fixed_baseline_count_verified'])
        self.assertFalse(report['ready_for_segmentation_fit'])

    def test_runtime_fixed_calibration_set_rejects_same_count_scene_swap_and_nonbool(self):
        from copy import deepcopy
        parents = PARTITION_IDS[28:]
        records = [dict(parent_id=p, scene_id=p+'/'+s, role='calib',
            oracle_geometry_eligible=s in SHOT_IDS[2:7],
            baseline_observable_eligible=s in SHOT_IDS[2:7],
            eligibility_protocol=evaluation.BASELINE_ELIGIBILITY_PROTOCOL)
            for p in parents for s in SHOT_IDS]
        runtime._ledger(records, parents, ['calib']*4)
        runtime._fixed_baseline_ledger(records, parents)
        swapped = deepcopy(records)
        for i in (0, 2):
            flag = not swapped[i]['oracle_geometry_eligible']
            swapped[i]['oracle_geometry_eligible'] = flag
            swapped[i]['baseline_observable_eligible'] = flag
        self.assertEqual(sum(r['oracle_geometry_eligible'] for r in swapped), 20)
        with self.assertRaises(ValueError): runtime._fixed_baseline_ledger(swapped, parents)
        for field, poison in (('oracle_geometry_eligible', 1), ('baseline_observable_eligible', 1),
                              ('eligibility_protocol', 'new_F_eligible')):
            changed = deepcopy(records); changed[2][field] = poison
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime._fixed_baseline_ledger(changed, parents)

    def test_runtime_rejects_old_versions_wrong_source_and_geometry_before_NN(self):
        from copy import deepcopy
        from our_mani.near_goal_depth_refiner_20261103_v2.training import VERSION as OLD_VERSION
        current = {'/synthetic/current_v4.py': 'a'*64}
        current_geometry = {'/synthetic/current_geometry.py': 'b'*64}
        meta = dict(scope='full32', dataset_split='TRAIN', TEST_read=False,
            training_complete=True, epochs_completed=160, actual_optimizer_steps=1920,
            schedule_protocol_sha256=training.PROTOCOL_SHA256,
            label_protocol=runtime.LABEL_PROTOCOL, quality_protocol=runtime.QUALITY_PROTOCOL,
            completion=False, current_depth_influence_only=True, complete_visible_instance_claimed=False,
            geometry_erosions=1, calibration_cached=False, TCP_GRU_GT_input=False,
            absolute_worldXYZ_input=False, feature_names=list(runtime.FEATURE_NAMES),
            class_names=list(runtime.CLASS_NAMES), model_config=dict(input_channels=9, classes=2, base_channels=8),
            selected_epoch=160, model_qualified=False, calibration_parent_ids=list(PARTITION_IDS[28:]),
            geometry_version=training.EXPECTED_GEOMETRY_VERSION,
            geometry_protocol_sha256=training.GEOMETRY_PROTOCOL_SHA256,
            geometry_source_sha256=current_geometry,
            baseline_eligibility_protocol=evaluation.BASELINE_ELIGIBILITY_PROTOCOL,
            identity=dict(config=training.fixed_config(), code_sha256=current))
        variants = [dict(version=OLD_VERSION, metadata={})]
        for bad in ({}, {**current, '/foreign.py':'c'*64}, {'/synthetic/current_v4.py':'c'*64}):
            changed = deepcopy(meta); changed['identity']['code_sha256'] = bad
            variants.append(dict(version=training.VERSION, metadata=changed))
        for field, bad in (('geometry_version', 'legacy_integer_ray'), ('geometry_source_sha256', {}),
                           ('geometry_protocol_sha256', '0'*64), ('baseline_eligibility_protocol', 'F_only')):
            changed = deepcopy(meta); changed[field] = bad
            variants.append(dict(version=training.VERSION, metadata=changed))
        with tempfile.TemporaryDirectory(prefix='synthetic_v4_runtime_') as directory:
            path = Path(directory)/'placeholder.pt'; path.write_bytes(b'unit fixture; no fitted weights')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for payload in variants:
                with self.subTest(payload_version=payload['version']), \
                     patch.object(torch, 'load', return_value=payload), \
                     patch.object(runtime, 'implementation_source_map', return_value=current), \
                     patch.object(runtime, 'checked_geometry_sources', return_value=current_geometry), \
                     patch.object(runtime, 'TinyDepthInfluenceUNet', side_effect=AssertionError('NN forbidden')):
                    with self.assertRaises(ValueError):
                        runtime.CurrentCubeDepthRefinerV4(path, digest)

    def test_qualified_CPU2_and_live_geometry_guard_precede_crop_or_forward(self):
        obj = runtime.CurrentCubeDepthRefinerV4.__new__(runtime.CurrentCubeDepthRefinerV4)
        obj.qualified, obj.model = True, torch.nn.Linear(1, 1).eval()
        obj.calibration_environment = calibration.calibration_environment()
        obj.metadata = dict(geometry_source_sha256={'expected': 'a'*64})
        try:
            torch.set_num_threads(4)
            with patch.object(runtime, 'make_near_goal_crops', side_effect=AssertionError('crop forbidden')) as crop:
                with self.assertRaises(ValueError): obj.predict({})
                crop.assert_not_called()
            self.assertEqual(torch.get_num_threads(), 4)
            torch.set_num_threads(2)
            with patch.object(runtime, 'checked_geometry_sources', return_value={'changed': 'b'*64}), \
                 patch.object(runtime, 'make_near_goal_crops', side_effect=AssertionError('crop forbidden')) as crop:
                with self.assertRaises(ValueError): obj.predict({})
                crop.assert_not_called()
        finally:
            torch.set_num_threads(2)
        obj.qualified, obj.model, obj.checkpoint_sha256 = False, object(), 'synthetic_unqualified'
        probabilities = np.zeros((4, 2, 64, 64), np.float32); probabilities[:, 1] = 1
        with patch.object(runtime, 'predict_probabilities', return_value=probabilities), \
             patch.object(runtime, 'probability_to_observation', side_effect=AssertionError('measurement forbidden')):
            output = obj.predict(public())
        self.assertFalse(output['refiner_available']); self.assertIsNone(output['centre'])

    def test_original_raw995_coverage_grid_and_negative_gates_are_not_softened(self):
        from copy import deepcopy
        reports = [dict(threshold=t, geometry_version=evaluation.GEOMETRY_VERSION,
                baseline_eligibility_protocol=evaluation.BASELINE_ELIGIBILITY_PROTOCOL,
                pixels=dict(predicted_positive_count=1000, U=0,
                    conservative_precision_lowerbound=1., known_precision=1.),
                positions=dict(oracle_geometry_eligible=20, oracle_eligible_coverage=.9,
                    qualified_visible_centre_error=dict(p95_m=.001), owner_unobserved_falseavailable=0,
                    negatives={k: dict(samples=4, falseavailable=0) for k in ('sphere_only', 'cube_far')}))
                for t in evaluation.THRESHOLD_GRID]
        self.assertEqual(evaluation.choose_calibrated_threshold(reports)['selected_threshold'], .5)
        modifiers = [lambda r: r['pixels'].update(conservative_precision_lowerbound=.994, U=6),
                     lambda r: r['positions'].update(oracle_eligible_coverage=.89),
                     lambda r: r['positions'].update(oracle_geometry_eligible=0, oracle_eligible_coverage=None),
                     lambda r: r['positions']['qualified_visible_centre_error'].update(p95_m=.0051),
                     lambda r: r['positions'].update(owner_unobserved_falseavailable=1),
                     lambda r: r['positions']['negatives']['sphere_only'].update(falseavailable=1)]
        for modify in modifiers:
            changed = deepcopy(reports)
            for report in changed: modify(report)
            selected = evaluation.choose_calibrated_threshold(changed)
            self.assertFalse(selected['calibration_passed']); self.assertIsNone(selected['selected_threshold'])


if __name__ == '__main__':
    unittest.main()
