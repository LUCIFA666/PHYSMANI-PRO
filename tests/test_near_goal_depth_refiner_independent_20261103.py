"""Independent synthetic CPU boundaries; no fitting, Native or real TEST data."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

from our_mani.near_goal_depth_refiner_20261103 import data, evaluation, model, training
from our_mani.near_goal_segmentation_20261103 import data as old_data
from our_mani.local_mpc_20261103.depth_owner_capture_sensors import _labels as actual_capture_label_producer

torch.set_num_threads(2)


def public():
    return dict(rgb=np.zeros((4, 256, 256, 3), np.uint8),
        depth_m=np.full((4, 256, 256), 1.025, np.float32),
        K=np.repeat(np.array([[-200., 0, 128], [0, -200., 128], [0, 0, 1.]])[None], 4, 0),
        T=np.repeat(np.eye(4)[None], 4, 0), public_sphere_center=np.array([0., 0., 1.]),
        camera_names=np.array(data.CAMERAS), timestamp=np.array(0.))


def producer_labels(inputs, cube=None, goal=None, *, negative=None, rgb_only=None):
    base = dict(rgb=inputs['rgb'], depth_m=inputs['depth_m'])
    without_cube = {k: v.copy() for k, v in base.items()}
    without_goal = {k: v.copy() for k, v in base.items()}
    if cube is not None: without_cube['depth_m'][cube] += np.float32(.1)
    if goal is not None: without_goal['depth_m'][goal] += np.float32(.1)
    if negative is not None: without_cube['depth_m'][negative] -= np.float32(.1)
    if rgb_only is not None: without_goal['rgb'][rgb_only] = [10, 20, 30]
    return actual_capture_label_producer(base, dict(cube=without_cube, goal=without_goal))


def mask_full():
    value = np.zeros((4, 256, 256), bool)
    value[:, 124:133, 124:133] = True
    return value


def sample(semantic=None, condition='near'):
    inp = public(); crops = data.make_near_goal_crops(inp)
    if semantic is None:
        semantic = np.zeros((4, 64, 64), np.uint8)
        semantic[:, 28:37, 28:37] = 1
    return dict(inputs=inp, crops=crops, features=crops['features'], semantic=semantic,
        cube_center_world=np.array([0., 0., 1.]), parent_id='synthetic_parent',
        scene_id='synthetic_'+condition, role='calib', condition=condition,
        unknown_reason_masks={'synthetic_unknown': semantic == 255})


def reports():
    return [dict(threshold=t,
        pixels=dict(TP=1000, FP=0, U=0, predicted_positive_count=1000,
                    known_precision=1., conservative_precision_lowerbound=1.),
        positions=dict(oracle_geometry_eligible=10, oracle_eligible_coverage=1.,
            owner_unobserved_falseavailable=0,
            qualified_visible_centre_error=dict(p95_m=.001),
            negatives={k: dict(samples=1, falseavailable=0) for k in ('sphere_only', 'cube_far')}))
        for t in evaluation.THRESHOLD_GRID]


class IndependentDepthRefinerTests(unittest.TestCase):
    def test_raw_positive_innerAA_retained_outer_ring_unknown_and_GT_ignored(self):
        inp, raw = public(), mask_full()
        labels = producer_labels(inp, raw)
        self.assertEqual(labels['semantic_role_ids'][0, 124, 124], 255)
        target = data.binary_depth_labels(inp, labels)
        self.assertTrue(np.all(target[raw] == 1))
        self.assertEqual(target[0, 123, 124], 255)
        self.assertEqual(target[0, 122, 124], 0)
        poisoned = dict(labels, cube_pose_world=np.full(7, 1e20), cube_present=np.array(False),
                        semantic_role_ids=np.full(raw.shape, 2, np.uint8), task_id='privileged')
        np.testing.assert_array_equal(target, data.binary_depth_labels(inp, poisoned))
        wrong = deepcopy(labels); wrong['cube_raw_positive'][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, 'actual depth'):
            data.binary_depth_labels(inp, wrong)

    def test_negative_invalid_conflict_and_RGBonly_are_unknown_not_background(self):
        inp = public(); raw = np.zeros((4, 256, 256), bool)
        raw[:, 128, 128] = True
        negative = np.zeros_like(raw); negative[:, 145, 145] = True
        rgb_only = np.zeros_like(raw); rgb_only[:, 150, 150] = True
        inp['depth_m'][:, 155, 155] = 0
        labels = producer_labels(inp, raw, raw, negative=negative, rgb_only=rgb_only)
        target = data.binary_depth_labels(inp, labels)
        for point in (128, 145, 150, 155): self.assertTrue(np.all(target[:, point, point] == 255))
        self.assertFalse((target == 1).any())
        self.assertTrue(np.all(target[:, 10, 10] == 0))

    def test_unknown_positive_mass_is_in_conservative_denominator_end_to_end(self):
        semantic = np.zeros((4, 64, 64), np.uint8); semantic[:, 28:37, 28:37] = 1
        semantic[0, 10:12, 10:12] = 255
        s = sample(semantic)
        p = np.zeros((4, 2, 64, 64), np.float32); p[:, 0] = .999; p[:, 1] = .001
        predicted = semantic == 1; predicted[0, 10:12, 10:12] = True
        p[:, 1][predicted] = .999; p[:, 0][predicted] = .001
        report = evaluation.evaluate_probabilities([s], [p], .5)
        pix = report['pixels']
        self.assertEqual(pix['TP'], 324); self.assertEqual(pix['FP'], 0); self.assertEqual(pix['U'], 4)
        self.assertEqual(pix['predicted_positive_count'], 328)
        self.assertEqual(pix['known_precision'], 1.)
        self.assertAlmostEqual(pix['conservative_precision_lowerbound'], 324/328)
        self.assertEqual(report['per_camera_pixels']['0']['U'], 4)
        self.assertEqual(report['UNKNOWN_predictions_by_reason']['synthetic_unknown']['Cube'], 4)
        empty = evaluation.conservative_pixel_metrics(np.full((2, 2), 255, np.uint8), np.zeros((2, 2), bool))
        self.assertIsNone(empty['conservative_precision_lowerbound'])

    def test_calibration_cannot_hide_empty_eligibility_abstention_unknown_or_falseavailability(self):
        self.assertEqual(evaluation.choose_calibrated_threshold(reports())['selected_threshold'], .5)
        for reason in ('empty', 'low_coverage', 'U', 'unobserved_falseavailable', 'missing_negative'):
            r = reports()
            for row in r:
                if reason == 'empty': row['positions'].update(oracle_geometry_eligible=0, oracle_eligible_coverage=None)
                if reason == 'low_coverage': row['positions']['oracle_eligible_coverage'] = .1
                if reason == 'U': row['pixels'].update(U=1000, predicted_positive_count=2000, conservative_precision_lowerbound=.5)
                if reason == 'unobserved_falseavailable': row['positions']['owner_unobserved_falseavailable'] = 1
                if reason == 'missing_negative': row['positions']['negatives']['cube_far']['samples'] = 0
            self.assertFalse(evaluation.choose_calibrated_threshold(r)['calibration_passed'], reason)
        with self.assertRaises(ValueError): evaluation.choose_calibrated_threshold(reports()[:-1])

    def test_measurement_checks_raw_border_before_single_erosion(self):
        inp = public(); crop = data.make_near_goal_crops(inp)
        raw = np.zeros((4, 64, 64), bool); raw[:, 28:37, 28:37] = True
        good = data.make_measurement(inp, crop, raw)
        self.assertTrue(good['refiner_available'])
        self.assertEqual(good['eroded_pixels'], 4*7*7)
        self.assertEqual(good['geometry_erosions'], 1)
        edge = raw.copy(); edge[0, 0, 32] = True
        # The spur vanishes during erosion; checking only afterwards is unsafe.
        np.testing.assert_array_equal(data.erode_one_pixel(edge), data.erode_one_pixel(raw))
        bad = data.make_measurement(inp, crop, edge)
        self.assertFalse(bad['refiner_available']); self.assertIsNone(bad['centre'])
        self.assertIn('raw_Cube_mask_touches_crop_boundary_before_erosion', bad['reasons'])
        single = np.zeros_like(raw); single[:, 32, 32] = True
        m = data.make_measurement(inp, crop, single)
        self.assertFalse(m['refiner_available']); self.assertEqual(m['eroded_pixels'], 0)
        self.assertEqual(m['raw_positive_pixels'], 4)
        self.assertFalse(m['complete_visible_instance_claimed'])

    def test_outside_observed_raw_pixel_is_not_erased_into_a_valid_measurement(self):
        inp = public(); inp['public_sphere_center'][0] = .63
        crop = data.make_near_goal_crops(inp)
        raw = np.zeros((4, 64, 64), bool); raw[:, 28:37, 40:49] = True
        raw[0, 30, 5] = True
        self.assertFalse(crop['valid_crop'][0, 30, 5])
        result = data.make_measurement(inp, crop, raw)
        self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
        self.assertIn('raw_Cube_prediction_outside_observed_crop', result['reasons'])

    def test_empty_owner_oracle_stops_before_model_or_optimizer(self):
        values = [sample(np.full((4, 64, 64), 255, np.uint8), 'occluded')]
        class Samples(list): scope = 'full32'
        dataset = Samples(values)
        report = evaluation.oracle_diagnostic(dataset)
        self.assertFalse(report['ready_for_segmentation_fit'])
        self.assertEqual(report['aggregate']['oracle_geometry_eligible'], 0)
        self.assertIsNone(report['aggregate']['qualified_visible_centre_error']['p95_m'])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); manifest = base/'manifest.json'; manifest.write_text('{}')
            identity = dict(code_sha256={}, config=json.loads(json.dumps(training.fixed_config())))
            with patch.object(training, 'DepthInfluenceDataset', return_value=dataset), \
                 patch.object(training, 'source_identity', return_value=identity), \
                 patch.object(training, 'TinyDepthInfluenceUNet', side_effect=AssertionError('NN must not exist')), \
                 patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')):
                with self.assertRaisesRegex(ValueError, 'Oracle5mm'):
                    training.train_capture(manifest, base/'blocked')
            self.assertFalse(json.loads((base/'blocked/status.json').read_text())['training_started'])

    def test_public_features_only_forward_and_allunknown_loss_no_GT_gradient(self):
        self.assertIs(data.make_near_goal_crops, old_data.make_near_goal_crops)
        inp = public(); features = data.make_near_goal_crops(inp)['features']
        net = model.TinyDepthInfluenceUNet().eval(); x = torch.tensor(features)
        out = net(x); self.assertEqual(tuple(out.shape), (4, 2, 64, 64))
        with self.assertRaises(TypeError): net(x, cube_center_world=torch.zeros(4, 3))
        with self.assertRaises(ValueError): net(dict(features=x, labels=torch.ones(4, 64, 64)))
        with self.assertRaises(ValueError): data.make_near_goal_crops(dict(inp, cube_pose_world=np.ones(7)))
        unknown = torch.full((4, 64, 64), 255, dtype=torch.long)
        loss = model.binary_segmentation_loss(out, unknown)
        torch.testing.assert_close(loss, torch.zeros(4), atol=0., rtol=0.)
        loss.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_pilot_metadata_cannot_construct_runtime_model_even_with_synthetic_attestations(self):
        # Deliberately synthetic in-memory metadata and temporary text files;
        # there are no trained weights, completed optimizer run or real evidence.
        with tempfile.TemporaryDirectory(prefix='synthetic_refiner_guard_') as directory:
            base = Path(directory); ckpt = base/'not_trained.pt'; ckpt.write_bytes(b'synthetic guard fixture only')
            digest = data.sha256(ckpt)
            identity = dict(config=json.loads(json.dumps(training.fixed_config())), synthetic_fixture_not_training=True)
            cal = dict(reports=reports(), selection=evaluation.choose_calibrated_threshold(reports()))
            oracle = dict(ready_for_segmentation_fit=True, synthetic_only=True)
            for name, value in (('cal.json', cal), ('oracle.json', oracle)):
                (base/name).write_text(json.dumps(value))
            meta = dict(dataset_split='TRAIN', TEST_read=False, training_complete=True, epochs_completed=40,
                label_protocol=data.LABEL_PROTOCOL, quality_protocol=evaluation.QUALITY_PROTOCOL,
                completion=False, current_depth_influence_only=True, complete_visible_instance_claimed=False,
                TCP_GRU_GT_input=False, absolute_worldXYZ_input=False, geometry_erosions=1,
                feature_names=data.FEATURE_NAMES, class_names=data.CLASS_NAMES,
                model_config=dict(input_channels=9, classes=2, base_channels=8), identity=identity,
                scope='pilot_smoke', pilot_is_smoke_only=True, model_qualified=False, selected_threshold=.5,
                calibration_evidence=dict(path='cal.json', sha256=data.sha256(base/'cal.json')),
                oracle_evidence=dict(path='oracle.json', sha256=data.sha256(base/'oracle.json')))
            status = dict(complete=True, training_complete=True, completed_epochs=40, best_checkpoint_sha256=digest,
                          identity=identity, model_qualified=False, scope='pilot_smoke', synthetic_only=True)
            (base/'status.json').write_text(json.dumps(status))
            payload = dict(version=training.VERSION, metadata=meta, model_state_dict={})
            with patch.object(torch, 'load', return_value=payload), \
                 patch.object(model, 'TinyDepthInfluenceUNet', side_effect=AssertionError('pilot must reject before NN')):
                with self.assertRaisesRegex(ValueError, 'Pilot/unqualified'):
                    model.CurrentCubeDepthRefiner(ckpt, digest)

    def test_unqualified_diagnostic_predict_never_completes_a_measurement(self):
        obj = model.CurrentCubeDepthRefiner.__new__(model.CurrentCubeDepthRefiner)
        obj.qualified, obj.model, obj.checkpoint_sha256 = False, object(), 'synthetic_only'
        probs = np.zeros((4, 2, 64, 64), np.float32); probs[:, 1] = 1.
        with patch.object(model, 'predict_probabilities', return_value=probs), \
             patch.object(evaluation, 'probability_to_observation', side_effect=AssertionError('unqualified measurement forbidden')):
            result = obj.predict(public())
        self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
        self.assertFalse(result['completion'])


if __name__ == '__main__': unittest.main()
