"""CPU contracts using frozen assets and synthetic public observations only."""
import json
import inspect
import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from our_mani.contracts import ObjectState, Pose
from our_mani.perception import CameraCalibration
from our_mani.hierarchy_20261030 import tracking
from our_mani.hierarchy_20261030.data import CAMERAS


ROOT = Path(__file__).resolve().parents[1]


def public_box_frame(timestamp, visible=True):
    calibration = CameraCalibration(
        np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]),
        Pose(np.array([.2, 0., 0.])))
    images, depths, calibrations = {}, {}, {}
    for camera in CAMERAS:
        rgb = np.zeros((64, 64, 3), np.uint8)
        if visible:
            rgb[29:35, 29:35] = [255, 0, 0]
        images[camera] = rgb
        depths[camera] = np.full((64, 64), .8, np.float32)
        calibrations[camera] = calibration
    robot = np.zeros(15, np.float32)
    robot[6] = robot[7] = 1
    return dict(timestamp=timestamp, **{
        'observation/images': images, 'observation/depths': depths,
        'observation/calibrations': calibrations}), robot


class PublicTrackingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_checkpoint_and_insert_cad_d4_load_without_relaxed_proofs(self):
        for task in tracking.FROZEN_CONFIGURATIONS:
            tracker, receipt = tracking.build_insert_tracker(ROOT, task)
            self.assertEqual(set(tracker.templates), set(tracking.ROLE_IDS))
            ids = [value for template in tracker.templates.values()
                   for component in template.components for value in component.instance_ids]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), set(tracking.ROLE_IDS.values()))
            asset = json.loads((ROOT / receipt['TRAIN_CAD']['path']).read_text())
            for role, template in tracker.templates.items():
                self.assertEqual(len(template.components), 1)
                expected = np.concatenate([part['points_body'] for part in asset['bodies'][role]['components']])
                np.testing.assert_array_equal(template.components[0].points_body, expected)
                self.assertEqual(template.provenance.calibration_split, 'train')
            self.assertEqual(tracker.templates['ring'].orientation_observability, 'none')
            self.assertIsNotNone(tracker.templates['ring'].finite_rotation_symmetry)
            self.assertFalse(receipt['ring_full_orientation_observable'])
            self.assertFalse(receipt['legacy_task_policy_installed'])
            self.assertTrue(receipt['symmetry_replay']['accepted'])

    def test_public_channels_and_camera_times_cannot_contain_privileged_inputs(self):
        observation, _ = public_box_frame(0.)
        tracking.validate_public_observation(observation)
        with self.assertRaises(ValueError):
            tracking.validate_public_observation(dict(observation, expert_stage='grasp'))
        with self.assertRaises(ValueError):
            tracking.validate_public_observation(dict(observation, **{'observation/instance_masks': {}}))
        with self.assertRaises(ValueError):
            tracking.validate_public_observation(dict(observation, **{'observation/camera_timestamps': {'front': .05}}))
        with self.assertRaises(ValueError):
            tracking.validate_public_observation(dict(observation, **{'observation/camera_timestamps': {'front': -.05}}))

    def test_frozen_gru_waits_five_real_observations_and_preserves_weights(self):
        pipeline = tracking.PublicMotionPipeline(ROOT, 'pick_moving_target_on_the_table')
        state_before = {key: value.clone() for key, value in pipeline.forecaster.predictor.model.state_dict().items()}
        outputs = []
        for frame in range(5):
            observation, robot = public_box_frame(frame * .05)
            outputs.append(pipeline.observe(observation, robot))
        for output in outputs[:4]:
            self.assertEqual(output['motion']['forecasts'], {})
            self.assertFalse(output['motion']['status']['target']['available'])
        forecast = outputs[4]['motion']['forecasts']['target']
        self.assertEqual(forecast['trajectory'].shape, (23, 20))
        self.assertEqual(forecast['history'].shape, (20, 21))
        self.assertEqual(forecast['history_length'], 5)
        np.testing.assert_allclose(forecast['trajectory'][:, -1], np.arange(23) * .05, atol=1e-7)
        self.assertTrue(np.isfinite(forecast['trajectory']).all())
        self.assertTrue(np.all(forecast['trajectory'][:, 18] == 0))
        self.assertFalse(forecast['full_orientation_observable'])
        self.assertEqual(outputs[4]['motion']['checkpoint_sha256'], tracking.FORECAST_SHA256)
        for name, value in pipeline.forecaster.predictor.model.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in pipeline.forecaster.predictor.model.parameters()))

    def test_closed_gripper_and_reopening_do_not_fabricate_held_or_released_modes(self):
        pipeline = tracking.PublicMotionPipeline(ROOT, 'pick_moving_target_on_the_table')
        for frame in range(5):
            observation, robot = public_box_frame(frame * .05)
            output = pipeline.observe(observation, robot)
        self.assertTrue(output['motion']['status']['target']['available'])
        observation, robot = public_box_frame(.25)
        robot[7] = 0
        closed = pipeline.observe(observation, robot)
        self.assertEqual(closed['modes']['target'], 'unknown')
        self.assertFalse(closed['motion']['status']['target']['available'])
        observation, robot = public_box_frame(.30)
        reopened = pipeline.observe(observation, robot)
        self.assertEqual(reopened['modes']['target'], 'unknown')
        self.assertEqual(reopened['motion']['forecasts'], {})
        self.assertNotIn('target', pipeline.forecaster.predictor.history)
        pipeline.reset()
        observation, robot = public_box_frame(0.)
        reset = pipeline.observe(observation, robot)
        self.assertEqual(reset['modes']['target'], 'external')
        self.assertEqual(reset['motion']['status']['target']['history_frames'], 1)

    def test_occlusion_clears_history_instead_of_continuing_a_fake_track(self):
        pipeline = tracking.PublicMotionPipeline(ROOT, 'pick_moving_target_on_the_table')
        for frame in range(5):
            observation, robot = public_box_frame(frame * .05)
            pipeline.observe(observation, robot)
        observation, robot = public_box_frame(.25, visible=False)
        missing = pipeline.observe(observation, robot)
        self.assertEqual(missing['objects'], [])
        self.assertEqual(missing['motion']['forecasts'], {})
        self.assertEqual(missing['motion']['status']['target']['reason'], 'no_fresh_public_pose')
        observation, robot = public_box_frame(.30)
        regained = pipeline.observe(observation, robot)
        self.assertEqual(regained['motion']['status']['target']['history_frames'], 1)
        self.assertEqual(regained['motion']['forecasts'], {})

    def test_case_arrays_keep_nan_masks_and_per_role_valid_ratio(self):
        pipeline = tracking.PublicMotionPipeline(ROOT, 'pick_moving_target_on_the_table')
        frames = [public_box_frame(i * .05) for i in range(6)]
        arrays, summary, receipts = tracking.collect_motion_case(pipeline, frames)
        self.assertEqual(arrays['predicted_trajectory'].shape, (6, 1, 23, 20))
        self.assertEqual(arrays['trajectory_mask'].shape, (6, 1, 23))
        self.assertEqual(arrays['observed_public_state'].shape, (6, 1, 20))
        self.assertTrue(np.isnan(arrays['predicted_trajectory'][:4]).all())
        self.assertFalse(arrays['trajectory_mask'][:4].any())
        self.assertTrue(np.isfinite(arrays['predicted_trajectory'][4:]).all())
        self.assertTrue(arrays['trajectory_mask'][4:].all())
        self.assertAlmostEqual(summary['role_valid_ratio']['target'], 2/6)
        self.assertEqual(summary['role_valid_frames']['target'], 2)
        self.assertTrue(summary['multi_body_trajectories_kept_separate'])
        self.assertFalse(summary['expert_stage_or_target_inputs'])
        self.assertEqual(len(receipts), 6)

    def test_insert_requires_fitted_segmentation_and_empty_predictions_do_not_invent_cad_pose(self):
        with self.assertRaises(FileNotFoundError):
            tracking.PublicMotionPipeline(ROOT, 'insert_onto_rotating_peg', segment_checkpoint='/missing/model.pt')
        class EmptyPublicSegmenter:
            def predict(self, observation):
                return dict(semantic_maps={camera: np.full_like(depth, -1, dtype=np.int16)
                                           for camera, depth in observation['observation/depths'].items()}, quality={})
        pipeline = tracking.PublicMotionPipeline(ROOT, 'insert_onto_rotating_peg', segmenter=EmptyPublicSegmenter())
        observation, robot = public_box_frame(0.)
        result = pipeline.observe(observation, robot)
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['motion']['forecasts'], {})
        self.assertTrue(all(not value['available'] for value in result['motion']['status'].values()))

    def test_hdf5_reader_uses_original_256_calibration_and_opens_no_expert_arrays(self):
        requested = []
        class Source(dict):
            attrs = {'task': 'pick_moving_target_on_the_table', 'num_frames': 1}
            def __getitem__(self, name):
                requested.append(name)
                if 'expert' in name or 'mask' in name:
                    raise AssertionError('forbidden label/native mask read')
                return super().__getitem__(name)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        source = Source()
        source['low_dim_obs/gripper_pose'] = np.array([[.2, 0, .8, 0, 0, 0, 1]], np.float32)
        source['low_dim_obs/gripper_open'] = np.array([1], np.float32)
        source['low_dim_obs/joint_positions'] = np.zeros((1, 7), np.float32)
        intrinsic = np.array([[-210., 0., 128.], [0., 210., 128.], [0., 0., 1.]])
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [.1, .2, .3]
        for camera in CAMERAS:
            source[f'cam_params/{camera}'] = dict(intrinsics=intrinsic, extrinsics=extrinsic,
                                                 near=np.asarray(.1), far=np.asarray(2.))
            source[f'observations/{camera}/rgb'] = np.zeros((1, 256, 256, 3), np.uint8)
            source[f'observations/{camera}/depth'] = np.zeros((1, 256, 256, 3), np.uint8)
        row = dict(split='train', task='pick_moving_target_on_the_table', frames=1,
                   source_path='/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench/train_package_compressed/unit_contract/data.h5')
        with patch.object(tracking.h5py, 'File', return_value=source), \
             patch.object(tracking, 'source_stat', return_value={'mtime_ns': 1}):
            frames = list(tracking.iter_public_hdf5_frames(row))
        self.assertEqual(len(frames), 1)
        observation, robot = frames[0]
        self.assertEqual(observation['observation/images']['front'].shape, (256, 256, 3))
        np.testing.assert_array_equal(observation['observation/calibrations']['front'].intrinsics, intrinsic)
        np.testing.assert_allclose(observation['observation/depths']['front'], .1)
        mask = np.zeros((256, 256), bool)
        mask[128, 128] = True
        point = observation['observation/calibrations']['front'].backproject(observation['observation/depths']['front'], mask)
        np.testing.assert_allclose(point, [[.1, .2, .4]], atol=1e-7)
        self.assertEqual(robot.shape, (15,))
        self.assertFalse(any('expert' in name or 'mask' in name for name in requested))

    def test_forbidden_source_splits_and_wrong_checkpoint_are_rejected(self):
        with patch.object(tracking.h5py, 'File', side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):
                list(tracking.iter_public_hdf5_frames(dict(split='test', task='pick_moving_target_on_the_table', source_path='/TEST/data.h5')))
            with self.assertRaises(ValueError):
                list(tracking.iter_public_hdf5_frames(dict(split='train', task='pick_moving_target_on_the_table', source_path='/outside/data.h5')))
        with self.assertRaises(ValueError):
            tracking.FrozenGRUForecaster(ROOT / tracking.FORECAST_RELATIVE, ('target',), expected_sha256='0' * 64)

    def test_blind_pipeline_runs_both_frontends_without_task_or_speed_input(self):
        class EmptySegmenter:
            calls = 0
            def predict(self, observation):
                self.calls += 1
                return dict(semantic_maps={camera: np.full_like(depth, -1, dtype=np.int16)
                                           for camera, depth in observation['observation/depths'].items()}, quality={})
        segmenter = EmptySegmenter()
        pipeline = tracking.BlindPublicMotionPipeline(ROOT, segmenter=segmenter)
        self.assertNotIn('task', inspect.signature(tracking.BlindPublicMotionPipeline).parameters)
        self.assertEqual(pipeline.roles, ('target', 'frame', 'peg0', 'peg1', 'peg2', 'ring'))
        for frame in range(5):
            observation, robot = public_box_frame(frame * .05)
            result = pipeline.observe(observation, robot)
        self.assertEqual(segmenter.calls, 5)
        self.assertEqual(result['all_frontends_evaluated'], ('pick', 'insert'))
        self.assertEqual(set(result['motion']['status']), set(pipeline.roles))
        self.assertTrue(result['motion']['status']['target']['available'])
        self.assertTrue(all(not result['motion']['status'][role]['available'] for role in tracking.ROLE_IDS))
        self.assertFalse(result['tracking']['source_task_or_speed_used'])
        self.assertFalse(pipeline.prior_receipt['task_id_or_description_input'])
        self.assertEqual(pipeline.role_prior_ids, dict(target=0, frame=1, peg0=1, peg1=1, peg2=1, ring=1))
        with self.assertRaises(TypeError):
            tracking.BlindPublicMotionPipeline(ROOT, task='pick_moving_target_on_the_table', segmenter=segmenter)

    def test_common_prior_audit_discloses_actual_differences_without_threshold_changes(self):
        audit = tracking.audit_canonical_public_priors(ROOT)
        self.assertEqual(set(audit['frozen_insert_sources']), set(tracking.FROZEN_CONFIGURATIONS))
        self.assertTrue(all(audit['configuration_equality'].values()))
        self.assertFalse(audit['source_task_or_speed_selects_frontend'])
        self.assertFalse(audit['registration_thresholds_changed'])
        self.assertFalse(audit['full_orientation_promoted'])
        self.assertFalse(audit['native_VAL_or_TEST_used'])
        self.assertGreater(audit['per_role_CAD_differences']['ring']['surface_maximum_absolute_difference_m'], 0.)
        self.assertLess(audit['per_role_CAD_differences']['ring']['surface_maximum_absolute_difference_m'], 1e-15)
        self.assertLess(max(row['maximum_position_difference_m'] for row in audit['rigid_layout_differences']), 1e-6)

    def test_parallel_source_plan_is_disjoint_and_aggregate_never_drops_zero_valid_cases(self):
        path = ROOT / 'scripts/prepare_hierarchy_motion_inputs_parallel_20261030.py'
        spec = importlib.util.spec_from_file_location('hierarchy_motion_owner_contract', path)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        rows = [dict(split='train', task='pick_moving_target_on_the_table', raw_episode_id=f'episode{i}')
                for i in range(9)]
        shards = owner.split_source_plan(rows, 4)
        flattened = [owner.source_key(row) for shard in shards for row in shard]
        self.assertEqual(set(flattened), {owner.source_key(row) for row in rows})
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaises(ValueError):
            owner.split_source_plan(rows + rows[:1], 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker_results = []
            for index, shard in enumerate(shards):
                folder = output / f'shard_{index:03d}'
                folder.mkdir()
                cases = [dict(row, state='complete', zero_valid_forecast_case=row['raw_episode_id'] == 'episode0')
                         for row in shard]
                (folder / 'manifest.json').write_text(json.dumps(dict(
                    stage='complete', complete=True, expected_cases=len(shard), cases=cases,
                    process=dict(pid=10+index, start_ticks=100+index), resources={})))
                worker_results.append(dict(shard=index, complete=True))
            complete = owner.aggregate(output, rows, {}, worker_results, time.perf_counter())
            self.assertTrue(complete['complete'])
            self.assertEqual(len(complete['cases']), 9)
            self.assertEqual(complete['zero_valid_cases'], 1)
            self.assertEqual(complete['unattempted_sources'], [])
            # Even if all shard status files claim complete, the owner requires
            # every planned source and every actual worker return.
            incomplete = owner.aggregate(output, rows, {}, worker_results[:-1], time.perf_counter())
            self.assertFalse(incomplete['complete'])
            pending_source = owner.aggregate(output, rows + [dict(rows[0], raw_episode_id='episode_missing')], {}, worker_results, time.perf_counter())
            self.assertFalse(pending_source['complete'])
            self.assertEqual(pending_source['unattempted_sources'][0]['raw_episode_id'], 'episode_missing')


if __name__ == '__main__':
    unittest.main()
