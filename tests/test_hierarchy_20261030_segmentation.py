import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.hierarchy_20261030.segmentation import (
    CAMERAS, PointSemanticMLP, balanced_pixel_indices, fit_feature_normalization,
    load_public_inputs, load_training_frame, pixel_features, predict_input_arrays,
    public_observation_inputs,
)


class PublicSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [.2, -.3, .8]
        intrinsic = np.array([[-2., 0., .5], [0., 3., .25], [0., 0., 1.]])
        return dict(rgb=np.full((4, 2, 3, 3), 128, dtype=np.uint8),
                    depth_m=np.full((4, 2, 3), 1.5, dtype=np.float32),
                    camera_intrinsics=np.repeat(intrinsic[None], 4, axis=0),
                    camera_to_world=np.repeat(transform[None], 4, axis=0),
                    camera_names=np.asarray(CAMERAS), timestamp=np.asarray(.25),
                    counted_tick=np.asarray(5))

    def test_backprojection_matches_existing_signed_camera_contract(self):
        inputs = self.inputs()
        features, valid = pixel_features(inputs)
        self.assertEqual(features.shape, (4, 2, 3, 7))
        self.assertTrue(valid.all())
        for index in range(4):
            matrix = inputs["camera_to_world"][index]
            calibration = CameraCalibration(inputs["camera_intrinsics"][index],
                                             Pose(matrix[:3, 3], matrix[:3, :3]))
            reference = calibration.backproject(inputs["depth_m"][index], np.ones((2, 3), dtype=bool))
            np.testing.assert_allclose(features[index, ..., 4:].reshape(-1, 3), reference, atol=1e-7)
        np.testing.assert_allclose(features[..., :3], 128 / 255., atol=1e-7)
        np.testing.assert_allclose(features[..., 3], np.log(1.5), atol=1e-7)

    def test_privileged_arrays_are_rejected_and_invalid_depth_stays_unknown(self):
        inputs = self.inputs()
        with self.assertRaises(ValueError):
            pixel_features({**inputs, "semantic_role_ids": np.zeros((4, 2, 3))})
        inputs["depth_m"][0, 0, 0] = np.nan
        inputs["depth_m"][1, 0, 0] = 0.
        model = PointSemanticMLP()
        for value in model.parameters():
            torch.nn.init.zeros_(value)
        result = predict_input_arrays(model, inputs)
        for camera in CAMERAS:
            self.assertTrue((result["semantic_maps"][camera] == -1).all())
        self.assertEqual(result["probabilities"]["front"][0, 0].sum(), 0.)
        self.assertFalse(result["quality"]["unknown_filled_from_native_labels"])
        self.assertTrue(all(result["quality"]["sparse_roles"].values()))

    def test_input_label_files_are_separate_and_label_values_do_not_change_features(self):
        inputs = self.inputs()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "inputs").mkdir(); (base / "labels").mkdir()
            input_path, label_path = base / "inputs/frame.npz", base / "labels/frame.npz"
            np.savez(input_path, **inputs)
            labels = dict(semantic_role_ids=np.zeros((4, 2, 3), dtype=np.uint8),
                          timestamp=inputs["timestamp"], counted_tick=inputs["counted_tick"])
            np.savez(label_path, **labels)
            first, mask, target = load_training_frame(input_path, label_path)
            self.assertTrue((target == 0).all())
            labels["semantic_role_ids"].fill(5)
            np.savez(label_path, **labels)
            second, other_mask, other_target = load_training_frame(input_path, label_path)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(mask, other_mask)
            self.assertTrue((other_target == 5).all())
            with self.assertRaises(ValueError):
                load_public_inputs(label_path)
            with self.assertRaises(ValueError):
                load_training_frame(input_path, input_path)

    def test_class_balancing_and_fit_only_normalization(self):
        labels = np.repeat(np.arange(6), np.arange(1, 7))
        indices = balanced_pixel_indices(labels, np.random.default_rng(20261030), 20)
        np.testing.assert_array_equal(np.bincount(labels[indices], minlength=6), np.full(6, 20))
        features = np.arange(140, dtype=np.float32).reshape(20, 7)
        mean, std = fit_feature_normalization(features)
        np.testing.assert_allclose(mean, features.mean(0))
        np.testing.assert_allclose(std, features.std(0), rtol=1e-6)
        model = PointSemanticMLP(16, mean, std)
        logits = model(torch.from_numpy(features))
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(20) % 6)
        loss.backward()
        self.assertEqual(logits.shape, (20, 6))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_observation_extraction_ignores_native_masks_and_targets(self):
        inputs = self.inputs()
        calibration = {}
        for index, camera in enumerate(CAMERAS):
            matrix = inputs["camera_to_world"][index]
            calibration[camera] = CameraCalibration(inputs["camera_intrinsics"][index],
                                                     Pose(matrix[:3, 3], matrix[:3, :3]))
        observation = {
            "observation/images": dict(zip(CAMERAS, inputs["rgb"])),
            "observation/depths": dict(zip(CAMERAS, inputs["depth_m"])),
            "observation/calibrations": calibration,
            "observation/instance_masks": {name: np.zeros((2, 3)) for name in CAMERAS},
            "privileged_target_position": np.ones(3), "task_id": 99, "timestamp": .25,
        }
        first = public_observation_inputs(observation)
        observation["observation/instance_masks"] = None
        observation["privileged_target_position"] = np.full(3, np.nan)
        observation["task_id"] = -1
        second = public_observation_inputs(observation)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertNotIn("task_id", first)
        self.assertNotIn("semantic_role_ids", first)


if __name__ == "__main__":
    unittest.main()
