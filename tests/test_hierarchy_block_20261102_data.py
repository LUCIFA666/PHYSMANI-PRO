"""Synthetic CPU data/calibration tests; no GPU, Native, or real TEST reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.hierarchy_block_20261102 import data as d


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, unknown_target=False):
    rows, cases = [], []
    for split in ("train", "validation"):
        for task in d.TASK_NAMES:
            n, close = 36, 29
            rgb = np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None, None, None], (n, 4, 64, 64, 3)).copy()
            depth = np.full((n, 4, 64, 64), 2, np.float32)
            robot = np.zeros((n, 15), np.float32)
            robot[:, 0] = np.arange(n) / 100
            robot[:, 6] = 1
            action = np.zeros((n, 8), np.float32)
            action[:, 0] = np.arange(n) / 100
            action[:, 2] = .75
            action[:, 6:] = 1
            action[close:, 7] = 0
            k = np.tile(np.array([[-200, 0, 128], [0, 220, 127], [0, 0, 1]], np.float64), (4, 1, 1))
            extrinsic = np.tile(np.eye(4), (4, 1, 1))
            extrinsic[:, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            extrinsic[:, :3, 3] = np.arange(4)[:, None] * np.array([.1, -.2, .3])
            scene = root / split / task / "episode1.npz"
            scene.parent.mkdir(parents=True, exist_ok=True)
            np.savez(scene, rgb=rgb, depth=depth, robot=robot, expert_actions=action,
                expert_command_run=np.arange(n), skill_labels=np.arange(n) % 4,
                timestamps=np.arange(n, dtype=np.float64) * d.DT, camera_intrinsics=k,
                camera_extrinsics=extrinsic, camera_near_far=np.tile([.01, 10], (4, 1)), camera_names=np.array(d.CAMERAS))
            row = dict(split=split, task=task, raw_episode_id="episode1", parent_source_group=f"{split}:{task}:episode1",
                source_path=f"/unopened-author-source/{split}/{task}/episode1.h5", npz_path=str(scene.relative_to(root)),
                cache_sha256=d.file_sha256(scene), frames=n)
            rows.append(row)
            trajectory = np.full((n, 6, 23, 20), np.nan, np.float32)
            mask = np.zeros((n, 6, 23), np.bool_)
            for f in range(4, n):
                role = 1 if f == 8 or (unknown_target and split == "train" and task == d.TASK_NAMES[0]) else 0
                if f == 6:
                    continue
                mask[f, role] = True
                trajectory[f, role] = 0
                trajectory[f, role, :, 0] = f / 100
                trajectory[f, role, :, -1] = np.arange(23) * d.DT
            mp, ap = scene.with_name("motion.npz"), scene.with_name("motion.json")
            np.savez(mp, role_names=np.array(d.ROLE_NAMES), timestamps=np.arange(n) * d.DT,
                     predicted_trajectory=trajectory, trajectory_mask=mask)
            frames = [dict(timestamp=f * d.DT, motion_status={role: dict(history_frames=min(f+1, 5),
                       available=bool(mask[f, r].all())) for r, role in enumerate(d.ROLE_NAMES)}) for f in range(n)]
            write_json(ap, dict(frames=frames))
            cases.append(dict(**{k: row[k] for k in ("split", "task", "raw_episode_id", "parent_source_group", "source_path")},
                npz_path=str(mp.relative_to(root)), sha256=d.file_sha256(mp), audit_path=str(ap.relative_to(root)), audit_sha256=d.file_sha256(ap),
                summary=dict(checkpoint_sha256=d.FORECAST_SHA256, original_gru_weights_unchanged=True,
                    expert_stage_or_target_inputs=False, native_instance_ids_used=False,
                    prior_receipt=dict(all_frontends_run_on_every_scene=True, task_id_or_description_input=False))))
    # These unrelated cache paths intentionally do not exist. Selection must
    # happen before any attempt to open other-task arrays.
    other = dict(split="train", task="insert_onto_rotating_peg", raw_episode_id="unread",
                 npz_path="FORBIDDEN_insert.npz", cache_sha256="bad", frames=1)
    rows.append(other)
    cases.append(other.copy())
    manifest = root / "manifest.json"
    write_json(manifest, dict(schema=d.DATA_SCHEMA, complete=True, smoke_only=False, TEST_read=False,
        source_debug_target_positions_read=False, consumed_input_keys=["rgb", "depth", "robot"], episodes=rows))
    write_json(root / "motion_manifest.json", dict(schema=d.MOTION_SCHEMA, complete=True, smoke_only=False,
        TEST_read=False, source_debug_target_positions_read=False, source_task_selects_frontend=False,
        frontend="blind_parallel_fixed_common_TRAIN_priors", identity=dict(data_manifest_sha256=d.file_sha256(manifest)), cases=cases))
    return root


class TestBlockData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = fixture(Path(self.temp.name))

    def load(self, **kwargs):
        # Production has no small/smoke membership escape hatch. The tiny test
        # changes only the in-memory constant, never production source files.
        with patch.dict(d.EXPECTED_COUNTS, {(s, t): 1 for s in ("train", "validation") for t in d.TASK_NAMES}, clear=True):
            return d.BlockData(self.root, "manifest.json", "motion_manifest.json", **kwargs)

    def test_signed_calibration_matches_tracking_at_actual_source_pixels(self):
        data = self.load()
        a = data.episodes["train"][0]["scene"]
        point, mask = d.geometry_points(a["rgb"][4], a["depth"][4], a["camera_intrinsics"], a["camera_extrinsics"])
        self.assertEqual(point.shape, (4096, 7))
        self.assertTrue(mask.all())
        pixels = np.arange(32) * 2 + 1
        source_pixels = d.resize_source_coordinates()
        np.testing.assert_array_equal(source_pixels, np.arange(64) * 4 + 2)
        selected = source_pixels[pixels]
        source_mask = np.zeros((256, 256), bool)
        source_mask[np.ix_(selected, selected)] = True
        for c in range(4):
            calibration = CameraCalibration(a["camera_intrinsics"][c], Pose(a["camera_extrinsics"][c, :3, 3], a["camera_extrinsics"][c, :3, :3]))
            expected = calibration.backproject(np.full((256, 256), 2.), source_mask)
            np.testing.assert_allclose(point[c*1024:(c+1)*1024, :3], expected, atol=1e-6)
        np.testing.assert_allclose(point[:, 3:6], 4 / 255 - .5, atol=1e-7)
        np.testing.assert_allclose(point[:, 6], np.log(3), atol=1e-7)
        self.assertEqual(data.manifest["geometry"]["world_units"], "meters")

    def test_geometry_unknown_depth_is_masked_zero(self):
        data = self.load()
        a = data.episodes["train"][0]["scene"]
        depth = a["depth"][0].copy()
        depth[0, 1, 1] = np.nan
        point, mask = d.geometry_points(a["rgb"][0], depth, a["camera_intrinsics"], a["camera_extrinsics"])
        self.assertFalse(mask[0])
        self.assertTrue(np.isfinite(point).all())
        np.testing.assert_array_equal(point[0], np.zeros(7))

    def test_unknown_nan_allowed_known_nan_rejected_and_target_qualification(self):
        data = self.load()
        ds = data.dataset("train", "candidate")
        first = ds[0]
        self.assertEqual(first["frame_index"], 4)
        self.assertTrue(np.isfinite(first["inputs"]["trajectory"]).all())
        np.testing.assert_array_equal(first["inputs"]["trajectory"][~first["inputs"]["trajectory_mask"]], 0)
        self.assertGreater(ds.summary["masked_nonfinite_values"], 0)
        self.assertNotIn((0, 6), ds.anchors)  # No real target forecast.
        self.assertNotIn((0, 8), ds.anchors)  # Other role known, target unknown.
        self.assertNotIn((0, 29), ds.anchors)  # At first close, not before it.
        self.assertIn((0, 5), ds.anchors)  # No inherited stride-two deletion.
        self.assertGreater(float(first["supervision"]["handoff_remaining_seconds"]), 1.1)
        motion = data.episodes["train"][0]["motion"]
        raw = motion["trajectory"].copy(); raw[4, 0, 0, 0] = np.nan
        arrays = dict(role_names=np.array(d.ROLE_NAMES), timestamps=np.arange(36)*d.DT,
                      predicted_trajectory=raw, trajectory_mask=motion["trajectory_mask"])
        case = data.cases[("train", d.TASK_NAMES[0], "episode1")]
        audit = json.loads((self.root / case["audit_path"]).read_text())
        with self.assertRaisesRegex(ValueError, "VALID"):
            d.validate_public_motion(arrays, audit, 36)

    def test_h4_ten_frame_spacing_and_explicit_tail_mask(self):
        ds = self.load().dataset("train", "skill")
        self.assertEqual(len(ds), 72)
        a = ds[4]["supervision"]
        np.testing.assert_array_equal(a["action_mask"], [True]*4)
        np.testing.assert_allclose(a["action_sequence"][:, 0], [.04, .14, .24, .34])
        tail = ds[35]["supervision"]
        np.testing.assert_array_equal(tail["action_mask"], [True, False, False, False])
        np.testing.assert_array_equal(tail["action_sequence"][1:], np.zeros((3, 8)))
        self.assertEqual(float(tail["handoff_remaining_seconds"]), 0)
        np.testing.assert_allclose(a["handoff_position"], [.29, 0, .75])
        np.testing.assert_array_equal(a["handoff_quaternion"], [0, 0, 0, 1])
        self.assertEqual(int(a["skill_label"]), 0)
        self.assertFalse(ds.summary["endpoint_padding_is_supervision"])

    def test_causal_prefix_and_labels_cannot_change_inputs(self):
        data = self.load(); ds = data.dataset("train", "skill")
        early = ds[0]
        self.assertEqual(early["inputs"]["lengths"], 1)
        np.testing.assert_array_equal(early["inputs"]["rgbd"][1:], 0)
        before = ds[4]["inputs"]
        a = data.episodes["train"][0]["scene"]
        a["rgb"][5:] = 255
        a["robot"][5:] = 99
        a["expert_actions"][:] = -123
        a["skill_labels"][:] = 3
        after = ds[4]["inputs"]
        self.assertEqual(set(before), {"rgbd", "robot_history", "lengths", "robot", "point_features", "point_mask"})
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertNotIn("task_index", before)
        self.assertNotIn("skill_label", before)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        self.assertEqual(tuple(batch["inputs"]["point_features"].shape), (2, 4096, 7))

    def test_exact_membership_selection_hashes_and_test_read_before_open(self):
        data = self.load()
        self.assertEqual(len(data.manifest["episodes"]), 4)
        self.assertFalse(any("FORBIDDEN" in p for p in data.identity["files"]))
        self.assertEqual({r["task"] for r in data.manifest["episodes"]}, set(d.TASK_NAMES))
        self.assertTrue(data.verify_identity())
        ds = data.dataset("validation", "candidate")
        self.assertEqual(ds.source_count, 2)
        self.assertEqual(ds.task_names, d.TASK_NAMES)
        self.assertTrue(ds.sample_ids[0].endswith("|episode1|4"))
        self.assertEqual(ds.expected_samples[0]["head"], "candidate")
        with self.assertRaisesRegex(ValueError, "exact two-task"):
            d.BlockData(self.root, "manifest.json", "motion_manifest.json")
        with self.assertRaisesRegex(ValueError, "TRAIN/offline"):
            data.dataset("test")
        with self.assertRaisesRegex(ValueError, "TEST input"):
            d.BlockData(self.root, "TEST/nonexistent.json", "motion_manifest.json")
        path = self.root / data.rows["train"][0]["npz_path"]
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA256|changed"):
            data.verify_identity()

    def test_zero_candidate_parent_retained_and_all_skill_frames_preserved(self):
        fixture(self.root, unknown_target=True)
        data = self.load()
        ds = data.dataset("train", "candidate")
        self.assertEqual(ds.source_count, 2)
        self.assertEqual(ds.summary["zero_sample_sources"], 1)
        self.assertEqual(data.dataset("train", "skill").source_count, 2)
        self.assertEqual(len(data.dataset("train", "skill")), 72)
        self.assertEqual(len([r for r in data.source_ledger if r["split"] == "train"]), 2)
        source = next(r for r in data.source_ledger if r["split"] == "train" and r["task"] == d.TASK_NAMES[0])
        self.assertTrue(source["zero_candidate_parent"])
        self.assertEqual(source["candidate_anchors"], 0)
        self.assertEqual(source["skill_anchors"], 36)


if __name__ == "__main__":
    unittest.main()
