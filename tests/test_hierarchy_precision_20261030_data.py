"""Synthetic cache tests only: no native execution or real TEST reads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from our_mani.hierarchy_precision_20261030.data import (
    CANDIDATE_VERSION, DATA_SCHEMA, ROLE_NAMES, SKILL_VERSION, TASK_NAMES,
    PrecisionData, PrecisionDataset,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def build_caches(tmp_path):
    skill, candidate = tmp_path / "skill", tmp_path / "candidate"
    skill.mkdir(); candidate.mkdir()
    rows, ledger, arrays = [], [], {}
    for split in ("train", "validation"):
        pieces = {k: [] for k in ("scene_features", "skill_probabilities", "robot_current", "action", "goal_xyz", "weight", "episode", "frame")}
        ci = {k: [] for k in ("scene_features", "skill_latent", "robot", "trajectory", "trajectory_mask")}
        cl = {k: [] for k in ("target_xyz", "weight", "episode", "frame")}
        for e, task in enumerate(TASK_NAMES):
            n, first_close = 30, 28
            robot = np.zeros((n, 15), np.float32)
            robot[:, 0] = np.arange(n) / 100
            robot[:, 6] = 1
            action = np.zeros((n, 8), np.float32)
            action[:, :3] = np.arange(n)[:, None] / 100 + e
            action[:, 6] = 1
            action[:, 7] = 1
            action[first_close:, 7] = 0
            rgb = np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None, None, None], (n, 4, 64, 64, 3)).copy()
            depth = np.full((n, 4, 64, 64), 2., np.float32)
            path = tmp_path / split / task / "episode1.npz"
            path.parent.mkdir(parents=True)
            np.savez(path, rgb=rgb, depth=depth, robot=robot, expert_actions=action,
                     expert_command_run=np.arange(n), timestamps=np.arange(n, dtype=np.float64) * .05)
            row = dict(split=split, task=task, raw_episode_id="episode1", parent_source_group=f"{split}:{task}:episode1",
                       source_path=f"/unopened-author-data/{split}/{task}/episode1.h5",
                       npz_path=str(path.relative_to(tmp_path)), cache_sha256=sha(path), frames=n)
            rows.append(row)
            trajectory = np.zeros((n, 6, 23, 20), np.float32)
            trajectory[:, :, :, 0] = np.arange(n)[:, None, None]
            mask = np.ones((n, 6, 23), dtype=np.bool_)
            mp, ap = path.with_name("motion.npz"), path.with_name("motion.json")
            np.savez(mp, role_names=np.array(ROLE_NAMES), predicted_trajectory=trajectory, trajectory_mask=mask)
            write_json(ap, {"public_only": True})
            decisions = [dict(frame=f, motion_row=f, eligible=True, weight=.5) for f in (4, 26)]
            ledger.append(dict(**{k: row[k] for k in ("split", "task", "raw_episode_id", "parent_source_group", "source_path")},
                scene_cache=dict(path=row["npz_path"], sha256=row["cache_sha256"]), source_frames=n,
                first_closed_command_frame=first_close, handoff_goal_xyz=action[first_close, :3].tolist(),
                motion_cache=dict(path=str(mp.relative_to(tmp_path)), sha256=sha(mp)),
                motion_audit=dict(path=str(ap.relative_to(tmp_path)), sha256=sha(ap)),
                anchor_decisions=decisions, valid_anchors=2, zero_weight_parent=False))
            for f in range(n):
                values = dict(scene_features=torch.full((128,), e + f / 100), skill_probabilities=torch.full((11,), 1 / 11),
                              robot_current=torch.from_numpy(robot[f]), action=torch.from_numpy(action[f]),
                              goal_xyz=torch.from_numpy(action[first_close, :3]), weight=torch.tensor(1.),
                              episode=torch.tensor(e), frame=torch.tensor(f))
                for k, v in values.items():
                    pieces[k].append(v)
            for f in (26, 4):  # Deliberately nonchronological candidate rows.
                for k, v in dict(scene_features=torch.full((128,), e + f / 100), skill_latent=torch.zeros(64),
                    robot=torch.from_numpy(robot[f]), trajectory=torch.from_numpy(trajectory[f]),
                    trajectory_mask=torch.from_numpy(mask[f])).items():
                    ci[k].append(v)
                for k, v in dict(target_xyz=torch.from_numpy(action[first_close, :3]), weight=torch.tensor(1.),
                                 episode=torch.tensor(e), frame=torch.tensor(f)).items():
                    cl[k].append(v)
        arrays[split] = ({k: torch.stack(v) for k, v in pieces.items()},
                         {k: torch.stack(v) for k, v in ci.items()}, {k: torch.stack(v) for k, v in cl.items()})
    manifest = dict(schema=DATA_SCHEMA, complete=True, smoke_only=False, TEST_read=False,
                    expected_episodes=8, completed_episodes=8, episodes=rows, consumed_input_keys=["rgb", "depth", "robot"])
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(skill / "effective_config.json", {})
    write_json(skill / "training_sources.json", dict(version=SKILL_VERSION, episodes=rows, TEST_read=False,
                                                     expert_goal_is_supervision_only=True))
    si = dict(data_manifest=dict(path="manifest.json", sha256=sha(manifest_path)),
              effective_config_sha256=sha(skill / "effective_config.json"),
              training_sources_sha256=sha(skill / "training_sources.json"))
    torch.save(dict(version=SKILL_VERSION, metadata=dict(identity=si, stage="skill")), skill / "best_skill.pt")
    ss = dict(version=SKILL_VERSION, complete=True, smoke_only=False, identity=si, best_skill_sha256=sha(skill / "best_skill.pt"))
    write_json(skill / "status.json", ss)
    write_json(candidate / "config.json", {})
    write_json(candidate / "source_ledger.json", ledger)
    identity = dict(data_manifest=si["data_manifest"], frozen_skill_identity=si,
        skill_checkpoint=dict(path="skill/best_skill.pt", sha256=ss["best_skill_sha256"]),
        skill_completed_status=dict(path="skill/status.json", sha256=sha(skill / "status.json")),
        source_ledger_sha256=sha(candidate / "source_ledger.json"), config_sha256=sha(candidate / "config.json"))
    cm = dict(identity=identity, training_complete=True, role_names=list(ROLE_NAMES), skill_checkpoint_sha256=ss["best_skill_sha256"])
    torch.save(dict(version=CANDIDATE_VERSION, metadata=cm), candidate / "best_candidate.pt")
    write_json(candidate / "status.json", dict(version=CANDIDATE_VERSION, complete=True, TEST_read=False,
                identity=identity, best_candidate_sha256=sha(candidate / "best_candidate.pt")))
    for split, (s, i, l) in arrays.items():
        # This obsolete cached value must not determine either new time target.
        s["goal_time"] = torch.full((len(s["frame"]),), .5)
        torch.save(dict(arrays=s, split=split, scene_input_uses_labels=False), skill / f"{split}_frozen_features.pt")
        torch.save(dict(input_arrays=i, identity=identity, split=split, goal_labels_consumed_by_encoder=False),
                   candidate / f"{split}_candidate_inputs.pt")
        torch.save(dict(supervision_arrays=l, identity=identity, split=split), candidate / f"{split}_candidate_labels.pt")
    return tmp_path


def load(root, **kwargs):
    return PrecisionData(root, "manifest.json", "skill", "candidate", **kwargs)


def test_current_pixels_labels_and_new_nominal_time(caches):
    data = load(caches)
    ds = PrecisionDataset(data, "train", "skill")
    sample = ds[4]
    assert sample["rgbd_current"].shape == (4, 4, 64, 64)
    assert torch.allclose(sample["rgbd_current"][:, :3], torch.full((4, 3, 64, 64), 4 / 255 - .5))
    assert torch.allclose(sample["rgbd_current"][:, 3], torch.full((4, 64, 64), np.log(3)))
    assert np.isclose(sample["goal_time"], 1.2)
    assert np.isclose(sample["handoff_remaining_seconds"], 1.2)
    assert np.allclose(sample["target_xyz"].tolist(), [.04] * 3)
    assert np.allclose(sample["goal_xyz"].tolist(), [.28] * 3)
    assert sample["target_quaternion"].tolist() == [0, 0, 0, 1]
    assert sample["target_gripper"].item() == 1
    assert ds[29]["goal_time"] == 0
    assert ds.task_names == TASK_NAMES and ds.source_count == 4 and len(ds) == 120
    assert ds.summary["handoff_time_above_1_1_seconds"] == 24
    assert np.isclose(ds.summary["handoff_time_max_seconds"], 1.4)
    assert "nominal" in data.identity["timestamp_definition"]
    assert ds.sample_ids[4] == TASK_NAMES[0] + "|episode1|4"
    assert ds.expected_samples[4]["head"] == "tcp"
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["sample_index"].tolist() == [0, 1]
    assert data.verify_identity()


def test_candidate_mapping_not_row_order_goal_not_scene_input(caches):
    data = load(caches)
    ds = data.dataset("train", "candidate")
    a, b = ds[0], ds[1]
    assert (a["frame_index"], b["frame_index"]) == (26, 4)
    assert np.isclose(a["frozen_scene"][0].item(), .26)
    assert np.isclose(a["robot"][0].item(), .26)
    assert np.isclose(a["target_time_seconds"], .1)
    assert np.isclose(b["target_time_seconds"], 1.2)  # Kept past the GRU 1.1s horizon.
    assert a["trajectory"].shape == (6, 23, 20) and a["trajectory_mask"].dtype == torch.bool
    assert torch.equal(a["trajectory"], torch.full((6, 23, 20), 0.).index_fill_(2, torch.tensor([0]), 26.))
    assert "goal_xyz" not in a and "goal_time" not in a and "skill_latent" not in a
    assert len(ds) == 8 and ds.source_count == 4 and ds.summary["long_handoff_anchors_filtered"] == 0
    assert data.dataset("train", "skill").data.scenes is ds.data.scenes


def test_hash_snapshot_rejects_changed_input(caches):
    data = load(caches)
    with (caches / "skill/train_frozen_features.pt").open("ab") as stream:
        stream.write(b"tamper")
    with unittest.TestCase().assertRaisesRegex(ValueError, "changed|SHA256"):
        data.verify_identity()
    with unittest.TestCase().assertRaisesRegex(ValueError, "SHA256"):
        load(caches, expected_identity=data.identity)


def test_mapped_robot_and_action_tamper_refused(caches):
    path = caches / "skill/train_frozen_features.pt"
    payload = torch.load(path, weights_only=True)
    payload["arrays"]["robot_current"][4, 0] += .1
    torch.save(payload, path)
    with unittest.TestCase().assertRaisesRegex(ValueError, "source frame"):
        load(caches)


def test_candidate_cached_trajectory_must_equal_bound_public_motion(caches):
    path = caches / "candidate/train_candidate_inputs.pt"
    payload = torch.load(path, weights_only=True)
    payload["input_arrays"]["trajectory"][0, 0, 0, 0] += 1
    torch.save(payload, path)
    with unittest.TestCase().assertRaisesRegex(ValueError, "public motion"):
        load(caches)


def test_checkpoint_readiness_and_identity_refused(caches):
    path = caches / "candidate/status.json"
    value = json.loads(path.read_text()); value["complete"] = False; write_json(path, value)
    with unittest.TestCase().assertRaisesRegex(ValueError, "completed v1 candidate"):
        load(caches)


def test_test_path_refused_before_open_and_no_future_pixels(caches):
    data = load(caches)
    with unittest.TestCase().assertRaisesRegex(ValueError, "only TRAIN"):
        data.dataset("test", "skill")
    with unittest.TestCase().assertRaisesRegex(ValueError, "TEST input"):
        PrecisionData(caches, "TEST/nonexistent.json", "skill", "candidate")
    ds = data.dataset("train", "tcp")
    before = ds[4]["rgbd_current"].clone()
    data.scenes["train"][0]["rgb"][5:] = 255
    assert torch.equal(ds[4]["rgbd_current"], before)


class TestPrecisionData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = build_caches(Path(self.temp.name))

    def test_current_pixels_labels_and_new_nominal_time(self):
        test_current_pixels_labels_and_new_nominal_time(self.root)

    def test_candidate_mapping_and_conditions(self):
        test_candidate_mapping_not_row_order_goal_not_scene_input(self.root)

    def test_hash_snapshot(self):
        test_hash_snapshot_rejects_changed_input(self.root)

    def test_mapped_robot(self):
        test_mapped_robot_and_action_tamper_refused(self.root)

    def test_public_trajectory(self):
        test_candidate_cached_trajectory_must_equal_bound_public_motion(self.root)

    def test_readiness(self):
        test_checkpoint_readiness_and_identity_refused(self.root)

    def test_test_rejected_and_causality(self):
        test_test_path_refused_before_open_and_no_future_pixels(self.root)


if __name__ == "__main__":
    unittest.main()
