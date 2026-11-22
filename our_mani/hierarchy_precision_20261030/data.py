"""Hash-bound TRAIN/offline-validation inputs for the new precision heads.

No HDF5, simulator, TEST reader, outcome filter, or task-conditioned frontend is
used here. Expert commands are supervision (and explicit skill goal conditions),
never inputs to the shared scene encoder. The old constant .5-second cache field
is deliberately ignored: remaining times use a *nominal reconstructed* 50ms clock.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


VERSION = "hierarchy_precision_data_20261030_v1"
DATA_SCHEMA = "hierarchy_four_task_causal_scene_proprio_v1_20261030"
SKILL_VERSION = "hierarchy_episode_handoff_skill_pretraining_20261030_v2"
CANDIDATE_VERSION = "hierarchy_multibody_candidate_pretraining_20261030_v1"
TASK_NAMES = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed",
              "insert_onto_rotating_peg", "insert_onto_rotating_peg_high_speed")
ROLE_NAMES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
BASE = "artifacts/framework/multitask_r10/hierarchy_20261030"
DEFAULT_DATA_MANIFEST = BASE + "/data_v1/manifest.json"
DEFAULT_SKILL_CACHE = BASE + "/skill_handoff_pretrain_v2"
DEFAULT_CANDIDATE_CACHE = BASE + "/candidate_pretrain_v1"
DT = .05


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row):
    return row["split"], row["task"], row["raw_episode_id"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor_arrays(payload, field, shapes):
    arrays = payload.get(field, {})
    _require(isinstance(arrays, dict), "missing tensor arrays")
    n = None
    for name, shape in shapes.items():
        value = arrays.get(name)
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu", "missing CPU array: " + name)
        n = len(value) if n is None else n
        _require(tuple(value.shape) == (n, *shape), "invalid array shape: " + name)
        _require(bool(torch.isfinite(value).all()), "nonfinite array: " + name)
    _require(n is not None and n > 0, "empty cached dataset")
    return arrays


class PrecisionData:
    """One shared, read-only-in-practice in-memory store for all four datasets.

    ``identity['files']`` snapshots SHA256 of every file consumed. The legacy
    frozen-feature PT files have no historical checksums of their own: their new
    checksums are therefore captured here, while their labels, indices and robot
    inputs are independently checked against the hash-bound source caches.
    Supply ``expected_identity`` on later opens to require that exact snapshot.
    """
    def __init__(self, root, data_manifest=DEFAULT_DATA_MANIFEST,
                 skill_cache_dir=DEFAULT_SKILL_CACHE,
                 candidate_cache_dir=DEFAULT_CANDIDATE_CACHE, *, expected_identity=None):
        self.root = Path(root).resolve(strict=True)
        self._hashes = {}
        self._expected_files = None if expected_identity is None else expected_identity["files"]
        self.manifest_path = self._path(data_manifest)
        self.skill_dir, self.candidate_dir = self._path(skill_cache_dir), self._path(candidate_cache_dir)
        self.manifest = self._json(self.manifest_path)
        m = self.manifest
        _require(m.get("schema") == DATA_SCHEMA and m.get("complete") is True
                 and m.get("smoke_only") is False and m.get("TEST_read") is False,
                 "completed nonsmoke v1 TRAIN/validation manifest required")
        rows = m.get("episodes", [])
        _require(len(rows) == m.get("expected_episodes") == m.get("completed_episodes"), "incomplete episode manifest")
        _require(all(r.get("split") in ("train", "validation") and r.get("task") in TASK_NAMES for r in rows),
                 "only the four tasks and TRAIN/offline validation are permitted")
        _require(len({_key(row) for row in rows}) == len(rows), "duplicate source identity")
        _require(set(m.get("consumed_input_keys", [])) == {"rgb", "depth", "robot"}, "scene input boundary changed")
        for ref in m.get("source_manifests", {}).values():
            self._reference(ref)
        self.rows = {split: [r for r in rows if r["split"] == split] for split in ("train", "validation")}
        _require(all(set(r["task"] for r in values) == set(TASK_NAMES) for values in self.rows.values()),
                 "each offline split must preserve all four source tasks")
        self.skill_status = self._json(self.skill_dir / "status.json")
        self.candidate_status = self._json(self.candidate_dir / "status.json")
        s, c = self.skill_status, self.candidate_status
        _require(s.get("version") == SKILL_VERSION and s.get("complete") is True and s.get("smoke_only") is False,
                 "completed nonsmoke v2 skill cache required")
        _require(c.get("version") == CANDIDATE_VERSION and c.get("complete") is True and c.get("TEST_read") is False,
                 "completed v1 candidate cache required")
        self.skill_identity, self.candidate_identity = s["identity"], c["identity"]
        manifest_sha = self._hash(self.manifest_path)
        _require(s["identity"]["data_manifest"]["sha256"] == manifest_sha
                 and c["identity"]["data_manifest"]["sha256"] == manifest_sha
                 and c["identity"].get("frozen_skill_identity") == s["identity"], "cache source identities disagree")
        self._verify_references(s["identity"])
        self._verify_references(c["identity"])
        self._hash(self.skill_dir / "effective_config.json", s["identity"]["effective_config_sha256"])
        catalog = self._json(self.skill_dir / "training_sources.json", s["identity"]["training_sources_sha256"])
        _require(catalog.get("version") == SKILL_VERSION and catalog.get("episodes") == rows
                 and catalog.get("TEST_read") is False and catalog.get("expert_goal_is_supervision_only") is True,
                 "skill source catalog differs from manifest")
        self._hash(self.candidate_dir / "config.json", c["identity"]["config_sha256"])
        ledger = self._json(self.candidate_dir / "source_ledger.json", c["identity"]["source_ledger_sha256"])
        _require(isinstance(ledger, list) and len(ledger) == len(rows), "source ledger must preserve every source")
        self.ledger = {_key(row): row for row in ledger}
        _require(len(self.ledger) == len(rows) and set(self.ledger) == {_key(row) for row in rows}, "source ledger identity mismatch")
        skill_checkpoint = self._load(self.skill_dir / "best_skill.pt", s["best_skill_sha256"])
        candidate_checkpoint = self._load(self.candidate_dir / "best_candidate.pt", c["best_candidate_sha256"])
        _require(skill_checkpoint.get("version") == SKILL_VERSION
                 and skill_checkpoint.get("metadata", {}).get("identity") == s["identity"]
                 and skill_checkpoint["metadata"].get("stage") == "skill", "skill checkpoint identity mismatch")
        cm = candidate_checkpoint.get("metadata", {})
        _require(candidate_checkpoint.get("version") == CANDIDATE_VERSION
                 and cm.get("training_complete") is True and cm.get("identity") == c["identity"]
                 and cm.get("skill_checkpoint_sha256") == s["best_skill_sha256"]
                 and cm.get("role_names") == list(ROLE_NAMES), "trained candidate checkpoint identity mismatch")
        del skill_checkpoint, candidate_checkpoint
        self.scenes, self.skill, self.candidate, self.candidate_to_skill = {}, {}, {}, {}
        for split in ("train", "validation"):
            self._load_split(split)
        self.identity = {"version": VERSION, "files": dict(sorted(self._hashes.items())),
                         "timestamp_definition": "frame_index * 0.05 seconds; nominal reconstructed clock, not verified HDF5 timestamps",
                         "TEST_read": False, "success_filter": False}
        if expected_identity is not None:
            _require(self.identity == expected_identity, "precision data identity snapshot changed")
        self.summary = {split: {"source_count": len(self.rows[split]), "skill_samples": len(self.skill[split]["frame"]),
                               "candidate_samples": len(self.candidate[split]["labels"]["frame"])}
                        for split in self.rows}

    def _path(self, path):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        _require(path.is_relative_to(self.root), "input path must remain within the project")
        parts = path.relative_to(self.root).parts
        _require(not any(p.lower() == "test" or p.lower().startswith("test_")
                         or "official_test" in p.lower() or "test_package" in p.lower() for p in parts),
                 "TEST input is forbidden before opening files")
        return path

    def _hash(self, path, expected=None):
        path = self._path(path)
        name = str(path.relative_to(self.root))
        actual = _sha(path)
        prior = self._hashes.get(name)
        _require(prior is None or actual == prior, "input changed during loading: " + name)
        _require(expected is None or actual == expected, "file SHA256 mismatch: " + name)
        if self._expected_files is not None:
            _require(self._expected_files.get(name) == actual, "expected input SHA256 mismatch: " + name)
        self._hashes[name] = actual
        return actual

    def _json(self, path, expected=None):
        self._hash(path, expected)
        result = json.loads(self._path(path).read_text())
        self._hash(path, expected)
        return result

    def _load(self, path, expected=None):
        self._hash(path, expected)
        result = torch.load(self._path(path), map_location="cpu", weights_only=True)
        self._hash(path, expected)
        return result

    def _reference(self, reference):
        _require(isinstance(reference, dict) and "path" in reference and "sha256" in reference, "invalid file reference")
        return self._hash(reference["path"], reference["sha256"])

    def _verify_references(self, tree):
        if isinstance(tree, dict):
            if "path" in tree and "sha256" in tree:
                self._reference(tree)
            for key, value in tree.items():
                if key == "code_sha256":
                    for path, digest in value.items():
                        self._hash(path, digest)
                elif key.endswith("_path") and key[:-5] + "_sha256" in tree:
                    self._hash(value, tree[key[:-5] + "_sha256"])
                else:
                    self._verify_references(value)
        elif isinstance(tree, list):
            for value in tree:
                self._verify_references(value)

    def verify_identity(self):
        """Recheck every consumed file immediately before freezing/using a run."""
        for name, digest in self.identity["files"].items():
            self._hash(name, digest)
        return True

    def dataset(self, split="train", head="skill"):
        return PrecisionDataset(self, split=split, head=head)

    def _load_split(self, split):
        episodes = []
        for row in self.rows[split]:
            path = self._path(row["npz_path"])
            self._hash(path, row["cache_sha256"])
            with np.load(path, allow_pickle=False) as cache:
                a = {key: cache[key] for key in ("rgb", "depth", "robot", "expert_actions", "expert_command_run", "timestamps")}
            self._hash(path, row["cache_sha256"])
            n = row["frames"]
            _require(a["rgb"].shape == (n, 4, 64, 64, 3) and a["rgb"].dtype == np.uint8
                     and a["depth"].shape == (n, 4, 64, 64) and a["robot"].shape == (n, 15)
                     and a["expert_actions"].shape == (n, 8), "scene cache shapes changed")
            _require(all(np.isfinite(a[k]).all() for k in ("depth", "robot", "expert_actions"))
                     and bool((a["depth"] >= 0).all()), "invalid scene/action values")
            _require(np.array_equal(a["timestamps"], np.arange(n, dtype=np.float64) * DT), "nominal frame clock differs")
            closed = np.flatnonzero(a["expert_actions"][:, 7] < .5)
            _require(len(closed) > 0, "source has no first closed-gripper command")
            a["first_closed_frame"] = int(closed[0])
            a["handoff_goal_xyz"] = a["expert_actions"][closed[0], :3].copy()
            ledger = self.ledger[_key(row)]
            _require(ledger["scene_cache"] == {"path": row["npz_path"], "sha256": row["cache_sha256"]}
                     and ledger["parent_source_group"] == row["parent_source_group"]
                     and ledger["source_path"] == row["source_path"]
                     and ledger["source_frames"] == n
                     and ledger["first_closed_command_frame"] == int(closed[0])
                     and np.array_equal(ledger["handoff_goal_xyz"], a["handoff_goal_xyz"]), "ledger source/label identity mismatch")
            for key in ("source_small_arrays_sha256", "source_file_stat"):
                _require(ledger.get(key) == row.get(key), "ledger source provenance changed")
            for key in ("motion_cache", "motion_audit"):
                if ledger.get(key) is not None:
                    self._reference(ledger[key])
            if ledger.get("motion_cache") is not None:
                with np.load(self._path(ledger["motion_cache"]["path"]), allow_pickle=False) as motion:
                    _require(tuple(motion["role_names"].tolist()) == ROLE_NAMES, "motion role identity changed")
                    a["motion_trajectory"] = motion["predicted_trajectory"]
                    a["motion_mask"] = motion["trajectory_mask"]
                self._reference(ledger["motion_cache"])
            episodes.append(a)
        self.scenes[split] = episodes
        payload = self._load(self.skill_dir / f"{split}_frozen_features.pt")
        _require(payload.get("split") == split and payload.get("scene_input_uses_labels") is False,
                 "skill feature split/input boundary mismatch")
        skill = _tensor_arrays(payload, "arrays", {"scene_features": (128,), "skill_probabilities": (11,),
            "robot_current": (15,), "action": (8,), "goal_xyz": (3,), "weight": (), "episode": (), "frame": ()})
        mapping, expected = {}, set()
        for e, a in enumerate(episodes):
            expected.update((e, f) for f in range(len(a["robot"])))
        for i, (episode, frame) in enumerate(zip(skill["episode"].tolist(), skill["frame"].tolist())):
            _require(isinstance(episode, int) and isinstance(frame, int) and (episode, frame) in expected
                     and (episode, frame) not in mapping, "invalid/duplicate skill episode/frame mapping")
            mapping[episode, frame] = i
            a = episodes[episode]
            _require(torch.equal(skill["robot_current"][i], torch.from_numpy(a["robot"][frame]))
                     and torch.equal(skill["action"][i], torch.from_numpy(a["expert_actions"][frame]))
                     and torch.equal(skill["goal_xyz"][i], torch.from_numpy(a["handoff_goal_xyz"])),
                     "skill features/labels do not match their source frame")
        _require(set(mapping) == expected, "skill cache omitted source anchors")
        probabilities = skill["skill_probabilities"]
        _require(bool((probabilities >= 0).all()) and torch.allclose(probabilities.sum(-1), torch.ones(len(probabilities)), atol=1e-5)
                 and bool((skill["weight"] > 0).all()), "invalid skill probabilities/weights")
        self.skill[split] = skill
        ip = self._load(self.candidate_dir / f"{split}_candidate_inputs.pt")
        lp = self._load(self.candidate_dir / f"{split}_candidate_labels.pt")
        _require(ip.get("split") == lp.get("split") == split
                 and ip.get("identity") == lp.get("identity") == self.candidate_identity
                 and ip.get("goal_labels_consumed_by_encoder") is False, "candidate input/label identity mismatch")
        inputs = _tensor_arrays(ip, "input_arrays", {"scene_features": (128,), "skill_latent": (64,), "robot": (15,),
                                                     "trajectory": (6, 23, 20), "trajectory_mask": (6, 23)})
        labels = _tensor_arrays(lp, "supervision_arrays", {"target_xyz": (3,), "weight": (), "episode": (), "frame": ()})
        _require(len(inputs["robot"]) == len(labels["frame"]) and inputs["trajectory_mask"].dtype == torch.bool
                 and bool((labels["weight"] > 0).all()), "candidate row count/mask/weight mismatch")
        indices, seen = [], set()
        for i, (e, f) in enumerate(zip(labels["episode"].tolist(), labels["frame"].tolist())):
            _require(isinstance(e, int) and isinstance(f, int) and (e, f) in mapping and (e, f) not in seen,
                     "candidate episode/frame mapping mismatch")
            seen.add((e, f)); j = mapping[e, f]; indices.append(j)
            a, row = episodes[e], self.rows[split][e]
            ledger = self.ledger[_key(row)]
            decisions = [d for d in ledger["anchor_decisions"] if d["frame"] == f and d["eligible"]]
            _require(len(decisions) == 1 and f < a["first_closed_frame"] and bool(inputs["trajectory_mask"][i].any()),
                     "candidate anchor lacks positive pre-handoff ledger evidence")
            motion_row = decisions[0]["motion_row"]
            _require("motion_trajectory" in a and isinstance(motion_row, int)
                     and 0 <= motion_row < len(a["motion_trajectory"])
                     and torch.equal(inputs["trajectory"][i], torch.from_numpy(a["motion_trajectory"][motion_row]))
                     and torch.equal(inputs["trajectory_mask"][i], torch.from_numpy(a["motion_mask"][motion_row])),
                     "candidate trajectory differs from hash-bound public motion cache")
            _require(torch.equal(inputs["robot"][i], skill["robot_current"][j])
                     and torch.equal(labels["target_xyz"][i], torch.from_numpy(a["handoff_goal_xyz"]))
                     and torch.allclose(inputs["scene_features"][i], skill["scene_features"][j], atol=1e-3, rtol=1e-4),
                     "candidate feature/goal does not match mapped skill source")
        eligible = {(e, d["frame"]) for e, row in enumerate(self.rows[split])
                    for d in self.ledger[_key(row)]["anchor_decisions"] if d["eligible"]}
        _require(seen == eligible, "candidate cache omitted/added eligible source anchors")
        self.candidate[split] = {"inputs": inputs, "labels": labels}
        self.candidate_to_skill[split] = np.asarray(indices, dtype=np.int64)


class PrecisionDataset(Dataset):
    """Flat tensors; field selection by the trainer keeps labels out of encoders.

    Shared scene inputs: rgbd_current[4,4,64,64], frozen_scene[128],
    skill_probabilities[11], robot[15]. TCP conditions: goal_xyz[3], goal_time[].
    TCP labels: target_xyz[3], target_quaternion[4], target_gripper[]. Candidate
    inputs additionally contain trajectory[6,23,20], trajectory_mask[6,23]; its
    labels are target_xyz[3], target_time_seconds[]. Integer task_index and all
    indices/weights are bookkeeping only, never scene/model inputs.
    """
    def __init__(self, data, split="train", head="skill"):
        _require(split in ("train", "validation"), "only TRAIN/offline validation datasets permitted")
        _require(head in ("skill", "tcp", "candidate"), "head must be skill/tcp/candidate")
        self.data, self.split, self.head = data, split, "tcp" if head in ("skill", "tcp") else head
        self.task_names = TASK_NAMES
        self.rows = data.rows[split]
        self.source_count = len(self.rows)
        self.arrays = data.skill[split] if self.head == "tcp" else data.candidate[split]["labels"]
        self.weights = self.arrays["weight"].numpy().astype(np.float64, copy=True)
        self.sample_ids, self.expected_samples, times = [], [], []
        for e, f in zip(self.arrays["episode"].tolist(), self.arrays["frame"].tolist()):
            row = self.rows[e]
            sample_id = f"{row['task']}|{row['raw_episode_id']}|{f}"
            self.sample_ids.append(sample_id)
            self.expected_samples.append(dict(sample_id=sample_id, split=split, task=row["task"], head=self.head))
            times.append(max(data.scenes[split][e]["first_closed_frame"] - f, 0) * DT)
        self.remaining_seconds = np.asarray(times, dtype=np.float32)
        parents_with_samples = len(set(self.arrays["episode"].tolist()))
        self.summary = dict(split=split, head=self.head, samples=len(self), source_count=self.source_count,
            sources_with_samples=parents_with_samples, zero_sample_sources=self.source_count-parents_with_samples,
            task_sources=dict(Counter(r["task"] for r in self.rows)), success_filter=False, TEST_read=False,
            timestamp_definition="nominal frame index * .05 seconds; no verified original HDF5 timestamps",
            handoff_time_above_1_1_seconds=int(np.sum(self.remaining_seconds > 1.1 + 1e-6)),
            handoff_time_not_0_5_seconds=int(np.sum(~np.isclose(self.remaining_seconds, .5, atol=1e-7, rtol=0))),
            handoff_time_max_seconds=float(self.remaining_seconds.max(initial=0)), long_handoff_anchors_filtered=0)

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, index):
        e, f = int(self.arrays["episode"][index]), int(self.arrays["frame"][index])
        source = self.data.scenes[self.split][e]
        j = index if self.head == "tcp" else int(self.data.candidate_to_skill[self.split][index])
        skill = self.data.skill[self.split]
        rgbd = np.empty((4, 4, 64, 64), dtype=np.float32)
        rgbd[:, :3] = source["rgb"][f].transpose(0, 3, 1, 2).astype(np.float32) / 255. - .5
        rgbd[:, 3] = np.log1p(source["depth"][f])
        result = dict(rgbd_current=torch.from_numpy(rgbd), frozen_scene=skill["scene_features"][j],
            skill_probabilities=skill["skill_probabilities"][j], robot=skill["robot_current"][j],
            target_xyz=self.arrays["action"][index, :3] if self.head == "tcp" else self.arrays["target_xyz"][index],
            task_index=TASK_NAMES.index(self.rows[e]["task"]), weight=np.float32(self.weights[index]),
            sample_index=index, episode_index=e, frame_index=f)
        if self.head == "tcp":
            result.update(goal_xyz=torch.from_numpy(source["handoff_goal_xyz"].copy()),
                goal_time=self.remaining_seconds[index], handoff_remaining_seconds=self.remaining_seconds[index],
                target_quaternion=self.arrays["action"][index, 3:7], target_gripper=self.arrays["action"][index, 7])
        else:
            inputs = self.data.candidate[self.split]["inputs"]
            result.update(trajectory=inputs["trajectory"][index], trajectory_mask=inputs["trajectory_mask"][index],
                          target_time_seconds=self.remaining_seconds[index])
        return result
