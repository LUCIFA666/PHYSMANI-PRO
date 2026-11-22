"""Public RGBD/geometry and separated expert supervision for the block model.

Only the two block TRAIN/offline-VAL sources are opened. No HDF5, TEST data,
success labels, stage labels, old policy features or model weights are read.
All times below use the declared nominal 50ms frame clock, not measured HDF5
timestamps. Missing/unknown motion tokens may contain NaN on disk; valid tokens
must be finite, and unknown tokens become masked zeros before model consumption.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


VERSION = "hierarchy_block_public_data_20261102_v1"
DATA_SCHEMA = "hierarchy_four_task_causal_scene_proprio_v1_20261030"
MOTION_SCHEMA = "hierarchy_public_motion_inputs_20261030_v1"
TASK_NAMES = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed")
CAMERAS = ("front", "8", "16", "36")
ROLE_NAMES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
EXPECTED_COUNTS = {(split, task): count for split, count in (("train", 100), ("validation", 20)) for task in TASK_NAMES}
BASE = "artifacts/framework/multitask_r10/hierarchy_20261030"
DEFAULT_DATA_MANIFEST = BASE + "/data_v1/manifest.json"
DEFAULT_MOTION_MANIFEST = BASE + "/motion_inputs_blind_v1/manifest.json"
DT, HISTORY_FRAMES, ACTION_HORIZON, DECISION_FRAME_STRIDE = .05, 5, 4, 10
POINT_GRID_SIDE, POINT_COUNT = 32, 4096
FORECAST_SHA256 = "ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row):
    return row["split"], row["task"], row["raw_episode_id"]


def resize_source_coordinates(source_size=256, cached_size=64):
    """Exact integer source pixels chosen by the original Pillow NEAREST cache.

    In this frozen 256->64 cache they are 4*u+2. Using the actual Pillow mapping
    also documents the pixel-centre convention rather than guessing a scaled K.
    """
    pixels = np.arange(source_size, dtype=np.int32)[None, :]
    return np.asarray(Image.fromarray(pixels).resize((cached_size, 1), Image.Resampling.NEAREST))[0].astype(np.int64)


def resize_public_rgbd(rgb, depth):
    """Resize camera-ordered public RGB/metric-depth exactly as the frozen cache.

    Inputs are [4,H,W,3] uint8 and [4,H,W] metric float32, H=W=256
    (or already-cached 64). Returned arrays are [4,64,64,3]/[4,64,64].
    Intrinsics retain the original 256-pixel coordinate convention separately.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth, np.float32)
    _require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[0] == 4
             and rgb.shape[-1] == 3 and rgb.shape[1:3] in ((256, 256), (64, 64))
             and depth.shape == rgb.shape[:3], "four aligned public RGBD cameras required")
    if rgb.shape[1] == 64:
        return rgb.copy(), depth.copy()
    rgb64 = np.stack([np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR)) for frame in rgb])
    depth64 = np.stack([np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.NEAREST), np.float32) for frame in depth])
    return rgb64, depth64


def encode_public_history(rgb_history, depth_history, robot_history):
    """Shared offline/online causal prefix helper, using the latest <=5 frames.

    Arrays: RGB[L,4,64,64,3], metric depth[L,4,64,64], robot[L,15].
    Returns only public inputs: rgbd[5,4,4,64,64], robot_history[5,15],
    lengths (leading valid prefix), and current robot[15]. No labels are read.
    """
    rgb, depth, robot = np.asarray(rgb_history), np.asarray(depth_history, np.float32), np.asarray(robot_history, np.float32)
    n = len(robot)
    _require(n > 0 and rgb.shape == (n, 4, 64, 64, 3) and rgb.dtype == np.uint8
             and depth.shape == (n, 4, 64, 64) and robot.shape == (n, 15), "aligned nonempty public histories required")
    _require(np.isfinite(depth).all() and np.all(depth >= 0) and np.isfinite(robot).all(), "nonfinite public scene/proprioception")
    length = min(n, HISTORY_FRAMES)
    rgbd = np.zeros((HISTORY_FRAMES, 4, 4, 64, 64), np.float32)
    rgbd[:length, :, :3] = rgb[-length:].transpose(0, 1, 4, 2, 3).astype(np.float32) / 255. - .5
    rgbd[:length, :, 3] = np.log1p(depth[-length:])
    robots = np.zeros((HISTORY_FRAMES, 15), np.float32)
    robots[:length] = robot[-length:]
    return dict(rgbd=rgbd, robot_history=robots, lengths=length, robot=robot[-1].copy())


def geometry_points(rgb, depth, intrinsics, extrinsics):
    """Return 4096 public XYZ(m), RGB(/255-.5), log1p(depth(m)) tokens.

    Same signed-K camera-to-world convention as CameraCalibration.backproject:
    ray=K^-1[u256,v256,1], world=R*(depth*ray)+translation. No segmentation,
    privileged object identifier, goal, stage, task or speed is used.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    k, transform = np.asarray(intrinsics, np.float64), np.asarray(extrinsics, np.float64)
    _require(rgb.shape == (4, 64, 64, 3) and rgb.dtype == np.uint8 and depth.shape == (4, 64, 64),
             "geometry requires four cached 64x64 RGBD cameras")
    _require(k.shape == (4, 3, 3) and transform.shape == (4, 4, 4)
             and np.isfinite(k).all() and np.isfinite(transform).all()
             and np.all(np.abs(np.linalg.det(k)) > 1e-12)
             and np.allclose(transform[:, 3], [0, 0, 0, 1]), "invalid public calibration")
    samples = np.floor((np.arange(POINT_GRID_SIDE) + .5) * 64 / POINT_GRID_SIDE).astype(np.int64)
    vv, uu = np.meshgrid(samples, samples, indexing="ij")
    source_pixels = resize_source_coordinates()
    uv1 = np.stack((source_pixels[uu], source_pixels[vv], np.ones_like(uu)), -1).reshape(-1, 3)
    features = np.zeros((4, POINT_GRID_SIDE ** 2, 7), np.float32)
    masks = np.zeros((4, POINT_GRID_SIDE ** 2), np.bool_)
    for camera in range(4):
        z = depth[camera, vv, uu].reshape(-1).astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        rays = uv1 @ np.linalg.inv(k[camera]).T
        safe_z = np.where(valid, z, 0.)
        world = (rays * safe_z[:, None]) @ transform[camera, :3, :3].T + transform[camera, :3, 3]
        valid &= np.isfinite(world).all(-1)
        values = np.c_[world, rgb[camera, vv, uu].reshape(-1, 3).astype(np.float64) / 255. - .5, np.log1p(safe_z)]
        features[camera] = np.where(valid[:, None], values, 0).astype(np.float32)
        masks[camera] = valid
    return features.reshape(POINT_COUNT, 7), masks.reshape(POINT_COUNT)


def validate_public_motion(arrays, audit, source_frames):
    """Reject NaN in known tokens; accept and mask NaN in genuinely unknown ones."""
    _require(tuple(arrays["role_names"].tolist()) == ROLE_NAMES, "fixed public motion role order differs")
    times = np.asarray(arrays["timestamps"], np.float64)
    raw = np.asarray(arrays["predicted_trajectory"], np.float32)
    mask = np.asarray(arrays["trajectory_mask"])
    n = len(times)
    _require(raw.shape == (n, 6, 23, 20) and mask.shape == (n, 6, 23) and mask.dtype == np.bool_,
             "public forecast/mask shape differs")
    _require(np.isfinite(times).all() and np.all(times >= 0) and np.all(np.diff(times) > 0), "invalid public motion clock")
    frames = np.rint(times / DT).astype(np.int64)
    _require(np.all(frames < source_frames) and np.allclose(times, frames * DT, atol=1e-8, rtol=0),
             "public motion timestamps do not align with nominal source frames")
    _require(np.array_equal(mask.any(-1), mask.all(-1)), "a body forecast must contain all 23 steps or be unknown")
    _require(np.isfinite(raw[mask]).all(), "nonfinite VALID public forecast token")
    for row, body in np.argwhere(mask.all(-1)):
        _require(np.allclose(raw[row, body, :, -1], np.arange(23, dtype=np.float32) * DT, atol=1e-6, rtol=0),
                 "public forecast horizon clock changed")
    receipts = audit.get("frames", [])
    _require(len(receipts) == n, "public forecast causal audit length differs")
    histories = np.zeros((n, 6), np.int64)
    for row, receipt in enumerate(receipts):
        _require(abs(float(receipt["timestamp"]) - times[row]) <= 1e-8, "public forecast audit clock mismatch")
        statuses = receipt["motion_status"]
        _require(set(statuses) == set(ROLE_NAMES), "public forecast causal role audit missing")
        for body, role in enumerate(ROLE_NAMES):
            status = statuses[role]
            count, known = int(status["history_frames"]), bool(mask[row, body].all())
            _require(count >= 0 and bool(status["available"]) == known and (not known or count >= HISTORY_FRAMES),
                     "known public forecast lacks five causal body observations")
            histories[row, body] = count
    clean = np.where(mask[..., None], raw, np.float32(0)).astype(np.float32)
    return dict(trajectory=clean, trajectory_mask=mask.copy(), frames=frames, histories=histories,
                masked_nonfinite_values=int(np.sum(~np.isfinite(raw) & ~mask[..., None])),
                unknown_body_forecasts=int(np.sum(~mask.all(-1))))


class BlockData:
    """Load the exact 200 TRAIN + 40 offline-VAL block sources once into RAM."""
    def __init__(self, root, data_manifest=DEFAULT_DATA_MANIFEST,
                 motion_manifest=DEFAULT_MOTION_MANIFEST, *, expected_identity=None):
        self.root = Path(root).resolve(strict=True)
        self._files, self._expected = {}, None if expected_identity is None else expected_identity["files"]
        self.data_path, self.motion_path = self._path(data_manifest), self._path(motion_manifest)
        source = self._json(self.data_path)
        motion = self._json(self.motion_path)
        _require(source.get("schema") == DATA_SCHEMA and source.get("complete") is True
                 and source.get("smoke_only") is False and source.get("TEST_read") is False
                 and source.get("source_debug_target_positions_read") is False, "complete public v1 scene manifest required")
        _require(motion.get("schema") == MOTION_SCHEMA and motion.get("complete") is True
                 and motion.get("smoke_only") is False and motion.get("TEST_read") is False
                 and motion.get("source_debug_target_positions_read") is False
                 and motion.get("source_task_selects_frontend") is False
                 and motion.get("frontend") == "blind_parallel_fixed_common_TRAIN_priors"
                 and motion.get("identity", {}).get("data_manifest_sha256") == self._hash(self.data_path),
                 "completed source-blind v1 public motion manifest required")
        _require(set(source.get("consumed_input_keys", [])) == {"rgb", "depth", "robot"}, "public scene input contract changed")
        selected = [r for r in source["episodes"] if r.get("task") in TASK_NAMES]
        _require(all(r.get("split") in ("train", "validation") for r in selected), "TEST or other split selected")
        _require(dict(Counter((r["split"], r["task"]) for r in selected)) == EXPECTED_COUNTS,
                 "exact two-task TRAIN200/offline-VAL40 source membership required")
        _require(len({_key(r) for r in selected}) == len(selected), "duplicate block source identity")
        cases = [r for r in motion["cases"] if r.get("task") in TASK_NAMES]
        self.cases = {_key(r): r for r in cases}
        _require(len(self.cases) == len(cases) == len(selected) and set(self.cases) == {_key(r) for r in selected},
                 "block scene and motion source membership differs")
        self.rows = {s: [r for r in selected if r["split"] == s] for s in ("train", "validation")}
        self.episodes, self.source_ledger = {}, []
        self._scene_cache = {}
        for split, rows in self.rows.items():
            self.episodes[split] = [self._load_episode(row) for row in rows]
        self.manifest = dict(schema=VERSION, complete=True, episodes=selected,
            source_count=len(selected), tasks=list(TASK_NAMES), TEST_read=False,
            other_task_arrays_read=False, success_filter=False, public_input_keys=["rgb", "depth", "robot", "calibration", "public_motion"],
            timestamp_definition="frame_index * .05 seconds; nominal reconstructed clock, not verified original timestamps",
            geometry=dict(points=POINT_COUNT, dimensions=7, cameras=list(CAMERAS), world_units="meters",
                intrinsic_resolution=[256, 256], cached_resolution=[64, 64], source_pixel_mapping="u256=4*u64+2, v256=4*v64+2",
                signed_intrinsics_preserved=True, extrinsics="camera_to_world", pixel_grid="32x32 per camera, cached indices 1+2*i"),
            history_frames=HISTORY_FRAMES, action_horizon=ACTION_HORIZON,
            decision_frame_stride=DECISION_FRAME_STRIDE, endpoint_padding_is_supervision=False)
        self.identity = dict(version=VERSION, files=dict(sorted(self._files.items())), loader_source_sha256=file_sha256(__file__),
            selected_sources=[dict(split=r["split"], task=r["task"], raw_episode_id=r["raw_episode_id"],
                                   parent_source_group=r["parent_source_group"]) for r in selected],
            TEST_read=False, other_task_arrays_read=False, success_filter=False)
        if expected_identity is not None:
            _require(self.identity == expected_identity, "block data identity snapshot differs")
        self.summary = {s: dict(source_count=len(rows), frames=sum(r["frames"] for r in rows),
            task_sources=dict(Counter(r["task"] for r in rows))) for s, rows in self.rows.items()}

    def _path(self, path):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        _require(path.is_relative_to(self.root), "input must stay within project")
        _require(not any(p.lower() == "test" or p.lower().startswith("test_") or "official_test" in p.lower()
                         or "test_package" in p.lower() for p in path.relative_to(self.root).parts), "TEST input forbidden before opening")
        return path

    def _hash(self, path, expected=None):
        path = self._path(path)
        name, actual = str(path.relative_to(self.root)), file_sha256(path)
        _require(expected is None or actual == expected, "file SHA256 mismatch: " + name)
        _require(name not in self._files or self._files[name] == actual, "input changed during loading: " + name)
        _require(self._expected is None or self._expected.get(name) == actual, "expected input SHA256 mismatch: " + name)
        self._files[name] = actual
        return actual

    def _json(self, path, expected=None):
        self._hash(path, expected)
        result = json.loads(self._path(path).read_text())
        self._hash(path, expected)
        return result

    def verify_identity(self):
        for path, digest in self.identity["files"].items():
            self._hash(path, digest)
        _require(file_sha256(__file__) == self.identity["loader_source_sha256"], "block loader changed")
        return True

    def dataset(self, split="train", head="skill"):
        return BlockDataset(self, split=split, head=head)

    def _load_episode(self, row):
        path = self._path(row["npz_path"])
        self._hash(path, row["cache_sha256"])
        if row["npz_path"] not in self._scene_cache:
            with np.load(path, allow_pickle=False) as cache:
                a = {k: cache[k] for k in ("rgb", "depth", "robot", "expert_actions", "expert_command_run", "skill_labels", "timestamps",
                                          "camera_intrinsics", "camera_extrinsics", "camera_near_far", "camera_names")}
            self._scene_cache[row["npz_path"]] = a
        else:
            a = self._scene_cache[row["npz_path"]]
        self._hash(path, row["cache_sha256"])
        n = row["frames"]
        _require(a["rgb"].shape == (n, 4, 64, 64, 3) and a["rgb"].dtype == np.uint8
                 and a["depth"].shape == (n, 4, 64, 64) and a["depth"].dtype == np.float32
                 and a["robot"].shape == (n, 15) and a["expert_actions"].shape == (n, 8), "block scene shapes differ")
        _require(np.isfinite(a["robot"]).all() and np.isfinite(a["expert_actions"]).all(), "invalid public robot/action arrays")
        _require(a["skill_labels"].shape == (n,) and np.all((a["skill_labels"] >= 0) & (a["skill_labels"] < 4)),
                 "only the four pick skill labels are permitted as supervision")
        _require(np.isfinite(a["depth"]).all() and np.all(a["depth"] > 0), "cached public metric depth must be positive finite")
        _require(np.array_equal(a["timestamps"], np.arange(n, dtype=np.float64) * DT), "nominal scene clock differs")
        _require(tuple(a["camera_names"].tolist()) == CAMERAS and a["camera_intrinsics"].shape == (4, 3, 3)
                 and a["camera_extrinsics"].shape == (4, 4, 4) and a["camera_near_far"].shape == (4, 2), "public calibration fields differ")
        geometry_points(a["rgb"][0], a["depth"][0], a["camera_intrinsics"], a["camera_extrinsics"])
        starts = np.r_[True, np.any(a["expert_actions"][1:] != a["expert_actions"][:-1], axis=1)]
        _require(np.array_equal(a["expert_command_run"], np.cumsum(starts) - 1), "expert command run metadata changed")
        closed = np.flatnonzero(a["expert_actions"][:, 7] < .5)
        _require(len(closed) > 0, "source has no first expert closed command; no handoff replacement allowed")
        close = int(closed[0])
        case = self.cases[_key(row)]
        _require(case.get("parent_source_group") == row["parent_source_group"] and case.get("source_path") == row["source_path"],
                 "public motion and scene parent provenance differ")
        failure = case.get("error") or case.get("infrastructure_error") or case.get("failed")
        if not case.get("npz_path"):
            _require(bool(failure), "missing motion cache without explicit failure evidence")
            motion = dict(trajectory=np.empty((0, 6, 23, 20), np.float32), trajectory_mask=np.empty((0, 6, 23), np.bool_),
                          frames=np.empty(0, np.int64), histories=np.empty((0, 6), np.int64), masked_nonfinite_values=0,
                          unknown_body_forecasts=0)
        else:
            summary = case.get("summary", {})
            _require(summary.get("checkpoint_sha256") == FORECAST_SHA256
                     and summary.get("original_gru_weights_unchanged") is True
                     and summary.get("expert_stage_or_target_inputs") is False
                     and summary.get("native_instance_ids_used") is False
                     and summary.get("prior_receipt", {}).get("all_frontends_run_on_every_scene") is True
                     and summary.get("prior_receipt", {}).get("task_id_or_description_input") is False,
                     "motion case lacks source-blind original-GRU public-input proof")
            mp = self._path(case["npz_path"])
            self._hash(mp, case["sha256"])
            audit = self._json(case["audit_path"], case["audit_sha256"])
            with np.load(mp, allow_pickle=False) as cache:
                arrays = {k: cache[k] for k in ("timestamps", "role_names", "predicted_trajectory", "trajectory_mask")}
            self._hash(mp, case["sha256"])
            motion = validate_public_motion(arrays, audit, n)
        lookup = {int(frame): i for i, frame in enumerate(motion["frames"])}
        decisions = []
        for frame in range(n):
            m = lookup.get(frame)
            if frame >= close:
                reason = "at_or_after_first_expert_close"
            elif frame < HISTORY_FRAMES - 1:
                reason = "fewer_than_five_scene_frames"
            elif m is None:
                reason = "documented_motion_failure" if failure else "motion_frame_missing"
            elif not (motion["trajectory_mask"][m, 0].all() and motion["histories"][m, 0] >= HISTORY_FRAMES):
                reason = "target_forecast_unknown"
            else:
                reason = None
            decisions.append(dict(frame=frame, motion_row=m, eligible=reason is None, zero_weight_reason=reason))
        eligible = [d["frame"] for d in decisions if d["eligible"]]
        self.source_ledger.append(dict(split=row["split"], task=row["task"], raw_episode_id=row["raw_episode_id"],
            parent_source_group=row["parent_source_group"], source_frames=n, first_closed_command_frame=close,
            skill_anchors=n, candidate_anchors=len(eligible), zero_candidate_parent=not bool(eligible),
            zero_weight_reasons=dict(Counter(d["zero_weight_reason"] for d in decisions if d["zero_weight_reason"])),
            anchor_decisions=decisions, masked_nonfinite_values=motion["masked_nonfinite_values"],
            documented_motion_failure=str(failure) if failure else None))
        return dict(scene=a, first_close=close, handoff_position=a["expert_actions"][close, :3].copy(),
                    handoff_quaternion=a["expert_actions"][close, 3:7].copy(), motion=motion,
                    motion_lookup=lookup, candidate_frames=eligible)


class BlockDataset(Dataset):
    """Explicit ``inputs`` and ``supervision`` groups; indices/weights outside.

    Candidate requires a current real *target* forecast and five causal frames.
    Skill uses every source frame, including episode tails. Action sequence row
    h is the expert command at frame+10*h, with an explicit false mask if absent.
    Handoff pose/time are supervision only; a trainer may pass them explicitly as
    low-level policy goal conditions, never to the scene/geometry encoder.
    """
    def __init__(self, data, split="train", head="skill"):
        _require(split in ("train", "validation"), "only block TRAIN/offline validation permitted")
        _require(head in ("skill", "candidate"), "head must be skill or candidate")
        self.data, self.split, self.head = data, split, head
        self.rows, self.task_names = data.rows[split], TASK_NAMES
        self.source_count = len(self.rows)
        self.anchors, weights = [], []
        task_parents = Counter(row["task"] for row in self.rows)
        for e, (row, episode) in enumerate(zip(self.rows, data.episodes[split])):
            a = episode["scene"]
            frames = range(len(a["robot"])) if head == "skill" else episode["candidate_frames"]
            if head == "skill":
                runs, counts = np.unique(a["expert_command_run"], return_counts=True)
                raw = [1 / (task_parents[row["task"]] * len(runs) * counts[a["expert_command_run"][f]]) for f in frames]
            else:
                raw = [1 / len(frames)] * len(frames) if len(frames) else []
            self.anchors.extend((e, int(f)) for f in frames)
            weights.extend(raw)
        self.weights = np.asarray(weights, np.float64)
        if len(self.weights):
            self.weights *= len(self.weights) / self.weights.sum()
        self.sample_ids = [f"{self.rows[e]['task']}|{self.rows[e]['raw_episode_id']}|{f}" for e, f in self.anchors]
        self.expected_samples = [dict(sample_id=sample_id, split=split, task=self.rows[e]["task"], head=head)
                                 for sample_id, (e, _) in zip(self.sample_ids, self.anchors)]
        times = np.asarray([max(data.episodes[split][e]["first_close"] - f, 0) * DT for e, f in self.anchors])
        self.summary = dict(split=split, head=head, source_count=self.source_count, samples=len(self),
            sources_with_samples=len({e for e, _ in self.anchors}), zero_sample_sources=self.source_count-len({e for e, _ in self.anchors}),
            task_sources=dict(task_parents), TEST_read=False, other_task_arrays_read=False, success_filter=False,
            history_frames=HISTORY_FRAMES, action_horizon=ACTION_HORIZON, decision_frame_stride=DECISION_FRAME_STRIDE,
            endpoint_padding_is_supervision=False, handoff_time_above_1_1_seconds=int(np.sum(times > 1.1 + 1e-8)),
            handoff_time_not_0_5_seconds=int(np.sum(~np.isclose(times, .5, atol=1e-8, rtol=0))),
            handoff_time_max_seconds=float(times.max(initial=0)), long_handoff_anchors_filtered=0,
            timestamp_definition="nominal reconstructed frame_index * .05 seconds",
            candidate_requires_current_target_forecast=True,
            masked_nonfinite_values=sum(e["motion"]["masked_nonfinite_values"] for e in data.episodes[split]))

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        e, frame = self.anchors[index]
        episode = self.data.episodes[self.split][e]
        a = episode["scene"]
        start = max(0, frame-HISTORY_FRAMES+1)
        inputs = encode_public_history(a["rgb"][start:frame+1], a["depth"][start:frame+1], a["robot"][start:frame+1])
        points, mask = geometry_points(a["rgb"][frame], a["depth"][frame], a["camera_intrinsics"], a["camera_extrinsics"])
        inputs.update(point_features=points, point_mask=mask)
        if self.head == "candidate":
            m = episode["motion_lookup"][frame]
            inputs.update(trajectory=episode["motion"]["trajectory"][m].copy(),
                          trajectory_mask=episode["motion"]["trajectory_mask"][m].copy())
        future = frame + np.arange(ACTION_HORIZON) * DECISION_FRAME_STRIDE
        valid = future < len(a["expert_actions"])
        commands = np.zeros((ACTION_HORIZON, 8), np.float32)
        commands[valid] = a["expert_actions"][future[valid]]
        supervision = dict(action_sequence=commands, action_mask=valid,
            handoff_position=episode["handoff_position"].copy(), handoff_quaternion=episode["handoff_quaternion"].copy(),
            handoff_remaining_seconds=np.float32(max(episode["first_close"]-frame, 0) * DT),
            skill_label=np.int64(a["skill_labels"][frame]),
            task_index=TASK_NAMES.index(self.rows[e]["task"]))
        return dict(inputs=inputs, supervision=supervision, weight=np.float32(self.weights[index]),
                    sample_index=index, episode_index=e, frame_index=frame)


# Descriptive aliases for callers that keep multiple model data stores in scope.
BlockManipulationData = BlockData
BlockManipulationDataset = BlockDataset
