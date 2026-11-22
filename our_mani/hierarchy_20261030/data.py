"""Causal scene/proprioception caches with explicitly separate expert labels.

This module reads only the exact four-task TRAIN/validation protocol sources.
The HDF5 world-model predictions and privileged debug target positions are never
opened. No simulator, torch model, or accelerator is used during preparation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image


SCHEMA = "hierarchy_four_task_causal_scene_proprio_v1_20261030"
CAMERAS = ("front", "8", "16", "36")
DT = 0.05
IMAGE_SIZE = 64
TASK_FAMILIES = {
    "pick_moving_target_on_the_table": 0,
    "pick_moving_target_on_the_table_high_speed": 0,
    "insert_onto_rotating_peg": 1,
    "insert_onto_rotating_peg_high_speed": 1,
}
SKILL_IDS = {
    "pick_moving_target_on_the_table": {
        "reach": 0, "pre-grasp": 1, "grasp": 2, "lift": 3,
    },
    "insert_onto_rotating_peg": {
        "wp0": 4, "wp1": 5, "wp1-grasp": 6, "wp2": 7,
        "wp3": 8, "wp4": 9, "wp4-open": 10,
    },
}
INPUT_KEYS = ("rgb", "depth", "robot")
LABEL_KEYS = ("expert_actions", "task_labels", "skill_labels")
METADATA_KEYS = (
    "timestamps", "expert_command_run", "camera_intrinsics",
    "camera_extrinsics", "camera_near_far", "camera_names",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps(
            {"key": key, "shape": list(value.shape), "dtype": value.dtype.str},
            sort_keys=True, separators=(",", ":"),
        ).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_stat(path):
    value = Path(path).stat()
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns,
                inode=value.st_ino, device=value.st_dev)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def decode_metric_depth(encoded, near, far):
    """Decode the author's RGB24 linear depth before any resizing."""
    encoded = np.asarray(encoded)
    if encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3:
        raise ValueError("expected HxWx3 uint8 RGB24 depth")
    if not (np.isfinite(near) and np.isfinite(far) and 0 <= near < far):
        raise ValueError("invalid public camera clipping planes")
    values = encoded.astype(np.uint32)
    normalized = ((values[..., 0] << 16) | (values[..., 1] << 8)
                  | values[..., 2]).astype(np.float64) / (2**24 - 1)
    return (float(near) + normalized * (float(far) - float(near))).astype(np.float32)


def validate_episode_arrays(arrays):
    expected = set(INPUT_KEYS + LABEL_KEYS + METADATA_KEYS)
    if set(arrays) != expected:
        raise ValueError(f"episode keys differ: {set(arrays) ^ expected}")
    n = len(arrays["robot"])
    if n < 1:
        raise ValueError("empty episode")
    shapes = {
        "rgb": (n, 4, IMAGE_SIZE, IMAGE_SIZE, 3),
        "depth": (n, 4, IMAGE_SIZE, IMAGE_SIZE), "robot": (n, 15),
        "expert_actions": (n, 8), "task_labels": (n,), "skill_labels": (n,),
        "timestamps": (n,), "expert_command_run": (n,),
        "camera_intrinsics": (4, 3, 3), "camera_extrinsics": (4, 4, 4),
        "camera_near_far": (4, 2), "camera_names": (4,),
    }
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"{key} shape differs: {value.shape} != {shape}")
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite {key}")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError("RGB must remain uint8")
    for key in ("depth", "robot", "expert_actions"):
        if arrays[key].dtype != np.float32:
            raise ValueError(f"{key} must be float32")
    if not np.all(arrays["depth"] > 0):
        raise ValueError("metric depth must be positive")
    if np.any((arrays["task_labels"] < 0) | (arrays["task_labels"] > 1)):
        raise ValueError("invalid task labels")
    if np.any((arrays["skill_labels"] < 0) | (arrays["skill_labels"] > 10)):
        raise ValueError("invalid skill labels")
    if not np.array_equal(arrays["timestamps"], np.arange(n, dtype=np.float64) * DT):
        raise ValueError("timestamps differ from declared nominal clock")
    starts = np.r_[True, np.any(arrays["expert_actions"][1:] !=
                              arrays["expert_actions"][:-1], axis=1)]
    if not np.array_equal(arrays["expert_command_run"], np.cumsum(starts) - 1):
        raise ValueError("command-run deduplication differs")
    if tuple(arrays["camera_names"].tolist()) != CAMERAS:
        raise ValueError("camera order differs")
    if not np.allclose(arrays["camera_extrinsics"][:, 3], [0, 0, 0, 1]):
        raise ValueError("invalid camera extrinsic homogeneous row")
    return n


def load_episode_inputs(path):
    """Return only current/past scene and proprioception, never expert labels."""
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in INPUT_KEYS}


def load_episode_labels(path):
    """Offline supervision; callers must keep these out of model inputs."""
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in LABEL_KEYS}


def prepare_episode(row, split, *, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if split not in ("train", "validation") or row["task"] not in TASK_FAMILIES:
        raise ValueError("only the declared four TRAIN/validation tasks are allowed")
    task, episode = row["task"], row["raw_episode_id"]
    if not episode or Path(episode).name != episode:
        raise ValueError("invalid episode identity")
    source_path = Path(row["path"]).resolve()
    package_name = "train_package_compressed" if split == "train" else "val_package_compressed"
    expected_root = Path("/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench") / package_name
    if not source_path.is_relative_to(expected_root.resolve()) or source_path.name != "data.h5":
        raise ValueError("source path is outside the exact TRAIN/validation package")
    destination = output / split / task / (episode + ".npz")
    if destination.exists():
        raise FileExistsError(f"preserve existing cache: {destination}")
    before = source_stat(source_path)
    started = time.perf_counter()
    with h5py.File(source_path, "r") as source:
        if str(source.attrs["task"]) != task:
            raise ValueError("source task attribute differs")
        n = int(source.attrs["num_frames"])
        raw = {key: source[f"low_dim_obs/{key}"][()] for key in (
            "expert_trajectory", "gripper_pose", "gripper_open", "joint_positions",
        )}
        stage_values = source["low_dim_obs/expert_trajectory_stage"][()].reshape(-1)
        stages = np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v)
                             for v in stage_values])
        if (raw["expert_trajectory"].shape != (n, 1, 8)
                or raw["gripper_pose"].shape != (n, 7)
                or raw["gripper_open"].shape != (n,)
                or raw["joint_positions"].shape != (n, 7)
                or stages.shape != (n,)):
            raise ValueError("source low-dimensional array shapes differ")
        if any(not np.isfinite(v).all() for v in raw.values()):
            raise ValueError("nonfinite source low-dimensional arrays")
        family = task.removesuffix("_high_speed")
        mapping = SKILL_IDS[family]
        unknown = set(stages.tolist()) - set(mapping)
        if unknown:
            raise ValueError(f"unmapped source stages: {unknown}")
        actions = raw["expert_trajectory"][:, 0].astype(np.float32)
        arrays = {
            "rgb": np.empty((n, 4, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "depth": np.empty((n, 4, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
            "robot": np.column_stack((raw["gripper_pose"], raw["gripper_open"],
                                      raw["joint_positions"])).astype(np.float32),
            "expert_actions": actions,
            "task_labels": np.full(n, TASK_FAMILIES[task], dtype=np.int64),
            "skill_labels": np.asarray([mapping[v] for v in stages], dtype=np.int64),
            "timestamps": np.arange(n, dtype=np.float64) * DT,
            "expert_command_run": np.cumsum(np.r_[True, np.any(actions[1:] != actions[:-1], axis=1)]) - 1,
            "camera_intrinsics": np.empty((4, 3, 3), dtype=np.float64),
            "camera_extrinsics": np.empty((4, 4, 4), dtype=np.float64),
            "camera_near_far": np.empty((4, 2), dtype=np.float64),
            "camera_names": np.asarray(CAMERAS),
        }
        for camera_index, camera in enumerate(CAMERAS):
            params = source[f"cam_params/{camera}"]
            near, far = float(params["near"][()]), float(params["far"][()])
            arrays["camera_intrinsics"][camera_index] = params["intrinsics"][()]
            arrays["camera_extrinsics"][camera_index] = params["extrinsics"][()]
            arrays["camera_near_far"][camera_index] = near, far
            # Read one camera stream at a time to bound memory and avoid repeated
            # HDF5 chunk decompression when accessing consecutive frames.
            rgb = source[f"observations/{camera}/rgb"][()]
            depth = source[f"observations/{camera}/depth"][()]
            if rgb.shape != (n, 256, 256, 3) or rgb.dtype != np.uint8 or depth.shape != rgb.shape:
                raise ValueError("source RGBD stream shape or RGB dtype differs")
            for frame in range(n):
                arrays["rgb"][frame, camera_index] = np.asarray(Image.fromarray(rgb[frame]).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
                metric = decode_metric_depth(depth[frame], near, far)
                arrays["depth"][frame, camera_index] = np.asarray(Image.fromarray(metric).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32)
            del rgb, depth
        small_arrays_sha = array_sha256({**raw, "expert_trajectory_stage": stages})
    after = source_stat(source_path)
    if before != after:
        raise ValueError("source file changed during preparation")
    validate_episode_arrays(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)
    # Round-trip the compact cache; no source image or Native state is reopened.
    with np.load(destination, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            raise ValueError("cache round-trip schema differs")
        for key, value in arrays.items():
            if not np.array_equal(stored[key], value):
                raise ValueError(f"cache round-trip differs: {key}")
    return dict(
        split=split, task=task, raw_episode_id=episode,
        parent_raw_episode_id=episode, parent_source_group=f"{split}:{task}:{episode}",
        source_path=str(source_path), source_file_stat=before,
        whole_HDF5_sha256_recomputed=False,
        source_integrity_boundary="stat before/after; exact small-array hash; prepared cache hash",
        source_small_arrays_sha256=small_arrays_sha,
        npz_path=str(destination.relative_to(root)), cache_sha256=file_sha256(destination),
        cache_bytes=destination.stat().st_size, frames=n,
        expert_commands=int(arrays["expert_command_run"][-1]) + 1,
        skill_frames={str(k): int(v) for k, v in sorted(Counter(arrays["skill_labels"].tolist()).items())},
        scene_proprio_input_sha256=array_sha256({key: arrays[key] for key in INPUT_KEYS}),
        seconds=time.perf_counter() - started,
    )


def exact_sources(root):
    root = Path(root).resolve()
    sources, manifests = [], {}
    expected = {"train": {
        "pick_moving_target_on_the_table": 100, "pick_moving_target_on_the_table_high_speed": 100,
        "insert_onto_rotating_peg": 100, "insert_onto_rotating_peg_high_speed": 100,
    }, "validation": {
        "pick_moving_target_on_the_table": 20, "pick_moving_target_on_the_table_high_speed": 20,
        "insert_onto_rotating_peg": 18, "insert_onto_rotating_peg_high_speed": 19,
    }}
    for split in ("train", "validation"):
        path = root / f"artifacts/framework/protocol/{split}_manifest.json"
        document = json.loads(path.read_text())
        rows = [row for row in document["episodes"] if row["task"] in TASK_FAMILIES]
        if dict(Counter(row["task"] for row in rows)) != expected[split]:
            raise ValueError(f"{split} exact task/source counts differ")
        identities = [(row["task"], row["raw_episode_id"]) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate parent identities in {split}")
        manifests[split] = dict(path=str(path.relative_to(root)), sha256=file_sha256(path))
        sources.extend((split, row) for row in rows)
    train = {(row["task"], row["raw_episode_id"]) for split, row in sources if split == "train"}
    validation = {(row["task"], row["raw_episode_id"]) for split, row in sources if split == "validation"}
    if train & validation:
        raise ValueError("TRAIN/validation parent source identities overlap")
    return sources, manifests


def prepare_dataset(root, output, *, smoke=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    permitted_root = root / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if not output.is_relative_to(permitted_root):
        raise ValueError("all new caches must use the new hierarchy namespace")
    if output.exists():
        raise FileExistsError(f"choose a fresh output namespace: {output}")
    sources, source_manifests = exact_sources(root)
    if smoke:
        sources = [next(item for item in sources if item[0] == "train")]
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    started = time.perf_counter()
    manifest = dict(
        schema=SCHEMA, created_utc=utc_now(), updated_utc=utc_now(), stage="preparing",
        complete=False, smoke_only=smoke, expected_episodes=len(sources), episodes=[],
        source_manifests=source_manifests,
        tasks=TASK_FAMILIES, skill_label_mapping=SKILL_IDS,
        consumed_input_keys=list(INPUT_KEYS), privileged_supervision_keys=list(LABEL_KEYS),
        supervision_only_command_run="expert_command_run is weighting metadata, never a model input",
        input_contract="current and past RGBD scene + measured TCP/gripper/joints only",
        output_shapes=dict(rgb="N,4,64,64,3 uint8", depth="N,4,64,64 float32 meters",
                           robot="N,15 float32: TCP xyz/quaternion_xyzw, gripper, joints7",
                           expert_actions="N,8 float32: target xyz/quaternion_xyzw, gripper"),
        camera_order=list(CAMERAS),
        rgb_resize="Pillow bilinear, 256x256 to 64x64",
        depth_resize="author RGB24 linear decode to metric float32 then nearest-neighbor 64x64",
        camera_calibration_resolution="intrinsics retained at original 256x256 resolution; scale for resized inputs before backprojection",
        timestamp_definition="frame_index * 0.05 seconds, nominal reconstructed clock",
        command_run_definition="new run iff any of eight exact float32 target components changes",
        tracking_status=dict(
            public_tracked_history_prepared=False,
            insert_public_tracks_missing=True,
            original_GRU_inputs_fabricated=False,
            note="Cache contains scene+proprio only; no replacement GRU, fake trajectories, or privileged target inputs.",
        ),
        TEST_read=False, mixed_split_fingerprint_files_read=False,
        source_debug_target_positions_read=False, world_model_predictions_read=False,
        simulator_started=False, GPU_used=False, automatic_skips=False,
        code_sha256={
            "our_mani/hierarchy_20261030/data.py": file_sha256(Path(__file__)),
            "scripts/prepare_hierarchy_four_task_20261030.py": file_sha256(root / "scripts/prepare_hierarchy_four_task_20261030.py"),
        },
    )
    atomic_json(manifest_path, manifest)
    try:
        for split, row in sources:
            record = prepare_episode(row, split, root=root, output=output)
            manifest["episodes"].append(record)
            manifest["updated_utc"] = utc_now()
            manifest["completed_episodes"] = len(manifest["episodes"])
            manifest["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(manifest_path, manifest)
            print(json.dumps(dict(stage="prepared_episode", split=split,
                                  task=row["task"], episode=row["raw_episode_id"],
                                  completed=len(manifest["episodes"]), expected=len(sources),
                                  frames=record["frames"], commands=record["expert_commands"],
                                  episode_seconds=round(record["seconds"], 3),
                                  elapsed_seconds=round(manifest["elapsed_seconds"], 3))), flush=True)
        if any(file_sha256(root / item["path"]) != item["sha256"]
               for item in source_manifests.values()):
            raise ValueError("source protocol manifest changed during preparation")
        summary = {}
        for split in ("train", "validation"):
            summary[split] = {}
            for task in TASK_FAMILIES:
                records = [r for r in manifest["episodes"] if r["split"] == split and r["task"] == task]
                summary[split][task] = dict(
                    episodes=len(records), frames=sum(r["frames"] for r in records),
                    expert_commands=sum(r["expert_commands"] for r in records),
                    cache_bytes=sum(r["cache_bytes"] for r in records),
                )
        manifest.update(stage="complete", complete=True, summary=summary,
                        elapsed_seconds=time.perf_counter() - started, finished_utc=utc_now())
        atomic_json(manifest_path, manifest)
        print(json.dumps(dict(stage="complete", manifest=str(manifest_path),
                              episodes=len(manifest["episodes"]),
                              elapsed_seconds=round(manifest["elapsed_seconds"], 3),
                              smoke_only=smoke)), flush=True)
        return manifest
    except BaseException as error:
        manifest.update(stage="failed", complete=False, error=repr(error),
                        elapsed_seconds=time.perf_counter() - started, updated_utc=utc_now())
        atomic_json(manifest_path, manifest)
        raise
