"""Learned current-frame semantic masks from public RGBD and calibration.

TRAIN annotation files are opened only by the offline data loader. The runtime
segmenter accepts public sensor arrays; task IDs, instance IDs, native handles,
expert targets, and future observations are not feature channels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCHEMA = "hierarchy_public_rgbd_semantic_mlp_v1_20261030"
CAPTURE_SCHEMA = "hierarchy_visual_train_capture_v1_20261030"
CAMERAS = ("front", "8", "16", "36")
CLASS_NAMES = ("background", "frame", "peg0", "peg1", "peg2", "ring")
FEATURE_NAMES = ("red_0_1", "green_0_1", "blue_0_1", "log_depth_m",
                 "world_x_m", "world_y_m", "world_z_m")
PUBLIC_KEYS = frozenset(("rgb", "depth_m", "camera_intrinsics", "camera_to_world",
                         "camera_names", "timestamp", "counted_tick"))
REQUIRED_PUBLIC_KEYS = frozenset(("rgb", "depth_m", "camera_intrinsics", "camera_to_world"))
CAPTURE_ROOT = Path("artifacts/framework/multitask_r10/hierarchy_20261030/visual_train_v1")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_observation_inputs(observation):
    """Extract only current RGB, metric depth and public camera matrices."""
    images = observation["observation/images"]
    depths = observation["observation/depths"]
    calibrations = observation["observation/calibrations"]
    if any(set(mapping) != set(CAMERAS) for mapping in (images, depths, calibrations)):
        raise ValueError("the declared four synchronized camera views are required")
    rgb, depth, intrinsics, extrinsics = [], [], [], []
    for camera in CAMERAS:
        calibration = calibrations[camera]
        if not np.isfinite(calibration.depth_scale) or calibration.depth_scale <= 0:
            raise ValueError("invalid public camera depth scale")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = calibration.camera_to_world.rotation
        matrix[:3, 3] = calibration.camera_to_world.position
        rgb.append(np.asarray(images[camera]))
        depth.append(np.asarray(depths[camera], dtype=np.float32) * float(calibration.depth_scale))
        intrinsics.append(np.asarray(calibration.intrinsics, dtype=np.float64))
        extrinsics.append(matrix)
    return {"rgb": np.stack(rgb), "depth_m": np.stack(depth),
            "camera_intrinsics": np.stack(intrinsics), "camera_to_world": np.stack(extrinsics),
            "camera_names": np.asarray(CAMERAS),
            "timestamp": np.asarray(float(observation["timestamp"]))}


def pixel_features(inputs):
    """Return [camera,H,W,7] features and valid-depth mask, using no labels.

    Pixel coordinates are zero-based (u,v,1). Signed focal lengths are retained:
    world_xyz = camera_to_world @ (inverse(K) @ pixel * metric_depth).
    """
    if not REQUIRED_PUBLIC_KEYS <= set(inputs) or set(inputs) - PUBLIC_KEYS:
        raise ValueError("only declared public RGBD/calibration input keys are allowed")
    rgb = np.asarray(inputs["rgb"])
    depth = np.asarray(inputs["depth_m"], dtype=np.float64)
    intrinsics = np.asarray(inputs["camera_intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(inputs["camera_to_world"], dtype=np.float64)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != 4 or rgb.shape[-1] != 3:
        raise ValueError("RGB must be four HxWx3 uint8 camera images")
    if depth.shape != rgb.shape[:-1] or intrinsics.shape != (4, 3, 3) or extrinsics.shape != (4, 4, 4):
        raise ValueError("RGB, metric depth and camera matrix shapes differ")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("nonfinite public camera matrices")
    if not np.allclose(extrinsics[:, 3], [0., 0., 0., 1.], atol=1e-8):
        raise ValueError("invalid homogeneous camera matrices")
    if "camera_names" in inputs and tuple(np.asarray(inputs["camera_names"]).tolist()) != CAMERAS:
        raise ValueError("public camera ordering differs")
    for key in ("timestamp", "counted_tick"):
        if key in inputs and not np.isfinite(np.asarray(inputs[key])).all():
            raise ValueError("nonfinite public sensor timing metadata")
    valid = np.isfinite(depth) & (depth > 0)
    height, width = depth.shape[1:]
    v, u = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    features = np.zeros((*depth.shape, 7), dtype=np.float32)
    features[..., :3] = rgb.astype(np.float32) / 255.
    for index in range(4):
        if abs(np.linalg.det(intrinsics[index])) < 1e-12:
            raise ValueError("singular public camera intrinsics")
        rotation = extrinsics[index, :3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError("camera-to-world rotation must be a proper rotation")
        safe_depth = np.where(valid[index], depth[index], 1.)
        rays = pixels @ np.linalg.inv(intrinsics[index]).T
        camera_xyz = rays * safe_depth[..., None]
        world_xyz = camera_xyz @ rotation.T + extrinsics[index, :3, 3]
        features[index, ..., 3] = np.log(safe_depth).astype(np.float32)
        features[index, ..., 4:] = world_xyz.astype(np.float32)
    features[~valid] = 0.
    if not np.isfinite(features).all():
        raise ValueError("nonfinite derived public pixel features")
    return features, valid


def load_public_inputs(path):
    path = Path(path)
    if path.parent.name != "inputs":
        raise ValueError("public arrays must be loaded from the physical inputs directory")
    with np.load(path, allow_pickle=False) as stored:
        if not REQUIRED_PUBLIC_KEYS <= set(stored.files) or set(stored.files) - PUBLIC_KEYS:
            raise ValueError("capture inputs contain an undeclared or privileged channel")
        result = {key: stored[key].copy() for key in stored.files}
    pixel_features(result)
    return result


def load_training_frame(input_path, label_path):
    """Load input and semantic targets separately; never use instance IDs."""
    input_path, label_path = Path(input_path), Path(label_path)
    if (input_path.parent.name != "inputs" or label_path.parent.name != "labels"
            or input_path.parent.parent != label_path.parent.parent
            or input_path.name != label_path.name):
        raise ValueError("same-frame inputs and labels must be physically separate siblings")
    inputs = load_public_inputs(input_path)
    features, valid = pixel_features(inputs)
    with np.load(label_path, allow_pickle=False) as labels:
        semantic = labels["semantic_role_ids"].copy()
        for key in ("timestamp", "counted_tick"):
            if key not in inputs or key not in labels.files or not np.array_equal(inputs[key], labels[key]):
                raise ValueError("input and label counted frame identities differ")
    if semantic.shape != valid.shape or semantic.dtype.kind not in "ui":
        raise ValueError("semantic labels must align with public camera pixels")
    if np.any(semantic >= len(CLASS_NAMES)):
        raise ValueError("unknown annotation class IDs")
    return features, valid, semantic.astype(np.int64)


def fit_feature_normalization(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 7 or not len(features) or not np.isfinite(features).all():
        raise ValueError("finite fit-only Nx7 features required")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features.std(axis=0, dtype=np.float64), 1e-3).astype(np.float32)
    return mean, std


class PointSemanticMLP(nn.Module):
    def __init__(self, hidden=64, feature_mean=None, feature_std=None):
        super().__init__()
        if hidden < 1:
            raise ValueError("positive hidden width required")
        mean = torch.zeros(7) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(7) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (7,) or std.shape != (7,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("finite seven-feature normalization required")
        self.hidden = int(hidden)
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.network = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, len(CLASS_NAMES)))

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 7:
            raise ValueError("pixel model expects Nx7 public features")
        return self.network((features - self.feature_mean) / self.feature_std)


@torch.inference_mode()
def predict_input_arrays(model, inputs, *, device="cpu", confidence_threshold=.6,
                         min_role_pixels=12, batch_size=65536):
    """Predict current semantic roles; invalid/low-confidence pixels stay -1."""
    if not 0 <= confidence_threshold <= 1 or min_role_pixels < 1 or batch_size < 1:
        raise ValueError("invalid fixed prediction quality settings")
    features, valid = pixel_features(inputs)
    flat_valid = valid.reshape(-1)
    indices = np.flatnonzero(flat_valid)
    probabilities = np.zeros((len(flat_valid), len(CLASS_NAMES)), dtype=np.float32)
    flat_features = features.reshape(-1, 7)
    model.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        tensor = torch.as_tensor(flat_features[selected], dtype=torch.float32, device=device)
        probabilities[selected] = model(tensor).softmax(-1).cpu().numpy()
    confidence = probabilities.max(axis=-1)
    semantic = probabilities.argmax(axis=-1).astype(np.int16)
    known = flat_valid & (confidence >= confidence_threshold)
    semantic[~known] = -1
    maps, probs, confidence_maps, quality = {}, {}, {}, {}
    shape = valid.shape
    semantic = semantic.reshape(shape)
    probabilities = probabilities.reshape(*shape, len(CLASS_NAMES))
    confidence = confidence.reshape(shape)
    for index, camera in enumerate(CAMERAS):
        maps[camera] = semantic[index]
        probs[camera] = probabilities[index]
        confidence_maps[camera] = confidence[index]
        role_counts = {name: int((semantic[index] == class_id).sum())
                       for class_id, name in enumerate(CLASS_NAMES)}
        quality[camera] = dict(
            valid_depth_pixels=int(valid[index].sum()),
            unknown_pixels=int((semantic[index] == -1).sum()),
            known_role_pixels=role_counts,
            sparse_roles={name: role_counts[name] < min_role_pixels for name in CLASS_NAMES[1:]},
        )
    total = {name: sum(v["known_role_pixels"][name] for v in quality.values()) for name in CLASS_NAMES}
    return {"semantic_maps": maps, "probabilities": probs,
            "confidence_maps": confidence_maps,
            "quality": {"per_camera": quality, "total_known_role_pixels": total,
                        "sparse_roles": {name: total[name] < min_role_pixels for name in CLASS_NAMES[1:]},
                        "confidence_threshold": confidence_threshold, "min_role_pixels": min_role_pixels,
                        "probabilities_are_calibrated": False,
                        "unknown_filled_from_native_labels": False},
            "provenance": {"schema": SCHEMA, "input_features": list(FEATURE_NAMES),
                           "task_id_used": False, "instance_or_native_pose_inputs_used": False}}


class PublicSemanticSegmenter:
    def __init__(self, checkpoint, *, device="cpu", confidence_threshold=.6,
                 min_role_pixels=12, batch_size=65536):
        self.path = Path(checkpoint).resolve(strict=True)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if (metadata.get("schema") != SCHEMA or metadata.get("training_complete") is not True
                or metadata.get("fit_split") != "train" or metadata.get("development_split") != "train"
                or metadata.get("TEST_read") is not False
                or metadata.get("input_features") != list(FEATURE_NAMES)
                or metadata.get("class_names") != list(CLASS_NAMES)
                or metadata.get("native_or_instance_inputs_used") is not False):
            raise ValueError("completed TRAIN-only public-feature segmentation checkpoint required")
        self.device = torch.device(device)
        self.model = PointSemanticMLP(hidden=int(payload["spec"]["hidden"]))
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.metadata = metadata
        self.checkpoint_sha256 = sha256(self.path)
        self.confidence_threshold = float(confidence_threshold)
        self.min_role_pixels = int(min_role_pixels)
        self.batch_size = int(batch_size)

    def predict(self, observation):
        inputs = public_observation_inputs(observation)
        result = self.predict_inputs(inputs)
        result["provenance"]["observed_until"] = float(observation["timestamp"])
        return result

    def predict_inputs(self, inputs):
        result = predict_input_arrays(
            self.model, inputs, device=self.device,
            confidence_threshold=self.confidence_threshold,
            min_role_pixels=self.min_role_pixels, batch_size=self.batch_size,
        )
        result["provenance"].update(
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_path=str(self.path),
            training_source="first source of each variant; second source TRAIN development holdout",
        )
        return result


def capture_training_records(root, plan_path):
    """Bind all four completed fixed TRAIN captures before any fitting begins."""
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve(strict=True)
    capture_root = (root / CAPTURE_ROOT).resolve()
    if not plan_path.is_relative_to(capture_root):
        raise ValueError("capture plan must use the new TRAIN visual namespace")
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != CAPTURE_SCHEMA or plan.get("split") != "train"
            or len(plan.get("cases", [])) != 4):
        raise ValueError("exact four-case TRAIN capture plan required")
    expected_tasks = {"insert_onto_rotating_peg", "insert_onto_rotating_peg_high_speed"}
    if set(plan.get("tasks", [])) != expected_tasks:
        raise ValueError("capture variants differ")
    plan_hash = sha256(plan_path)
    by_task = Counter()
    records = []
    for case in plan["cases"]:
        task, case_key = case["task"], case["case_key"]
        if task not in expected_tasks or Path(case_key).name != case_key:
            raise ValueError("invalid declared capture case identity")
        partition = "fit" if by_task[task] == 0 else "development"
        by_task[task] += 1
        if by_task[task] > 2:
            raise ValueError("more than two declared sources per variant")
        case_path = capture_root / "cases" / case_key / "case.json"
        record = json.loads(case_path.read_text())
        terminal = record.get("terminal", {})
        if (record.get("schema") != CAPTURE_SCHEMA or record.get("split") != "train"
                or record.get("case") != case or record["plan"]["sha256"] != plan_hash
                or terminal.get("complete") is not True
                or terminal.get("natural_20_tick_capture") is not True
                or terminal.get("administrative_stop") is not False
                or terminal.get("infrastructure_failed") is not False):
            raise ValueError("all planned TRAIN captures must finish naturally before fitting")
        if [frame["tick"] for frame in record["frames"]] != plan["sample_ticks"]:
            raise ValueError("capture frame roster differs from the fixed plan")
        for frame in record["frames"]:
            expected_name = f"frame_{frame['tick']:05d}.npz"
            pair = {}
            for kind in ("inputs", "labels"):
                ref = frame[kind]
                path = (root / ref["path"]).resolve(strict=True)
                expected_path = capture_root / "cases" / case_key / kind / expected_name
                if path != expected_path or sha256(path) != ref["sha256"]:
                    raise ValueError("capture pointer/hash or physical input-label separation differs")
                pair[kind] = path
            records.append(dict(partition=partition, task=task, case_key=case_key,
                                raw_episode_id=case["raw_episode_id"], tick=frame["tick"],
                                input_path=pair["inputs"], label_path=pair["labels"],
                                input_sha256=frame["inputs"]["sha256"],
                                label_sha256=frame["labels"]["sha256"],
                                case_path=str(case_path.relative_to(root)), case_sha256=sha256(case_path)))
    if dict(by_task) != {task: 2 for task in expected_tasks}:
        raise ValueError("fit/development source counts differ")
    return records, {"path": str(plan_path.relative_to(root)), "sha256": plan_hash}


def load_fit_and_development_pixels(records):
    features, labels = {"fit": [], "development": []}, {"fit": [], "development": []}
    counts = {"fit": np.zeros(6, dtype=np.int64), "development": np.zeros(6, dtype=np.int64)}
    for row in records:
        x, valid, y = load_training_frame(row["input_path"], row["label_path"])
        selected_x, selected_y = x[valid], y[valid]
        if len(selected_x) == 0:
            raise ValueError("capture frame has no valid public RGBD pixels")
        key = row["partition"]
        features[key].append(selected_x)
        labels[key].append(selected_y)
        counts[key] += np.bincount(selected_y, minlength=6)
    result = {key: (np.concatenate(features[key]), np.concatenate(labels[key])) for key in features}
    if np.any(counts["fit"] == 0):
        missing = [CLASS_NAMES[i] for i in np.flatnonzero(counts["fit"] == 0)]
        raise ValueError(f"fixed fit captures have no annotations for classes {missing}; no synthetic fill is permitted")
    return result, counts


def balanced_pixel_indices(labels, rng, samples_per_class):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "ui" or samples_per_class < 1:
        raise ValueError("integer pixel labels and positive class sample count required")
    selected = []
    for class_id in range(6):
        candidates = np.flatnonzero(labels == class_id)
        if not len(candidates):
            raise ValueError("every class needs real fit labels")
        selected.append(rng.choice(candidates, samples_per_class,
                                   replace=len(candidates) < samples_per_class))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices


@torch.inference_mode()
def evaluate_pixels(model, features, labels, *, device="cpu", batch_size=65536):
    confusion = np.zeros((6, 6), dtype=np.int64)
    model.eval()
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
        prediction = model(x).argmax(-1).cpu().numpy()
        y = labels[start:start + batch_size]
        confusion += np.bincount(y * 6 + prediction, minlength=36).reshape(6, 6)
    support = confusion.sum(axis=1)
    union = confusion.sum(0) + support - confusion.diagonal()
    iou = np.divide(confusion.diagonal(), union, out=np.full(6, np.nan), where=union > 0)
    present = support > 0
    return {"pixels": int(support.sum()), "confusion": confusion.tolist(),
            "support": {name: int(support[i]) for i, name in enumerate(CLASS_NAMES)},
            "IoU": {name: float(iou[i]) if union[i] > 0 else None for i, name in enumerate(CLASS_NAMES)},
            "macro_IoU_on_present_classes": float(np.nanmean(iou[present])),
            "absent_annotation_classes": [CLASS_NAMES[i] for i in np.flatnonzero(~present)],
            "pixel_accuracy": float(confusion.diagonal().sum() / max(1, support.sum()))}
