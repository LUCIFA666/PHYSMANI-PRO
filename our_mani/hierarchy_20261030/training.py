"""Audited, goal-conditioned skill pretraining; no simulator or TEST reads.

The first phase learns scene-to-skill classification from causal observations.
The second freezes that encoder and learns a distinct skill policy, conditioned
on the first expert target in each contiguous skill segment. This is pretraining,
not an evaluation of the learned candidate selector or native task success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import SCHEMA, INPUT_KEYS, TASK_FAMILIES, file_sha256
from .networks import CausalSceneSkillEncoder, PretrainedSkillPolicy


ROOT = Path(__file__).resolve().parents[2]
VERSION = "hierarchy_four_task_skill_pretraining_20261030_v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


class CausalSkillWindows(Dataset):
    """Keep complete episodes in host RAM; expert labels never enter history.

    Each parent carries equal mass within its task. Each distinct submitted
    command run carries equal mass within its parent; repeated hold rows do
    not masquerade as independent expert decisions.
    """
    def __init__(self, root, manifest, split, history_frames=5, smoke=False):
        if split not in ("train", "validation"):
            raise ValueError("only TRAIN and offline validation are permitted")
        self.root, self.split, self.history_frames = Path(root), split, history_frames
        rows = [row for row in manifest["episodes"] if row["split"] == split]
        if smoke:
            rows = [next(row for row in rows if row["task"] == task) for task in TASK_FAMILIES]
        self.rows, self.episodes, self.anchors, weights = rows, [], [], []
        task_parents = Counter(row["task"] for row in rows)
        started = time.perf_counter()
        for episode_index, row in enumerate(rows):
            path = self.root / row["npz_path"]
            if file_sha256(path) != row["cache_sha256"]:
                raise ValueError("cache fingerprint changed: " + str(path))
            with np.load(path, allow_pickle=False) as cached:
                arrays = {key: cached[key] for key in (*INPUT_KEYS, "expert_actions", "skill_labels", "expert_command_run")}
            n = len(arrays["robot"])
            if n != row["frames"]:
                raise ValueError("episode frame count changed")
            runs, counts = np.unique(arrays["expert_command_run"], return_counts=True)
            if not np.array_equal(runs, np.arange(len(runs))):
                raise ValueError("command runs are not consecutive")
            labels = arrays["skill_labels"]
            starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
            ends = np.r_[starts[1:], n]
            goals = np.empty((n, 3), np.float32)
            for begin, end in zip(starts, ends):
                goals[begin:end] = arrays["expert_actions"][begin, :3]
            arrays["skill_entry_goals"] = goals
            self.episodes.append(arrays)
            for frame in range(n):
                self.anchors.append((episode_index, frame))
                run_count = counts[arrays["expert_command_run"][frame]]
                weights.append(1 / (task_parents[row["task"]] * len(runs) * run_count))
            if (episode_index + 1) % 50 == 0:
                print(json.dumps(dict(stage="preload", split=split, episodes=episode_index + 1,
                                      expected=len(rows), seconds=round(time.perf_counter() - started, 2))), flush=True)
        self.weights = np.asarray(weights, np.float64)
        self.weights *= len(weights) / self.weights.sum()
        self.summary = dict(split=split, episodes=len(rows), frames=len(self.anchors),
                            task_parents=dict(task_parents), seconds=time.perf_counter() - started)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        episode_index, frame = self.anchors[index]
        arrays = self.episodes[episode_index]
        begin = max(0, frame - self.history_frames + 1)
        length = frame - begin + 1
        rgbd = np.zeros((self.history_frames, 4, 4, 64, 64), np.float32)
        rgbd[:length, :, :3] = arrays["rgb"][begin:frame + 1].transpose(0, 1, 4, 2, 3) / 255.0 - .5
        # Log metric depth is a fixed, invertible feature transform; no task
        # metadata, future frame, expert pose, or mask identifier enters it.
        rgbd[:length, :, 3] = np.log1p(arrays["depth"][begin:frame + 1])
        robot = np.zeros((self.history_frames, 15), np.float32)
        robot[:length] = arrays["robot"][begin:frame + 1]
        return dict(rgbd=rgbd, robot_history=robot, lengths=length,
                    skill_label=arrays["skill_labels"][frame],
                    goal_xyz=arrays["skill_entry_goals"][frame],
                    goal_time=np.float32(.5),
                    action=arrays["expert_actions"][frame],
                    weight=np.float32(self.weights[index]),
                    episode=episode_index, frame=frame)


class FrozenSkillFeatures(Dataset):
    """Exact frozen encoder outputs; labels remain separate from input fields."""
    def __init__(self, source, arrays):
        self.rows, self.weights, self.arrays = source.rows, source.weights, arrays

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}


@torch.inference_mode()
def freeze_features(source, encoder, device, *, batch_size, workers, output):
    """Materialize only inputs to the frozen visual/classification pathway.

    E is deliberately excluded: the skill embedding table still learns while
    p(skill|past observations) and scene features remain fixed.
    """
    encoder.eval()
    loader = DataLoader(source, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    pieces = {}
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        data = inputs_to_device(batch, device)
        encoded = encoder(data["rgbd"], data["robot_history"], data["lengths"])
        robot = data["robot_history"][torch.arange(len(data["lengths"]), device=device), data["lengths"]-1]
        values = {key: encoded[key] for key in ("scene_features", "skill_logits", "skill_probabilities")}
        values["robot_current"] = robot
        values.update({key: data[key] for key in ("skill_label", "goal_xyz", "goal_time", "action", "weight", "episode", "frame")})
        for key, value in values.items():
            pieces.setdefault(key, []).append(value.cpu())
        if step % 50 == 0 or step == 1:
            print(json.dumps(dict(stage="frozen_features", split=source.split, step=step,
                                  steps=len(loader), seconds=round(time.perf_counter()-started, 2))), flush=True)
    arrays = {key: torch.cat(values) for key, values in pieces.items()}
    torch.save(dict(arrays=arrays, split=source.split,
                    input_path="same causal scene encoder, frozen visual/history/classifier",
                    scene_input_uses_labels=False), output)
    return FrozenSkillFeatures(source, arrays)


def encoded_from_features(batch, encoder):
    return dict(scene_features=batch["scene_features"], skill_logits=batch["skill_logits"],
                skill_probabilities=batch["skill_probabilities"],
                skill_latent=batch["skill_probabilities"] @ F.normalize(encoder.skill_embeddings.weight, dim=-1))


def quaternion_matrix(quaternion):
    q = F.normalize(quaternion, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def skill_loss(prediction, action, scale):
    position = ((prediction["xyz"] - action[:, :3]) / scale).square().mean(-1)
    rotation = (prediction["rotation_matrix"] - quaternion_matrix(action[:, 3:7])).square().mean((-1, -2))
    gripper = F.cross_entropy(prediction["gripper_logits"], (action[:, 7] >= .5).long(), reduction="none")
    return position + .2 * rotation + .1 * gripper


def inputs_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate(encoder, policy, loader, device, stage, scale):
    encoder.eval()
    if policy is not None:
        policy.eval()
    confusion = torch.zeros(11, 11, dtype=torch.float64, device=device)
    totals = dict(weight=0., loss=0., position_m=0., angle_rad=0., gripper_correct=0., goal_copy_m=0.)
    by_task = {name: dict(weight=0., position_m=0., angle_rad=0., gripper_correct=0.) for name in TASK_FAMILIES}
    task_names = [row["task"] for row in loader.dataset.rows]
    for batch in loader:
        data = inputs_to_device(batch, device)
        encoded = (encoded_from_features(data, encoder) if "scene_features" in data else
                   encoder(data["rgbd"], data["robot_history"], data["lengths"]))
        labels, weights = data["skill_label"], data["weight"].double()
        selected = encoded["skill_logits"].argmax(-1)
        confusion += torch.bincount(labels * 11 + selected, weights=weights, minlength=121).reshape(11, 11)
        losses = F.cross_entropy(encoded["skill_logits"], labels, reduction="none")
        if stage == "skill":
            robot = (data["robot_current"] if "robot_current" in data else
                     data["robot_history"][torch.arange(len(labels), device=device), data["lengths"] - 1])
            prediction = policy(encoded["scene_features"], encoded["skill_latent"], robot,
                                data["goal_xyz"], data["goal_time"])
            if prediction["trajectory_consumed"]:
                raise ValueError("this pretraining phase must not fabricate or consume trajectories")
            losses = skill_loss(prediction, data["action"], scale)
            distances = (prediction["xyz"] - data["action"][:, :3]).norm(dim=-1)
            target_rotation = quaternion_matrix(data["action"][:, 3:7])
            relative = prediction["rotation_matrix"].transpose(-1, -2) @ target_rotation
            angles = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1).acos()
            grip = (prediction["gripper_logits"].argmax(-1) == (data["action"][:, 7] >= .5).long()).double()
            for key, values in (("position_m", distances), ("angle_rad", angles), ("gripper_correct", grip)):
                totals[key] += float((values * weights).sum())
            totals["goal_copy_m"] += float(((data["goal_xyz"]-data["action"][:, :3]).norm(dim=-1)*weights).sum())
            episodes = data["episode"].cpu().tolist()
            for task in TASK_FAMILIES:
                keep = torch.tensor([task_names[e] == task for e in episodes], device=device)
                task_weights = weights * keep
                row = by_task[task]
                row["weight"] += float(task_weights.sum())
                for key, values in (("position_m", distances), ("angle_rad", angles), ("gripper_correct", grip)):
                    row[key] += float((values * task_weights).sum())
        totals["weight"] += float(weights.sum())
        totals["loss"] += float((losses.double() * weights).sum())
    rows = confusion.sum(-1)
    output = dict(loss=totals["loss"]/totals["weight"],
                  skill_accuracy=float(confusion.diag().sum()/confusion.sum()),
                  skill_macro_recall=float((confusion.diag()/rows.clamp_min(1))[rows > 0].mean()),
                  skill_confusion=confusion.cpu().tolist(),
                  native_success_measured=False, candidate_network_evaluated=False,
                  goal_source="expert_skill_segment_entry", trajectory_consumed=False)
    if stage == "skill":
        output.update(position_error_m=totals["position_m"]/totals["weight"],
                      rotation_error_deg=totals["angle_rad"]/totals["weight"] * 180/np.pi,
                      gripper_accuracy=totals["gripper_correct"]/totals["weight"],
                      copy_segment_entry_goal_position_error_m=totals["goal_copy_m"]/totals["weight"])
        output["per_task"] = {task: dict(position_error_m=row["position_m"]/row["weight"],
                                         rotation_error_deg=row["angle_rad"]/row["weight"]*180/np.pi,
                                         gripper_accuracy=row["gripper_correct"]/row["weight"])
                              for task, row in by_task.items() if row["weight"]}
    return output


def save_checkpoint(path, encoder, policy, metadata):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(version=VERSION, encoder_state_dict=encoder.state_dict(),
                    skill_state_dict=None if policy is None else policy.state_dict(), metadata=metadata), temporary)
    temporary.replace(path)


def run(args):
    root = ROOT
    config_path = root / args.config
    config = json.loads(config_path.read_text())
    manifest_path = root / args.data_manifest
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != SCHEMA or not manifest.get("complete") or manifest.get("smoke_only"):
        raise ValueError("complete full four-task cache required")
    if manifest["TEST_read"] or manifest["source_debug_target_positions_read"]:
        raise ValueError("forbidden source input in prepared data")
    output = (root / args.out).resolve()
    permitted = root / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if not output.is_relative_to(permitted) or output.exists():
        raise ValueError("fresh versioned pretraining output required")
    output.mkdir(parents=True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.backends.cudnn.benchmark = True
    training = CausalSkillWindows(root, manifest, "train", config["scene_history_frames"], args.smoke)
    validation = CausalSkillWindows(root, manifest, "validation", config["scene_history_frames"], args.smoke)
    positions = np.concatenate([episode["expert_actions"][:, :3] for episode in training.episodes])
    lower, upper = positions.min(0) - .05, positions.max(0) + .05
    center, half_width = (upper + lower)/2, (upper - lower)/2
    encoder = CausalSceneSkillEncoder().to(device)
    policy = PretrainedSkillPolicy(center.tolist(), half_width.tolist()).to(device)
    batch_size = args.batch_size or config["skill_pretraining"]["batch_size"]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.from_numpy(training.weights), len(training), replacement=True, generator=generator)
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda",
                                   persistent_workers=args.workers > 0)
    scale = torch.tensor(half_width, device=device)
    code_paths = [Path(__file__), root / "our_mani/hierarchy_20261030/networks.py",
                  root / "our_mani/hierarchy_20261030/data.py", root / "scripts/train_hierarchy_skills_20261030.py"]
    identity = dict(config_sha256=file_sha256(config_path), data_manifest_sha256=file_sha256(manifest_path),
                    code_sha256={str(path.relative_to(root)): file_sha256(path) for path in code_paths})
    status = dict(version=VERSION, created_utc=utc(), pid=os.getpid(),
                  process_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                  device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  identity=identity, train=training.summary, offline_validation=validation.summary,
                  workspace_center=center.tolist(), workspace_half_width=half_width.tolist(),
                  stage="starting", complete=False, epochs=[],
                  architecture_scope="scene/action classifier + independent goal-conditioned skill policy pretraining",
                  original_motion_forecaster_used=False, candidate_transformer_trained=False,
                  native_VAL_started=False, TEST_started=False, native_success_measured=False,
                  expert_labels_as_scene_inputs=False, held_repeat_rows_counted_as_independent_commands=False,
                  goal_conditioning="first expert TCP target per contiguous skill segment; fixed 0.5s contract",
                  next_gate="public tracks, real forecaster outputs, candidate training and fixed TRAIN audit before native VAL")
    atomic_json(output / "status.json", status)
    started = time.perf_counter()
    class_epochs = 1 if args.smoke else config["skill_pretraining"]["classification_epochs"]
    skill_epochs = 1 if args.smoke else config["skill_pretraining"]["skill_epochs"]
    try:
        for stage, epochs in (("classification", class_epochs), ("skill", skill_epochs)):
            if stage == "skill":
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
                encoder.load_state_dict(torch.load(output/"best_classification.pt", map_location=device, weights_only=False)["encoder_state_dict"])
                encoder.skill_embeddings.weight.requires_grad_(True)
                status.update(stage="materializing_frozen_encoder_features", updated_utc=utc())
                atomic_json(output/"status.json", status)
                training_features = freeze_features(training, encoder, device, batch_size=batch_size,
                                                    workers=args.workers, output=output/"train_frozen_features.pt")
                validation_features = freeze_features(validation, encoder, device, batch_size=batch_size,
                                                      workers=args.workers, output=output/"validation_frozen_features.pt")
                feature_sampler = WeightedRandomSampler(torch.from_numpy(training.weights), len(training),
                                                       replacement=True, generator=generator)
                loader = DataLoader(training_features, batch_size=batch_size, sampler=feature_sampler,
                                    num_workers=0, pin_memory=device.type == "cuda")
                validation_loader = DataLoader(validation_features, batch_size=batch_size, shuffle=False,
                                               num_workers=0, pin_memory=device.type == "cuda")
            parameters = list(encoder.parameters()) if stage == "classification" else [*policy.parameters(), encoder.skill_embeddings.weight]
            optimizer = torch.optim.AdamW(parameters, lr=config["skill_pretraining"]["learning_rate"],
                                          weight_decay=config["skill_pretraining"]["weight_decay"])
            best = float("inf")
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                encoder.train(stage == "classification")
                policy.train(stage == "skill")
                losses, examples = 0., 0
                status.update(stage=stage, current_epoch=epoch, expected_stage_epochs=epochs, updated_utc=utc())
                atomic_json(output/"status.json", status)
                for step, batch in enumerate(loader, 1):
                    data = inputs_to_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    if stage == "classification":
                        encoded = encoder(data["rgbd"], data["robot_history"], data["lengths"])
                    else:
                        encoded = encoded_from_features(data, encoder)
                    if stage == "classification":
                        loss = F.cross_entropy(encoded["skill_logits"], data["skill_label"])
                    else:
                        robot = data["robot_current"]
                        # The visual/classification pathway is frozen; the same
                        # learned embedding table will condition both downstream
                        # modules. Recompute e in the gradient domain so E learns.
                        skill_latent = encoded["skill_probabilities"].detach() @ F.normalize(encoder.skill_embeddings.weight, dim=-1)
                        predicted = policy(encoded["scene_features"], skill_latent, robot,
                                           data["goal_xyz"], data["goal_time"])
                        loss = skill_loss(predicted, data["action"], scale).mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.)
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("nonfinite gradient")
                    optimizer.step()
                    count = len(data["skill_label"])
                    losses += float(loss.detach()) * count
                    examples += count
                    if step % 50 == 0 or step == 1:
                        status.update(step=step, expected_steps=len(loader), processed_examples=examples,
                                      epoch_elapsed_seconds=time.perf_counter()-epoch_start, updated_utc=utc())
                        atomic_json(output/"status.json", status)
                        print(json.dumps(dict(stage=stage, epoch=epoch, step=step, steps=len(loader),
                                              train_loss=losses/examples, seconds=round(time.perf_counter()-epoch_start, 2))), flush=True)
                    if args.smoke and step >= 4:
                        break
                metrics = evaluate(encoder, policy, validation_loader, device, stage, scale)
                row = dict(stage=stage, epoch=epoch, train_loss=losses/examples,
                           validation=metrics, seconds=time.perf_counter()-epoch_start,
                           examples=examples)
                status["epochs"].append(row)
                metadata = dict(identity=identity, stage=stage, epoch=epoch,
                                workspace_center=center.tolist(), workspace_half_width=half_width.tolist(),
                                validation=metrics, skill_policy_trajectory_consumed=False,
                                candidate_network_trained=False, motion_forecaster_used=False,
                                native_success_measured=False, smoke_only=args.smoke)
                save_checkpoint(output/"last.pt", encoder, policy if stage == "skill" else None, metadata)
                if metrics["loss"] < best:
                    best = metrics["loss"]
                    save_checkpoint(output/f"best_{stage}.pt", encoder, policy if stage == "skill" else None, metadata)
                status.update(updated_utc=utc(), elapsed_seconds=time.perf_counter()-started)
                atomic_json(output/"status.json", status)
                print(json.dumps(row), flush=True)
        if file_sha256(config_path) != identity["config_sha256"] or file_sha256(manifest_path) != identity["data_manifest_sha256"]:
            raise ValueError("config/data identity changed during training")
        if any(file_sha256(root/name) != digest for name, digest in identity["code_sha256"].items()):
            raise ValueError("training source changed during fitting")
        status.update(stage="complete", complete=True, finished_utc=utc(),
                      elapsed_seconds=time.perf_counter()-started, smoke_only=args.smoke,
                      full_hierarchy_ready=False)
        atomic_json(output/"status.json", status)
        return status
    except BaseException as error:
        status.update(stage="failed", complete=False, error=repr(error), updated_utc=utc(),
                      elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/"status.json", status)
        raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default="config/framework/hierarchy_20261030_v1.json")
    command.add_argument("--data-manifest", default="artifacts/framework/multitask_r10/hierarchy_20261030/data_v1/manifest.json")
    command.add_argument("--out", required=True)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--smoke", action="store_true")
    return command
