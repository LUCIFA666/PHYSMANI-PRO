"""Joint geometry/pose-time proposals and feedback action-sequence imitation.

All fitting and checkpoint selection use the exact block TRAIN/offline-VAL
cohorts. Neither simulator task outcomes nor any TEST source is read here.
Qualification is measured from complete ledgers, never inferred from low loss.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from our_mani.hierarchy_20261030.training import quaternion_matrix
from our_mani.hierarchy_precision_20261030.precision import precision_report
from .data import BlockData, BlockDataset, TASK_NAMES, ROLE_NAMES, DT
from .networks import BlockHierarchyNetwork


ROOT = Path(__file__).resolve().parents[2]
VERSION = "hierarchy_block_training_20261102_v1"
BASE = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102")
OUT = BASE / "training_v1"
RUNTIME = Path("runtime/framework/hierarchy_block_20261102/training_v1")
INITIAL = Path("artifacts/framework/multitask_r10/hierarchy_20261030/skill_pretrain_v1/best_classification.pt")
SCORE = "expert_joint_handoff_pose_time_imitation_not_calibrated_success"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity():
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def move(value, device):
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


def matrix(quaternions):
    return quaternion_matrix(quaternions.reshape(-1, 4)).reshape(*quaternions.shape[:-1], 3, 3)


def position_penalty(distance):
    """Physical thresholds; no prediction clipping or distant-sample removal."""
    return ((distance / .03).square() + 2 * (F.relu(distance - .03) / .03).square()
            + 4 * (F.relu(distance - .05) / .03).square())


def weighted(values, weights):
    if (values.ndim != 1 or values.shape != weights.shape or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all())):
        raise ValueError("finite aligned per-sample losses and positive weights required")
    return (values * weights).sum() / weights.sum()


def candidate_loss(output, labels, weights):
    distances = (output["positions"] - labels["handoff_position"][:, None]).norm(dim=-1)
    rotations = (output["rotation_matrices"] - matrix(labels["handoff_quaternion"])[:, None]).square().sum((-1, -2)) / (2 * .15 ** 2)
    times = ((torch.log1p(output["times"]) - torch.log1p(labels["handoff_remaining_seconds"])[:, None]) / .2).square()
    cost = position_penalty(distances) + .2 * rotations + .1 * times
    # A fixed-scale pose/time mixture keeps gradients to all proposals. Its
    # weights are imitation scores, not learned Native-success probabilities.
    nll = -2 * torch.logsumexp(F.log_softmax(output["logits"], -1) - .5 * cost, -1)
    ranking = F.cross_entropy(output["logits"], cost.detach().argmin(-1), reduction="none")
    return weighted(nll + .1 * ranking, weights)


def skill_loss(output, labels, weights, step_weights):
    target, valid = labels["action_sequence"], labels["action_mask"]
    if valid.dtype != torch.bool or not bool(valid[:, 0].all()):
        raise ValueError("every skill anchor requires a true first-step expert command")
    distance = (output["positions"] - target[..., :3]).norm(dim=-1)
    rotation = (output["rotation_matrices"] - matrix(target[..., 3:7])).square().sum((-1, -2)) / (2 * .15 ** 2)
    gripper = F.cross_entropy(output["gripper_logits"].reshape(-1, 2),
                              (target[..., 7] >= .5).long().reshape(-1), reduction="none").reshape_as(distance)
    values = position_penalty(distance) + .2 * rotation + .2 * gripper
    mass = valid.to(values.dtype) * values.new_tensor(step_weights)[None]
    return weighted((values * mass).sum(-1) / mass.sum(-1), weights)


def encode(model, inputs):
    if "frozen_scene" in inputs:
        return model.encode_frozen_features(inputs["frozen_scene"], inputs["frozen_latent"],
                                             inputs["frozen_logits"], inputs["point_features"], inputs["point_mask"])
    return model.encode(inputs["rgbd"], inputs["robot_history"], inputs["lengths"],
                         inputs["point_features"], inputs["point_mask"])


class CachedDataset(Dataset):
    """Input-only classifier cache; geometry and shared e are trained live."""
    def __init__(self, base, features, skill_indices):
        self.base, self.features = base, features
        self.indices = [skill_indices[anchor] for anchor in base.anchors]
        self.head, self.split, self.anchors = base.head, base.split, base.anchors
        self.expected_samples, self.sample_ids = base.expected_samples, base.sample_ids
        self.summary, self.weights = base.summary, base.weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        result = self.base[index]
        inputs = result["inputs"]
        for key in ("rgbd", "robot_history", "lengths"):
            inputs.pop(key)
        row = self.indices[index]
        inputs.update(frozen_scene=self.features["scene_features"][row],
                      frozen_latent=self.features["scene_latent"][row], frozen_logits=self.features["skill_logits"][row])
        return result


@torch.inference_mode()
def materialize_classifier(model, dataset, device, path, *, workers):
    model.eval()
    pieces = {key: [] for key in ("scene_features", "scene_latent", "skill_logits")}
    observed = []
    for raw in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=workers,
                          pin_memory=device.type == "cuda"):
        inputs = move(raw["inputs"], device)
        scene = model.encoder(inputs["rgbd"], inputs["robot_history"], inputs["lengths"])
        for key in pieces:
            value = scene[key][:, :4] if key == "skill_logits" else scene[key]
            pieces[key].append(value.cpu())
        observed.extend(raw["sample_index"].tolist())
    if observed != list(range(len(dataset))):
        raise ValueError("classifier cache must account for every anchor in source order")
    arrays = {key: torch.cat(values) for key, values in pieces.items()}
    torch.save(dict(arrays=arrays, sample_ids=dataset.sample_ids, split=dataset.split,
                    input_only=True, expert_or_task_labels_consumed=False), path)
    return arrays


@torch.inference_mode()
def goal_cache(model, dataset, device):
    """Predict one first-eligible goal per source, with no label input."""
    model.eval()
    first = {}
    for index, (episode, frame) in enumerate(dataset.anchors):
        first.setdefault(episode, index)
    cache = {}
    indices = list(first.values())
    for start in range(0, len(indices), 32):
        rows = indices[start:start + 32]
        raw = torch.utils.data.default_collate([dataset[index] for index in rows])
        batch = move(raw, device)
        inputs = batch["inputs"]
        encoded = encode(model, inputs)
        proposal = model.forward_candidate(encoded, inputs["robot"], inputs["trajectory"], inputs["trajectory_mask"])
        for j, index in enumerate(rows):
            episode, frame = dataset.anchors[index]
            cache[episode] = dict(frame=frame, xyz=proposal["goal_position"][j].cpu(),
                                  rotation=proposal["goal_rotation"][j].cpu(), time=float(proposal["goal_time"][j]))
    return cache


def conditions(batch, cache=None, *, training=False, config=None):
    labels = batch["supervision"]
    xyz = labels["handoff_position"].clone()
    rotation = matrix(labels["handoff_quaternion"])
    remaining = labels["handoff_remaining_seconds"].clone()
    predicted_used = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    if cache is not None:
        for i, (episode, frame) in enumerate(zip(batch["episode_index"].cpu().tolist(), batch["frame_index"].cpu().tolist())):
            row = cache.get(episode)
            if row is None or frame < row["frame"]:
                continue
            if training and float(torch.rand(())) >= config["predicted_goal_probability"]:
                continue
            xyz[i] = row["xyz"].to(xyz.device)
            rotation[i] = row["rotation"].to(xyz.device)
            remaining[i] = max(row["time"] - (frame - row["frame"]) * DT, 0.)
            predicted_used[i] = True
    if training:
        noise = torch.randn_like(xyz) * config["goal_noise_std_m"]
        noise = noise * (config["goal_noise_max_norm_m"] / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1.)
        xyz = xyz + noise
    return xyz, rotation, remaining, predicted_used


def ledger_entry(dataset, index, *, head, suffix=""):
    row = dict(dataset.expected_samples[index])
    row.update(head=head, sample_id=row["sample_id"] + suffix)
    return row


@torch.inference_mode()
def evaluate(model, datasets, device, *, workers):
    model.eval()
    predicted_goals = goal_cache(model, datasets["candidate"], device)
    observed, expected, sequence_observed, sequence_expected = [], [], [], []
    predicted_observed, predicted_expected = [], []
    # Declare every expected identity before consuming model outputs. A missing
    # batch cannot shrink the denominator or silently improve qualification.
    for head, dataset in datasets.items():
        for index, (episode, frame) in enumerate(dataset.anchors):
            row = ledger_entry(dataset, index, head="candidate" if head == "candidate" else "tcp")
            expected.append(row)
            if head == "skill":
                base = getattr(dataset, "base", dataset)
                n = len(base.data.episodes[dataset.split][episode]["scene"]["robot"])
                for h in range(4):
                    if frame + 10 * h < n:
                        sequence_expected.append(ledger_entry(dataset, index, head="tcp", suffix=f"|h{h}"))
                if episode in predicted_goals and frame >= predicted_goals[episode]["frame"]:
                    predicted_expected.append(row)
    aux = dict(candidate_rotation_deg=[], candidate_time_error_seconds=[], skill_rotation_deg=[], gripper_correct=[], skill_class_correct=[])
    for head, dataset in datasets.items():
        for raw in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda"):
            batch = move(raw, device)
            inputs, labels = batch["inputs"], batch["supervision"]
            encoded = encode(model, inputs)
            indices = raw["sample_index"].tolist()
            if head == "candidate":
                result = model.forward_candidate(encoded, inputs["robot"], inputs["trajectory"], inputs["trajectory_mask"])
                positions = result["goal_position"]
                relative = result["goal_rotation"].transpose(-1, -2) @ matrix(labels["handoff_quaternion"])
                angles = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
                aux["candidate_rotation_deg"].extend((angles * (180 / math.pi)).cpu().tolist())
                aux["candidate_time_error_seconds"].extend((result["goal_time"] - labels["handoff_remaining_seconds"]).abs().cpu().tolist())
                for i, index in enumerate(indices):
                    row = ledger_entry(dataset, index, head="candidate")
                    observed.append(dict(**row, pred_xyz=positions[i].cpu().tolist(), target_xyz=labels["handoff_position"][i].cpu().tolist()))
            else:
                xyz, rotation, seconds, _ = conditions(batch)
                result = model.forward_skill(encoded, inputs["robot"], xyz, rotation, seconds)
                px, pr, pt, used = conditions(batch, predicted_goals)
                predicted = model.forward_skill(encoded, inputs["robot"], px, pr, pt)
                target = labels["action_sequence"]
                relative = result["rotation_matrices"][:, 0].transpose(-1, -2) @ matrix(target[:, 0, 3:7])
                angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
                aux["skill_rotation_deg"].extend((angle * (180 / math.pi)).cpu().tolist())
                aux["gripper_correct"].extend((result["gripper_logits"][:, 0].argmax(-1) == (target[:, 0, 7] >= .5).long()).cpu().tolist())
                aux["skill_class_correct"].extend((encoded["skill_logits"].argmax(-1) == labels["skill_label"]).cpu().tolist())
                for i, index in enumerate(indices):
                    row = ledger_entry(dataset, index, head="tcp")
                    observed.append(dict(**row, pred_xyz=result["positions"][i, 0].cpu().tolist(), target_xyz=target[i, 0, :3].cpu().tolist()))
                    for h in range(4):
                        if bool(labels["action_mask"][i, h]):
                            seq_row = ledger_entry(dataset, index, head="tcp", suffix=f"|h{h}")
                            sequence_observed.append(dict(**seq_row, pred_xyz=result["positions"][i, h].cpu().tolist(), target_xyz=target[i, h, :3].cpu().tolist()))
                    if bool(used[i]):
                        predicted_observed.append(dict(**row, pred_xyz=predicted["positions"][i, 0].cpu().tolist(), target_xyz=target[i, 0, :3].cpu().tolist()))
    report = precision_report(observed, expected_samples=expected, usage="selection")
    report["sequence_precision"] = precision_report(sequence_observed, expected_samples=sequence_expected, usage="selection")
    report["predicted_goal_conditioned_precision"] = precision_report(predicted_observed, expected_samples=predicted_expected, usage="selection")
    report["predicted_goal_conditioning_scope"] = "same offline expert states at or after first eligible scene frame; not a Native oracle paired experiment"
    report["auxiliary"] = {key: float(np.mean(values)) for key, values in aux.items()}
    report["scope"] = "offline expert command/goal imitation; no Native tracking-error or success claim"
    source_count = datasets["candidate"].base.source_count
    report["source_coverage"] = dict(expected_sources=source_count, predicted_goal_sources=len(predicted_goals),
                                     complete=len(predicted_goals) == source_count,
                                     candidate_anchor_scope="before first close, five causal frames, genuine current target forecast",
                                     missing_forecast_sources_never_count_as_precision_pass=True)
    return report, observed


def groups(report):
    return [row for split in report["by_split"].values() for task in split["tasks"].values() for row in task.values()]


def selection_key(report):
    if set(report["by_split"]) != {"validation"}:
        raise ValueError("only offline validation may select the checkpoint")
    rows = groups(report) + groups(report["sequence_precision"]) + groups(report["predicted_goal_conditioned_precision"])
    if len(groups(report)) != 4 or not all(row["cohort_complete"] for row in rows):
        raise ValueError("complete two-task candidate/TCP cohorts required")
    return (max(1 - row["fraction_le_5cm"] for row in rows), max(row["max_m"] for row in rows), max(row["mean_m"] for row in rows))


def qualified(report, split):
    return report["source_coverage"]["complete"] and all(value["by_split"][split]["hard_gate_passed"] for value in
               (report, report["sequence_precision"], report["predicted_goal_conditioned_precision"]))


def mean_target_met(report, split):
    return report["source_coverage"]["complete"] and all(value["by_split"][split]["mean_target_met"] for value in
               (report, report["sequence_precision"], report["predicted_goal_conditioned_precision"]))


def summary(report):
    return {split: {task: {head: {key: row[key] for key in ("mean_m", "p95_m", "max_m", "fraction_le_5cm")}
                           for head, row in heads.items()} for task, heads in item["tasks"].items()}
            for split, item in report["by_split"].items()}


def source_hashes():
    # Freeze fitting code now; evaluation concurrency can be developed while
    # optimization runs. Its final runtime code is frozen before Native use.
    paths = [ROOT / f"our_mani/hierarchy_block_20261102/{name}.py" for name in
             ("__init__", "data", "networks", "training")]
    paths += [ROOT / "scripts/train_hierarchy_block_20261102.py"]
    paths += [ROOT / "our_mani/hierarchy_20261030/networks.py", ROOT / "our_mani/hierarchy_20261030/training.py",
              ROOT / "our_mani/hierarchy_precision_20261030/precision.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def check_sources(identity):
    for name, expected in identity["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("bound source changed during training: " + name)
    for name, expected in identity.get("classifier_input_caches", {}).get("files", {}).items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("frozen classifier input cache changed: " + name)
    for reference in (identity["config"], identity["inherited_classifier"]):
        if sha(ROOT / reference["path"]) != reference["sha256"]:
            raise RuntimeError("bound configuration or initialization changed")


def checkpoint(path, model, metadata):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(version=VERSION, model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    metadata=metadata), temporary)
    temporary.replace(path)


def run(args):
    torch.set_num_threads(4)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if (config["epochs"] != 120 or config["position_target_m"] != .03 or config["position_limit_m"] != .05
            or config["TEST_used_for_fitting_or_selection"] is not False):
        raise ValueError("declared two-task training and physical precision protocol changed")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    store = BlockData(ROOT)
    base = {s: {h: BlockDataset(store, split=s, head=h) for h in ("skill", "candidate")}
            for s in ("train", "validation")}
    positions = np.concatenate([e["scene"]["expert_actions"][:, :3] for e in store.episodes["train"]])
    lower, upper = positions.min(0) - .05, positions.max(0) + .05
    center, scale = (upper + lower) / 2, (upper - lower) / 2
    initial = torch.load(ROOT / INITIAL, map_location="cpu", weights_only=True)
    if initial.get("metadata", {}).get("stage") != "classification":
        raise ValueError("completed inherited scene-classification initialization required")
    model = BlockHierarchyNetwork(center.tolist(), scale.tolist(), initial["encoder_state_dict"], **config["model_config"])
    if args.check:
        for head in ("skill", "candidate"):
            raw = next(iter(DataLoader(base["train"][head], batch_size=2)))
            inputs, labels = raw["inputs"], raw["supervision"]
            encoded = encode(model, inputs)
            if head == "candidate":
                result = model.forward_candidate(encoded, inputs["robot"], inputs["trajectory"], inputs["trajectory_mask"])
                loss = candidate_loss(result, labels, raw["weight"])
            else:
                goal, rotation, seconds, _ = conditions(raw)
                result = model.forward_skill(encoded, inputs["robot"], goal, rotation, seconds)
                loss = skill_loss(result, labels, raw["weight"], config["sequence_step_weights"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite real-source CPU dry run")
        print(json.dumps(dict(version=VERSION, mode="CPU_check_only", optimizer_steps=0, TEST_read=False,
                              summaries={s: {h: d.summary for h, d in ds.items()} for s, ds in base.items()}), ensure_ascii=False), flush=True)
        return
    output, runtime = ROOT / OUT, ROOT / RUNTIME
    if output.exists() or runtime.exists():
        raise ValueError("fresh training namespace required; never overwrite or silently resume")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("explicit available cuda:0 required for optimization")
    output.mkdir(parents=True)
    runtime.mkdir(parents=True)
    device = torch.device(args.device)
    model.to(device)
    identity = dict(data=store.identity, code_sha256=source_hashes(),
                    config=dict(path=str(config_path.relative_to(ROOT)), sha256=sha(config_path)),
                    inherited_classifier=dict(path=str(INITIAL), sha256=sha(ROOT / INITIAL)),
                    inherited_classifier_scope="previous four-variant classification pretraining; no other-task arrays fitted in this run",
                    TEST_read=False)
    status = dict(version=VERSION, complete=False, stage="classifier_input_cache", process=process_identity(),
                  identity=identity, precision_qualified=False, epochs=[], completed_epochs=0, expected_epochs=120,
                  TEST_read=False, native_started=False, created_utc=utc(),
                  data_summaries={s: {h: d.summary for h, d in ds.items()} for s, ds in base.items()})
    atomic_json(output / "config.json", config)
    atomic_json(output / "source_ledger.json", store.source_ledger)
    stopped = False
    started = time.monotonic()

    def signal_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)

    def stop_check():
        if stopped or (runtime / "STOP").exists():
            raise InterruptedError("new block training administrative STOP")

    def write_status():
        status.update(updated_utc=utc(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "status.json", status)
        atomic_json(runtime / "status.json", status)

    lock_path = Path("/tmp/our-mani-hierarchy-20261030-gpu.lock")
    try:
        with lock_path.open("a+b") as gpu_lock:
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_status()
            datasets = {}
            cache_identity = {}
            for split in ("train", "validation"):
                stop_check()
                path = output / f"{split}_classifier_inputs.pt"
                arrays = materialize_classifier(model, base[split]["skill"], device, path, workers=args.workers)
                cache_identity[str(path.relative_to(ROOT))] = sha(path)
                indices = {anchor: i for i, anchor in enumerate(base[split]["skill"].anchors)}
                datasets[split] = {h: CachedDataset(d, arrays, indices) for h, d in base[split].items()}
            identity["classifier_input_caches"] = dict(files=cache_identity)
            atomic_json(output / "identity.json", identity)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=config["learning_rate"], weight_decay=config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=10, min_lr=1e-6)
            skill_loader = DataLoader(datasets["train"]["skill"], batch_size=config["skill_batch_size"], shuffle=True,
                                       num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
            candidate_loader = DataLoader(datasets["train"]["candidate"], batch_size=config["candidate_batch_size"], shuffle=True,
                                           num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
            best, selected_epoch = None, None
            for epoch in range(1, 121):
                stop_check()
                check_sources(identity)
                cache = (goal_cache(model, datasets["train"]["candidate"], device)
                         if epoch >= config["predicted_goal_training_from_epoch"] else None)
                model.train()
                status.update(stage="joint_geometry_candidate_skill", current_epoch=epoch)
                write_status()
                iterator, total = iter(candidate_loader), 0.
                epoch_start = time.monotonic()
                for step, raw_skill in enumerate(skill_loader, 1):
                    stop_check()
                    try:
                        raw_candidate = next(iterator)
                    except StopIteration:
                        iterator = iter(candidate_loader)
                        raw_candidate = next(iterator)
                    sb, cb = move(raw_skill, device), move(raw_candidate, device)
                    optimizer.zero_grad(set_to_none=True)
                    se, ce = encode(model, sb["inputs"]), encode(model, cb["inputs"])
                    gx, gr, gt, _ = conditions(sb, cache, training=True, config=config)
                    skill = model.forward_skill(se, sb["inputs"]["robot"], gx, gr, gt)
                    candidate = model.forward_candidate(ce, cb["inputs"]["robot"], cb["inputs"]["trajectory"], cb["inputs"]["trajectory_mask"])
                    loss = skill_loss(skill, sb["supervision"], sb["weight"], config["sequence_step_weights"])
                    loss = loss + candidate_loss(candidate, cb["supervision"], cb["weight"])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite joint physical pose training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach())
                    if step == 1 or step % 25 == 0:
                        status.update(step=step, steps=len(skill_loader), train_loss=total / step)
                        write_status()
                        print(json.dumps(dict(epoch=epoch, step=step, steps=len(skill_loader), loss=total / step,
                                              seconds=time.monotonic() - epoch_start)), flush=True)
                report, observed = evaluate(model, datasets["validation"], device, workers=args.workers)
                key = selection_key(report)
                scheduler.step(key[0] + key[1])
                row = dict(epoch=epoch, train_loss=total / len(skill_loader), selection_key=key,
                           offline_validation=summary(report), auxiliary=report["auxiliary"],
                           precision_qualified=qualified(report, "validation"), seconds=time.monotonic() - epoch_start)
                status["epochs"].append(row)
                metadata = dict(identity=identity, epoch=epoch, workspace_center=center.tolist(), workspace_half_width=scale.tolist(),
                                model_config=config["model_config"], model_contract=model.metadata, role_names=list(ROLE_NAMES),
                                training_complete=False, completed_epochs=epoch, TEST_read=False, score_semantics=SCORE,
                                validation_precision=report, precision_qualified=False)
                checkpoint(output / "last.pt", model, metadata)
                if best is None or key < best:
                    best, selected_epoch = key, epoch
                    checkpoint(output / "best.pt", model, metadata)
                    atomic_json(output / "best_validation_precision.json", report)
                    atomic_json(output / "best_validation_predictions.json", observed)
                status.update(completed_epochs=epoch, selected_epoch=selected_epoch, best_selection_key=best)
                write_status()
                print(json.dumps(row, ensure_ascii=False), flush=True)
            check_sources(identity)
            store.verify_identity()
            selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(selected["model_state_dict"], strict=True)
            reports = {}
            for split in ("train", "validation"):
                stop_check()
                reports[split], observed = evaluate(model, datasets[split], device, workers=args.workers)
                atomic_json(output / f"selected_{split}_predictions.json", observed)
            precision_qualified = all(qualified(report, split) for split, report in reports.items())
            final_report = dict(schema="hierarchy_block_offline_precision_20261102_v1", reports=reports,
                                precision_qualified=precision_qualified, task_success_measured=False,
                                precision_qualification_definition="all declared available candidate and expert-conditioned sequence outputs, plus predicted-goal first outputs, <=5cm; all parents have a genuine forecast",
                                mean_3cm_target_met=all(mean_target_met(report, split) for split, report in reports.items()),
                                TEST_read=False, samples_filtered=False, selected_epoch=selected_epoch)
            precision_path = output / "offline_precision_report.json"
            atomic_json(precision_path, final_report)
            selected["metadata"].update(training_complete=True, completed_epochs=120, selected_epoch=selected_epoch,
                                         precision_qualified=precision_qualified,
                                         precision_qualification_definition=final_report["precision_qualification_definition"],
                                         mean_3cm_target_met=final_report["mean_3cm_target_met"],
                                         precision_report=dict(path=str(precision_path.relative_to(ROOT)), sha256=sha(precision_path)))
            temporary = output / "best.pt.partial"
            torch.save(selected, temporary)
            temporary.replace(output / "best.pt")
            status.update(stage="complete", complete=True, completed_epochs=120, selected_epoch=selected_epoch,
                          best_checkpoint_sha256=sha(output / "best.pt"), precision_qualified=precision_qualified,
                          precision_qualification_definition=final_report["precision_qualification_definition"],
                          mean_3cm_target_met=final_report["mean_3cm_target_met"],
                          precision_report=selected["metadata"]["precision_report"], finished_utc=utc(),
                          next_stage="same-checkpoint fixed Native TRAIN8 then VAL40 and TEST80; precision not inferred from success")
            write_status()
    except BaseException as error:
        status.update(stage="administrative_stopped" if isinstance(error, InterruptedError) else "failed",
                      error=repr(error), complete=False, finished_utc=utc())
        write_status()
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="config/framework/hierarchy_block_20261102_v1.json")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--check", action="store_true")
    return result


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
