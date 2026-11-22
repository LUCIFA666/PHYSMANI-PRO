"""Train a fresh precision version without changing the live v1 evaluation.

Positions use physical Euclidean metres. All offline validation samples enter
checkpoint selection; neither Native outcomes nor TEST sources are opened.
Completing optimization never grants a 5cm qualification by itself.
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
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from our_mani.hierarchy_20261030.training import quaternion_matrix
from .data import PrecisionData, PrecisionDataset
from .networks import PrecisionHeads
from .precision import position_loss_m, precision_report


ROOT = Path(__file__).resolve().parents[2]
VERSION = "hierarchy_precision_training_20261030_v1"
BASE = Path("artifacts/framework/multitask_r10/hierarchy_20261030")
OUTPUT = Path("artifacts/framework/multitask_r10/hierarchy_precision_20261030/train_v1")
RUNTIME = Path("runtime/framework/hierarchy_precision_20261030/train_v1")
SOURCES = (
    "our_mani/hierarchy_precision_20261030/__init__.py",
    "our_mani/hierarchy_precision_20261030/data.py",
    "our_mani/hierarchy_precision_20261030/networks.py",
    "our_mani/hierarchy_precision_20261030/precision.py",
    "our_mani/hierarchy_precision_20261030/training.py",
    "scripts/train_hierarchy_precision_20261030.py",
    "our_mani/hierarchy_20261030/networks.py",
    "our_mani/hierarchy_20261030/candidate_multibody.py",
    "our_mani/hierarchy_20261030/training.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity():
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(stat[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward(model, batch, head):
    shared = (batch["rgbd_current"], batch["frozen_scene"], batch["skill_probabilities"], batch["robot"])
    if head == "candidate":
        return model.forward_candidate(*shared, batch["trajectory"], batch["trajectory_mask"])
    return model.forward_skill(*shared, batch["goal_xyz"], batch["goal_time"])


def metric_position_objective(predicted, target):
    """Dimensionless training penalty with 3cm/5cm physical thresholds.

    Nothing clips a prediction or discards a distant point. The hard gate is
    computed later from every observed physical distance, never from this loss.
    """
    distances = position_loss_m(predicted, target, split="train", reduction="none")
    return ((distances / .03).square()
            + 4 * (F.relu(distances - .03) / .03).square()
            + 8 * (F.relu(distances - .05) / .03).square())


def weighted_mean(values, weights):
    if (values.ndim != 1 or values.shape != weights.shape or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all())):
        raise ValueError("finite per-sample losses and strictly positive aligned weights required")
    return (values * weights).sum() / weights.sum()


def losses(prediction, batch, head, model=None):
    xyz = prediction["goal_xyz"] if head == "candidate" else prediction["xyz"]
    values = metric_position_objective(xyz, batch["target_xyz"])
    if head == "candidate":
        if model is None:
            raise ValueError("candidate model workspace required for grid-cell supervision")
        normalized = ((batch["target_xyz"] - model.candidate.workspace_center)
                      / model.candidate.workspace_scale)
        if not bool((normalized.abs() <= 1).all()):
            raise ValueError("TRAIN handoff outside its declared TRAIN-fitted workspace")
        cells = ((normalized + 1) * 8).floor().long().clamp(0, 15)
        indices = cells[:, 0] * 256 + cells[:, 1] * 16 + cells[:, 2]
        values = values + F.cross_entropy(prediction["point_logits"], indices, reduction="none")
        supervised_cell_xyz = prediction["candidate_xyz"][torch.arange(len(indices), device=indices.device), indices]
        values = values + metric_position_objective(supervised_cell_xyz, batch["target_xyz"])
        # Expected spatial distance sends a physical-position gradient to the
        # grid probabilities even when argmax selects a wrong cell.
        distances_squared = (prediction["candidate_xyz"] - batch["target_xyz"][:, None]).square().sum(-1)
        values = values + .25 * (prediction["point_probabilities"] * distances_squared / (.03 ** 2)).sum(-1)
        # Continuous timing uses the actual per-anchor nominal label. No fake
        # .5s classification label and no truncation at the 1.1s forecast edge.
        values = values + .1 * ((prediction["goal_time"] - batch["target_time_seconds"]) / .5).square()
    else:
        matrix = quaternion_matrix(batch["target_quaternion"])
        rotation = (prediction["rotation_matrix"] - matrix).square().mean((-1, -2))
        gripper = F.cross_entropy(prediction["gripper_logits"],
                                  (batch["target_gripper"] >= .5).long(), reduction="none")
        values = values + .2 * rotation + .1 * gripper
    return weighted_mean(values, batch["weight"])


def group_metrics(report, split):
    return [metrics for heads in report["by_split"][split]["tasks"].values() for metrics in heads.values()]


def selection_key(report):
    """All-head, all-task offline-VAL precision; never consult TEST."""
    if set(report["by_split"]) != {"validation"}:
        raise ValueError("only offline validation may select a precision checkpoint")
    groups = group_metrics(report, "validation")
    if len(groups) != 8 or not all(row["cohort_complete"] for row in groups):
        raise ValueError("four tasks and both complete finite heads required for selection")
    return (max(1 - row["fraction_le_5cm"] for row in groups),
            max(row["max_m"] for row in groups), max(row["mean_m"] for row in groups))


@torch.inference_mode()
def evaluate(model, datasets, device, *, workers=0):
    model.eval()
    records, expected, auxiliary = [], [], {}
    for dataset in datasets:
        head = "candidate" if dataset.head == "candidate" else "tcp"
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        expected.extend(dataset.expected_samples)
        angle_errors, gripper_correct, time_errors = [], [], []
        for raw in loader:
            batch = move(raw, device)
            prediction = forward(model, batch, head)
            xyz = prediction["goal_xyz"] if head == "candidate" else prediction["xyz"]
            indices = raw["sample_index"].tolist()
            for index, predicted, target in zip(indices, xyz.cpu().tolist(), raw["target_xyz"].tolist()):
                records.append(dict(**dataset.expected_samples[index], pred_xyz=predicted, target_xyz=target))
            if head == "candidate":
                time_errors.extend((prediction["goal_time"] - batch["target_time_seconds"]).abs().cpu().tolist())
            else:
                rotation = prediction["rotation_matrix"].transpose(-1, -2) @ quaternion_matrix(batch["target_quaternion"])
                angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
                angle_errors.extend((angle * (180 / math.pi)).cpu().tolist())
                gripper_correct.extend((prediction["gripper_logits"].argmax(-1)
                                        == (batch["target_gripper"] >= .5).long()).cpu().tolist())
        auxiliary[head] = (dict(nominal_time_MAE_seconds=float(np.mean(time_errors))) if head == "candidate" else
                           dict(rotation_mean_deg=float(np.mean(angle_errors)),
                                gripper_accuracy=float(np.mean(gripper_correct))))
    report = precision_report(records, expected_samples=expected,
                              usage="selection" if datasets[0].split == "validation" else "evaluation")
    report["auxiliary_not_position_qualification"] = auxiliary
    report["metric_scope"] = "offline expert-target prediction, not actual Native TCP/object tracking or task success"
    return report, records


def brief(report, split):
    return {task: {head: {key: row[key] for key in
                            ("mean_m", "p95_m", "max_m", "fraction_le_5cm", "hard_gate_passed")}
                    for head, row in heads.items()}
            for task, heads in report["by_split"][split]["tasks"].items()}


def save_checkpoint(path, model, base_encoder, metadata):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(version=VERSION,
                    heads_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    base_encoder_state_dict=base_encoder,
                    metadata=metadata), temporary)
    temporary.replace(path)


def verify_sources(identity):
    for name, expected in identity["code_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("precision source changed during optimization: " + name)


@contextmanager
def exclusive_gpu():
    # The existing v1 owner uses this same training lock and has released it.
    path = ROOT / "runtime/framework/hierarchy_20261030/gpu_training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args):
    torch.set_num_threads(4)
    config_path = (ROOT / args.config).resolve(strict=True)
    config = json.loads(config_path.read_text())
    if (config["target_error_m"] != .03 or config["maximum_error_m"] != .05
            or config["epochs"] != 120 or config["TEST_used_for_training_or_selection"] is not False):
        raise ValueError("declared fresh 3cm/5cm training contract differs")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    store = PrecisionData(ROOT, ROOT / BASE / "data_v1/manifest.json",
                          ROOT / BASE / "skill_handoff_pretrain_v2", ROOT / BASE / "candidate_pretrain_v1")
    datasets = {split: {head: PrecisionDataset(store, split=split, head=head)
                        for head in ("skill", "candidate")} for split in ("train", "validation")}
    skill_path = ROOT / BASE / "skill_handoff_pretrain_v2/best_skill.pt"
    candidate_path = ROOT / BASE / "candidate_pretrain_v1/best_candidate.pt"
    skill, candidate = (torch.load(path, map_location="cpu", weights_only=True)
                        for path in (skill_path, candidate_path))
    metadata = candidate["metadata"]
    model = PrecisionHeads(metadata["workspace_center"], metadata["workspace_half_width"], metadata["role_names"],
                           skill["encoder_state_dict"], skill["skill_state_dict"], candidate["candidate_state_dict"])
    identity = dict(config=dict(path=str(config_path.relative_to(ROOT)), sha256=digest(config_path)),
                    code_sha256={name: digest(ROOT / name) for name in SOURCES},
                    data=store.identity,
                    initial_skill=dict(path=str(skill_path.relative_to(ROOT)), sha256=digest(skill_path)),
                    initial_candidate=dict(path=str(candidate_path.relative_to(ROOT)), sha256=digest(candidate_path)),
                    Native_or_TEST_used_for_training_or_selection=False)
    if args.check:
        for head, dataset in datasets["train"].items():
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
            result = forward(model, batch, head)
            loss = losses(result, batch, head, model)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite real-source dry run")
        print(json.dumps(dict(version=VERSION, mode="CPU_check_only", optimizer_steps=0,
                              dataset_summaries={s: {h: d.summary for h, d in ds.items()} for s, ds in datasets.items()},
                              identity=identity, native_started=False, TEST_read=False), ensure_ascii=False), flush=True)
        return
    output, runtime = ROOT / OUTPUT, ROOT / RUNTIME
    if output.exists() or runtime.exists():
        raise ValueError("fresh precision namespace required; never overwrite an old run")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("precision training explicitly requires available cuda:0")
    output.mkdir(parents=True)
    runtime.mkdir(parents=True)
    atomic_json(output / "config.json", config)
    atomic_json(output / "identity.json", identity)
    device = torch.device(args.device)
    model.to(device)
    status = dict(version=VERSION, process=process_identity(), stage="initial_precision_audit", complete=False,
                  current_epoch=0, expected_epochs=config["epochs"], TEST_read=False,
                  original_v1_evaluation_modified=False, native_started=False,
                  precision_qualified=False, identity=identity, epochs=[], created_utc=utc())
    stopped = False

    def handle_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    def check_stop():
        if stopped or (runtime / "STOP").exists():
            raise InterruptedError("precision training administrative STOP")

    def write_status():
        status.update(updated_utc=utc(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "status.json", status)
        atomic_json(runtime / "status.json", status)

    started = time.monotonic()
    try:
        with exclusive_gpu():
            write_status()
            for split in ("train", "validation"):
                report, _ = evaluate(model, list(datasets[split].values()), device, workers=args.workers)
                atomic_json(output / f"initial_{split}_precision.json", report)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=config["learning_rate"], weight_decay=config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=.5, min_lr=1e-6)
            skill_loader = DataLoader(datasets["train"]["skill"], batch_size=config["skill_batch_size"], shuffle=True,
                                       num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
            candidate_loader = DataLoader(datasets["train"]["candidate"], batch_size=config["candidate_batch_size"], shuffle=True,
                                           num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
            best_key, best_epoch = None, None
            for epoch in range(1, config["epochs"] + 1):
                verify_sources(identity)
                check_stop()
                model.train()
                status.update(stage="precision_joint_training", current_epoch=epoch)
                write_status()
                candidate_iterator = iter(candidate_loader)
                epoch_started, loss_sum = time.monotonic(), 0.
                for step, skill_raw in enumerate(skill_loader, 1):
                    check_stop()
                    try:
                        candidate_raw = next(candidate_iterator)
                    except StopIteration:
                        candidate_iterator = iter(candidate_loader)
                        candidate_raw = next(candidate_iterator)
                    optimizer.zero_grad(set_to_none=True)
                    skill_batch, candidate_batch = move(skill_raw, device), move(candidate_raw, device)
                    loss = losses(forward(model, skill_batch, "tcp"), skill_batch, "tcp", model)
                    loss = loss + losses(forward(model, candidate_batch, "candidate"), candidate_batch, "candidate", model)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite physical precision training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    if step == 1 or step % 50 == 0:
                        status.update(step=step, steps=len(skill_loader), train_loss=loss_sum / step,
                                      gradient_norm=float(norm))
                        write_status()
                        print(json.dumps(dict(epoch=epoch, step=step, steps=len(skill_loader),
                                              train_loss=loss_sum / step, seconds=time.monotonic() - epoch_started)), flush=True)
                report, records = evaluate(model, list(datasets["validation"].values()), device, workers=args.workers)
                key = selection_key(report)
                scheduler.step(key[0] + key[1])
                row = dict(epoch=epoch, train_loss=loss_sum / len(skill_loader), selection_key=key,
                           validation=brief(report, "validation"),
                           hard_5cm_gate=report["by_split"]["validation"]["hard_gate_passed"],
                           mean_3cm_target=report["by_split"]["validation"]["mean_target_met"],
                           learning_rate=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - epoch_started)
                status["epochs"].append(row)
                checkpoint_metadata = dict(identity=identity, epoch=epoch, validation=report,
                                           workspace_center=metadata["workspace_center"],
                                           workspace_half_width=metadata["workspace_half_width"], role_names=metadata["role_names"],
                                           model_contract=model.metadata, precision_qualified=False,
                                           optimization_complete=False, TEST_read=False, Native_success_measured=False)
                save_checkpoint(output / "last.pt", model, skill["encoder_state_dict"], checkpoint_metadata)
                if best_key is None or key < best_key:
                    best_key, best_epoch = key, epoch
                    save_checkpoint(output / "best.pt", model, skill["encoder_state_dict"], checkpoint_metadata)
                    atomic_json(output / "best_validation_precision.json", report)
                    atomic_json(output / "best_validation_predictions.json", records)
                atomic_json(output / "last_validation_precision.json", report)
                status.update(selected_epoch=best_epoch, best_selection_key=best_key)
                write_status()
                print(json.dumps(row, ensure_ascii=False), flush=True)
            verify_sources(identity)
            if hasattr(store, "verify_identity"):
                store.verify_identity()
            selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(selected["heads_state_dict"], strict=True)
            final_reports = {}
            for split in ("train", "validation"):
                check_stop()
                report, records = evaluate(model, list(datasets[split].values()), device, workers=args.workers)
                final_reports[split] = report
                atomic_json(output / f"selected_{split}_precision.json", report)
                atomic_json(output / f"selected_{split}_predictions.json", records)
            qualified = all(r["by_split"][s]["hard_gate_passed"] for s, r in final_reports.items())
            selected["metadata"].update(optimization_complete=True, completed_epochs=config["epochs"],
                                         precision_qualified=qualified,
                                         qualification_scope="complete offline expert prediction cohort only; no Native accuracy/success claim",
                                         mean_3cm_target_met=all(r["by_split"][s]["mean_target_met"] for s, r in final_reports.items()))
            torch.save(selected, output / "qualified_offline.pt" if qualified else output / "unqualified_offline.pt")
            status.update(stage="complete" if qualified else "complete_not_precision_qualified", complete=True,
                          precision_qualified=qualified, selected_epoch=best_epoch,
                          final_precision={s: brief(r, s) for s, r in final_reports.items()}, finished_utc=utc(),
                          next_stage="Native TRAIN accuracy audit required" if qualified else "repair training based on TRAIN/offline-VAL only; do not launch Native TEST")
            write_status()
    except BaseException as error:
        status.update(stage="administrative_stopped" if isinstance(error, InterruptedError) else "failed",
                      complete=False, error=repr(error), finished_utc=utc())
        write_status()
        raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default="config/framework/hierarchy_precision_20261030_v1.json")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--check", action="store_true")
    return command


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
