"""Train only a multi-body handoff selector on real blind public forecasts.

The fixed episode handoff label is privileged offline supervision. It is never
fed to the frozen scene encoder or used to invent a missing body trajectory.
Unknown frames and parents remain in an explicit zero-weight source ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .candidate_multibody import MultiBodyCandidateTransformer, candidate_imitation_loss
from .data import SCHEMA as DATA_SCHEMA, TASK_FAMILIES, file_sha256
from .handoff_training_v2 import (
    VERSION as SKILL_VERSION, GOAL_SOURCE, EpisodeHandoffSkillWindows,
    episode_handoff_goal,
)
from .networks import CausalSceneSkillEncoder, PretrainedSkillPolicy
from .training import atomic_json, inputs_to_device, utc


ROOT = Path(__file__).resolve().parents[2]
VERSION = "hierarchy_multibody_candidate_pretraining_20261030_v1"
MOTION_SCHEMA = "hierarchy_public_motion_inputs_20261030_v1"
ROLE_NAMES = ("target", "frame", "peg0", "peg1", "peg2", "ring")
FORECAST_SHA256 = "ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef"
SCORE_SEMANTICS = "expert_handoff_point_imitation_probability_not_native_success_probability"
DEFAULT_DATA = "artifacts/framework/multitask_r10/hierarchy_20261030/data_v1/manifest.json"
DEFAULT_MOTION = "artifacts/framework/multitask_r10/hierarchy_20261030/motion_inputs_blind_v1/manifest.json"
DEFAULT_SKILL = "artifacts/framework/multitask_r10/hierarchy_20261030/skill_handoff_pretrain_v2/best_skill.pt"
DEFAULT_SKILL_STATUS = "artifacts/framework/multitask_r10/hierarchy_20261030/skill_handoff_pretrain_v2/status.json"
DEFAULT_OUTPUT = "artifacts/framework/multitask_r10/hierarchy_20261030/candidate_pretrain_v1"
DEFAULT_READINESS = "runtime/framework/hierarchy_20261030/candidate_readiness_v1.json"
FIXED_CONFIG = dict(seed=20261030, epochs=40, batch_size=32, learning_rate=.0003,
                    weight_decay=.0001, optimizer="AdamW", frame_stride=2,
                    minimum_history_frames=5, fixed_goal_time_seconds=.5,
                    time_target_index=10, selection="best offline validation imitation loss",
                    weighting="equal mass per parent with eligible anchors; equal anchors within that parent; zero-parent ledger retained",
                    role_names=list(ROLE_NAMES), score_semantics=SCORE_SEMANTICS,
                    model_dim=128, heads=4, layers=2)


def _key(row):
    return row["split"], row["task"], row["raw_episode_id"]


def check_full_source_coverage(data_manifest, motion_manifest, data_manifest_sha256):
    """Pure membership/provenance gate, independent of success or label values."""
    if (data_manifest.get("schema") != DATA_SCHEMA or data_manifest.get("complete") is not True
            or data_manifest.get("smoke_only") is not False
            or data_manifest.get("TEST_read") is not False
            or data_manifest.get("source_debug_target_positions_read") is not False):
        raise ValueError("complete exact public-input data manifest required")
    expected = {("train", task): 100 for task in TASK_FAMILIES}
    expected.update({("validation", "pick_moving_target_on_the_table"): 20,
                     ("validation", "pick_moving_target_on_the_table_high_speed"): 20,
                     ("validation", "insert_onto_rotating_peg"): 18,
                     ("validation", "insert_onto_rotating_peg_high_speed"): 19})
    rows = data_manifest["episodes"]
    if dict(Counter((r["split"], r["task"]) for r in rows)) != expected:
        raise ValueError("exact TRAIN400/offline validation77 membership differs")
    if (motion_manifest.get("schema") != MOTION_SCHEMA or motion_manifest.get("complete") is not True
            or motion_manifest.get("smoke_only") is not False
            or motion_manifest.get("expected_cases") != 477
            or motion_manifest.get("TEST_read") is not False
            or motion_manifest.get("source_debug_target_positions_read") is not False
            or motion_manifest.get("frontend") != "blind_parallel_fixed_common_TRAIN_priors"
            or motion_manifest.get("source_task_selects_frontend") is not False
            or motion_manifest.get("identity", {}).get("data_manifest_sha256") != data_manifest_sha256):
        raise ValueError("complete non-smoke source-blind 477-parent motion manifest required")
    cases = motion_manifest["cases"]
    keys, motion_keys = [_key(r) for r in rows], [_key(r) for r in cases]
    if len(set(keys)) != 477 or len(motion_keys) != 477 or len(set(motion_keys)) != 477 or set(keys) != set(motion_keys):
        raise ValueError("motion manifest must retain each of the 477 exact parent sources once")
    source_by_key = {_key(r): r for r in rows}
    for case in cases:
        row = source_by_key[_key(case)]
        if case["parent_source_group"] != row["parent_source_group"] or case.get("source_path") != row["source_path"]:
            raise ValueError("motion and scene source identities differ")
    return source_by_key


def validate_motion_arrays(arrays, audit, source_frames):
    """Validate real per-body clocks, explicit unknown masks and history proof."""
    if tuple(arrays["role_names"].tolist()) != ROLE_NAMES:
        raise ValueError("fixed source-blind body-role ordering differs")
    times = np.asarray(arrays["timestamps"], dtype=np.float64)
    trajectory = np.asarray(arrays["predicted_trajectory"], dtype=np.float32)
    mask = np.asarray(arrays["trajectory_mask"])
    n = len(times)
    if trajectory.shape != (n, 6, 23, 20) or mask.dtype != np.bool_ or mask.shape != (n, 6, 23):
        raise ValueError("real motion forecast/mask shapes differ")
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("motion timestamps must be finite and strictly causal")
    frame_ids = np.rint(times / .05).astype(np.int64)
    if np.any(frame_ids >= source_frames) or not np.allclose(times, frame_ids * .05, atol=1e-8, rtol=0):
        raise ValueError("motion timestamps do not align with the original scene frame clock")
    if np.any(mask.any(-1) != mask.all(-1)):
        raise ValueError("partial invented body forecasts are forbidden; complete 23 steps or unknown required")
    if not np.isfinite(trajectory[mask]).all():
        raise ValueError("nonfinite valid forecast tokens")
    expected_times = np.arange(23, dtype=np.float32) * .05
    for row, body in np.argwhere(mask.all(-1)):
        if not np.allclose(trajectory[row, body, :, -1], expected_times, atol=1e-6, rtol=0):
            raise ValueError("each body must retain its own true 0..1.1 second clock")
    receipts = audit["frames"]
    if len(receipts) != n:
        raise ValueError("motion frame receipt count differs")
    histories = np.zeros((n, 6), dtype=np.int64)
    for index, receipt in enumerate(receipts):
        if abs(float(receipt["timestamp"]) - times[index]) > 1e-8:
            raise ValueError("motion audit timestamps differ from cached forecasts")
        statuses = receipt["motion_status"]
        if set(statuses) != set(ROLE_NAMES):
            raise ValueError("motion audit lacks fixed-role availability evidence")
        for body, role in enumerate(ROLE_NAMES):
            status = statuses[role]
            count = int(status["history_frames"])
            if count < 0:
                raise ValueError("negative causal body history length")
            histories[index, body] = count
            known = bool(mask[index, body].all())
            if bool(status["available"]) != known or known and count < 5:
                raise ValueError("known forecast lacks its five-frame causal history receipt")
    return trajectory, mask, frame_ids, histories


def select_candidate_anchors(actions, frame_ids, masks, histories, *, stride=2):
    """Keep a zero-weight decision for every fixed-stride source frame."""
    _, first_close = episode_handoff_goal(actions)
    n = len(actions)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    masks, histories = np.asarray(masks), np.asarray(histories)
    if (masks.shape != (len(frame_ids), 6, 23) or masks.dtype != np.bool_
            or histories.shape != (len(frame_ids), 6) or stride != 2
            or len(set(frame_ids.tolist())) != len(frame_ids)):
        raise ValueError("aligned real motion frames and the fixed stride-two contract required")
    lookup = {int(frame): index for index, frame in enumerate(frame_ids)}
    eligible, decisions = [], []
    for frame in range(0, n, stride):
        row = lookup.get(frame)
        if frame >= first_close:
            reason = "at_or_after_first_expert_close"
        elif frame + 1 < 5:
            reason = "scene_history_fewer_than_five_frames"
        elif row is None:
            reason = "motion_frame_missing"
        elif not bool((masks[row].all(-1) & (histories[row] >= 5)).any()):
            reason = "no_current_real_forecast_with_five_frame_history"
        else:
            reason = None
            eligible.append((frame, row))
        decisions.append(dict(frame=frame, motion_row=row, eligible=reason is None,
                              zero_weight_reason=reason, weight=0.))
    return eligible, decisions, first_close


class CandidateWindows(EpisodeHandoffSkillWindows):
    """Exact causal scene histories plus independently bound real forecasts."""
    def __init__(self, root, data_manifest, motion_manifest, split, history_frames=5):
        if history_frames != 5:
            raise ValueError("candidate scene-history length is fixed at five")
        super().__init__(root, data_manifest, split, history_frames, smoke=False)
        self.motion_arrays, self.source_ledger = [], []
        source_frames = len(self.anchors)
        new_anchors, motion_rows, raw_weights = [], [], []
        cases = {_key(row): row for row in motion_manifest["cases"]}
        for episode_index, (row, scene) in enumerate(zip(self.rows, self.episodes)):
            case = cases[_key(row)]
            n = len(scene["robot"])
            with np.load(self.root / row["npz_path"], allow_pickle=False) as cached:
                if not np.array_equal(cached["timestamps"], np.arange(n, dtype=np.float64) * .05):
                    raise ValueError("scene cache clock changed")
            failure = case.get("error") or case.get("infrastructure_error") or case.get("failed")
            if failure and not case.get("npz_path"):
                trajectory = np.empty((0, 6, 23, 20), dtype=np.float32)
                mask = np.empty((0, 6, 23), dtype=np.bool_)
                frames, histories = np.empty(0, dtype=np.int64), np.empty((0, 6), dtype=np.int64)
                motion_ref, audit_ref = None, None
            else:
                motion_path = (self.root / case["npz_path"]).resolve(strict=True)
                audit_path = (self.root / case["audit_path"]).resolve(strict=True)
                if (not motion_path.is_relative_to(self.root) or not audit_path.is_relative_to(self.root)
                        or file_sha256(motion_path) != case["sha256"]
                        or file_sha256(audit_path) != case["audit_sha256"]):
                    raise ValueError("motion cache/audit content identity differs")
                with np.load(motion_path, allow_pickle=False) as stored:
                    arrays = {key: stored[key] for key in ("timestamps", "role_names", "predicted_trajectory", "trajectory_mask")}
                audit = json.loads(audit_path.read_text())
                summary = case["summary"]
                if (summary.get("checkpoint_sha256") != FORECAST_SHA256
                        or summary.get("original_gru_weights_unchanged") is not True
                        or summary.get("expert_stage_or_target_inputs") is not False
                        or summary.get("native_instance_ids_used") is not False
                        or summary.get("prior_receipt", {}).get("all_frontends_run_on_every_scene") is not True
                        or summary.get("prior_receipt", {}).get("task_id_or_description_input") is not False):
                    raise ValueError("motion case lacks frozen original-GRU/source-blind public-input evidence")
                trajectory, mask, frames, histories = validate_motion_arrays(arrays, audit, n)
                motion_ref = dict(path=case["npz_path"], sha256=case["sha256"])
                audit_ref = dict(path=case["audit_path"], sha256=case["audit_sha256"])
            eligible, decisions, first_close = select_candidate_anchors(scene["expert_actions"], frames, mask, histories)
            if failure:
                for decision in decisions:
                    if decision["zero_weight_reason"] == "motion_frame_missing":
                        decision["zero_weight_reason"] = "documented_motion_case_failure"
            mass = 1. / len(eligible) if eligible else 0.
            for frame, motion_row in eligible:
                new_anchors.append((episode_index, frame)); motion_rows.append(motion_row); raw_weights.append(mass)
            for decision in decisions:
                if decision["eligible"]:
                    decision["weight"] = mass
            self.motion_arrays.append((trajectory, mask))
            counts = Counter(d["zero_weight_reason"] for d in decisions if d["zero_weight_reason"] is not None)
            self.source_ledger.append(dict(
                split=split, task=row["task"], raw_episode_id=row["raw_episode_id"],
                parent_source_group=row["parent_source_group"], source_path=row["source_path"],
                scene_cache={"path": row["npz_path"], "sha256": row["cache_sha256"]},
                source_small_arrays_sha256=row.get("source_small_arrays_sha256"), source_file_stat=row.get("source_file_stat"),
                motion_cache=motion_ref, motion_audit=audit_ref,
                source_frames=n, first_closed_command_frame=first_close,
                handoff_goal_xyz=self.handoff_goals[episode_index].tolist(),
                valid_anchors=len(eligible), parent_mass_before_global_normalization=1. if eligible else 0.,
                zero_weight_parent=not bool(eligible), zero_weight_reasons=dict(counts),
                unsampled_stride_frames=n - len(decisions), anchor_decisions=decisions,
                documented_motion_failure=str(failure) if failure else None,
            ))
        self.anchors, self.motion_rows = new_anchors, motion_rows
        self.weights = np.asarray(raw_weights, dtype=np.float64)
        if len(self.weights):
            self.weights *= len(self.weights) / self.weights.sum()
        by_task = {}
        for task in TASK_FAMILIES:
            parents = [r for r in self.source_ledger if r["task"] == task]
            by_task[task] = dict(parents=len(parents), valid_parents=sum(not r["zero_weight_parent"] for r in parents),
                                 zero_weight_parents=sum(r["zero_weight_parent"] for r in parents),
                                 valid_anchors=sum(r["valid_anchors"] for r in parents))
        self.summary.update(source_frames=source_frames, eligible_anchors=len(self.anchors),
                            frame_stride=2, per_task=by_task,
                            weighting=FIXED_CONFIG["weighting"], role_names=list(ROLE_NAMES))

    def __getitem__(self, index):
        base = super().__getitem__(index)
        episode, _ = self.anchors[index]
        motion_row = self.motion_rows[index]
        trajectory, mask = self.motion_arrays[episode]
        return {"rgbd": base["rgbd"], "robot_history": base["robot_history"], "lengths": base["lengths"],
                "trajectory": trajectory[motion_row], "trajectory_mask": mask[motion_row],
                "target_xyz": base["goal_xyz"], "weight": base["weight"],
                "episode": episode, "frame": base["frame"]}


@torch.inference_mode()
def encode_candidate_batch(encoder, batch):
    """The label target is deliberately absent from the encoder call."""
    encoded = encoder(batch["rgbd"], batch["robot_history"], batch["lengths"])
    rows = torch.arange(len(batch["lengths"]), device=batch["robot_history"].device)
    robot = batch["robot_history"][rows, batch["lengths"] - 1]
    return dict(scene_features=encoded["scene_features"], skill_latent=encoded["skill_latent"], robot=robot,
                trajectory=batch["trajectory"], trajectory_mask=batch["trajectory_mask"])


class FrozenCandidateFeatures(Dataset):
    def __init__(self, source, arrays):
        self.arrays, self.weights, self.rows = arrays, source.weights, source.rows

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}


@torch.inference_mode()
def materialize_candidate_features(source, encoder, device, output, identity, workers=0):
    pieces = {}
    loader = DataLoader(source, batch_size=32, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    encoder.eval()
    for step, batch in enumerate(loader, 1):
        data = inputs_to_device(batch, device)
        values = encode_candidate_batch(encoder, data)
        values.update({key: data[key] for key in ("target_xyz", "weight", "episode", "frame")})
        for key, value in values.items():
            pieces.setdefault(key, []).append(value.cpu())
        if step == 1 or step % 50 == 0:
            print(json.dumps(dict(stage="candidate_frozen_features", split=source.split,
                                  step=step, steps=len(loader))), flush=True)
    arrays = {key: torch.cat(values) for key, values in pieces.items()}
    inputs = {key: arrays[key] for key in ("scene_features", "skill_latent", "robot", "trajectory", "trajectory_mask")}
    labels = {key: arrays[key] for key in ("target_xyz", "weight", "episode", "frame")}
    torch.save(dict(identity=identity, split=source.split, input_arrays=inputs,
                    goal_labels_consumed_by_encoder=False), output / f"{source.split}_candidate_inputs.pt")
    torch.save(dict(identity=identity, split=source.split, supervision_arrays=labels,
                    label_usage="expert handoff target and source weighting only"), output / f"{source.split}_candidate_labels.pt")
    return FrozenCandidateFeatures(source, arrays)


def module_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def load_bound_inputs(args):
    data_path, motion_path = (ROOT / args.data_manifest).resolve(strict=True), (ROOT / args.motion_manifest).resolve(strict=True)
    skill_path, status_path = (ROOT / args.skill_checkpoint).resolve(strict=True), (ROOT / args.skill_status).resolve(strict=True)
    for path in (data_path, motion_path, skill_path, status_path):
        if not path.is_relative_to(ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030"):
            raise ValueError("all candidate sources must use the new hierarchy namespace")
    data, motion, status = json.loads(data_path.read_text()), json.loads(motion_path.read_text()), json.loads(status_path.read_text())
    check_full_source_coverage(data, motion, file_sha256(data_path))
    payload = torch.load(skill_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if (payload.get("version") != SKILL_VERSION or metadata.get("stage") != "skill"
            or metadata.get("smoke_only") is not False or metadata.get("goal_source") != GOAL_SOURCE
            or metadata.get("goal_latched_for_whole_episode") is not True
            or metadata.get("goal_time_seconds") != .5
            or status.get("version") != SKILL_VERSION or status.get("complete") is not True
            or status.get("smoke_only") is not False or status.get("best_skill_sha256") != file_sha256(skill_path)
            or metadata.get("identity") != status.get("identity")
            or metadata["identity"]["data_manifest"]["sha256"] != file_sha256(data_path)):
        raise ValueError("completed episode-handoff v2 skill/encoder checkpoint required")
    for name, digest in metadata["identity"]["code_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("frozen skill source code identity changed")
    for name, digest in motion.get("identity", {}).get("code_sha256", {}).items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("motion preparation source code identity changed")
    center, scale = metadata["workspace_center"], metadata["workspace_half_width"]
    encoder = CausalSceneSkillEncoder(); encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    policy = PretrainedSkillPolicy(center, scale); policy.load_state_dict(payload["skill_state_dict"], strict=True)
    for module in (encoder, policy):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    identity = dict(data_manifest={"path": str(data_path.relative_to(ROOT)), "sha256": file_sha256(data_path)},
                    motion_manifest={"path": str(motion_path.relative_to(ROOT)), "sha256": file_sha256(motion_path)},
                    skill_checkpoint={"path": str(skill_path.relative_to(ROOT)), "sha256": file_sha256(skill_path)},
                    skill_completed_status={"path": str(status_path.relative_to(ROOT)), "sha256": file_sha256(status_path)},
                    frozen_skill_identity=metadata["identity"], motion_identity=motion["identity"],
                    role_names=list(ROLE_NAMES), original_motion_checkpoint_sha256=FORECAST_SHA256)
    return data, motion, encoder, policy, center, scale, identity


def _readiness(args, diagnostic):
    path = (ROOT / args.readiness_out).resolve()
    if not path.is_relative_to(ROOT / "runtime/framework/hierarchy_20261030"):
        raise ValueError("readiness diagnostics must use the new runtime namespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, diagnostic)


@torch.inference_mode()
def evaluate_candidates(model, loader, device):
    totals = dict(weight=0., loss=0., goal_error_m=0., time_correct=0., outside_weight=0.)
    by_task = {task: dict(weight=0., loss=0., goal_error_m=0.) for task in TASK_FAMILIES}
    task_names = [row["task"] for row in loader.dataset.rows]
    model.eval()
    for batch in loader:
        data = inputs_to_device(batch, device)
        prediction = model(data["trajectory"], data["trajectory_mask"], data["skill_latent"], data["robot"], data["scene_features"])
        losses, outside = candidate_imitation_loss(model, prediction, data["target_xyz"])
        weights = data["weight"].double()
        error = (prediction["goal_xyz"] - data["target_xyz"]).norm(dim=-1)
        totals["weight"] += float(weights.sum()); totals["loss"] += float((losses * weights).sum())
        totals["goal_error_m"] += float((error * weights).sum())
        totals["time_correct"] += float(((prediction["time_index"] == 10) * weights).sum())
        totals["outside_weight"] += float((outside * weights).sum())
        episodes = data["episode"].cpu().tolist()
        for task in TASK_FAMILIES:
            keep = torch.tensor([task_names[i] == task for i in episodes], device=device)
            selected = weights * keep
            by_task[task]["weight"] += float(selected.sum())
            by_task[task]["loss"] += float((losses * selected).sum())
            by_task[task]["goal_error_m"] += float((error * selected).sum())
    if totals["weight"] <= 0:
        raise ValueError("no valid offline validation weight")
    return dict(loss=totals["loss"] / totals["weight"],
                goal_error_m=totals["goal_error_m"] / totals["weight"],
                fixed_time_index_accuracy=totals["time_correct"] / totals["weight"],
                outside_workspace_weight_fraction=totals["outside_weight"] / totals["weight"],
                per_task={task: dict(loss=value["loss"] / value["weight"],
                                     goal_error_m=value["goal_error_m"] / value["weight"])
                          for task, value in by_task.items() if value["weight"] > 0},
                score_semantics=SCORE_SEMANTICS, native_success_measured=False,
                goal_source=GOAL_SOURCE)


def _save_checkpoint(path, model, metadata):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(version=VERSION, candidate_state_dict=model.state_dict(), metadata=metadata), temporary)
    temporary.replace(path)


def run(args):
    started = time.perf_counter()
    readiness = dict(version=VERSION, checked_utc=utc(), ready=False,
                     candidate_fitted=False, native_success_measured=False,
                     TEST_read=False, config=FIXED_CONFIG)
    try:
        data, motion, encoder, policy, center, scale, source_identity = load_bound_inputs(args)
        training = CandidateWindows(ROOT, data, motion, "train")
        validation = CandidateWindows(ROOT, data, motion, "validation")
        readiness.update(source_identity=source_identity, train=training.summary,
                         offline_validation=validation.summary,
                         source_ledger=training.source_ledger + validation.source_ledger)
        missing = {split: [task for task, values in dataset.summary["per_task"].items() if values["valid_parents"] == 0]
                   for split, dataset in (("train", training), ("validation", validation))}
        readiness["zero_valid_parent_tasks"] = missing
        if any(missing.values()):
            raise ValueError("at least one task has zero eligible real-forecast parents; readiness ledger retained")
        readiness.update(ready=True, checked_seconds=time.perf_counter() - started,
                         encoder_trainable_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                         skill_trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
        _readiness(args, readiness)
        if args.check or args.dry_run:
            if args.dry_run:
                torch.set_num_threads(4)
                torch.manual_seed(FIXED_CONFIG["seed"])
                model = MultiBodyCandidateTransformer(center, scale, role_names=ROLE_NAMES)
                batch = next(iter(DataLoader(training, batch_size=min(2, len(training)), shuffle=False)))
                encoded = encode_candidate_batch(encoder, batch)
                with torch.no_grad():
                    prediction = model(encoded["trajectory"], encoded["trajectory_mask"], encoded["skill_latent"], encoded["robot"], encoded["scene_features"])
                    losses, outside = candidate_imitation_loss(model, prediction, batch["target_xyz"])
                readiness["CPU_dry_run"] = dict(examples=len(losses), loss=float(losses.mean()),
                    outside_workspace=int(outside.sum()), optimizer_steps=0, GPU_used=False)
                _readiness(args, readiness)
            print(json.dumps({key: value for key, value in readiness.items() if key != "source_ledger"}, ensure_ascii=False, allow_nan=False), flush=True)
            return readiness
        output = (ROOT / args.out).resolve()
        if not output.is_relative_to(ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030") or output.exists():
            raise ValueError("fresh candidate output namespace required")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested candidate CUDA device unavailable")
        seed = FIXED_CONFIG["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(8)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        encoder.to(device); policy.to(device)
        frozen_encoder_hash, frozen_policy_hash = module_sha256(encoder), module_sha256(policy)
        model = MultiBodyCandidateTransformer(center, scale, role_names=ROLE_NAMES).to(device)
        output.mkdir(parents=True)
        atomic_json(output / "source_ledger.json", readiness["source_ledger"])
        atomic_json(output / "config.json", FIXED_CONFIG)
        code_names = ("our_mani/hierarchy_20261030/candidate_training.py", "scripts/train_hierarchy_candidates_20261030.py",
                      "our_mani/hierarchy_20261030/candidate_multibody.py", "our_mani/hierarchy_20261030/networks.py",
                      "our_mani/hierarchy_20261030/data.py", "our_mani/hierarchy_20261030/handoff_training_v2.py",
                      "our_mani/hierarchy_20261030/training.py")
        identity = dict(**source_identity, config_sha256=file_sha256(output / "config.json"),
                        source_ledger_sha256=file_sha256(output / "source_ledger.json"),
                        code_sha256={name: file_sha256(ROOT / name) for name in code_names},
                        frozen_encoder_sha256=frozen_encoder_hash, frozen_skill_sha256=frozen_policy_hash)
        status = dict(version=VERSION, created_utc=utc(), complete=False, stage="frozen_features",
                      identity=identity, config=FIXED_CONFIG, train=training.summary,
                      offline_validation=validation.summary, epochs=[], device=str(device), pid=os.getpid(),
                      process_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                      score_semantics=SCORE_SEMANTICS, native_success_measured=False,
                      scene_encoder_or_skill_updated=False, TEST_read=False,
                      source_task_selects_frontend=False, fake_forecasts_used=False,
                      workspace_center=center, workspace_half_width=scale, role_names=list(ROLE_NAMES))
        atomic_json(output / "status.json", status)
        try:
            train_features = materialize_candidate_features(training, encoder, device, output, identity, args.workers)
            val_features = materialize_candidate_features(validation, encoder, device, output, identity, args.workers)
            generator = torch.Generator().manual_seed(seed)
            sampler = WeightedRandomSampler(torch.from_numpy(training.weights), len(training), replacement=True, generator=generator)
            loader = DataLoader(train_features, batch_size=32, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
            val_loader = DataLoader(val_features, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
            best, best_epoch = float("inf"), None
            for epoch in range(1, 41):
                model.train(); total_loss, examples = 0., 0
                epoch_start = time.perf_counter()
                status.update(stage="candidate", current_epoch=epoch, expected_epochs=40, updated_utc=utc())
                atomic_json(output / "status.json", status)
                for step, batch in enumerate(loader, 1):
                    batch = inputs_to_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch["trajectory"], batch["trajectory_mask"], batch["skill_latent"], batch["robot"], batch["scene_features"])
                    losses, outside = candidate_imitation_loss(model, prediction, batch["target_xyz"])
                    if bool(outside.any()):
                        raise ValueError("TRAIN handoff label lies outside the frozen TRAIN-only workspace")
                    loss = losses.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite candidate imitation loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite candidate gradient")
                    optimizer.step()
                    count = len(losses); total_loss += float(loss.detach()) * count; examples += count
                    if step == 1 or step % 50 == 0:
                        status.update(step=step, expected_steps=len(loader), processed_examples=examples, updated_utc=utc())
                        atomic_json(output / "status.json", status)
                        print(json.dumps(dict(stage="candidate", epoch=epoch, step=step, steps=len(loader), train_loss=total_loss / examples)), flush=True)
                validation_metrics = evaluate_candidates(model, val_loader, device)
                row = dict(epoch=epoch, train_loss=total_loss / examples, validation=validation_metrics,
                           seconds=time.perf_counter() - epoch_start, examples=examples)
                status["epochs"].append(row)
                metadata = dict(training_complete=False, stage="candidate", epoch=epoch, identity=identity,
                                workspace_center=center, workspace_half_width=scale, role_names=list(ROLE_NAMES),
                                skill_checkpoint_sha256=source_identity["skill_checkpoint"]["sha256"],
                                score_semantics=SCORE_SEMANTICS, fixed_goal_time_seconds=.5, time_target_index=10,
                                goal_source=GOAL_SOURCE, validation=validation_metrics,
                                encoder_or_skill_updated=False, native_success_measured=False, TEST_read=False,
                                model_spec=dict(model_dim=128, heads=4, layers=2))
                _save_checkpoint(output / "last.pt", model, metadata)
                if validation_metrics["loss"] < best:
                    best, best_epoch = validation_metrics["loss"], epoch
                    _save_checkpoint(output / "best_candidate.pt", model, metadata)
                if module_sha256(encoder) != frozen_encoder_hash or module_sha256(policy) != frozen_policy_hash:
                    raise ValueError("candidate fitting modified frozen encoder/E/skill weights")
                status.update(elapsed_seconds=time.perf_counter() - started, updated_utc=utc())
                atomic_json(output / "status.json", status)
                print(json.dumps(row, allow_nan=False), flush=True)
            for ref in source_identity.values():
                if isinstance(ref, dict) and "path" in ref and "sha256" in ref and file_sha256(ROOT / ref["path"]) != ref["sha256"]:
                    raise ValueError("bound candidate source identity changed")
            for name, digest in identity["code_sha256"].items():
                if file_sha256(ROOT / name) != digest:
                    raise ValueError("candidate implementation changed during training")
            for row in readiness["source_ledger"]:
                for key in ("scene_cache", "motion_cache", "motion_audit"):
                    ref = row[key]
                    if ref is not None and file_sha256(ROOT / ref["path"]) != ref["sha256"]:
                        raise ValueError("a scene/motion/audit source changed during training")
            for name in ("best_candidate.pt", "last.pt"):
                checkpoint = torch.load(output / name, map_location="cpu", weights_only=True)
                checkpoint["metadata"].update(training_complete=True, completed_epochs=40,
                    selected_epoch=best_epoch if name == "best_candidate.pt" else 40)
                temporary = output / (name + ".partial")
                torch.save(checkpoint, temporary); temporary.replace(output / name)
            status.update(stage="complete", complete=True, finished_utc=utc(),
                          elapsed_seconds=time.perf_counter() - started, selected_epoch=best_epoch,
                          best_candidate_sha256=file_sha256(output / "best_candidate.pt"),
                          full_hierarchy_ready=False,
                          next_gate="new full hierarchy CPU contracts and fixed TRAIN Native development audit before new native VAL")
            atomic_json(output / "status.json", status)
            return status
        except BaseException as error:
            status.update(stage="failed", complete=False, error=repr(error), updated_utc=utc(), elapsed_seconds=time.perf_counter() - started)
            atomic_json(output / "status.json", status)
            raise
    except BaseException as error:
        if not readiness["ready"]:
            readiness.update(error=repr(error), checked_seconds=time.perf_counter() - started)
            _readiness(args, readiness)
        raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="full source/readiness CPU check, no optimizer")
    mode.add_argument("--dry-run", action="store_true", help="readiness plus two real-anchor CPU forward/loss checks, no optimizer")
    command.add_argument("--data-manifest", default=DEFAULT_DATA)
    command.add_argument("--motion-manifest", default=DEFAULT_MOTION)
    command.add_argument("--skill-checkpoint", default=DEFAULT_SKILL)
    command.add_argument("--skill-status", default=DEFAULT_SKILL_STATUS)
    command.add_argument("--out", default=DEFAULT_OUTPUT)
    command.add_argument("--readiness-out", default=DEFAULT_READINESS)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=4)
    return command
