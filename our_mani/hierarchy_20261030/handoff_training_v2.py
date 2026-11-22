"""Episode-handoff goal skill pretraining, isolated from the frozen v1 run.

Every action in an episode is conditioned on the same first closed-gripper
expert TCP target. That target is offline supervision only. The scene encoder
continues to receive current/past RGBD and measured robot state, never labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import INPUT_KEYS, SCHEMA, TASK_FAMILIES, file_sha256
from .networks import CausalSceneSkillEncoder, PretrainedSkillPolicy
from .training import (
    CausalSkillWindows, VERSION as V1_VERSION, atomic_json, encoded_from_features,
    evaluate, freeze_features, inputs_to_device, skill_loss, utc,
)


ROOT = Path(__file__).resolve().parents[2]
VERSION = "hierarchy_episode_handoff_skill_pretraining_20261030_v2"
GOAL_SOURCE = "episode_first_closed_expert_TCP_target"
DEFAULT_ENCODER = "artifacts/framework/multitask_r10/hierarchy_20261030/skill_pretrain_v1/best_classification.pt"
DEFAULT_ENCODER_STATUS = "artifacts/framework/multitask_r10/hierarchy_20261030/skill_pretrain_v1/status.json"
DEFAULT_OUTPUT = "artifacts/framework/multitask_r10/hierarchy_20261030/skill_handoff_pretrain_v2"


def episode_handoff_goal(actions):
    """Return the first expert *commanded* closed-gripper TCP xyz and row.

    The author convention is 1=open, 0=closed. No successful grasp, held object,
    future observed pose, or Native task outcome is used to choose the row.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("finite nonempty expert Nx8 action labels required")
    closed = np.flatnonzero(actions[:, 7] < .5)
    if not len(closed):
        raise ValueError("episode has no expert closed-gripper command; no substitute handoff is permitted")
    frame = int(closed[0])
    return actions[frame, :3].astype(np.float32).copy(), frame


class EpisodeHandoffSkillWindows(CausalSkillWindows):
    """Reuse v1 causal inputs and exact weights; replace only offline goals."""
    def __init__(self, root, manifest, split, history_frames=5, smoke=False):
        super().__init__(root, manifest, split, history_frames, smoke)
        self.handoff_goals = []
        self.handoff_records = []
        for row, arrays in zip(self.rows, self.episodes):
            goal, frame = episode_handoff_goal(arrays["expert_actions"])
            self.handoff_goals.append(goal)
            self.handoff_records.append(dict(
                split=split, task=row["task"], raw_episode_id=row["raw_episode_id"],
                parent_source_group=row["parent_source_group"],
                first_closed_command_frame=frame, goal_xyz=goal.tolist(),
                goal_time_seconds=.5, goal_source=GOAL_SOURCE,
            ))
        self.summary.update(goal_source=GOAL_SOURCE, episode_goals=len(self.handoff_goals),
                            expert_goals_used_as_scene_inputs=False)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        episode_index, _ = self.anchors[index]
        result["goal_xyz"] = self.handoff_goals[episode_index].copy()
        result["goal_time"] = np.float32(.5)
        return result


def frozen_encoder_sha256(encoder):
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        if name.startswith("skill_embeddings."):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialize_handoff_models(encoder_state_dict, center, half_width, *, seed=20261030, device="cpu"):
    """Load/freeze v1 visual/classification state; newly initialize E + policy."""
    if seed != 20261030:
        raise ValueError("the fixed v2 initialization seed is 20261030")
    torch.manual_seed(seed)
    encoder = CausalSceneSkillEncoder()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    # Reinitialize the table rather than silently continuing the v1 skill E.
    torch.manual_seed(seed)
    with torch.no_grad():
        torch.nn.init.normal_(encoder.skill_embeddings.weight, std=.1)
    encoder.skill_embeddings.weight.requires_grad_(True)
    policy = PretrainedSkillPolicy(center, half_width)
    encoder.to(device).eval()
    policy.to(device)
    return encoder, policy


def evaluate_handoff(encoder, policy, loader, device, scale):
    """Reuse numerical metrics while explicitly correcting their goal semantics."""
    result = evaluate(encoder, policy, loader, device, "skill", scale)
    result["goal_source"] = GOAL_SOURCE
    result["goal_latched_for_whole_episode"] = True
    result["copy_episode_handoff_goal_position_error_m"] = result.pop("copy_segment_entry_goal_position_error_m")
    return result


def _check_protocol(config, manifest):
    if (config.get("seed") != 20261030 or config.get("scene_history_frames") != 5
            or config.get("action_classes") != 11 or set(config.get("tasks", [])) != set(TASK_FAMILIES)):
        raise ValueError("frozen four-task encoder/seed/history contract differs")
    expected_hyperparameters = dict(skill_epochs=40, batch_size=64,
                                    learning_rate=.0003, weight_decay=.0001,
                                    goal_time_seconds=.5)
    if any(config["skill_pretraining"].get(key) != value for key, value in expected_hyperparameters.items()):
        raise ValueError("v2 retains the declared v1 skill hyperparameters")
    if (manifest.get("schema") != SCHEMA or not manifest.get("complete")
            or manifest.get("smoke_only") or manifest.get("TEST_read") is not False
            or manifest.get("source_debug_target_positions_read") is not False
            or manifest.get("consumed_input_keys") != list(INPUT_KEYS)):
        raise ValueError("complete exact public-input four-task cache required")
    counts = Counter((row["split"], row["task"]) for row in manifest["episodes"])
    expected = {("train", task): 100 for task in TASK_FAMILIES}
    expected.update({("validation", "pick_moving_target_on_the_table"): 20,
                     ("validation", "pick_moving_target_on_the_table_high_speed"): 20,
                     ("validation", "insert_onto_rotating_peg"): 18,
                     ("validation", "insert_onto_rotating_peg_high_speed"): 19})
    if dict(counts) != expected or len({row["parent_source_group"] for row in manifest["episodes"]}) != 477:
        raise ValueError("fixed TRAIN400/offline-VAL77 source membership differs")


def load_bound_inputs(args):
    config_path = (ROOT / args.config).resolve(strict=True)
    manifest_path = (ROOT / args.data_manifest).resolve(strict=True)
    checkpoint_path = (ROOT / args.encoder_checkpoint).resolve(strict=True)
    encoder_status_path = (ROOT / args.encoder_status).resolve(strict=True)
    permitted = ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if (not checkpoint_path.is_relative_to(permitted / "skill_pretrain_v1")
            or checkpoint_path.name != "best_classification.pt"
            or not encoder_status_path.is_relative_to(permitted / "skill_pretrain_v1")):
        raise ValueError("v2 must bind the completed v1 best-classification source")
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    _check_protocol(config, manifest)
    status = json.loads(encoder_status_path.read_text())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    identity = metadata.get("identity", {})
    if (status.get("version") != V1_VERSION or status.get("complete") is not True
            or status.get("smoke_only") is not False
            or payload.get("version") != V1_VERSION or metadata.get("stage") != "classification"
            or metadata.get("smoke_only") is not False or payload.get("skill_state_dict") is not None
            or identity != status.get("identity")
            or identity.get("config_sha256") != file_sha256(config_path)
            or identity.get("data_manifest_sha256") != file_sha256(manifest_path)):
        raise ValueError("v1 classification checkpoint/status/config/data binding differs")
    if any(file_sha256(ROOT / name) != digest for name, digest in identity["code_sha256"].items()):
        raise ValueError("frozen v1 code changed; do not mutate its source identity")
    return config, manifest, payload, dict(
        base_config={"path": str(config_path.relative_to(ROOT)), "sha256": file_sha256(config_path)},
        data_manifest={"path": str(manifest_path.relative_to(ROOT)), "sha256": file_sha256(manifest_path)},
        v1_classification_checkpoint={"path": str(checkpoint_path.relative_to(ROOT)), "sha256": file_sha256(checkpoint_path)},
        v1_completed_status={"path": str(encoder_status_path.relative_to(ROOT)), "sha256": file_sha256(encoder_status_path)},
        v1_identity=identity,
    )


def _save_checkpoint(path, encoder, policy, metadata):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(version=VERSION, encoder_state_dict=encoder.state_dict(),
                    skill_state_dict=policy.state_dict(), metadata=metadata), temporary)
    temporary.replace(path)


def run(args):
    config, manifest, classification, source_identity = load_bound_inputs(args)
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(8)
    training = EpisodeHandoffSkillWindows(ROOT, manifest, "train", config["scene_history_frames"], args.smoke)
    validation = EpisodeHandoffSkillWindows(ROOT, manifest, "validation", config["scene_history_frames"], args.smoke)
    positions = np.concatenate([episode["expert_actions"][:, :3] for episode in training.episodes])
    lower, upper = positions.min(0) - .05, positions.max(0) + .05
    center, half_width = (upper + lower) / 2, (upper - lower) / 2
    if args.check:
        encoder, policy = initialize_handoff_models(classification["encoder_state_dict"],
            center.tolist(), half_width.tolist(), seed=seed)
        print(json.dumps(dict(version=VERSION, mode="CPU_check_only", source_identity=source_identity,
                              train=training.summary, offline_validation=validation.summary,
                              goal_source=GOAL_SOURCE, goal_latched_for_whole_episode=True,
                              all_episodes_have_expert_closed_command=True,
                              trainable_encoder_parameters=[name for name, p in encoder.named_parameters() if p.requires_grad],
                              policy_parameters=sum(p.numel() for p in policy.parameters()),
                              optimizer_steps=0, GPU_used=False, candidate_network_trained=False,
                              native_success_measured=False), ensure_ascii=False, allow_nan=False), flush=True)
        return
    output = (ROOT / args.out).resolve()
    permitted = ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if not output.is_relative_to(permitted) or output.exists():
        raise ValueError("fresh isolated v2 output namespace required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    encoder, policy = initialize_handoff_models(classification["encoder_state_dict"],
        center.tolist(), half_width.tolist(), seed=seed, device=device)
    frozen_hash = frozen_encoder_sha256(encoder)
    output.mkdir(parents=True)
    effective_config = copy.deepcopy(config)
    effective_config.update(schema=VERSION, classification_training=False,
                            base_config=source_identity["base_config"],
                            pretrained_classifier=source_identity["v1_classification_checkpoint"])
    effective_config["skill_pretraining"].update(
        goal_input="one first expert closed-gripper command TCP xyz for every row of the complete episode; offline labels only",
        goal_time_seconds=.5, goal_latched_for_whole_episode=True,
        encoder_frozen_during_skill_pretraining="all v1 CNN/GRU/projection/classifier frozen; E reinitialized with fixed seed and trained with independent new policy",
        initialization="E and policy newly initialized, never resumed from best_skill.pt",
    )
    atomic_json(output / "effective_config.json", effective_config)
    catalog = dict(version=VERSION, source_identity=source_identity,
                   episodes=manifest["episodes"],
                   handoff_goals=training.handoff_records + validation.handoff_records,
                   expert_goal_is_supervision_only=True, TEST_read=False)
    atomic_json(output / "training_sources.json", catalog)
    code_paths = [Path(__file__), ROOT / "scripts/train_hierarchy_handoff_skills_20261030.py",
                  ROOT / "our_mani/hierarchy_20261030/training.py",
                  ROOT / "our_mani/hierarchy_20261030/networks.py",
                  ROOT / "our_mani/hierarchy_20261030/data.py"]
    identity = dict(**source_identity,
                    effective_config_sha256=file_sha256(output / "effective_config.json"),
                    training_sources_sha256=file_sha256(output / "training_sources.json"),
                    code_sha256={str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths})
    status = dict(version=VERSION, created_utc=utc(), identity=identity, stage="materializing_frozen_features",
                  complete=False, smoke_only=args.smoke, pid=os.getpid(),
                  process_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                  device=str(device), train=training.summary, offline_validation=validation.summary,
                  workspace_center=center.tolist(), workspace_half_width=half_width.tolist(),
                  goal_source=GOAL_SOURCE, goal_latched_for_whole_episode=True,
                  goal_time_seconds=.5, first_closed_command_is_success_evidence=False,
                  scene_input_uses_expert_labels=False, classifier_retrained=False,
                  original_motion_forecaster_used=False, candidate_transformer_trained=False,
                  native_success_measured=False, native_VAL_started=False, TEST_started=False,
                  encoder_frozen_sha256=frozen_hash, E_and_policy_reinitialized=True,
                  held_repeat_rows_counted_as_independent_commands=False, epochs=[])
    atomic_json(output / "status.json", status)
    started = time.perf_counter()
    batch_size = int(config["skill_pretraining"]["batch_size"])
    scale = torch.tensor(half_width, device=device)
    try:
        training_features = freeze_features(training, encoder, device, batch_size=batch_size,
            workers=args.workers, output=output / "train_frozen_features.pt")
        validation_features = freeze_features(validation, encoder, device, batch_size=batch_size,
            workers=args.workers, output=output / "validation_frozen_features.pt")
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.from_numpy(training.weights), len(training),
                                        replacement=True, generator=generator)
        loader = DataLoader(training_features, batch_size=batch_size, sampler=sampler,
                            num_workers=0, pin_memory=device.type == "cuda")
        validation_loader = DataLoader(validation_features, batch_size=batch_size, shuffle=False,
                                       num_workers=0, pin_memory=device.type == "cuda")
        parameters = [encoder.skill_embeddings.weight, *policy.parameters()]
        optimizer = torch.optim.AdamW(parameters,
            lr=config["skill_pretraining"]["learning_rate"],
            weight_decay=config["skill_pretraining"]["weight_decay"])
        best = float("inf")
        epochs = 1 if args.smoke else int(config["skill_pretraining"]["skill_epochs"])
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            encoder.eval(); policy.train()
            loss_sum, examples = 0., 0
            status.update(stage="skill", current_epoch=epoch, expected_stage_epochs=epochs, updated_utc=utc())
            atomic_json(output / "status.json", status)
            for step, batch in enumerate(loader, 1):
                data = inputs_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                encoded = encoded_from_features(data, encoder)
                latent = data["skill_probabilities"].detach() @ F.normalize(encoder.skill_embeddings.weight, dim=-1)
                predicted = policy(encoded["scene_features"], latent, data["robot_current"],
                                   data["goal_xyz"], data["goal_time"])
                if predicted["trajectory_consumed"]:
                    raise ValueError("v2 skill pretraining must not fabricate trajectories")
                loss = skill_loss(predicted, data["action"], scale).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite v2 handoff skill loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite v2 handoff gradient")
                optimizer.step()
                count = len(data["skill_label"])
                loss_sum += float(loss.detach()) * count; examples += count
                if step % 50 == 0 or step == 1:
                    status.update(step=step, expected_steps=len(loader), processed_examples=examples,
                                  epoch_elapsed_seconds=time.perf_counter() - epoch_start, updated_utc=utc())
                    atomic_json(output / "status.json", status)
                    print(json.dumps(dict(stage="skill_handoff_v2", epoch=epoch, step=step,
                                          steps=len(loader), train_loss=loss_sum / examples,
                                          seconds=round(time.perf_counter() - epoch_start, 2))), flush=True)
                if args.smoke and step >= 4:
                    break
            if frozen_encoder_sha256(encoder) != frozen_hash:
                raise ValueError("frozen CNN/GRU/projection/classifier changed during v2 fitting")
            metrics = evaluate_handoff(encoder, policy, validation_loader, device, scale)
            row = dict(stage="skill", epoch=epoch, train_loss=loss_sum / examples,
                       validation=metrics, seconds=time.perf_counter() - epoch_start, examples=examples)
            status["epochs"].append(row)
            metadata = dict(identity=identity, stage="skill", epoch=epoch,
                            workspace_center=center.tolist(), workspace_half_width=half_width.tolist(),
                            validation=metrics, goal_source=GOAL_SOURCE,
                            goal_latched_for_whole_episode=True, goal_time_seconds=.5,
                            encoder_frozen_sha256=frozen_hash, E_and_policy_reinitialized=True,
                            skill_policy_trajectory_consumed=False, candidate_network_trained=False,
                            motion_forecaster_used=False, native_success_measured=False,
                            scene_input_uses_expert_labels=False, smoke_only=args.smoke)
            _save_checkpoint(output / "last.pt", encoder, policy, metadata)
            if metrics["loss"] < best:
                best = metrics["loss"]
                _save_checkpoint(output / "best_skill.pt", encoder, policy, metadata)
            status.update(elapsed_seconds=time.perf_counter() - started, updated_utc=utc())
            atomic_json(output / "status.json", status)
            print(json.dumps(row, allow_nan=False), flush=True)
        for ref in source_identity.values():
            if isinstance(ref, dict) and "path" in ref and "sha256" in ref:
                if file_sha256(ROOT / ref["path"]) != ref["sha256"]:
                    raise ValueError("bound v1 source/config/data identity changed during fitting")
        if any(file_sha256(ROOT / name) != digest for name, digest in identity["code_sha256"].items()):
            raise ValueError("v2 implementation source changed during fitting")
        for name, digest in (("effective_config.json", identity["effective_config_sha256"]),
                             ("training_sources.json", identity["training_sources_sha256"])):
            if file_sha256(output / name) != digest:
                raise ValueError("v2 source catalog/effective contract changed during fitting")
        status.update(stage="complete", complete=True, finished_utc=utc(),
                      elapsed_seconds=time.perf_counter() - started,
                      full_hierarchy_ready=False,
                      best_skill_sha256=file_sha256(output / "best_skill.pt"),
                      next_gate="public tracks and real forecasts, candidate training, fixed TRAIN audit before new native VAL")
        atomic_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(stage="failed", complete=False, error=repr(error),
                      elapsed_seconds=time.perf_counter() - started, updated_utc=utc())
        atomic_json(output / "status.json", status)
        raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default="config/framework/hierarchy_20261030_v1.json")
    command.add_argument("--data-manifest", default="artifacts/framework/multitask_r10/hierarchy_20261030/data_v1/manifest.json")
    command.add_argument("--encoder-checkpoint", default=DEFAULT_ENCODER)
    command.add_argument("--encoder-status", default=DEFAULT_ENCODER_STATUS)
    command.add_argument("--out", default=DEFAULT_OUTPUT)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--check", action="store_true", help="CPU-only full source/goal/frozen-parameter check, no optimizer or output")
    command.add_argument("--smoke", action="store_true", help="explicit four-parent development smoke, never full qualification")
    return command
