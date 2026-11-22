"""Commit an immutable, actually trained block runtime freeze on CPU.

Completeness and artifact identity are required; offline precision qualification
is reported independently and may be false. No Native runtime, GPU work, TEST
recording, fitting or old STOP file is opened/changed by this module.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .data import EXPECTED_COUNTS, TASK_NAMES, VERSION as DATA_VERSION, file_sha256
from .evaluation_v2 import (
    EvaluationBlocked, FREEZE_SCHEMA, SOURCE_FILES, checked, pointer, validate_freeze,
)


BASE = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102")
LEGACY = Path("artifacts/framework/multitask_r10/hierarchy_20261030")
TRAINING_VERSION = "hierarchy_block_training_20261102_v1"
SCORE_SEMANTICS = "expert_joint_handoff_pose_time_imitation_not_calibrated_success"
FORECAST = Path("artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt")
FORECAST_SHA = "ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef"
SEGMENTATION = LEGACY / "segmentation_v1/model.pt"
SEGMENTATION_MANIFEST = LEGACY / "segmentation_v1/manifest.json"
MOTION = LEGACY / "motion_inputs_blind_v1/manifest.json"
DATA = LEGACY / "data_v1/manifest.json"
MODEL_CONFIG = dict(num_candidates=4, action_horizon=4, freeze_encoder_visual=True)
EXPECTED_MODEL_SHA = "cbe5477777ff508e7f70a034f91b038835786181f362118f1c3bfde769a594af"
NETWORKS = ("scene_classifier", "skill_policy", "candidate_network")
OLD_RUNTIME_FILES = (
    "our_mani/hierarchy_20261030/networks.py", "our_mani/hierarchy_20261030/data.py",
    "our_mani/hierarchy_20261030/tracking.py", "our_mani/hierarchy_20261030/segmentation.py",
    "our_mani/contracts.py", "our_mani/context_dynamics.py", "our_mani/perception.py",
    "our_mani/benchmark_perception.py", "our_mani/multitask/context_se3.py",
    "our_mani/multitask/factory.py", "our_mani/multitask/perception.py",
    "our_mani/multitask/training_geometry.py", "our_mani/multitask/rotation_equivalence.py",
)


def _require(condition, message):
    if not condition:
        raise EvaluationBlocked(message)


def _write_new(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _project_path(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    _require(path.is_relative_to(root), "freeze input/output must remain within project")
    _require(not any(p.lower() == "test" or p.lower().startswith("test_") or "official_test" in p.lower()
                     or "test_package" in p.lower() for p in path.relative_to(root).parts), "TEST input forbidden before opening")
    return path


def _read_json(root, path):
    return json.loads(_project_path(root, path).read_text())


def _verify_tree(root, tree):
    """Check all declared fitting code, consumed file and artifact references."""
    if isinstance(tree, dict):
        if "path" in tree and "sha256" in tree:
            checked(root, {"path": str(_project_path(root, tree["path"]).relative_to(root)), "sha256": tree["sha256"]})
        for key, value in tree.items():
            if key in ("files", "code_sha256"):
                _require(isinstance(value, dict), "training identity hashes must be mappings")
                for path, digest in value.items():
                    target = _project_path(root, path)
                    _require(file_sha256(target) == digest, "fitting input/source SHA256 changed: " + str(path))
            elif key.endswith("_path") and key[:-5] + "_sha256" in tree:
                _require(file_sha256(_project_path(root, value)) == tree[key[:-5] + "_sha256"], "fitting reference changed: " + key)
            else:
                _verify_tree(root, value)
    elif isinstance(tree, list):
        for item in tree:
            _verify_tree(root, item)


def _verify_roster(root, identity, data, motion):
    consumed = identity.get("data", {})
    _require(consumed.get("version") == DATA_VERSION and consumed.get("TEST_read") is False and consumed.get("other_task_arrays_read") is False
             and consumed.get("success_filter") is False, "new training must exclude TEST/other-task arrays and outcome filtering")
    roster = consumed.get("selected_sources", [])
    _require(dict(Counter((r["split"], r["task"]) for r in roster)) == EXPECTED_COUNTS,
             "new fitting must retain exact block TRAIN200/offline-VAL40 membership")
    key = lambda row: (row["split"], row["task"], row["raw_episode_id"])
    selected = [r for r in data.get("episodes", []) if r.get("task") in TASK_NAMES]
    _require(len({key(r) for r in roster}) == len(roster) and {key(r) for r in roster} == {key(r) for r in selected},
             "fitting source roster differs from completed original block sources")
    cases = {key(r): r for r in motion.get("cases", []) if r.get("task") in TASK_NAMES}
    _require(set(cases) == {key(r) for r in selected}, "fitting motion source membership differs")
    files = consumed.get("files", {})
    required = {str(DATA): file_sha256(root / DATA), str(MOTION): file_sha256(root / MOTION)}
    for row in selected:
        case = cases[key(row)]
        _require(case.get("parent_source_group") == row["parent_source_group"] and case.get("source_path") == row["source_path"],
                 "scene/motion source identity differs")
        required[row["npz_path"]] = row["cache_sha256"]
        if case.get("npz_path"):
            required[case["npz_path"]] = case["sha256"]
            required[case["audit_path"]] = case["audit_sha256"]
        else:
            _require(bool(case.get("error") or case.get("infrastructure_error") or case.get("failed")),
                     "missing motion parent must have explicit failure evidence")
    _require(files == required, "fitting consumed-file inventory must contain only the exact two-task scene/motion caches")
    for row in roster:
        original = next(r for r in selected if key(r) == key(row))
        _require(row.get("parent_source_group") == original.get("parent_source_group"), "fitting parent group identity differs")
    _require(consumed.get("loader_source_sha256") == file_sha256(root / "our_mani/hierarchy_block_20261102/data.py"),
             "actual block data loader changed since fitting")
    return selected


def _runtime_sources(root, status, motion, segmentation):
    names = set(SOURCE_FILES) | set(OLD_RUNTIME_FILES)
    names.update(str(p.relative_to(root)) for p in (root / "our_mani/hierarchy_block_20261102").glob("*.py"))
    names.update(str(p.relative_to(root)) for p in (root / "scripts").glob("*block*20261102*.py"))
    for owner in (status.get("identity", {}), motion.get("identity", {}), segmentation):
        names.update(owner.get("code_sha256", {}))
    return [pointer(root, root / name) for name in sorted(names)]


def _construct_cpu_actor(root, freeze):
    from .actor_v2 import create_actor
    actor = create_actor(root=root, freeze_manifest=freeze, device="cpu")
    try:
        _require(all(callable(getattr(actor, name, None)) for name in ("observe", "act", "close")), "CPU actor interface incomplete")
        _require(str(getattr(actor, "device", "")) == "cpu", "CPU actor construction must not use a device fallback")
        return dict(complete=True, device="cpu", actor_constructed=True, checkpoint_loaded=True,
                    tracking_constructed=True, actor_observe_act_interface=True, native_started=False,
                    GPU_started=False, TEST_read=False, native_success_certified=False)
    finally:
        close = getattr(actor, "close", None)
        if callable(close):
            close()


def build_freeze(root, *, training_dir=None, output_dir=None):
    """Build a fresh freeze only after complete training and real CPU readiness.

    A failed actor construction leaves evidence in the fresh directory, with no
    usable freeze.json. Existing directories are never overwritten or resumed.
    """
    root = Path(root).resolve(strict=True)
    training = _project_path(root, BASE / "training_v1" if training_dir is None else training_dir)
    output = _project_path(root, BASE / "freeze_model_v2" if output_dir is None else output_dir)
    _require(training.is_relative_to(root / BASE) and output.is_relative_to(root / BASE), "new block artifact namespace required")
    if output.exists():
        raise FileExistsError("preserve existing freeze directory: " + str(output))
    status_path, checkpoint_path = training / "status.json", training / "best.pt"
    status = _read_json(root, status_path)
    _require(status.get("version") == TRAINING_VERSION and status.get("complete") is True
             and status.get("completed_epochs") == 120 and status.get("TEST_read") is False,
             "actually completed 120-epoch new TRAIN-only training required")
    _require(type(status.get("precision_qualified")) is bool, "offline precision qualification must be explicit")
    digest = file_sha256(checkpoint_path)
    _require(digest == EXPECTED_MODEL_SHA, "v2 must reuse the exact completed v1 best.pt checkpoint")
    recorded = next((status.get(k) for k in ("best_checkpoint_sha256", "best_sha256", "checkpoint_sha256") if status.get(k)), None)
    if recorded is None and isinstance(status.get("best_checkpoint"), dict):
        recorded = status["best_checkpoint"].get("sha256")
    _require(recorded == digest, "completed training status does not bind best.pt bytes")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    identity = status.get("identity", {})
    _require(payload.get("version") == TRAINING_VERSION and meta.get("training_complete") is True
             and meta.get("completed_epochs") == 120 and meta.get("TEST_read") is False
             and meta.get("model_config") == MODEL_CONFIG and meta.get("score_semantics") == SCORE_SEMANTICS
             and meta.get("identity") == identity and bool(payload.get("model_state_dict")),
             "checkpoint training identity/configuration is incomplete")
    if "precision_qualified" in meta:
        _require(meta["precision_qualified"] == status["precision_qualified"], "checkpoint/status precision qualification differs")
    center, scale = np.asarray(meta.get("workspace_center")), np.asarray(meta.get("workspace_half_width"))
    _require(center.shape == (3,) and scale.shape == (3,) and np.isfinite(center).all()
             and np.isfinite(scale).all() and np.all(scale > 0), "finite TRAIN-only workspace required")
    _require(bool(identity.get("code_sha256")), "actual new training source hashes required")
    _verify_tree(root, identity)
    data, motion, segmentation = (_read_json(root, p) for p in (DATA, MOTION, SEGMENTATION_MANIFEST))
    _require(data.get("complete") is True and data.get("smoke_only") is False and data.get("TEST_read") is False
             and data.get("source_debug_target_positions_read") is False
             and motion.get("complete") is True and motion.get("smoke_only") is False and motion.get("TEST_read") is False
             and motion.get("source_debug_target_positions_read") is False
             and motion.get("frontend") == "blind_parallel_fixed_common_TRAIN_priors"
             and motion.get("source_task_selects_frontend") is False
             and motion.get("identity", {}).get("data_manifest_sha256") == file_sha256(root / DATA),
             "completed original public source evidence required")
    selected = _verify_roster(root, identity, data, motion)
    _require(file_sha256(root / FORECAST) == FORECAST_SHA
             and motion.get("identity", {}).get("motion_checkpoint_sha256") == FORECAST_SHA,
             "unchanged original v671 GRU/context forecaster required")
    segmentation_sha = file_sha256(root / SEGMENTATION)
    _require(segmentation.get("schema") == "hierarchy_public_rgbd_semantic_mlp_v1_20261030"
             and segmentation.get("training_complete") is True and segmentation.get("stage") == "complete"
             and segmentation.get("TEST_read") is False and segmentation.get("original_VAL_read") is False
             and segmentation.get("native_or_instance_inputs_used") is False
             and segmentation.get("model", {}).get("sha256") == segmentation_sha
             and motion.get("identity", {}).get("segmentation_checkpoint_sha256") == segmentation_sha,
             "completed TRAIN-only segmentation must match real public replay")
    _verify_tree(root, {"code_sha256": segmentation.get("code_sha256", {})})
    _verify_tree(root, {"code_sha256": motion.get("identity", {}).get("code_sha256", {})})
    sources = _runtime_sources(root, status, motion, segmentation)
    reports = []
    report_refs = status.get("offline_precision_reports", [])
    if status.get("precision_report") is not None:
        report_refs = [*report_refs, status["precision_report"]]
    for reference in report_refs:
        checked(root, reference)
        reports.append(reference)
    output.mkdir(parents=True, exist_ok=False)
    evidence = dict(schema="hierarchy_block_checkpoint_training_evidence_20261102_v1", complete=True,
        trained_components=list(NETWORKS), component_artifact_sha256={name: digest for name in NETWORKS},
        training_run=pointer(root, status_path), checkpoint=pointer(root, checkpoint_path), completed_epochs=120,
        source_count=len(selected), task_counts={s: dict(Counter(r["task"] for r in selected if r["split"] == s)) for s in ("train", "validation")},
        TEST_used=False, other_task_arrays_fitted=False, success_filter=False,
        score_semantics=SCORE_SEMANTICS, precision_qualified=status["precision_qualified"],
        native_success_certified=False, same_shared_latent_conditions_both_consumers=True)
    _write_new(output / "training_evidence.json", evidence)
    _write_new(output / "tracking_contract.json", dict(schema="hierarchy_block_public_tracking_contract_20261102_v1", complete=True,
        implementation="our_mani.hierarchy_20261030.tracking.BlindPublicMotionPipeline",
        source_task_or_speed_selects_frontend=False, target_role=0, target_forecast_requires_five_actual_causal_observations=True,
        motion_inputs=pointer(root, root / MOTION), segmentation_model=pointer(root, root / SEGMENTATION),
        segmentation_fit=pointer(root, root / SEGMENTATION_MANIFEST), selected_source_count=len(selected),
        fitting_tasks=list(TASK_NAMES), all_frontends_run_on_every_scene=True, masked_unknown_values_are_padding=True,
        native_instance_ids_as_inputs=False, quality_is_not_certified_perfect=True, TEST_used=False))
    _write_new(output / "forecaster_evidence.json", dict(schema="hierarchy_block_retained_forecaster_evidence_20261102_v1", complete=True,
        checkpoint_sha256=FORECAST_SHA, weights_changed=False, dt=.05, forecast_steps=22,
        history_source="five to twenty actual causal public tracked observations", historical_training="retained original v671 TRAIN model",
        unavailable_input="explicit unknown mask; never privileged pose or fabricated trajectory", TEST_used=False))
    shared = pointer(root, output / "training_evidence.json")
    components = {name: dict(artifact=pointer(root, checkpoint_path), evidence=shared, trained=True) for name in NETWORKS}
    components.update(forecaster=dict(artifact=pointer(root, root / FORECAST), evidence=pointer(root, output / "forecaster_evidence.json")),
        tracking=dict(artifact=pointer(root, output / "tracking_contract.json"), evidence=pointer(root, root / MOTION)),
        segmentation=dict(artifact=pointer(root, root / SEGMENTATION), evidence=pointer(root, root / SEGMENTATION_MANIFEST), trained=True))
    freeze = dict(schema=FREEZE_SCHEMA, frozen=True, created_utc=datetime.now(timezone.utc).isoformat(), tasks=list(TASK_NAMES),
        readiness={name: True for name in ("scene_classifier_trained", "skill_policy_trained", "candidate_trained", "public_tracking_ready", "forecaster_ready")},
        components=components, actor=dict(module="our_mani.hierarchy_block_20261102.actor_v2", factory="create_actor",
                                        source=pointer(root, root / "our_mani/hierarchy_block_20261102/actor_v2.py")),
        runtime_sources=sources, training_status=pointer(root, status_path), offline_precision_reports=reports,
        precision_qualified=status["precision_qualified"], evaluation_of_unqualified_model_allowed=True,
        runtime_revision="native_runtime_python_path_resolution_v2", reused_model_sha256=EXPECTED_MODEL_SHA,
        native_python=dict(selection="python after sourcing original runtime.sh", training_interpreter_override_allowed=False),
        protocol=dict(dt=.05, simulation_seconds=10., gripper_velocity=.2, baseline_decision_ticks=10, warmup_ticks=4,
            inference_device="cpu", baseline_repo="/home/ubuntu/桌面/phsimani/PhysMani", current_instance_masks=True,
            information_setting="native common sensor capture; policy receives causal RGBD/calibration/proprio only"),
        policy_inputs="causal_RGBD_calibration_robot_only", old_stage_policy_allowed=False, forecast_fallback_allowed=False,
        TEST_used_for_fitting=False, probability_is_calibrated_success=False, technical_readiness_is_not_success_qualification=True,
        time_contract=dict(goal_command_time_semantics="remaining_until_first_closed_advice",
            expert_grasp_lookahead_seconds=.5,
            implied_object_target_remaining_seconds="max(predicted_first_close_seconds + .5 - elapsed_since_handoff, 0)",
            time_is_reachability_guarantee=False,
            source_clock="nominal reconstructed frame_index * .05 seconds; not verified HDF5 timestamps"),
        handoff_time_clock="nominal reconstructed source frames * .05 seconds; not verified original HDF5 timestamps")
    validate_freeze(root, freeze)
    try:
        readiness = _construct_cpu_actor(root, freeze)
    except Exception as error:
        _write_new(output / "cpu_readiness_failure.json", dict(complete=False, ready=False,
            error_type=type(error).__name__, error=str(error), native_started=False, GPU_started=False, TEST_read=False))
        raise
    _write_new(output / "cpu_readiness.json", readiness)
    freeze["cpu_actor_readiness"] = pointer(root, output / "cpu_readiness.json")
    _verify_tree(root, identity)
    validate_freeze(root, freeze)
    checked(root, freeze["training_status"])
    checked(root, freeze["cpu_actor_readiness"])
    target = output / "freeze.json"
    _write_new(target, freeze)
    return target
