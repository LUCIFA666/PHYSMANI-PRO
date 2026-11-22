"""Freeze the actually completed hierarchy, without opening any TEST source.

Readiness here concerns reproducible interfaces and real training evidence.
It does not certify policy success or perfect segmentation/association.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import file_sha256, TASK_FAMILIES
from .evaluation import (
    FREEZE_SCHEMA, SOURCE_FILES, EvaluationBlocked, pointer, validate_freeze,
)


BASE = Path("artifacts/framework/multitask_r10/hierarchy_20261030")
FORECAST = Path("artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt")
FORECAST_SHA = "ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef"


def _read_complete(root, path):
    document = json.loads((root/path).read_text())
    if document.get("complete") is not True:
        raise EvaluationBlocked("completed evidence required: " + str(path))
    return document


def _write_new(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _read_segmentation(root, path, model_path):
    document = json.loads((root/path).read_text())
    if (document.get("schema") != "hierarchy_public_rgbd_semantic_mlp_v1_20261030"
            or document.get("training_complete") is not True or document.get("stage") != "complete"
            or document.get("TEST_read") is not False or document.get("original_VAL_read") is not False
            or document.get("native_or_instance_inputs_used") is not False
            or document.get("model", {}).get("sha256") != file_sha256(root/model_path)):
        raise EvaluationBlocked("completed TRAIN-only segmentation evidence bound to its model required")
    for name, digest in document["code_sha256"].items():
        if file_sha256(root/name) != digest:
            raise EvaluationBlocked("segmentation fitting source changed: " + name)
    return document


def build_freeze(root):
    root = Path(root).resolve()
    output = root/BASE/"freeze_model_v1"
    if output.exists():
        raise FileExistsError("preserve existing model freeze: " + str(output))
    data_path = BASE/"data_v1/manifest.json"
    skill_status_path = BASE/"skill_handoff_pretrain_v2/status.json"
    candidate_status_path = BASE/"candidate_pretrain_v1/status.json"
    motion_path = BASE/"motion_inputs_blind_v1/manifest.json"
    segment_path = BASE/"segmentation_v1/model.pt"
    segment_manifest_path = BASE/"segmentation_v1/manifest.json"
    skill_path = BASE/"skill_handoff_pretrain_v2/best_skill.pt"
    candidate_path = BASE/"candidate_pretrain_v1/best_candidate.pt"
    data = _read_complete(root, data_path)
    skill_status = _read_complete(root, skill_status_path)
    candidate_status = _read_complete(root, candidate_status_path)
    motion = _read_complete(root, motion_path)
    segment = _read_segmentation(root, segment_manifest_path, segment_path)
    if (len(data.get("episodes", [])) != 477 or data.get("TEST_read") is not False
            or motion.get("expected_cases") != 477 or len(motion.get("cases", [])) != 477
            or motion.get("TEST_read") is not False or motion.get("source_task_selects_frontend") is not False
            or motion.get("infrastructure_failed") != 0):
        raise EvaluationBlocked("exact four-task real public source coverage required")
    for name, digest in motion["identity"]["code_sha256"].items():
        if file_sha256(root/name) != digest:
            raise EvaluationBlocked("motion producer source changed: " + name)
    if file_sha256(root/FORECAST) != FORECAST_SHA:
        raise EvaluationBlocked("retained v671 forecaster identity changed")
    skill = torch.load(root/skill_path, map_location="cpu", weights_only=True)
    candidate = torch.load(root/candidate_path, map_location="cpu", weights_only=True)
    sm, cm = skill["metadata"], candidate["metadata"]
    if (skill.get("version") != "hierarchy_episode_handoff_skill_pretraining_20261030_v2"
            or sm.get("goal_latched_for_whole_episode") is not True
            or sm.get("skill_policy_trajectory_consumed") is not False
            or sm.get("scene_input_uses_expert_labels") is not False
            or sm.get("goal_time_seconds") != .5):
        raise EvaluationBlocked("completed whole-episode handoff skill required")
    if (cm.get("training_complete") is not True or cm.get("completed_epochs") != 40
            or cm.get("skill_checkpoint_sha256") != file_sha256(root/skill_path)
            or cm.get("score_semantics") != "expert_handoff_point_imitation_probability_not_native_success_probability"):
        raise EvaluationBlocked("candidate training must bind this exact trained skill/E")
    if cm.get("identity") != candidate_status.get("identity") or sm.get("identity") != skill_status.get("identity"):
        raise EvaluationBlocked("checkpoint and actual fitting status identities differ")
    for metadata in (sm, cm):
        for name, digest in metadata["identity"]["code_sha256"].items():
            if file_sha256(root/name) != digest:
                raise EvaluationBlocked("fitting source changed after its run: " + name)
    if motion["identity"].get("segmentation_checkpoint_sha256") != file_sha256(root/segment_path):
        raise EvaluationBlocked("runtime segmentation must match the source replay")
    output.mkdir(parents=True)
    network_evidence = dict(
        schema="hierarchy_checkpoint_bound_training_evidence_20261030_v1", complete=True,
        trained_components=["scene_classifier", "skill_policy", "candidate_network"],
        component_artifact_sha256={"scene_classifier": file_sha256(root/skill_path),
                                   "skill_policy": file_sha256(root/skill_path),
                                   "candidate_network": file_sha256(root/candidate_path)},
        skill_run=pointer(root, root/skill_status_path),
        candidate_run=pointer(root, root/candidate_status_path), data=pointer(root, root/data_path),
        TEST_used=False, native_success_certified=False,
        same_skill_embedding_conditions_both_consumers=True,
        policy_goal="one predicted handoff point held for the whole episode",
        candidate_probability_semantics=cm["score_semantics"],
        supervised_classification_is_not_task_success=True,
    )
    _write_new(output/"training_evidence.json", network_evidence)
    tracking_contract = dict(
        schema="hierarchy_source_blind_runtime_tracking_contract_20261030_v1", complete=True,
        implementation="our_mani.hierarchy_20261030.tracking.BlindPublicMotionPipeline",
        source_task_or_speed_selects_frontend=False,
        motion_inputs=pointer(root, root/motion_path), segmentation_model=pointer(root, root/segment_path),
        segmentation_fit=pointer(root, root/segment_manifest_path),
        actual_sources=477, learned_masks_only=True, native_instance_ids_as_inputs=False,
        quality_is_not_certified_perfect=True,
        known_limits=["TRAIN development peg identities are confused by the initial segmentation model",
                      "wrong-scene public priors can produce false positives",
                      "occluded/closed manipulated bodies remain explicit unknown",
                      "candidate uses only real, currently valid forecast bodies"],
        TEST_used=False,
    )
    _write_new(output/"tracking_contract.json", tracking_contract)
    forecaster_evidence = dict(
        schema="hierarchy_retained_forecaster_evidence_20261030_v1", complete=True,
        checkpoint_sha256=FORECAST_SHA, history_dim=21, state_dim=19,
        dt=.05, forecast_steps=22, weights_changed=False,
        history_source="five to twenty actual causal public tracked observations",
        unavailable_input="unknown mask; never privileged pose or fabricated trajectory",
        historical_training="v671 original TRAIN parents across sixteen versions; retained initialization",
        TEST_used=False,
    )
    _write_new(output/"forecaster_evidence.json", forecaster_evidence)
    new_sources = tuple(str(path.relative_to(root)) for path in sorted((root/"our_mani/hierarchy_20261030").glob("*.py")))
    old_sources = (
        "our_mani/contracts.py", "our_mani/context_dynamics.py", "our_mani/perception.py",
        "our_mani/benchmark_perception.py", "our_mani/multitask/context_se3.py",
        "our_mani/multitask/perception.py", "our_mani/multitask/training_geometry.py",
        "our_mani/multitask/rotation_equivalence.py",
        "scripts/evaluate_hierarchy_test_20261030.py",
        "scripts/run_hierarchy_workflow_20261030.py",
    )
    sources = sorted(set((*SOURCE_FILES, *new_sources, *old_sources)))
    shared_evidence = pointer(root, output/"training_evidence.json")
    freeze = dict(
        schema=FREEZE_SCHEMA, frozen=True, tasks=list(TASK_FAMILIES),
        readiness={name: True for name in ("scene_classifier_trained", "skill_policy_trained",
                                         "candidate_trained", "public_tracking_ready", "forecaster_ready")},
        components={
            "scene_classifier": {"artifact": pointer(root, root/skill_path), "evidence": shared_evidence, "trained": True},
            "skill_policy": {"artifact": pointer(root, root/skill_path), "evidence": shared_evidence, "trained": True},
            "candidate_network": {"artifact": pointer(root, root/candidate_path), "evidence": shared_evidence, "trained": True},
            "forecaster": {"artifact": pointer(root, root/FORECAST), "evidence": pointer(root, output/"forecaster_evidence.json")},
            "tracking": {"artifact": pointer(root, output/"tracking_contract.json"), "evidence": pointer(root, root/motion_path)},
            "segmentation": {"artifact": pointer(root, root/segment_path), "evidence": pointer(root, root/segment_manifest_path), "trained": True},
        },
        actor={"module": "our_mani.hierarchy_20261030.actor", "factory": "create_actor",
               "source": pointer(root, root/"our_mani/hierarchy_20261030/actor.py")},
        runtime_sources=[pointer(root, root/name) for name in sources],
        protocol={"dt": .05, "simulation_seconds": 10., "gripper_velocity": .2,
                  "baseline_decision_ticks": 10, "warmup_ticks": 4, "seed": 0,
                  "inference_device": "cpu", "baseline_repo": "/home/ubuntu/桌面/phsimani/PhysMani",
                  "current_instance_masks": True,
                  "information_setting": "native common sensor capture; actor sees causal RGBD/calibration/proprio only"},
        policy_inputs="causal_RGBD_calibration_robot_only", old_stage_policy_allowed=False,
        forecast_fallback_allowed=False, TEST_used_for_fitting=False,
        goal_time_seconds=.5, probability_is_calibrated_success=False,
        technical_readiness_is_not_success_qualification=True,
    )
    validate_freeze(root, freeze)
    # Construct the actual inference graph before committing a usable freeze.
    # This opens no native runtime or TEST recording.
    from .actor import create_actor
    actor = create_actor(root=root, freeze_manifest=freeze, device="cpu")
    actor.close()
    target = output/"freeze.json"
    _write_new(target, freeze)
    return target
