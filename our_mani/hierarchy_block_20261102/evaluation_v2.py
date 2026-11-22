"""Versioned two-task evaluation contracts, provenance and explicit execution.

CPU planning is allowed without a model. Native execution requires a complete
SHA-bound hierarchy freeze, public tracking and trained candidate selector.
This version deliberately has no TEST reader or TEST execution entry point.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import is_dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

import h5py
import numpy as np

SCHEMA = "hierarchy_block_native_evaluation_20261102_v2"
FREEZE_SCHEMA = "hierarchy_block_runtime_freeze_20261102_v2"
TASKS = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed")
BLOCKS = frozenset(TASKS[:2])
ARTIFACTS = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102/evaluation_v2")
RUNTIME = Path("runtime/framework/hierarchy_block_20261102/evaluation_v2")
FAMILY_ROOT = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102")
ENV_MANIFEST = Path("artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json")
SHARED_LOCKS = ("runtime/framework/fixed_TRAIN104_geometry_native.lock",
                "runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock")
PUBLIC_CHANNELS = ("timestamp", "observation/images", "observation/depths", "observation/calibrations",
                   "observation/camera_timestamps", "observation/sensor_protocol")
COMPONENTS = ("scene_classifier", "skill_policy", "candidate_network", "forecaster", "tracking")
READY_FIELDS = ("scene_classifier_trained", "skill_policy_trained", "candidate_trained",
                "public_tracking_ready", "forecaster_ready")
NATIVE_PYTHON = Path("/home/ubuntu/physmani-conda/envs/physmani5090/bin/python")
SOURCE_FILES = (
    "our_mani/hierarchy_block_20261102/evaluation_v2.py", "scripts/evaluate_hierarchy_case_block_20261102_v2.py",
    "our_mani/hierarchy_block_20261102/networks.py", "our_mani/hierarchy_block_20261102/data.py",
    "our_mani/hierarchy_block_20261102/isolation.py", "scripts/run_hierarchy_isolated_block_20261102.sh",
    "scripts/run_hierarchy_frozen_environment_block_20261102.py",
    "our_mani/multitask/author_aligned_geometry_executor_v1114.py",
    "our_mani/multitask/physmani_executor_v7.py", "our_mani/sensor_alignment/physmani_executor_v6.py",
    "our_mani/sensor_alignment/physmani_executor_v5.py", "our_mani/ablations/physmani_executor_v4.py",
    "our_mani/multitask/native_outcome_sampling_v741.py", "our_mani/multitask/physmani_block_receipt_v541.py",
    "our_mani/multitask/physmani_block_policy_v538.py", "our_mani/multitask/simulation.py",
    "our_mani/multitask/maps.py", "our_mani/multitask/native_tasks.py", "our_mani/multitask/specs.py",
    "our_mani/sensor_alignment/author_rgbd.py", "our_mani/aligned/simulation.py",
    "our_mani/aligned/protocol.py", "our_mani/pyrep_env.py", "scripts/collect_local_rollouts.py",
    "scripts/run_frozen_system_environment_v820_r10.py", "scripts/run_local_sim.sh",
)


class EvaluationBlocked(ValueError):
    """A missing dependency cannot become a substitute forecast or action."""


class ActorContractError(RuntimeError):
    pass


def native_interpreter_contract():
    """Check the actual ABI before any Native environment is constructed."""
    if sys.version_info[:2] != (3, 10) or Path(sys.executable).resolve() != NATIVE_PYTHON.resolve(strict=True):
        raise EvaluationBlocked("Native execution requires runtime.sh's verified Python 3.10; training Python is not a runtime substitute")
    import _cffi_backend
    return {"executable": sys.executable, "resolved_executable": str(Path(sys.executable).resolve()),
            "version": list(sys.version_info[:3]), "cffi_backend": str(Path(_cffi_backend.__file__).resolve()),
            "cffi_backend_sha256": sha(_cffi_backend.__file__), "python_binary_sha256": sha(NATIVE_PYTHON.resolve()),
            "selection": "python_resolved_from_frozen_runtime_sh_PATH", "native_started": False}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    partial.replace(path)


def pointer(root, path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def checked(root, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise EvaluationBlocked("exact project path/SHA256 pointer required")
    path = (root / reference["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or sha(path) != reference["sha256"]:
        raise EvaluationBlocked("bound project artifact changed: " + str(path))
    return path


def stop_evidence(root, *paths):
    """Bind direct administrative requests only in this experiment's namespace."""
    root = Path(root).resolve()
    family = root / "runtime/framework/hierarchy_block_20261102"
    evidence = []
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        if path.name != "STOP" or not path.is_relative_to(family):
            raise EvaluationBlocked("administrative STOP evidence must belong to the new block namespace")
        if path.is_file():
            evidence.append(pointer(root, path))
    return evidence


def allowed_split(split):
    # Deliberately before any manifest or artifact read. A later implementation
    # must first validate a new freeze plus full Native VAL release to add TEST.
    if split not in ("train", "validation"):
        raise EvaluationBlocked("TEST remains sealed; this entry has no TEST reader. New freeze and complete 40-case Native VAL audit release are required")


def select_sources(manifest, split):
    allowed_split(split)
    if manifest.get("split") != split:
        raise EvaluationBlocked("source manifest split differs")
    rows = []
    for task in TASKS:
        group = [row for row in manifest["episodes"] if row["task"] == task]
        if split == "train":
            if len(group) < 4:
                raise EvaluationBlocked("four fixed TRAIN sources per task required")
            group = group[:4]
        rows.extend(group)
    expected = {"train": 8, "validation": 40}[split]
    identities = [(row["task"], row["raw_episode_id"]) for row in rows]
    if len(rows) != expected or len(set(identities)) != expected:
        raise EvaluationBlocked("exact fixed source roster differs")
    if split == "validation":
        counts = {task: sum(row["task"] == task for row in rows) for task in TASKS}
        if counts != dict(zip(TASKS, (20, 20))):
            raise EvaluationBlocked("original two-task VAL40 shortfalls must remain unchanged")
    return rows


def validate_freeze(root, manifest):
    """Validate artifact bytes and explicit training/readiness evidence on CPU."""
    root = Path(root).resolve()
    if manifest.get("schema") != FREEZE_SCHEMA or manifest.get("frozen") is not True:
        raise EvaluationBlocked("explicit completed hierarchy runtime freeze required")
    if set(manifest.get("tasks", [])) != set(TASKS):
        raise EvaluationBlocked("freeze must contain only the two agreed tasks")
    missing = [key for key in READY_FIELDS if manifest.get("readiness", {}).get(key) is not True]
    if missing:
        raise EvaluationBlocked("hierarchy dependencies unready: " + ", ".join(missing))
    components = manifest.get("components", {})
    for name in COMPONENTS:
        value = components.get(name, {})
        checked(root, value.get("artifact"))
        evidence_path = checked(root, value.get("evidence"))
        if name in COMPONENTS[:3] and value.get("trained") is not True:
            raise EvaluationBlocked("untrained network cannot evaluate: " + name)
        if name in COMPONENTS[:3]:
            evidence = json.loads(evidence_path.read_text())
            if (evidence.get("complete") is not True or name not in evidence.get("trained_components", [])
                    or evidence.get("component_artifact_sha256", {}).get(name) != value["artifact"]["sha256"]
                    or evidence.get("TEST_used") is not False):
                raise EvaluationBlocked("complete checkpoint-bound TRAIN evidence required: " + name)
    actor = manifest.get("actor", {})
    source = checked(root, actor.get("source"))
    if not actor.get("module", "").startswith("our_mani.hierarchy_block_20261102.") or not actor.get("factory"):
        raise EvaluationBlocked("explicit new hierarchy actor module/factory required")
    if source.suffix != ".py" or not source.is_relative_to(root / "our_mani/hierarchy_block_20261102"):
        raise EvaluationBlocked("actor must use a bound new hierarchy source")
    for reference in manifest.get("runtime_sources", []):
        checked(root, reference)
    # The builder validates its complete in-memory draft before constructing
    # the actor. CPU readiness is added only after that real construction, so
    # this draft-level validator permits its absence, but never a bad pointer.
    if "training_status" in manifest:
        checked(root, manifest["training_status"])
    for reference in manifest.get("offline_precision_reports", []):
        checked(root, reference)
    if "cpu_actor_readiness" in manifest:
        readiness = json.loads(checked(root, manifest["cpu_actor_readiness"]).read_text())
        if readiness.get("complete") is not True or readiness.get("device") != "cpu":
            raise EvaluationBlocked("completed CPU actor readiness attestation required")
    protocol = manifest.get("protocol", {})
    for key, expected in (("dt", .05), ("simulation_seconds", 10.), ("gripper_velocity", .2),
                          ("baseline_decision_ticks", 10), ("warmup_ticks", 4)):
        if protocol.get(key) != expected:
            raise EvaluationBlocked("frozen physical/action cadence differs: " + key)
    if manifest.get("policy_inputs") != "causal_RGBD_calibration_robot_only":
        raise EvaluationBlocked("policy input contract must exclude native labels and task metadata")
    if manifest.get("old_stage_policy_allowed") is not False or manifest.get("forecast_fallback_allowed") is not False:
        raise EvaluationBlocked("old policy stages and substitute forecasts are forbidden")
    if protocol.get("baseline_repo") != "/home/ubuntu/桌面/phsimani/PhysMani":
        raise EvaluationBlocked("local frozen PhysMani repository required")
    if protocol.get("current_instance_masks") is not True or not protocol.get("information_setting"):
        raise EvaluationBlocked("native sensor protocol is separate from filtered RGBD-only actor inputs")
    if protocol.get("inference_device") != "cpu":
        raise EvaluationBlocked("this experiment freezes serial CPU inference")
    # Accuracy qualification and the user's authorized experimental evaluation
    # are distinct. A completed unqualified network is still measured honestly.
    if type(manifest.get("precision_qualified")) is not bool:
        raise EvaluationBlocked("freeze must explicitly disclose offline precision qualification")
    return manifest


def load_freeze(root, path):
    root, path = Path(root).resolve(), Path(path).resolve(strict=True)
    if not path.is_relative_to(root / FAMILY_ROOT):
        raise EvaluationBlocked("freeze must belong to the new hierarchy namespace")
    manifest = json.loads(path.read_text())
    if "cpu_actor_readiness" not in manifest:
        raise EvaluationBlocked("published runtime freeze requires actual CPU actor readiness evidence")
    return validate_freeze(root, manifest)


def native_source_binding(native_repo, task):
    """Legacy pick has no core R10 TaskSpec; bind its actual native files too."""
    from our_mani.multitask.native_tasks import verify_native_sources
    from our_mani.multitask.specs import NATIVE_COMMON_SOURCE_SHA256
    if task in BLOCKS:
        result = dict(NATIVE_COMMON_SOURCE_SHA256)
        for name, expected in result.items():
            if sha(native_repo / name) != expected:
                raise EvaluationBlocked("frozen native common source changed: " + name)
        name = "third_party/RLBench/rlbench/tasks/" + task + ".py"
        result[name] = sha(native_repo / name)
    else:
        result = dict(verify_native_sources(native_repo, task))
    scene = "third_party/RLBench/rlbench/task_ttms/" + task + ".ttm"
    result[scene] = sha(native_repo / scene)
    return result


def prepare_plan(root, split, *, freeze_path=None, plan_path=None, train_release_path=None):
    allowed_split(split)
    root = Path(root).resolve()
    manifest_path = root / f"artifacts/framework/protocol/{split}_manifest.json"
    rows = select_sources(json.loads(manifest_path.read_text()), split)
    native_repo = Path("/home/ubuntu/桌面/phsimani/PhysMani").resolve(strict=True)
    package = "train_package_compressed" if split == "train" else "val_package_compressed"
    cases = []
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        expected = native_repo / "data/rmt/physmani_bench" / package / row["task"] / "all_variations/episodes" / row["raw_episode_id"] / "data.h5"
        if path != expected:
            raise EvaluationBlocked("source path does not match its fixed split/task/episode")
        before = path.stat()
        source_sha = sha(path)
        with h5py.File(path, "r") as source:
            variation = int(source.attrs["variation"])
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino):
            raise EvaluationBlocked("source raced during provenance capture")
        cases.append({"case_key": row["task"] + "__" + row["raw_episode_id"], "task": row["task"],
                      "raw_episode_id": row["raw_episode_id"], "path": str(path), "variation": variation,
                      "source_sha256": source_sha, "source_bytes": before.st_size})
    freeze = None if freeze_path is None else pointer(root, Path(freeze_path))
    blockers = []
    if freeze is None:
        blockers.append("complete trained hierarchy/public-tracking freeze missing")
    else:
        try:
            load_freeze(root, root / freeze["path"])
        except (ValueError, OSError) as error:
            blockers.append(str(error))
    train_release = None if train_release_path is None else pointer(root, Path(train_release_path))
    if split == "validation":
        if train_release is None:
            blockers.append("complete exit-audited fixed TRAIN8 release missing")
        elif freeze is not None:
            try:
                validate_prior_train_release(root, train_release, freeze)
            except (ValueError, OSError) as error:
                blockers.append(str(error))
    plan = {"schema": SCHEMA, "split": split, "method": "new_learned_hierarchy_only",
            "tasks": list(TASKS), "source_manifest": pointer(root, manifest_path), "cases": cases,
            "expected_cases": len(cases), "selection": "first four declared TRAIN sources per task" if split == "train" else "all original two-task VAL40",
            "freeze": freeze, "native_execution_ready": not blockers, "blockers": blockers,
            "TRAIN_release": train_release,
            "automatic_retry": False, "dt": .05, "decision_ticks": 10, "budget_ticks": 200,
            "warmup_ticks": 4, "runtime_unknown_behavior": "explicit_counted_hold_with_unknown_diagnostics",
            "source_motion_replayed": False, "source_use": "original block initialization or source-seeded core development reset; labels excluded from actor",
            "environment": pointer(root, root / ENV_MANIFEST),
            "native_sources": {task: native_source_binding(native_repo, task) for task in TASKS},
            "runtime_sources": {name: sha(root / name) for name in SOURCE_FILES},
            "TEST_read": False}
    if freeze is not None and not blockers:
        plan["precision_qualified"] = load_freeze(root, root / freeze["path"])["precision_qualified"]
        plan["evaluation_of_unqualified_model_allowed"] = True
    path = root / ARTIFACTS / split / "plan.json" if plan_path is None else Path(plan_path).resolve()
    if not path.is_relative_to(root / ARTIFACTS):
        raise EvaluationBlocked("plan output must stay in the new evaluation namespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise EvaluationBlocked("immutable plan already exists with different provenance; use a new plan path")
    else:
        atomic_json(path, plan)
    return plan, path


def verify_plan(root, plan_path, *, require_ready=True):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve(strict=True)
    if not plan_path.is_relative_to(root / ARTIFACTS):
        raise EvaluationBlocked("new evaluation namespace plan required")
    plan = json.loads(plan_path.read_text())
    allowed_split(plan.get("split"))
    if plan.get("schema") != SCHEMA or plan.get("method") != "new_learned_hierarchy_only":
        raise EvaluationBlocked("new two-task plan identity required")
    if (set(plan.get("tasks", [])) != set(TASKS) or plan.get("automatic_retry") is not False
            or plan.get("dt") != .05 or plan.get("decision_ticks") != 10 or plan.get("budget_ticks") != 200
            or plan.get("warmup_ticks") != 4 or plan.get("TEST_read") is not False
            or set(plan.get("runtime_sources", {})) != set(SOURCE_FILES)):
        raise EvaluationBlocked("frozen task/runtime/cadence/source closure contract differs")
    rows = select_sources(json.loads(checked(root, plan["source_manifest"]).read_text()), plan["split"])
    if plan.get("expected_cases") != len(rows):
        raise EvaluationBlocked("complete fixed case count differs")
    if [(r["task"], r["raw_episode_id"], r["path"]) for r in rows] != [(r["task"], r["raw_episode_id"], r["path"]) for r in plan["cases"]]:
        raise EvaluationBlocked("case roster was changed or filtered")
    for name, expected in plan["runtime_sources"].items():
        if sha(root / name) != expected:
            raise EvaluationBlocked("evaluation source changed: " + name)
    for row in plan["cases"]:
        if sha(row["path"]) != row["source_sha256"]:
            raise EvaluationBlocked("fixed source hash changed: " + row["case_key"])
    checked(root, plan["environment"])
    repo = Path("/home/ubuntu/桌面/phsimani/PhysMani")
    for task in TASKS:
        if native_source_binding(repo, task) != plan["native_sources"][task]:
            raise EvaluationBlocked("native source identity changed")
    if require_ready:
        if plan.get("native_execution_ready") is not True or plan.get("freeze") is None:
            raise EvaluationBlocked("Native evaluation blocked: " + "; ".join(plan.get("blockers", [])))
        freeze = load_freeze(root, checked(root, plan["freeze"]))
        if plan["split"] == "validation":
            validate_prior_train_release(root, plan.get("TRAIN_release"), plan["freeze"])
        return plan, freeze
    return plan, None


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    value = Path(f"/proc/{pid}/stat").read_text()
    return {"pid": pid, "start_ticks": int(value.rsplit(")", 1)[1].split()[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


@contextmanager
def evaluation_locks(root, *, case_runtime=None, case_output=None, slot=0, parallel=False):
    if case_runtime is not None:
        from .isolation import resource_scope
        with resource_scope(root, case_runtime, slot=slot, parallel=parallel, case_output=case_output) as receipt:
            yield receipt
        return
    directory = root / RUNTIME
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for path, mode in [(directory / "native.lock", "a+b"), *((root / name, "rb") for name in SHARED_LOCKS)]:
            handle = path.open(mode)
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield {"process": process_identity(), "lock_paths": [str(Path(h.name).relative_to(root)) for h in handles]}
    finally:
        for handle in reversed(handles):
            handle.close()


def public_observation(observation):
    missing = set(PUBLIC_CHANNELS[:4]) - set(observation)
    if missing:
        raise ActorContractError("missing causal RGBD/calibration channels: " + str(sorted(missing)))
    # No observation/state, instance masks, task name, native handles or debug.
    return {name: observation[name] for name in PUBLIC_CHANNELS if name in observation}


def public_robot(robot):
    if not is_dataclass(robot):
        raise ActorContractError("numeric RobotState dataclass required")
    return replace(robot, task_success=False)


def validate_decision(decision, *, timestamp=None):
    if not isinstance(decision, dict) or set(decision) != {"action", "receipt"}:
        raise ActorContractError("actor must return action plus runtime evidence")
    evidence = decision["receipt"]
    if not isinstance(evidence, dict):
        raise ActorContractError("numeric actor evidence mapping required")
    if evidence.get("old_stage_rule_calls") != 0 or evidence.get("task_metadata_used") is not False:
        raise ActorContractError("old stage policy or hidden task metadata entered the action path")
    if decision["action"] is None:
        if evidence.get("event") != "perception_unknown" or not isinstance(evidence.get("reason"), str) or not evidence["reason"]:
            raise ActorContractError("missing input must be an explicit perception_unknown reason")
        if evidence.get("real_forecast_consumed") is not False:
            raise ActorContractError("unknown input cannot claim a consumed real forecast")
        json.dumps(evidence, allow_nan=False)
        return None, evidence
    action = np.asarray(decision["action"], dtype=float)
    if action.shape != (8,) or not np.isfinite(action).all() or not np.isclose(np.linalg.norm(action[3:7]), 1., atol=1e-5):
        raise ActorContractError("finite xyz/unit-xyzw/gripper action required")
    if action[7] not in (0., 1.):
        raise ActorContractError("network must select a discrete native gripper command")
    latent = evidence.get("shared_skill_latent_sha256")
    if not isinstance(latent, str) or len(latent) != 64 or any(x not in "0123456789abcdef" for x in latent):
        raise ActorContractError("numeric shared skill latent fingerprint required")
    if evidence.get("policy_skill_latent_sha256") != latent:
        raise ActorContractError("skill policy must consume the current shared skill latent")
    event = evidence.get("event")
    if event == "candidate_handoff":
        if evidence.get("candidate_skill_latent_sha256") != latent:
            raise ActorContractError("candidate and skill policy must consume the same skill latent at handoff")
        for key in ("public_tracks_valid", "real_forecast_consumed"):
            if evidence.get(key) is not True:
                raise ActorContractError("missing actual public tracking/forecast: " + key)
        if timestamp is None or evidence.get("forecast_generated_at") != timestamp:
            raise ActorContractError("handoff requires the actual current forecast timestamp")
    elif event == "skill_execution":
        if evidence.get("candidate_skill_latent_sha256") is not None:
            raise ActorContractError("cached candidate latent cannot be reported as current candidate consumption")
        if evidence.get("real_forecast_consumed") is not False:
            raise ActorContractError("skill continuation must explicitly report no new forecast consumption")
    else:
        raise ActorContractError("explicit candidate_handoff or learned skill_execution event required")
    json.dumps(evidence, allow_nan=False)
    return action.copy(), evidence


@contextmanager
def action_scope(env, block, *, label="learned_hierarchy_action", max_ticks=10):
    if not block:
        with env.native_action_scope(label=label, max_ticks=max_ticks):
            yield
        return
    if not isinstance(env.task._fail_conditions, list) or env.task._fail_conditions:
        raise RuntimeError("legacy block requires its original empty failure-condition list")
    before = env.ledger.ticks
    env._before_action()
    yield
    if env.task._fail_conditions or not before < env.ledger.ticks <= min(before + max_ticks, 200):
        raise RuntimeError("block action clock/native outcome contract changed")


def run_loop(env, actor, executor, *, block, stop_requested=lambda: False, warmup_ticks=4, trace=None, resource_receipt=None):
    """Learned actions, frozen initial observation warmup and explicit unknown holds."""
    if warmup_ticks != 4:
        raise ActorContractError("this version fixes four observation warmup ticks inside the 200-tick budget")
    if getattr(executor, "skill_policy", None) is not None:
        raise ActorContractError("executor qualification/stage policy must be absent")
    trace = {} if trace is None else trace
    steps, holds = trace.setdefault("steps", []), trace.setdefault("holds", [])
    last_observed_time = None
    def observe(observation, robot):
        nonlocal last_observed_time
        timestamp = float(robot.timestamp)
        if last_observed_time is not None:
            if timestamp < last_observed_time:
                raise ActorContractError("causal observation time moved backwards")
            if timestamp == last_observed_time:
                return
        actor.observe(public_observation(observation), public_robot(robot))
        last_observed_time = timestamp
    observation, robot = env.observe()
    observe(observation, robot)
    administrative_stop = False
    while env.ledger.ticks < 200 and not env._native_terminal_latched:
        if stop_requested():
            administrative_stop = True
            break
        if resource_receipt is not None:
            from .isolation import assert_live
            assert_live(resource_receipt)
        before = int(env.ledger.ticks)
        if before < warmup_ticks:
            with action_scope(env, block, label="frozen_observation_warmup", max_ticks=1):
                env.hold_tick()
            if env.ledger.ticks != before + 1:
                raise RuntimeError("warmup hold must account exactly one tick")
            holds.append({"start_tick": before, "end_tick": env.ledger.ticks,
                          "event": "frozen_observation_warmup", "inside_200_tick_budget": True})
            observation, robot = env.observe()
            observe(observation, robot)
            continue
        decision = actor.act(public_observation(observation), public_robot(robot), remaining_seconds=env.ledger.remaining)
        action, evidence = validate_decision(decision, timestamp=float(robot.timestamp))
        if action is None:
            with action_scope(env, block, label="perception_unknown_hold", max_ticks=1):
                env.hold_tick()
            if env.ledger.ticks != before + 1:
                raise RuntimeError("unknown-input hold must account exactly one tick")
            holds.append({"start_tick": before, "end_tick": env.ledger.ticks, "actor": evidence,
                          "event": "perception_unknown", "no_native_ID_or_pose_substitution": True})
            observation, robot = env.observe()
            observe(observation, robot)
            continue
        with action_scope(env, block):
            execution = executor.execute(env, action, observer=observe)
        if not before < env.ledger.ticks <= min(before + 10, 200):
            raise RuntimeError("executor exceeded the fixed ten-tick decision allowance")
        steps.append({"start_tick": before, "end_tick": env.ledger.ticks,
                      "action": action.tolist(), "actor": evidence, "execution": execution})
        observation, robot = env.observe()
        observe(observation, robot)
    success = bool(env._native_success_latched)
    failure = bool(getattr(env, "_native_failure_latched", False))
    natural = not administrative_stop and (env.ledger.ticks == 200 or env._native_terminal_latched)
    category = ("AdministrativeSTOP" if administrative_stop else "NativeTrue" if success
                else "NativeTerminalFailure" if failure else "NativeFalse" if natural else "Incomplete")
    return {"complete": natural, "natural_completion": natural, "category": category,
            "native_success": success if natural else None, "native_terminal_failure": failure if natural else None,
            "administrative_stop": administrative_stop, "infrastructure_failed": False,
            "counted_ticks": int(env.ledger.ticks), "steps": steps, "holds": holds,
            "perception_unknown_holds": sum(row["event"] == "perception_unknown" for row in holds),
            "warmup_ticks": warmup_ticks, "clock": env.ledger.receipt(),
            "actual_process_exit_observed": False, "automatic_retry": False}


def create_native_environment(case, split, protocol):
    """Instantiate only frozen physical classes; no old task policy factory."""
    if split not in ("train", "validation"):
        allowed_split(split)
    if case["task"] in BLOCKS:
        from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation
        from our_mani.multitask.physmani_block_receipt_v541 import ReceiptBlockView
        from scripts.collect_local_rollouts import initialization
        raw = AlignedLocalSimulation(case["task"], case["path"], protocol=protocol, seed=protocol.get("seed", 0), servo_lead=1.)
        try:
            x, velocity, acceleration, goal = initialization(case["path"])
            if not raw.reset(target_position=x, target_velocity=velocity, target_acceleration=acceleration, goal_position=goal):
                raise RuntimeError("original source-bound block reset IK failed")
            raw.finish_home_reset()
            return ReceiptBlockView(raw)
        except BaseException:
            raw.close()
            raise
    from our_mani.multitask.simulation import AlignedDevelopmentSimulation
    from our_mani.multitask.native_outcome_sampling_v741 import AuthorActionOutcomeSamplingMixin
    class OutcomeDevelopment(AuthorActionOutcomeSamplingMixin, AlignedDevelopmentSimulation):
        pass
    env = OutcomeDevelopment(case["task"], case["path"], source=case, split=split,
                             protocol=protocol, seed=protocol.get("seed", 0), replica=0)
    try:
        env.reset_development(variation=case["variation"])
        return env
    except BaseException:
        env.close()
        raise


def cpu_preflight(root, plan_path):
    plan, freeze = verify_plan(root, plan_path)
    report = {"schema": SCHEMA + "_CPU_preflight", "checked_utc": utc(), "plan": pointer(root, plan_path),
              "freeze": plan["freeze"], "expected_cases": plan["expected_cases"],
              "all_required_components_SHA_verified": True, "native_started": False,
              "actor_runtime_not_yet_exercised": True, "qualification_granted": False}
    destination = root / RUNTIME / plan["split"]
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / ("CPU_preflight_" + str(time.time_ns()) + ".json"), report)
    return report


def worker_case(root, plan_path, case_key, *, device="cpu", resource_slot=0, parallel=False):
    root = Path(root).resolve()
    plan, freeze = verify_plan(root, plan_path)
    cases = [case for case in plan["cases"] if case["case_key"] == case_key]
    if len(cases) != 1:
        raise EvaluationBlocked("one fixed planned case required")
    if device != "cpu" or not os.environ.get("DISPLAY"):
        raise EvaluationBlocked("CPU inference and independent DISPLAY / xvfb-run required")
    interpreter = native_interpreter_contract()
    if parallel:
        from scripts.run_hierarchy_frozen_environment_block_20261102 import namespace_context
    else:
        from scripts.run_frozen_system_environment_v820_r10 import namespace_context
    namespace_context(plan["environment"]["sha256"])
    runtime = root / RUNTIME / plan["split"] / "cases" / case_key
    launch = json.loads((runtime / "launch.json").read_text())
    if launch.get("token") != os.environ.get("HIERARCHY_EVALUATION_PARENT_TOKEN"):
        raise EvaluationBlocked("single-case parent launcher receipt required")
    if process_identity(launch["parent"]["pid"]) != launch["parent"]:
        raise EvaluationBlocked("original parent launcher is no longer live")
    owner_stop = None if launch.get("administrative_stop_path") is None else root / launch["administrative_stop_path"]
    output = root / ARTIFACTS / plan["split"] / "cases" / case_key
    if output.exists() or (runtime / "terminal.json").exists():
        raise EvaluationBlocked("preserve previous case attempt; no overwrite/retry")
    stop = root / RUNTIME / "STOP"
    if stop_evidence(root, stop, owner_stop):
        raise EvaluationBlocked("new evaluation namespace STOP exists")
    env = actor = None
    record = {"schema": SCHEMA, "plan": pointer(root, plan_path), "freeze": plan["freeze"],
              "case": cases[0], "split": plan["split"], "complete": False,
              "natural_completion": False, "actual_process_exit_observed": False,
              "native_interpreter": interpreter}
    error = None
    trace = {"steps": [], "holds": []}
    from .isolation import record_native_interval
    output_owned = False
    try:
        with evaluation_locks(root, case_runtime=runtime, case_output=output, slot=resource_slot, parallel=parallel) as resources:
            output.mkdir(parents=True)
            output_owned = True
            record["resources"] = resources
            try:
                module = importlib.import_module(freeze["actor"]["module"])
                if Path(module.__file__).resolve() != checked(root, freeze["actor"]["source"]):
                    raise EvaluationBlocked("actor module resolved to the wrong source file")
                actor = getattr(module, freeze["actor"]["factory"])(root=root, freeze_manifest=freeze, device=device)
                if not all(callable(getattr(actor, method, None)) for method in ("observe", "act")):
                    raise ActorContractError("actor.observe/act required")
                env = create_native_environment(cases[0], plan["split"], freeze["protocol"])
                record_native_interval(resources, phase="start")
                from our_mani.multitask.author_aligned_geometry_executor_v1114 import AuthorAlignedGeometryExecutor
                executor = AuthorAlignedGeometryExecutor(freeze["protocol"], observe_public_frame=lambda *_: None,
                    skill_policy=None, admit_pose_action=None, admit_joint_target=None, collision_diagnostic_limit=0)
                record.update(run_loop(env, actor, executor, block=cases[0]["task"] in BLOCKS,
                                       stop_requested=lambda: bool(stop_evidence(root, stop, owner_stop)), trace=trace,
                                       resource_receipt=resources))
            except BaseException as exc:
                error = exc
                record.update(category="InfrastructureFailure", infrastructure_failed=True, complete=False,
                              natural_completion=False, native_success=None, native_terminal_failure=None,
                              administrative_stop=False, error_type=type(exc).__name__, error=str(exc))
                record.update(trace)
                if env is not None:
                    record["counted_ticks"] = int(env.ledger.ticks)
                    record["clock"] = env.ledger.receipt()
            finally:
                for value in (actor, env):
                    if value is not None and callable(getattr(value, "close", None)):
                        try:
                            value.close()
                        except BaseException as exc:
                            record.update(category="InfrastructureFailure", infrastructure_failed=True,
                                          complete=False, natural_completion=False, native_success=None,
                                          native_terminal_failure=None, cleanup_error=repr(exc))
                            error = exc if error is None else error
                if env is not None and "start_monotonic" in resources.get("native_interval", {}):
                    record_native_interval(resources, phase="end")
                record.update(finished_utc=utc(), process=process_identity(), automatic_retry=False)
    except BaseException as scope_error:
        record.update(category="InfrastructureFailure", infrastructure_failed=True, complete=False,
                      natural_completion=False, native_success=None, native_terminal_failure=None,
                      administrative_stop=False, resource_scope_error_type=type(scope_error).__name__,
                      resource_scope_error=str(scope_error), finished_utc=utc(),
                      process=process_identity(), automatic_retry=False)
        record.update(trace)
    if not output_owned:
        if output.exists():
            raise EvaluationBlocked("preserve output owned by a prior/concurrent attempt")
        output.mkdir(parents=True)
    atomic_json(output / "case.json", record)
    atomic_json(runtime / "terminal.json", record)
    return record


def launch_case(root, plan_path, case_key, *, timeout=900, device="cpu", administrative_stop_path=None,
                resource_slot=0, parallel=False):
    """One subprocess, one actual wait/exit receipt; never a hidden retry loop."""
    root = Path(root).resolve()
    if device != "cpu" or timeout <= 0:
        raise EvaluationBlocked("CPU inference and positive case timeout required")
    stop_evidence(root, root / RUNTIME / "STOP", administrative_stop_path)
    plan, _ = verify_plan(root, plan_path)
    if case_key not in [case["case_key"] for case in plan["cases"]]:
        raise EvaluationBlocked("one fixed planned case required")
    directory = root / RUNTIME / plan["split"] / "cases" / case_key
    if directory.exists():
        raise EvaluationBlocked("prior launch exists; no silent retry")
    directory.mkdir(parents=True)
    token = secrets.token_hex(24)
    command = ["bash", "scripts/run_hierarchy_isolated_block_20261102.sh", str(directory), "--",
               "scripts/run_hierarchy_frozen_environment_block_20261102.py", "--manifest-sha256", plan["environment"]["sha256"],
               *([] if parallel else ["--shared-network"]), "--",
               "python", "-B",
               "scripts/evaluate_hierarchy_case_block_20261102_v2.py", "--worker-case", case_key,
               "--plan", str(Path(plan_path).resolve()), "--device", device, "--resource-slot", str(resource_slot)]
    if parallel:
        command.append("--parallel-native")
    launch = {"schema": SCHEMA + "_launch", "parent": process_identity(), "token": token,
              "command": command, "started_utc": utc(), "plan": pointer(root, plan_path),
              "administrative_stop_path": None if administrative_stop_path is None else str(Path(administrative_stop_path).resolve().relative_to(root)),
              "resource_slot": resource_slot, "parallel": parallel,
              "automatic_retry": False}
    atomic_json(directory / "launch.json", launch)
    started = time.monotonic()
    timed_out = False
    with (directory / "native.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, HIERARCHY_EVALUATION_PARENT_TOKEN=token, PYTHONDONTWRITEBYTECODE="1"), start_new_session=True)
        try:
            spawned = process_identity(process.pid)
        except FileNotFoundError:
            spawned = {"pid": process.pid, "start_ticks_unavailable": "process exited before initial proc stat"}
        atomic_json(directory / "spawned_process.json", {"observed_utc": utc(), "process": spawned})
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    receipt = {"schema": SCHEMA + "_actual_process_exit", "actual_process_exit_observed": True,
               "pid": process.pid, "returncode": returncode, "timed_out": timed_out,
               "spawned_process": pointer(root, directory / "spawned_process.json"),
               "finished_utc": utc(), "launch": pointer(root, directory / "launch.json"),
               "wall_seconds": time.monotonic() - started,
               "terminal": pointer(root, directory / "terminal.json") if (directory / "terminal.json").exists() else None,
               "case_complete": False, "automatic_retry": False}
    receipt["administrative_stop_evidence"] = stop_evidence(root, root / RUNTIME / "STOP", administrative_stop_path)
    if receipt["terminal"] is not None:
        terminal = json.loads((directory / "terminal.json").read_text())
        receipt["case_complete"] = bool(returncode == 0 and not timed_out and terminal.get("complete") is True and terminal.get("natural_completion") is True)
        receipt["worker_reported_category"] = terminal.get("category")
        receipt["category"] = (terminal.get("category") if receipt["case_complete"] else
                               "AdministrativeSTOP" if (receipt["administrative_stop_evidence"] or terminal.get("administrative_stop") is True) else
                               "InfrastructureFailure")
    else:
        receipt["category"] = "AdministrativeSTOP" if receipt["administrative_stop_evidence"] else "InfrastructureFailure"
    receipt["administrative_stop"] = receipt["category"] == "AdministrativeSTOP"
    atomic_json(directory / "parent_exit.json", receipt)
    return receipt


def audit_fixed_plan(root, plan_path, *, split):
    """Audit every fixed Native receipt; offline losses are not a release."""
    plan, _ = verify_plan(root, plan_path)
    count = {"train": 8, "validation": 40}[split]
    if plan["split"] != split or plan["expected_cases"] != count:
        raise EvaluationBlocked(f"release requires the complete fixed {split}{count} plan")
    audited = []
    for case in plan["cases"]:
        directory = root / RUNTIME / split / "cases" / case["case_key"]
        exit_path = directory / "parent_exit.json"
        exit_record = json.loads(exit_path.read_text())
        if (exit_record.get("actual_process_exit_observed") is not True or exit_record.get("case_complete") is not True
                or exit_record.get("returncode") != 0 or exit_record.get("timed_out") is not False):
            raise EvaluationBlocked("missing natural, exit-audited case: " + case["case_key"])
        terminal_path = checked(root, exit_record["terminal"])
        terminal = json.loads(terminal_path.read_text())
        validate_native_resource(root, terminal, directory, root / ARTIFACTS / split / "cases" / case["case_key"])
        if exit_record.get("category") != terminal.get("category"):
            raise EvaluationBlocked("actual exit and native terminal classification differ")
        if terminal.get("freeze") != plan["freeze"] or terminal.get("case") != case or terminal.get("plan") != pointer(root, plan_path):
            raise EvaluationBlocked("terminal model/source/plan identity differs")
        if (terminal.get("complete") is not True or terminal.get("natural_completion") is not True
                or terminal.get("administrative_stop") is not False or terminal.get("infrastructure_failed") is not False
                or terminal.get("category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")):
            raise EvaluationBlocked("case is incomplete, administrative or infrastructure")
        audited.append({"case_key": case["case_key"], "task": case["task"], "exit": pointer(root, exit_path),
                        "terminal": exit_record["terminal"], "native_success": terminal["native_success"],
                        "native_terminal_failure": terminal["native_terminal_failure"]})
    name = "Native_TRAIN_release" if split == "train" else "Native_VAL_release"
    release = {"schema": SCHEMA + "_" + name, "complete": True, "split": split, "expected_cases": count,
               "actual_process_exits_audited": count, "freeze": plan["freeze"], "plan": pointer(root, plan_path),
               "cases": audited, "issued_utc": utc(), "old_TEST_read": False,
               "TEST_execution_implemented": False, "offline_validation_is_native_release": False}
    release["per_task"] = task_outcomes(audited)
    release["precision_qualified"] = plan.get("precision_qualified", False)
    release["native_success_is_precision_qualification"] = False
    path = root / ARTIFACTS / split / (name + ".json")
    if path.exists():
        raise EvaluationBlocked("preserve prior release")
    atomic_json(path, release)
    return release


def validate_prior_train_release(root, reference, freeze_reference):
    """Revalidate all eight actual exits before Native VAL admission."""
    release_path = checked(root, reference)
    if not release_path.is_relative_to(root / ARTIFACTS / "train"):
        raise EvaluationBlocked("new evaluation namespace TRAIN release required")
    release = json.loads(release_path.read_text())
    if (release.get("schema") != SCHEMA + "_Native_TRAIN_release" or release.get("complete") is not True
            or release.get("expected_cases") != 8 or release.get("actual_process_exits_audited") != 8
            or release.get("freeze") != freeze_reference or len(release.get("cases", [])) != 8):
        raise EvaluationBlocked("complete same-freeze TRAIN8 exit audit required")
    train_plan_path = checked(root, release["plan"])
    train_plan, _ = verify_plan(root, train_plan_path)
    if train_plan["split"] != "train" or train_plan["freeze"] != freeze_reference:
        raise EvaluationBlocked("TRAIN release does not bind the same fixed model plan")
    if [row["case_key"] for row in release["cases"]] != [row["case_key"] for row in train_plan["cases"]]:
        raise EvaluationBlocked("TRAIN release roster differs")
    for row in release["cases"]:
        exited = json.loads(checked(root, row["exit"]).read_text())
        terminal = json.loads(checked(root, row["terminal"]).read_text())
        key = row["case_key"]
        validate_native_resource(root, terminal, root / RUNTIME / "train/cases" / key,
                                 root / ARTIFACTS / "train/cases" / key)
        if (exited.get("actual_process_exit_observed") is not True or exited.get("returncode") != 0
                or exited.get("case_complete") is not True or exited.get("timed_out") is not False
                or exited.get("terminal") != row["terminal"] or terminal.get("complete") is not True
                or terminal.get("natural_completion") is not True or terminal.get("freeze") != freeze_reference
                or terminal.get("administrative_stop") is not False or terminal.get("infrastructure_failed") is not False
                or terminal.get("category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")
                or exited.get("category") != terminal.get("category")
                or terminal.get("plan") != release["plan"]
                or terminal.get("case") != next(case for case in train_plan["cases"] if case["case_key"] == row["case_key"])):
            raise EvaluationBlocked("TRAIN actual exit/terminal identity changed")
    return release


def audit_training(root, plan_path):
    return audit_fixed_plan(root, plan_path, split="train")


def audit_validation(root, plan_path):
    return audit_fixed_plan(root, plan_path, split="validation")


def task_outcomes(rows):
    """Success rates retain every fixed source, including NativeFalse."""
    result = {}
    for task in TASKS:
        group = [row for row in rows if row["task"] == task]
        result[task] = {"cases": len(group), "native_successes": sum(row["native_success"] is True for row in group),
                        "native_terminal_failures": sum(row["native_terminal_failure"] is True for row in group)}
        result[task]["native_success_rate"] = result[task]["native_successes"] / len(group) if group else None
    return result


def validate_native_resource(root, terminal, runtime, output):
    """Revalidate closed per-case isolation evidence during every release gate."""
    from .isolation import validate_receipt
    receipt = validate_receipt(terminal.get("resources", {}), require_native_interval=True)
    for name, expected in (("case_runtime", runtime), ("case_output", output)):
        actual = Path(receipt.get(name, ""))
        actual = (Path(root) / actual).resolve() if not actual.is_absolute() else actual.resolve()
        if actual != Path(expected).resolve():
            raise EvaluationBlocked("native resource receipt belongs to another case: " + name)
    return receipt
