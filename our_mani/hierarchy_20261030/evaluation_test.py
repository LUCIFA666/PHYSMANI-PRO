"""Post-VAL-only TEST extension for the exact frozen four-task hierarchy.

Every public entry validates the new complete Native VAL77 release *before*
opening any TEST plan, source manifest or recording. No legacy TEST result is
an input. Legacy manifests supply only original official reset mappings.
"""
from __future__ import annotations

from collections import Counter
import importlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

from . import evaluation as base

SCHEMA = "hierarchy_four_task_native_TEST_v1_20261030"
ARTIFACTS = Path("artifacts/framework/multitask_r10/hierarchy_20261030/evaluation_test_v1")
RUNTIME = Path("runtime/framework/hierarchy_20261030/evaluation_test_v1")
EXTRA_SOURCES = (
    "our_mani/hierarchy_20261030/evaluation_test.py", "scripts/evaluate_hierarchy_test_20261030.py",
    "our_mani/hierarchy_20261030/actor.py", "our_mani/hierarchy_20261030/tracking.py",
    "our_mani/hierarchy_20261030/segmentation.py", "our_mani/hierarchy_20261030/candidate_multibody.py",
    "our_mani/official_simulation.py",
)
OFFICIAL_FILES = (
    "low_dim_obs.pkl", "variation_number.pkl", "target_state.pkl", "variation_descriptions.pkl",
    "front_rgb/0.png", "front_depth/0.png", "front_mask/0.png",
    *(f"nerf_data/0/{kind}/{camera}.{extension}" for camera in ("8", "16", "36")
      for kind, extension in (("images", "png"), ("depths", "png"), ("masks", "png"), ("poses", "pkl"))),
)


def validate_val_gate(root, freeze_path, val_release_path):
    """Recheck all 77 actual exits, original roster and the same model freeze."""
    root = Path(root).resolve()
    if freeze_path is None or val_release_path is None:
        raise base.EvaluationBlocked("explicit new freeze and complete Native VAL77 release required before any TEST read")
    freeze_path, val_release_path = Path(freeze_path).resolve(strict=True), Path(val_release_path).resolve(strict=True)
    freeze = base.load_freeze(root, freeze_path)
    if freeze["protocol"].get("inference_device") != "cpu":
        raise base.EvaluationBlocked("this TEST version freezes CPU inference")
    if not val_release_path.is_relative_to(root / base.ARTIFACTS / "validation"):
        raise base.EvaluationBlocked("only the new hierarchy Native VAL release can open TEST")
    freeze_ref, release_ref = base.pointer(root, freeze_path), base.pointer(root, val_release_path)
    release = json.loads(val_release_path.read_text())
    if (release.get("schema") != base.SCHEMA + "_Native_VAL_release"
            or release.get("split") != "validation" or release.get("complete") is not True
            or release.get("expected_cases") != 77 or release.get("actual_process_exits_audited") != 77
            or release.get("freeze") != freeze_ref or len(release.get("cases", [])) != 77
            or release.get("offline_validation_is_native_release") is not False):
        raise base.EvaluationBlocked("same-freeze complete actual Native VAL77 release required before TEST")
    val_plan_path = base.checked(root, release["plan"])
    val_plan, _ = base.verify_plan(root, val_plan_path)
    if val_plan["split"] != "validation" or val_plan["expected_cases"] != 77 or val_plan["freeze"] != freeze_ref:
        raise base.EvaluationBlocked("VAL release model/plan identity differs")
    expected = {case["case_key"]: case for case in val_plan["cases"]}
    if len(expected) != 77 or [row["case_key"] for row in release["cases"]] != list(expected):
        raise base.EvaluationBlocked("complete unfiltered Native VAL roster required")
    for row in release["cases"]:
        exit_record = json.loads(base.checked(root, row["exit"]).read_text())
        terminal = json.loads(base.checked(root, row["terminal"]).read_text())
        if (exit_record.get("actual_process_exit_observed") is not True
                or exit_record.get("returncode") != 0 or exit_record.get("timed_out") is not False
                or exit_record.get("case_complete") is not True or exit_record.get("terminal") != row["terminal"]
                or terminal.get("complete") is not True or terminal.get("natural_completion") is not True
                or terminal.get("administrative_stop") is not False or terminal.get("infrastructure_failed") is not False
                or terminal.get("category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")
                or exit_record.get("category") != terminal.get("category")
                or terminal.get("freeze") != freeze_ref or terminal.get("plan") != release["plan"]
                or terminal.get("case") != expected[row["case_key"]]):
            raise base.EvaluationBlocked("incomplete, administrative, infrastructure or changed Native VAL receipt")
    return {"freeze": freeze, "freeze_ref": freeze_ref, "VAL_release": release_ref,
            "VAL_plan": release["plan"], "actual_VAL_exits_revalidated": 77}


def select_test_rows(original, verified):
    """Pure metadata selection: every original source, no success-dependent filter."""
    rows = [row for row in original.get("episodes", []) if row["task"] in base.TASKS]
    if Counter(row["task"] for row in rows) != Counter({task: 40 for task in base.TASKS}):
        raise base.EvaluationBlocked("all original forty TEST sources per agreed task are required")
    keys = [(row["task"], row["raw_episode_id"]) for row in rows]
    if len(set(keys)) != 160:
        raise base.EvaluationBlocked("duplicate original TEST source identity")
    if verified.get("split") != "test" or verified.get("mapping_status") != "verified":
        raise base.EvaluationBlocked("existing verified official reset mapping required")
    selected_mapping = [row for row in verified.get("episodes", []) if row["task"] in base.TASKS]
    mapping = {(row["task"], row["raw_episode_id"]): row for row in selected_mapping}
    if len(mapping) != len(selected_mapping) or set(mapping) != set(keys):
        raise base.EvaluationBlocked("verified mapping must preserve the exact original four-task TEST160 set")
    result = []
    for row in rows:
        mapped = dict(mapping[(row["task"], row["raw_episode_id"])])
        if mapped.get("mapping_status") != "verified" or set(mapped.get("source_required_files", {})) != set(OFFICIAL_FILES):
            raise base.EvaluationBlocked("each official source needs its verified complete 19-file reset/calibration binding")
        mapped.update(case_key=row["task"] + "__" + row["raw_episode_id"], split="test",
                      original_recording_path=row["path"])
        result.append(mapped)
    return result


def _verified_manifest(root, original, explicit):
    """Called only after the new gate; inspect mapping manifests, never scores."""
    if explicit is not None:
        path = Path(explicit).resolve(strict=True)
        return path, select_test_rows(original, json.loads(path.read_text()))
    directory = root / "artifacts/framework/multitask_r10"
    candidates = sorted(directory.glob("formal_TEST*/**/official_test_manifest16.json"))
    for path in candidates:
        value = json.loads(path.read_text())
        try:
            rows = select_test_rows(original, value)
        except base.EvaluationBlocked:
            continue
        return path.resolve(), rows
    raise base.EvaluationBlocked("provide --verified-manifest with the original full16 verified 19-file official reset mappings; no TEST outcome file is accepted")


def verify_official_case(case, native_repo):
    """Byte verification only; privileged reset pickles are not policy inputs."""
    path = Path(case["path"]).resolve(strict=True)
    original = Path(case["original_recording_path"]).resolve(strict=True)
    if not path.is_dir() or not original.is_dir() or type(case.get("variation")) is not int:
        raise base.EvaluationBlocked("official recording directory and verified variation required")
    index = case.get("official_episode_index")
    if type(index) is not int or index < 0 or case.get("mapping_status") != "verified":
        raise base.EvaluationBlocked("original variation-local official episode index required")
    for name in OFFICIAL_FILES:
        reference = case["source_required_files"][name]
        source = (path / name).resolve(strict=True)
        if (not source.is_relative_to(path) or Path(reference["path"]).resolve(strict=True) != source
                or source.stat().st_size != reference.get("bytes") or base.sha(source) != reference.get("sha256")):
            raise base.EvaluationBlocked("official source/reset/calibration bytes changed: " + name)
    if path != original:
        for name in ("low_dim_obs.pkl", "variation_number.pkl", "target_state.pkl"):
            if base.sha(original / name) != case["source_required_files"][name]["sha256"]:
                raise base.EvaluationBlocked("verified alias is not the original raw source recording")
    if case["task"] not in base.BLOCKS:
        from our_mani.multitask.native_tasks import validate_reset_metadata
        from our_mani.multitask.simulation import verify_official_source_files
        from our_mani.multitask.specs import get_task_spec
        validate_reset_metadata(case["task"], case, archived_variation=case["variation"])
        verify_official_source_files(case, get_task_spec(case["task"]), native_repo)
        alias = Path(case.get("official_alias_path", "")).resolve(strict=True)
        named = Path(case["official_alias_path"])
        if alias != path or named.name != "episode" + str(index) or named.parent.parent.name != "variation" + str(case["variation"]):
            raise base.EvaluationBlocked("core TEST must preserve the original variation-local official alias")


def prepare_test_plan(root, freeze_path, val_release_path, plan_path=None, *,
                      original_manifest_path=None, verified_manifest_path=None):
    root = Path(root).resolve()
    gate = validate_val_gate(root, freeze_path, val_release_path)
    # No TEST-derived path is even resolved until the new gate succeeds.
    original_path = (root / "artifacts/framework/protocol/test_manifest_all16.json"
                     if original_manifest_path is None else Path(original_manifest_path)).resolve(strict=True)
    original = json.loads(original_path.read_text())
    mapping_path, cases = _verified_manifest(root, original, verified_manifest_path)
    repo = Path(gate["freeze"]["protocol"]["baseline_repo"])
    for case in cases:
        verify_official_case(case, repo)
    aliases = {(row["task"], row["variation"], row["official_episode_index"]) for row in cases}
    if len(aliases) != 160:
        raise base.EvaluationBlocked("duplicate variation-local official reset alias")
    source_files = tuple(dict.fromkeys((*base.SOURCE_FILES, *EXTRA_SOURCES)))
    plan = {"schema": SCHEMA, "split": "test", "method": "new_learned_hierarchy_only", "expected_cases": 160,
            "tasks": list(base.TASKS), "cases": cases, "freeze": gate["freeze_ref"],
            "Native_VAL_release": gate["VAL_release"], "Native_VAL_plan": gate["VAL_plan"],
            "complete_Native_VAL_observed_before_TEST_read": True, "actual_VAL_exits_revalidated": 77,
            "original_source_manifest": base.pointer(root, original_path),
            "verified_reset_manifest": base.pointer(root, mapping_path),
            "old_TEST_scores_read": False, "TEST_used_for_fitting_or_selection": False,
            "legacy_manifest_used_only_for_original_reset_mapping": True,
            "environment": base.pointer(root, root / base.ENV_MANIFEST),
            "native_sources": {task: base.native_source_binding(repo, task) for task in base.TASKS},
            "runtime_sources": {name: base.sha(root / name) for name in source_files},
            "automatic_retry": False, "warmup_ticks": 4, "dt": .05, "decision_ticks": 10, "budget_ticks": 200}
    path = root / ARTIFACTS / "plan_v1.json" if plan_path is None else Path(plan_path).resolve()
    if not path.is_relative_to(root / ARTIFACTS):
        raise base.EvaluationBlocked("new TEST extension namespace required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise base.EvaluationBlocked("prior TEST plan is immutable; no source/model replacement")
    else:
        base.atomic_json(path, plan)
    return plan, path


def verify_test_plan(root, plan_path, freeze_path, val_release_path, *, case_key=None):
    root = Path(root).resolve()
    gate = validate_val_gate(root, freeze_path, val_release_path)
    path = Path(plan_path).resolve(strict=True)
    if not path.is_relative_to(root / ARTIFACTS):
        raise base.EvaluationBlocked("new TEST namespace plan required")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != SCHEMA or plan.get("split") != "test" or plan.get("expected_cases") != 160
            or plan.get("freeze") != gate["freeze_ref"] or plan.get("Native_VAL_release") != gate["VAL_release"]
            or plan.get("complete_Native_VAL_observed_before_TEST_read") is not True
            or plan.get("automatic_retry") is not False or plan.get("TEST_used_for_fitting_or_selection") is not False
            or plan.get("dt") != .05 or plan.get("decision_ticks") != 10 or plan.get("budget_ticks") != 200
            or plan.get("warmup_ticks") != 4):
        raise base.EvaluationBlocked("same-freeze post-VAL fixed TEST160 plan contract required")
    original = json.loads(base.checked(root, plan["original_source_manifest"]).read_text())
    verified = json.loads(base.checked(root, plan["verified_reset_manifest"]).read_text())
    if select_test_rows(original, verified) != plan["cases"]:
        raise base.EvaluationBlocked("TEST source identities or original reset mappings changed")
    files = tuple(dict.fromkeys((*base.SOURCE_FILES, *EXTRA_SOURCES)))
    if set(plan.get("runtime_sources", {})) != set(files):
        raise base.EvaluationBlocked("complete new TEST runtime source closure required")
    for name, digest in plan["runtime_sources"].items():
        if base.sha(root / name) != digest:
            raise base.EvaluationBlocked("TEST runtime source changed: " + name)
    repo = Path(gate["freeze"]["protocol"]["baseline_repo"])
    for task in base.TASKS:
        if base.native_source_binding(repo, task) != plan["native_sources"][task]:
            raise base.EvaluationBlocked("native task/scene source changed")
    base.checked(root, plan["environment"])
    selected = plan["cases"] if case_key is None else [case for case in plan["cases"] if case["case_key"] == case_key]
    if case_key is not None and len(selected) != 1:
        raise base.EvaluationBlocked("one fixed TEST case required")
    for case in selected:
        verify_official_case(case, repo)
    return plan, gate["freeze"]


def cpu_preflight_test(root, plan_path, freeze_path, val_release_path):
    root = Path(root).resolve()
    plan, _ = verify_test_plan(root, plan_path, freeze_path, val_release_path)
    report = {"schema": SCHEMA + "_CPU_preflight", "plan": base.pointer(root, plan_path),
              "Native_VAL_release": plan["Native_VAL_release"], "expected_cases": 160,
              "complete_Native_VAL_gate_rechecked": True, "native_started": False, "checked_utc": base.utc()}
    directory = root / RUNTIME
    directory.mkdir(parents=True, exist_ok=True)
    base.atomic_json(directory / ("CPU_preflight_" + str(time.time_ns()) + ".json"), report)
    return report


def request_stop(root, reason):
    """Write only this new TEST control namespace; never touch a legacy STOP."""
    root = Path(root).resolve()
    if not isinstance(reason, str) or not reason:
        raise ValueError("explicit administrative stop reason required")
    path = root / RUNTIME / "STOP"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"path": str(path.relative_to(root)), "already_present": True}
    base.atomic_json(path, {"schema": SCHEMA + "_administrative_STOP", "reason": reason,
                            "requested_utc": base.utc(), "administrative_stop": True})
    return {"path": str(path.relative_to(root)), "already_present": False}


def official_environment(case, protocol):
    if case["task"] in base.BLOCKS:
        from our_mani.sensor_alignment.author_rgbd import AlignedOfficialSimulation
        from our_mani.multitask.physmani_block_receipt_v541 import ReceiptBlockView
        env = AlignedOfficialSimulation(case["task"], case["path"], protocol=protocol,
                                       seed=int(case.get("seed", protocol.get("seed", 0))), servo_lead=1.)
        try:
            env.reset_official(case)
            return ReceiptBlockView(env)
        except BaseException:
            env.close()
            raise
    from our_mani.multitask.simulation import AlignedOfficialSimulation
    from our_mani.multitask.native_outcome_sampling_v741 import AuthorActionOutcomeSamplingMixin
    class OutcomeOfficial(AuthorActionOutcomeSamplingMixin, AlignedOfficialSimulation):
        pass
    env = OutcomeOfficial(case["task"], case["path"], protocol=protocol, seed=int(case.get("seed", 0)), servo_lead=1.)
    try:
        env.reset_official(case)
        return env
    except BaseException:
        env.close()
        raise


def worker_test_case(root, plan_path, case_key, freeze_path, val_release_path, *, device="cpu"):
    root = Path(root).resolve()
    plan, freeze = verify_test_plan(root, plan_path, freeze_path, val_release_path, case_key=case_key)
    if device != "cpu" or not os.environ.get("DISPLAY"):
        raise base.EvaluationBlocked("explicit CPU inference and independent DISPLAY required")
    from scripts.run_frozen_system_environment_v820_r10 import namespace_context
    namespace_context(plan["environment"]["sha256"])
    case = next(row for row in plan["cases"] if row["case_key"] == case_key)
    runtime, output = root / RUNTIME / "cases" / case_key, root / ARTIFACTS / "cases" / case_key
    launch = json.loads((runtime / "launch.json").read_text())
    if launch.get("token") != os.environ.get("HIERARCHY_TEST_PARENT_TOKEN") or base.process_identity(launch["parent"]["pid"]) != launch["parent"]:
        raise base.EvaluationBlocked("live original TEST parent launcher required")
    stop_paths = (root / RUNTIME / "STOP", root / base.RUNTIME / "STOP")
    if output.exists() or (runtime / "terminal.json").exists() or any(path.exists() for path in stop_paths):
        raise base.EvaluationBlocked("prior TEST attempt or controlled STOP exists; no retry")
    record = {"schema": SCHEMA, "split": "test", "case": case, "plan": base.pointer(root, plan_path),
              "freeze": plan["freeze"], "Native_VAL_release": plan["Native_VAL_release"],
              "complete": False, "natural_completion": False, "actual_process_exit_observed": False}
    env = actor = None
    trace = {"steps": [], "holds": []}
    with base.evaluation_locks(root) as resources:
        output.mkdir(parents=True)
        record["resources"] = resources
        try:
            module = importlib.import_module(freeze["actor"]["module"])
            if Path(module.__file__).resolve() != base.checked(root, freeze["actor"]["source"]):
                raise base.EvaluationBlocked("bound new actor source required")
            actor = getattr(module, freeze["actor"]["factory"])(root=root, freeze_manifest=freeze, device=device)
            env = official_environment(case, freeze["protocol"])
            from our_mani.multitask.author_aligned_geometry_executor_v1114 import AuthorAlignedGeometryExecutor
            executor = AuthorAlignedGeometryExecutor(freeze["protocol"], observe_public_frame=lambda *_: None,
                skill_policy=None, admit_pose_action=None, admit_joint_target=None, collision_diagnostic_limit=0)
            record.update(base.run_loop(env, actor, executor, block=case["task"] in base.BLOCKS,
                stop_requested=lambda: any(path.exists() for path in stop_paths), trace=trace))
        except BaseException as error:
            record.update(category="InfrastructureFailure", infrastructure_failed=True, complete=False,
                natural_completion=False, native_success=None, native_terminal_failure=None, administrative_stop=False,
                error_type=type(error).__name__, error=str(error), **trace)
            if env is not None:
                record.update(counted_ticks=int(env.ledger.ticks), clock=env.ledger.receipt())
        finally:
            for value in (actor, env):
                if value is not None and callable(getattr(value, "close", None)):
                    try:
                        value.close()
                    except BaseException as error:
                        record.update(category="InfrastructureFailure", infrastructure_failed=True, complete=False,
                            natural_completion=False, native_success=None, native_terminal_failure=None, cleanup_error=repr(error))
            record.update(finished_utc=base.utc(), process=base.process_identity(), automatic_retry=False)
            base.atomic_json(output / "case.json", record)
            base.atomic_json(runtime / "terminal.json", record)
    return record


def launch_test_case(root, plan_path, case_key, freeze_path, val_release_path, *, device="cpu", timeout=900):
    root = Path(root).resolve()
    plan, _ = verify_test_plan(root, plan_path, freeze_path, val_release_path, case_key=case_key)
    if device != "cpu" or timeout <= 0:
        raise base.EvaluationBlocked("CPU inference and positive timeout required")
    directory = root / RUNTIME / "cases" / case_key
    if directory.exists():
        raise base.EvaluationBlocked("prior TEST launch exists; no silent retry")
    directory.mkdir(parents=True)
    token = secrets.token_hex(24)
    command = [str(root / ".conda-env/bin/python"), "-B", "scripts/run_frozen_system_environment_v820_r10.py",
               "--manifest-sha256", plan["environment"]["sha256"], "--", "bash", "scripts/run_local_sim.sh",
               "scripts/evaluate_hierarchy_test_20261030.py", "--worker-case", case_key, "--plan", str(Path(plan_path).resolve()),
               "--freeze", str(Path(freeze_path).resolve()), "--native-val-release", str(Path(val_release_path).resolve()), "--device", device]
    base.atomic_json(directory / "launch.json", {"schema": SCHEMA + "_launch", "parent": base.process_identity(),
        "token": token, "command": command, "started_utc": base.utc(), "plan": base.pointer(root, plan_path), "automatic_retry": False})
    timed_out = False
    with (directory / "native.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, HIERARCHY_TEST_PARENT_TOKEN=token, PYTHONDONTWRITEBYTECODE="1"), start_new_session=True)
        try:
            spawned = base.process_identity(process.pid)
        except FileNotFoundError:
            spawned = {"pid": process.pid, "start_ticks_unavailable": True}
        base.atomic_json(directory / "spawned_process.json", {"process": spawned, "observed_utc": base.utc()})
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    terminal_path = directory / "terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
    complete = code == 0 and not timed_out and terminal.get("complete") is True and terminal.get("natural_completion") is True
    record = {"schema": SCHEMA + "_actual_process_exit", "actual_process_exit_observed": True,
        "pid": process.pid, "returncode": code, "timed_out": timed_out, "case_complete": complete,
        "category": terminal.get("category") if complete else "AdministrativeSTOP" if terminal.get("administrative_stop") is True and not timed_out else "InfrastructureFailure",
        "worker_reported_category": terminal.get("category"), "finished_utc": base.utc(), "automatic_retry": False,
        "launch": base.pointer(root, directory / "launch.json"), "spawned_process": base.pointer(root, directory / "spawned_process.json"),
        "terminal": base.pointer(root, terminal_path) if terminal_path.exists() else None}
    base.atomic_json(directory / "parent_exit.json", record)
    return record


def audit_test(root, plan_path, freeze_path, val_release_path):
    root = Path(root).resolve()
    plan, _ = verify_test_plan(root, plan_path, freeze_path, val_release_path)
    audited = []
    for case in plan["cases"]:
        directory = root / RUNTIME / "cases" / case["case_key"]
        path = directory / "parent_exit.json"
        exited = json.loads(path.read_text())
        terminal = json.loads(base.checked(root, exited["terminal"]).read_text())
        if (exited.get("actual_process_exit_observed") is not True or exited.get("returncode") != 0
                or exited.get("timed_out") is not False or exited.get("case_complete") is not True
                or terminal.get("complete") is not True or terminal.get("natural_completion") is not True
                or terminal.get("case") != case or terminal.get("freeze") != plan["freeze"]
                or terminal.get("plan") != base.pointer(root, plan_path)
                or terminal.get("Native_VAL_release") != plan["Native_VAL_release"]
                or terminal.get("category") != exited.get("category")
                or terminal.get("category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")):
            raise base.EvaluationBlocked("all original TEST160 cases need complete same-freeze actual exit receipts")
        audited.append({"case_key": case["case_key"], "exit": base.pointer(root, path), "terminal": exited["terminal"],
                        "native_success": terminal["native_success"], "native_terminal_failure": terminal["native_terminal_failure"]})
    result = {"schema": SCHEMA + "_results", "complete": True, "expected_cases": 160,
              "freeze": plan["freeze"], "Native_VAL_release": plan["Native_VAL_release"], "plan": base.pointer(root, plan_path),
              "cases": audited, "native_successes": sum(row["native_success"] for row in audited),
              "TEST_used_for_fitting_or_selection": False, "actual_process_exits_audited": 160, "issued_utc": base.utc()}
    destination = root / ARTIFACTS / "TEST_results.json"
    if destination.exists():
        raise base.EvaluationBlocked("completed TEST results are immutable")
    base.atomic_json(destination, result)
    return result
