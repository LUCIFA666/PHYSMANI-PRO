#!/usr/bin/env python3
"""Read only the exact saved TEST outputs after the observed host reboot.

This audits recoverability, not full-suite release.  It neither invokes a
simulation nor recreates missing case/receipt/terminal bytes.  The only write
is a new, exclusively created v1054 observation report.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("runtime/framework/formal_TEST_composite_resume_v1041")
REPORT = Path("runtime/framework/reboot_TEST_evidence_v1054.json")
INPUTS = {
    RUNTIME / "launch.json": "b16de763e777cd1eea98fc8bbc28210e594f6a1c0e4df4ce2b29622792c76869",
    RUNTIME / "status.json": "71c7235ea0d5b9237d168ea34a741c225df1266816317d797e57532b0edf8f38",
    RUNTIME / "plan.json": "87e205b8c55fc6c247767fbea6048089973f2255e63e6fa8d6920e4c6f26a158",
}
KNOWN_RUNTIMES = (
    "runtime/framework/formal_TEST_after_environment_resume_v753",
    "runtime/framework/formal_TEST_concurrent_resume_v902",
    "runtime/framework/formal_TEST_composite_resume_v956",
    "runtime/framework/formal_TEST_composite_resume_v1011",
    str(RUNTIME),
)
METHODS = ("cv_geometry", "context_geometry", "physmani")
POLICY = "eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef"
DRIVER = {
    "path": "scripts/run_formal_case_author_outcome_v751_r10.py",
    "sha256": "06053736b8cb868c6783a4bb8a81c3f36698ef8137bf62f1e5f53e007ebe9d36",
}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def job_set_digest(ids):
    return digest(json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def exact_file(relative):
    """Reject traversal; all subsequent reads stay inside this project."""
    p = Path(relative)
    demand(not p.is_absolute() and ".." not in p.parts, "non-project path")
    q = ROOT / p
    demand(q.resolve().is_relative_to(ROOT), "path escapes project")
    return q


def read_json_observation(relative, expected_sha256=None):
    q = exact_file(relative)
    record = {"path": str(relative), "expected_sha256": expected_sha256}
    if not q.is_file():
        record.update(exists=False, valid_JSON=False, error="missing_file")
        return record, None
    raw = q.read_bytes()
    record.update(exists=True, bytes=len(raw), sha256=digest(raw))
    if expected_sha256 is not None:
        record["SHA_matches"] = record["sha256"] == expected_sha256
    try:
        data = json.loads(raw)
        demand(isinstance(data, dict), "JSON root is not an object")
    except (ValueError, UnicodeDecodeError) as e:
        record.update(valid_JSON=False, error=str(e))
        return record, None
    record["valid_JSON"] = True
    return record, data


def clean_terminal(data):
    return (
        isinstance(data, dict)
        and data.get("complete") is True
        and type(data.get("returncode")) is int
        and data["returncode"] == 0
        and data.get("timed_out") is False
        and data.get("interrupted") is False
        and data.get("automatic_retry") is False
    )


def pid_observation(pid, expected_start=None):
    p = Path("/proc") / str(pid) / "stat"
    try:
        raw = p.read_text()
    except FileNotFoundError:
        return {"pid": pid, "expected_start_ticks": expected_start,
                "PID_exists": False, "same_process_alive": False}
    except OSError as e:
        return {"pid": pid, "expected_start_ticks": expected_start,
                "PID_exists": None, "same_process_alive": None, "error": str(e)}
    tail = raw[raw.rfind(")") + 2:].split()
    actual = int(tail[19])
    return {"pid": pid, "expected_start_ticks": expected_start,
            "PID_exists": True, "observed_start_ticks": actual,
            "same_process_alive": actual == expected_start}


def matches(actual, expected):
    if isinstance(expected, bool):
        return type(actual) is bool and actual is expected
    return actual == expected


def validate_completed(job, row, policy):
    """Never infer a Native outcome when the saved evidence is unavailable."""
    result = {"job_id": job["job_id"], "evidence_runtime": job["evidence_runtime"],
              "original_status_native_success": row.get("native_success"),
              "original_status_motion_pass": row.get("motion_pass"),
              "recoverable_complete": False, "native_success": None,
              "files": {}, "errors": []}
    errors = result["errors"]
    if not clean_terminal(row) or type(row.get("native_success")) is not bool:
        errors.append("saved_completed_status_not_clean")
    output = Path(job["argv"][job["argv"].index("--out") + 1])
    demand(str(output).startswith("artifacts/framework/multitask_r10/"), "case output scope")
    demand(job["evidence_runtime"] in KNOWN_RUNTIMES, "unknown evidence owner")
    loaded = {}
    for label, filename in (("case", "case.json"),
                            ("admission_driver_receipt", "admission_driver_receipt.json"),
                            ("batch_driver_receipt", "batch_driver_receipt.json")):
        ptr = row.get(label)
        required = label != "batch_driver_receipt" or job["execution_entry"]["kind"] != "original751"
        if ptr is None:
            if required:
                errors.append(label + ":missing_pointer")
            continue
        demand(isinstance(ptr, dict) and ptr.get("path") == str(output / filename),
               "saved output pointer does not match exact plan output")
        observation, data = read_json_observation(ptr["path"], ptr.get("sha256"))
        result["files"][label] = observation
        loaded[label] = data
        if observation.get("SHA_matches") is not True or data is None:
            errors.append(label + ":missing_invalid_or_SHA_mismatch")
    terminal_path = Path(job["evidence_runtime"]) / "native" / job["job_id"] / "terminal.json"
    observation, terminal = read_json_observation(terminal_path)
    result["files"]["native_terminal"] = observation
    if not clean_terminal(terminal):
        errors.append("native_terminal:missing_invalid_or_not_clean")
    else:
        for label in ("case", "admission_driver_receipt", "batch_driver_receipt"):
            if label in row and terminal.get(label) != row[label]:
                errors.append("native_terminal:" + label + "_pointer_mismatch")
    case = loaded.get("case")
    if case is not None:
        expected = {
            "task": job["task"], "raw_episode_id": job["raw"],
            "method": job["method"], "split": "test", "complete": True,
            "infrastructure_error": None, "formal_TEST_metric": True,
            "formal_test_used": True, "demo_only": False, "TEST_used_for_fitting": False,
            "automatic_retry": False, "Value_NGFF_FASTER_used": False,
            "cleanup_completed": True, "official_task_success": row.get("native_success"),
        }
        for k, value in expected.items():
            if k not in case or not matches(case[k], value):
                errors.append("case:" + k)
        frozen = case.get("full16_frozen_execution", {})
        for k, value in {"policy_sha256": policy, "actual_split": "test",
                         "actual_component_purpose": "evaluation",
                         "VAL_TEST_relabelled_as_TRAIN": False,
                         "original_TEST_reset_unmodified": True}.items():
            if k not in frozen or not matches(frozen[k], value):
                errors.append("case:full16_frozen_execution." + k)
    receipt = loaded.get("admission_driver_receipt")
    if receipt is not None and not (
        receipt.get("schema") == "r10_explicit_author_outcome_formal_driver_v751"
        and receipt.get("complete") is True and receipt.get("actual_split") == "test"
        and receipt.get("case") == row.get("case")
        and receipt.get("admission_driver") == DRIVER
    ):
        errors.append("751_receipt:identity_or_case_link")
    batch = loaded.get("batch_driver_receipt")
    if batch is not None and not (
        batch.get("schema") == "r10_scoped_frozen_lookup_bootstrap_receipt_v901"
        and batch.get("complete") is True and batch.get("check_only") is False
        and batch.get("case") == row.get("case")
        and batch.get("scientific_policy_changed") is False
        and batch.get("original_admission_driver_receipt") == row.get("admission_driver_receipt")
    ):
        errors.append("901_receipt:identity_or_case_link")
    if not errors:
        result["recoverable_complete"] = True
        result["native_success"] = row["native_success"]
    return result


def metadata(relative):
    q = exact_file(relative)
    if not q.is_dir():
        return {"path": str(relative), "directory_exists": False, "files": {}}
    # One known job directory only: do not descend into any science input tree.
    files = {p.name: {"bytes": p.stat().st_size} for p in sorted(q.iterdir()) if p.is_file()}
    return {"path": str(relative), "directory_exists": True, "files": files}


def existence_observation(relative):
    q = exact_file(relative)
    observation = {"path": str(relative), "exists": q.exists(), "is_file": q.is_file()}
    if q.is_file():
        observation["bytes"] = q.stat().st_size
    return observation


def inspect_stage(job_id, kind):
    base = RUNTIME / kind / job_id
    observed = {"job_id": job_id, "kind": kind, "metadata": metadata(base)}
    observed["launch"], launch = read_json_observation(base / "launch.json")
    observed["terminal"], terminal = read_json_observation(base / "terminal.json")
    observed["saved_terminal_clean"] = clean_terminal(terminal)
    if launch is not None:
        observed["launcher_process"] = pid_observation(
            launch.get("launcher_pid"), launch.get("launcher_start_ticks"))
    else:
        observed["launcher_process"] = None
    if kind == "CPU":
        observed["CPU_preflight"], preflight = read_json_observation(base / "CPU_preflight.json")
        observed["saved_CPU_preflight_complete"] = bool(
            clean_terminal(terminal) and preflight is not None
            and preflight.get("complete") is True and preflight.get("native_started") is False
            and preflight.get("GPU_started") is False and preflight.get("actual_split") == "test")
    return observed


def audit():
    started = time.monotonic()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    demand(not exact_file(REPORT).exists(), "report namespace already exists; refusing overwrite")
    input_observations = {}
    data = {}
    for path, sha in INPUTS.items():
        observation, value = read_json_observation(path, sha)
        demand(observation.get("SHA_matches") is True and value is not None, "frozen audit input changed")
        input_observations[path.name] = observation
        data[path.name] = value
    status, plan, launch = data["status.json"], data["plan.json"], data["launch.json"]
    demand(status.get("policy_sha256") == plan.get("policy_sha256") == POLICY, "policy mismatch")
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    demand(len(jobs) == len(plan["jobs"]) == 1920, "plan is not exactly 1920 unique jobs")
    demand(status.get("expected") == 1920 and len(status["completed"]) == 1413, "saved count changed")
    demand(status.get("failed") == {} and status.get("owner_error") is None, "unexpected saved failure")
    state_lists = {k: list(status[k]) for k in ("active", "checking", "ready_jobs")}
    all_listed = list(status["completed"]) + sum(state_lists.values(), [])
    demand(len(all_listed) == len(set(all_listed)) and set(all_listed) <= set(jobs), "state partition mismatch")
    owner = pid_observation(launch["pid"], launch["start_ticks"])
    demand(owner["same_process_alive"] is False, "old owner liveness not disproven")
    completed = [validate_completed(jobs[k], row, POLICY) for k, row in status["completed"].items()]
    valid = [r for r in completed if r["recoverable_complete"]]
    damaged = [r for r in completed if not r["recoverable_complete"]]
    # Inspect exactly the current runtime's known job stage directories, including
    # any launch not yet registered in the last status.  No output is selected by success.
    stages = {}
    for kind in ("CPU", "native"):
        directory = exact_file(RUNTIME / kind)
        ids = sorted(x.name for x in directory.iterdir() if x.is_dir())
        demand(set(ids) <= set(jobs), "stage directory not declared by exact plan")
        stages[kind] = [inspect_stage(k, kind) for k in ids]
    unregistered_clean = [r for r in stages["native"]
                          if r["job_id"] not in status["completed"] and r["saved_terminal_clean"]]
    for kind in stages:
        for r in stages[kind]:
            process = r["launcher_process"]
            demand(process is None or process["same_process_alive"] is False,
                   "old launcher liveness not disproven")
    unknown = []
    for state, ids in state_lists.items():
        for k in ids:
            job = jobs[k]
            output = Path(job["argv"][job["argv"].index("--out") + 1])
            unknown.append({"job_id": k, "last_recorded_state": state,
                            "native_success": None, "outcome": "unknown_after_host_reboot",
                            "output_metadata": metadata(output)})
    task_counts = collections.defaultdict(lambda: collections.defaultdict(lambda: {"success": 0, "completed": 0}))
    tier_counts = collections.Counter()
    for row in valid:
        j = jobs[row["job_id"]]
        task_counts[j["task"]][j["method"]]["completed"] += 1
        task_counts[j["task"]][j["method"]]["success"] += int(row["native_success"])
        tier_counts[j["evidence_runtime"]] += 1
    for path, sha in INPUTS.items():
        demand(digest(exact_file(path).read_bytes()) == sha, "audit input changed during observation")
    sets = {
        "accepted_completed_job_ids": sorted(r["job_id"] for r in valid),
        "rejected_saved_completed_job_ids": sorted(r["job_id"] for r in damaged),
        "remaining_job_ids": sorted(set(jobs) - {r["job_id"] for r in valid}),
        "saved_completed_job_ids": sorted(status["completed"]),
    }
    absent = {
        str(base / name): existence_observation(base / name)
        for base in (RUNTIME, Path("runtime/framework/ordered_composite_TEST_v1043"))
        for name in ("terminal.json", "failure.json", "DRAIN", "STOP")
    }
    demand(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id,
           "boot identity changed during audit")
    report = {
        "schema": "r10_readonly_host_reboot_TEST_evidence_observation_v1054",
        "audit_complete": True, "formal_full_TEST_complete": False,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                   "sha256": digest(Path(__file__).read_bytes())},
        "inputs": input_observations, "inputs_unchanged_through_audit": True,
        "boot_id": boot_id, "current_boot_id": boot_id,
        "boot_id_unchanged_through_audit": True,
        "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
        "old_owner_process": owner,
        "old_owner_terminal": metadata(RUNTIME)["files"].get("terminal.json"),
        "absent_evidence": absent,
        "saved_status_updated_utc": status["updated_utc"],
        "saved_status_completed_count": len(completed),
        "recoverable_complete_count": len(valid), "damaged_registered_count": len(damaged),
        "recoverable_counts_by_runtime": dict(tier_counts),
        "recoverable_native_false_count": sum(r["native_success"] is False for r in valid),
        "recoverable_task_method_counts": task_counts,
        "completed_evidence": completed,
        "saved_current_runtime_CPU_stages": stages["CPU"],
        "saved_current_runtime_native_stages": stages["native"],
        "unregistered_clean_native_terminals": unregistered_clean,
        "unknown_interrupted_cases": unknown,
        "not_yet_launched_count": len(jobs) - len(all_listed),
        "still_unclosed_count": len(jobs) - len(valid),
        "scientific_inputs_read": False, "raw_data_read": False, "model_read": False,
        "full_source_graph_read": False, "Native_started": False,
        "old_outputs_modified": False, "missing_evidence_reconstructed": False,
        "unknown_counted_as_NativeFalse": False, "success_based_case_selection": False,
        "new_terminal_claimed_for_interrupted_owner": False,
        "job_id_set_digest_encoding": "sha256(UTF-8 json.dumps(sorted(ids), ensure_ascii=False, separators=(',',':')))",
        "recovery_constraints": [
            "Preserve all intact completed records literally, including NativeFalse.",
            "Preserve damaged outputs and unknown attempts; never reconstruct case or terminal files.",
            "A new explicit recovery branch must distinguish reboot interruption from Native outcome.",
            "Reuse the fixed task, raw source, method, policy, physical parameters and counted budget.",
            "Require current environment and process admission before any new simulation.",
        ],
        "wall_seconds": time.monotonic() - started,
    }
    report.update(sets)
    report.update({k + "_sha256": job_set_digest(ids) for k, ids in sets.items()})
    output = exact_file(REPORT)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"report": str(REPORT), "report_sha256": digest(output.read_bytes()),
                      "recoverable_complete_count": len(valid), "damaged_registered_count": len(damaged),
                      "unregistered_clean_native_count": len(unregistered_clean),
                      "unknown_interrupted_count": len(unknown),
                      "still_unclosed_count": len(jobs) - len(valid)}, ensure_ascii=False))


if __name__ == "__main__":
    audit()
