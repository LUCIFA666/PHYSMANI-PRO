#!/usr/bin/env python3
"""Observe the second abrupt reboot without recreating any TEST evidence.

The first 1411 results inherit the pinned v1054 audit and v1055 freeze.  Only
the eight new v1056 results have their saved case/receipt bytes reread here.
This program starts no simulator and exclusively creates one new report.
"""
from __future__ import annotations

import collections
from copy import deepcopy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OWNER = Path("runtime/framework/formal_TEST_reboot_resume_v1056")
CONTROLLER = Path("runtime/framework/ordered_reboot_TEST_v1058")
REPORT = Path("runtime/framework/second_reboot_TEST_evidence_v1067.json")
SCHEMA = "r10_readonly_host_second_reboot_TEST_evidence_observation_v1067"
POLICY = "eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef"
OLD_BOOT = "2be1f393-5cdf-4015-bb79-c43c5475ffc7"
OBSERVED_BOOT = "726b6813-97a8-4ccc-aa01-089a65334e5a"
RML = "d9a36e5ba8a73123c5deb65229d7fa151089145dca4b55c68c1e9ad1dec4e562"
METHODS = ("cv_geometry", "context_geometry", "physmani")
INPUTS = {
    "previous_owner_launch": (str(OWNER / "launch.json"), "1e549d2a4247a1a763ced9c7b483cbdf22023ee19d4621a898cad03d57317471"),
    "previous_owner_plan": (str(OWNER / "plan.json"), "15926bd7e06de021c83e9daeb7c3105642c23b1a3d79fe136df8f0457335c17b"),
    "previous_owner_status": (str(OWNER / "status.json"), "f4ab2931e662572ed53cb07592d47c2f9c9f303daae5a4095a1fa2c6418593d6"),
    "previous_controller_launch": (str(CONTROLLER / "launch.json"), "8b114b4fac5d0ae9a46f453c46ae64853e8b01861734416300ff36a7083ba5a0"),
    "previous_controller_status": (str(CONTROLLER / "status.json"), "cdd63d41add780041bc3c58ec8f9680ed4232b0ad48de57b686bd57c9b756712"),
    "old1054_audit": ("runtime/framework/reboot_TEST_evidence_v1054.json", "0bebb704674767eb2795b00e60dc3889956c6ccce3e1d5ed76399e3d717beecf"),
    "parent_status": ("runtime/framework/formal_TEST_composite_resume_v1041/status.json", "71c7235ea0d5b9237d168ea34a741c225df1266816317d797e57532b0edf8f38"),
    "previous_resume_control": ("artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056_setup/resume_control.json", "360aab569b33523968a170f892fd2ef7d78de4709cc777483fe4e363db81ed63"),
    "previous_resume_control_lock": ("artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056_setup/resume_inputs.json", "a7f5858885b1d8f08bee344eff99bdde2cded215ff23cc8487dba3972a83a485"),
}
SOURCES = {
    "scripts/audit_reboot_TEST_evidence_v1054_r10.py": "44b1fe8e5a68b814718772484ea7663f967d2ca74d5f7cf07fb712fe710c7096",
    "scripts/run_reboot_TEST_owner_v1056_r10.py": "37bca4c18598c10d473ec4c8acccfdce7eb93b2a0dbdbe8690aeaf12b73954e6",
    "our_mani/multitask/composite_TEST_reboot_resume_v1055.py": "72b335e672b1fcdfa486cd653970f47e469fec742303ecbe1737b13a80b2fddf",
    "scripts/prepare_composite_TEST_reboot_resume_v1055_r10.py": "f1b39310db0a959eec56c8b3d4e0d0789d9222f849bbc262b67119661ffd7682",
    "our_mani/multitask/evaluation.py": "f36811de5428b880ec4af474177c151f06599f5cc106594adb8a36522f2a7d1b",
    "our_mani/multitask/common_rml_protocol_v530.py": "65eed7c630a587ea019a3946db7ed524d197e4246568aa1932262efca124173c",
    "our_mani/multitask/three_method_release16.py": "faf77babdd431be41c5765a34dfb1b08a18e7bf8ec1a70b6f26b78f7210477f3",
    "scripts/run_formal_suite16_owner_v753_r10.py": "ece7f0d57f4e533b22a62d35a4b5220deedb88dbfc44767b060fc889b30cb183",
}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def job_set_digest(ids):
    return digest(json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode())


def project_file(relative):
    p = Path(relative)
    demand(not p.is_absolute() and ".." not in p.parts, "project-relative path required")
    target = ROOT / p
    demand(target.resolve().is_relative_to(ROOT), "project path escaped")
    return target


def read_input(relative, sha):
    target = project_file(relative)
    raw = target.read_bytes()
    demand(digest(raw) == sha, "frozen audit input changed: " + relative)
    obj = json.loads(raw)
    demand(isinstance(obj, dict), "object metadata required")
    return {"path": relative, "sha256": sha, "expected_sha256": sha,
            "SHA_matches": True, "exists": True, "valid_JSON": True,
            "bytes": len(raw)}, obj


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inherited_evidence(old_audit, parent_status, status, jobs, control, lock):
    """Compare literal frozen records; do not reread old physical case files."""
    accepted = set(old_audit["accepted_completed_job_ids"])
    imported = {k for k in status["completed"] if jobs[k]["imported_completed"] is True}
    demand(old_audit.get("audit_complete") is True
           and old_audit.get("schema") == "r10_readonly_host_reboot_TEST_evidence_observation_v1054"
           and old_audit.get("recoverable_complete_count") == len(accepted) == 1411,
           "first reboot audit is not the frozen 1411 result set")
    demand(imported == accepted == set(control["completed_job_ids"]), "inherited result keyset mismatch")
    demand(control.get("completed_key_sha256") == job_set_digest(accepted), "first recovery key digest mismatch")
    demand(lock.get("control") == status.get("resume_control"), "v1055 lock/control binding mismatch")
    old = {r["job_id"]: r for r in old_audit["completed_evidence"]}
    result = []
    for key in sorted(imported):
        row = status["completed"][key]
        observed = old[key]
        demand(row == parent_status["completed"][key], "inherited result was rewritten: " + key)
        demand(observed.get("recoverable_complete") is True
               and type(observed.get("native_success")) is bool
               and observed["native_success"] is row.get("native_success")
               and observed.get("evidence_runtime") == jobs[key]["evidence_runtime"],
               "inherited Native evidence mismatch: " + key)
        copy = deepcopy(observed)
        copy.update(evidence_inherited=True, physical_output_rechecked_this_audit=False,
                    files_observations_are_from_pinned_v1054=True,
                    literal_parent_row_matches=True, literal_parent_status_record_matches=True,
                    parent_status_record_sha256=digest(json.dumps(row, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()),
                    inherited_audit={"path": INPUTS["old1054_audit"][0], "sha256": INPUTS["old1054_audit"][1]},
                    inherited_resume_control={"path": INPUTS["previous_resume_control"][0], "sha256": INPUTS["previous_resume_control"][1]},
                    inherited_resume_control_lock={"path": INPUTS["previous_resume_control_lock"][0], "sha256": INPUTS["previous_resume_control_lock"][1]})
        result.append(copy)
    return result


def state_partition(status, jobs):
    states = {k: list(status[k]) for k in ("active", "checking", "ready_jobs")}
    listed = list(status["completed"]) + sum(states.values(), [])
    demand(len(listed) == len(set(listed)) and set(listed) <= set(jobs), "state partition mismatch")
    demand(status.get("expected") == 1920 and len(jobs) == 1920
           and status.get("failed") == {} and status.get("owner_error") is None,
           "unexpected previous owner state")
    return states, listed


def validate_new(helper, owner_helper, job, row, metadata):
    """Use the frozen admission acceptors on exact saved files, without main()."""
    result = helper.validate_completed(job, row, POLICY)
    result.update(evidence_inherited=False, physical_output_rechecked_this_audit=True)
    cpu_base = OWNER / "CPU" / job["job_id"]
    native_base = OWNER / "native" / job["job_id"]
    observations = {}
    loaded = {}
    for name, relative in (("CPU_preflight", cpu_base / "CPU_preflight.json"),
                           ("CPU_terminal", cpu_base / "terminal.json"),
                           ("CPU_launch", cpu_base / "launch.json"),
                           ("native_launch", native_base / "launch.json")):
        observations[name], loaded[name] = helper.read_json_observation(relative)
    result["files"].update(observations)
    try:
        demand(helper.clean_terminal(loaded["CPU_terminal"]), "CPU terminal not clean")
        identity = owner_helper.accept_CPU(loaded["CPU_preflight"], job,
                                           metadata["source_sha256"], metadata["batch_sources"])
        demand(loaded["native_launch"].get("expected_identity_sha256") == identity.get("sha256"),
               "native launcher not bound to CPU identity")
        demand(loaded["CPU_launch"].get("physical_argv") == job["physical_argv"] + ["--check"],
               "CPU launcher physical argv changed")
        demand(owner_helper.accept_native(job, row, identity, metadata["author_outcome_protocol"],
                                          POLICY, RML, metadata["batch_sources"]) == row,
               "Native acceptor would rewrite result")
        case = json.loads(project_file(row["case"]["path"]).read_bytes())
        result["shared_context_checkpoint_observed"] = case.get("shared_context_checkpoint")
        result["common_author_RML_parameters_observed"] = case.get("common_author_RML_parameters")
        result["CPU_identity_sha256"] = identity["sha256"]
        result["frozen_CPU_and_Native_acceptors_passed"] = True
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        result["errors"].append("frozen_acceptor:" + repr(error))
        result["frozen_CPU_and_Native_acceptors_passed"] = False
    if result["errors"]:
        result.update(recoverable_complete=False, native_success=None)
    return result


def canonical_sets(jobs, completed):
    accepted = {r["job_id"] for r in completed if r["recoverable_complete"] is True}
    rejected = {r["job_id"] for r in completed if r["recoverable_complete"] is not True}
    sets = {"accepted_completed_job_ids": sorted(accepted),
            "rejected_saved_completed_job_ids": sorted(rejected),
            "saved_completed_job_ids": sorted(r["job_id"] for r in completed),
            "remaining_job_ids": sorted(set(jobs) - accepted)}
    return sets | {k + "_sha256": job_set_digest(value) for k, value in sets.items()}


def write_exclusive_report(path, report):
    """Only the newly named report is writable; durable and never overwritten."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def audit():
    started = time.monotonic()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    demand(boot_id == OBSERVED_BOOT and boot_id != OLD_BOOT, "unexpected current reboot identity")
    output = project_file(REPORT)
    demand(not output.exists(), "report namespace already exists; refusing overwrite")
    inputs, data = {}, {}
    for name, (relative, sha) in INPUTS.items():
        inputs[name], data[name] = read_input(relative, sha)
    sources = {}
    frozen_files = data["previous_resume_control_lock"]["files"]
    for relative, sha in SOURCES.items():
        demand(digest(project_file(relative).read_bytes()) == sha, "audit dependency source changed")
        demand(frozen_files.get(str(ROOT / relative)) == sha, "source absent from v1055 frozen map")
        sources[relative] = {"path": relative, "sha256": sha}
    sys.path.insert(0, str(ROOT))
    helper = load_module("reboot_audit_v1054_for_v1067", "scripts/audit_reboot_TEST_evidence_v1054_r10.py")
    helper.ROOT, helper.RUNTIME = ROOT, OWNER
    helper.KNOWN_RUNTIMES += (str(OWNER),)
    owner_helper = load_module("reboot_owner_v1056_for_v1067", "scripts/run_reboot_TEST_owner_v1056_r10.py")
    status, plan, launch = (data["previous_owner_" + n] for n in ("status", "plan", "launch"))
    control, lock = data["previous_resume_control"], data["previous_resume_control_lock"]
    demand(status["policy_sha256"] == plan["policy_sha256"] == control["policy_sha256"] == POLICY,
           "frozen TEST policy changed")
    demand(status["boot_id"] == plan["boot_id"] == launch["boot_id"] == OLD_BOOT, "old boot binding changed")
    demand(status.get("resume_control") == launch.get("resume_control") == lock.get("control"),
           "old owner control binding changed")
    demand(status.get("resume_control_lock") == launch.get("resume_control_lock")
           == {"path": INPUTS["previous_resume_control_lock"][0], "sha256": INPUTS["previous_resume_control_lock"][1]},
           "old owner lock binding changed")
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    demand(len(jobs) == len(plan["jobs"]) == 1920 and {j["method"] for j in jobs.values()} == set(METHODS),
           "not the exact three-method 1920 plan")
    states, listed = state_partition(status, jobs)
    demand(len(status["completed"]) == 1419 and status["imported_completed"] == 1411, "saved count changed")
    owner = helper.pid_observation(launch["pid"], launch["start_ticks"])
    controller_launch = data["previous_controller_launch"]
    controller = helper.pid_observation(controller_launch["pid"], controller_launch["start_ticks"])
    demand(owner["same_process_alive"] is False and controller["same_process_alive"] is False,
           "old owner/controller liveness not disproven")
    inherited = inherited_evidence(data["old1054_audit"], data["parent_status"], status, jobs, control, lock)
    new_keys = sorted(set(status["completed"]) - {r["job_id"] for r in inherited})
    demand(len(new_keys) == 8 and all(jobs[k]["evidence_runtime"] == str(OWNER)
                                    and jobs[k]["imported_completed"] is False for k in new_keys),
           "new saved completion ownership changed")
    metadata = {"source_sha256": launch["split_source_lock_sha256"],
                "batch_sources": launch["batch_sources"], "author_outcome_protocol": plan["author_outcome_protocol"]}
    fresh = [validate_new(helper, owner_helper, jobs[k], status["completed"][k], metadata) for k in new_keys]
    completed = inherited + fresh
    sets = canonical_sets(jobs, completed)
    demand(len(sets["accepted_completed_job_ids"]) == 1419
           and not sets["rejected_saved_completed_job_ids"] and len(sets["remaining_job_ids"]) == 501,
           "new saved evidence is damaged; no report asserting 1419 may be written")
    stages = {}
    for kind in ("CPU", "native"):
        directory = project_file(OWNER / kind)
        ids = sorted(p.name for p in directory.iterdir() if p.is_dir())
        demand(set(ids) <= set(jobs), "stage not declared by pinned plan")
        stages[kind] = [helper.inspect_stage(k, kind) for k in ids]
        for stage in stages[kind]:
            process = stage["launcher_process"]
            demand(process is None or process["same_process_alive"] is False, "old launcher still alive")
    unregistered = [r for r in stages["native"] if r["job_id"] not in status["completed"]
                    and r["saved_terminal_clean"]]
    demand(not unregistered, "new clean terminal must be audited, never silently omitted")
    unknown = []
    for state, ids in states.items():
        for key in ids:
            job = jobs[key]
            out = Path(job["argv"][job["argv"].index("--out") + 1])
            saved = helper.metadata(out)
            demand(not any(n in saved["files"] for n in ("case.json", "admission_driver_receipt.json", "batch_driver_receipt.json")),
                   "unadopted result bytes need explicit audit")
            unknown.append({"job_id": key, "last_recorded_state": state, "native_success": None,
                            "outcome": "unknown_after_second_host_reboot", "output_metadata": saved})
    missing_native = [r["job_id"] for r in stages["native"] if not r["saved_terminal_clean"]]
    demand(len(unknown) == 29 and len(missing_native) == 9 and set(missing_native) == set(states["active"]),
           "interrupted stage counts changed")
    counts = collections.defaultdict(lambda: collections.defaultdict(lambda: {"success": 0, "completed": 0}))
    tiers = collections.Counter()
    for record in completed:
        j = jobs[record["job_id"]]
        counts[j["task"]][j["method"]]["completed"] += 1
        counts[j["task"]][j["method"]]["success"] += int(record["native_success"])
        tiers[j["evidence_runtime"]] += 1
    absent = {str(base / name): helper.existence_observation(base / name)
              for base in (OWNER, CONTROLLER) for name in ("terminal.json", "failure.json", "DRAIN", "STOP")}
    demand(all(not v["exists"] for v in absent.values()), "unexpected terminal/drain/stop evidence")
    for relative, sha in INPUTS.values():
        demand(digest(project_file(relative).read_bytes()) == sha, "input changed during audit")
    for relative, sha in SOURCES.items():
        demand(digest(project_file(relative).read_bytes()) == sha, "source changed during audit")
    demand(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id, "boot changed during audit")
    btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
    report = dict(schema=SCHEMA, audit_complete=True, formal_full_TEST_complete=False,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source={"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__).read_bytes())},
                  inputs=inputs, semantic_sources=sources, inputs_unchanged_through_audit=True,
                  semantic_sources_unchanged_through_audit=True, current_boot_id=boot_id, boot_id=boot_id,
                  previous_boot_id=OLD_BOOT, boot_id_unchanged_through_audit=True,
                  boot_time_epoch_seconds=btime, boot_time_utc=dt.datetime.fromtimestamp(btime, dt.timezone.utc).isoformat(),
                  uptime_seconds=float(Path("/proc/uptime").read_text().split()[0]),
                  old_owner_process=owner, old_controller_process=controller, old_owner_terminal=None,
                  absent_evidence=absent, saved_status_updated_utc=status["updated_utc"],
                  saved_status_completed_count=1419, recoverable_complete_count=1419,
                  inherited_completed_count=1411, new_completed_physically_rechecked_count=8,
                  inherited_all_literal_parent_rows_match=True, inherited_keyset_matches_pinned_first_reboot_audit=True,
                  old1411_physical_case_files_reread=False, damaged_registered_count=0,
                  recoverable_counts_by_runtime=dict(tiers), recoverable_task_method_counts=counts,
                  recoverable_native_false_count=sum(r["native_success"] is False for r in completed),
                  new_native_false_count=sum(r["native_success"] is False for r in fresh),
                  completed_evidence=completed, saved_current_runtime_CPU_stages=stages["CPU"],
                  saved_current_runtime_native_stages=stages["native"],
                  missing_native_terminal_job_ids=missing_native, unregistered_clean_native_terminals=unregistered,
                  unknown_interrupted_cases=unknown, not_yet_launched_count=len(jobs) - len(listed), still_unclosed_count=501,
                  scientific_inputs_read=False, raw_data_read=False, model_read=False, full_source_graph_read=False,
                  Native_started=False, GPU_started=False, old_outputs_modified=False, missing_evidence_reconstructed=False,
                  unknown_counted_as_NativeFalse=False, success_based_case_selection=False,
                  new_terminal_claimed_for_interrupted_owner=False,
                  job_id_set_digest_encoding="sha256(UTF-8 json.dumps(sorted(ids), ensure_ascii=False, separators=(',',':')))",
                  recovery_constraints=["Preserve all 1419 complete records literally, including NativeFalse.",
                                        "Retain both reboot attempt histories and all absent-terminal observations.",
                                        "Do not count any of the 29 interrupted attempts as NativeFalse.",
                                        "Use a new recovery namespace and recheck current environment/process admission.",
                                        "Keep the exact task/raw/method, original RML, policy, fitting and budget."],
                  wall_seconds=time.monotonic() - started)
    report.update(sets)
    write_exclusive_report(output, report)
    print(json.dumps({"report": str(REPORT), "report_sha256": digest(output.read_bytes()),
                      "recoverable_complete_count": 1419, "damaged_registered_count": 0,
                      "unregistered_clean_native_count": 0, "unknown_interrupted_count": 29,
                      "still_unclosed_count": 501}, ensure_ascii=False))


if __name__ == "__main__":
    audit()
