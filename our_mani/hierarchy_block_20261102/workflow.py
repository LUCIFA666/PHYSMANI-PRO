"""Fresh owner with bounded, independently qualified parallel Native waves.

No work starts on import or during --check. The running owner checks and trains the
TRAIN200/offline-VAL40 cohort, observes every subprocess exit, and admits TEST
only after a complete same-freeze Native VAL release. It never retries a case.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


VERSION = "hierarchy_controlled_workflow_block_20261102_v1"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_FAMILY = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102")
RUNTIME_FAMILY = Path("runtime/framework/hierarchy_block_20261102")
DEFAULT_TRAINING_CLI = "scripts/train_hierarchy_block_20261102.py"
GPU_LOCK = RUNTIME_FAMILY / "gpu_training.lock"
TASKS = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed")
SOURCE_FILES = (
    "our_mani/hierarchy_block_20261102/workflow.py", "scripts/run_hierarchy_workflow_block_20261102.py",
    "our_mani/hierarchy_block_20261102/freeze_builder.py", "our_mani/hierarchy_block_20261102/evaluation.py",
    "our_mani/hierarchy_block_20261102/evaluation_test.py", "our_mani/hierarchy_block_20261102/actor.py",
    "our_mani/hierarchy_block_20261102/data.py", "our_mani/hierarchy_block_20261102/networks.py",
    "our_mani/hierarchy_block_20261102/isolation.py", "scripts/run_hierarchy_isolated_block_20261102.sh",
    "scripts/run_hierarchy_frozen_environment_block_20261102.py",
    "our_mani/hierarchy_block_20261102/training.py", DEFAULT_TRAINING_CLI,
    "scripts/evaluate_hierarchy_case_block_20261102.py", "scripts/evaluate_hierarchy_test_block_20261102.py",
    "our_mani/hierarchy_20261030/tracking.py", "our_mani/hierarchy_20261030/segmentation.py",
    "our_mani/hierarchy_20261030/candidate_multibody.py", "our_mani/hierarchy_20261030/networks.py",
    "our_mani/hierarchy_20261030/handoff_training_v2.py", "our_mani/hierarchy_20261030/training.py",
    "our_mani/hierarchy_20261030/data.py", "our_mani/hierarchy_precision_20261030/precision.py",
)



class WorkflowBlocked(RuntimeError):
    pass


class WorkflowFailed(RuntimeError):
    pass


class AdministrativeStop(RuntimeError):
    pass


def qualify_parallel(records, workers):
    """Qualify resource isolation and actual overlap, without task outcomes."""
    if workers not in (2, 4) or len(records) != workers:
        raise WorkflowFailed("qualification needs the fixed complete two/four-case canary wave")
    resources = [row.get("resources", {}) for row in records]
    for row, resource in zip(records, resources):
        interval = resource.get("native_interval", {})
        if (row.get("outcome_category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")
                or resource.get("schema") != "hierarchy_block_native_isolation_20261102_v1"
                or resource.get("complete") is not True or resource.get("restored") is not True
                or resource.get("errors") != [] or resource.get("parallel") is not True
                or resource.get("listener_audit_complete") is not True
                or resource.get("netnamespace_isolated") is not True
                or resource.get("all_threads_within_assignment") is not True
                or not all(isinstance(interval.get(key), (float, int)) and math.isfinite(interval[key])
                           for key in ("start_monotonic", "end_monotonic"))
                or interval["start_monotonic"] >= interval["end_monotonic"]):
            raise WorkflowFailed("actual canary isolation/exit evidence invalid; preserve attempts and stop")
    cpu_assignments = [r.get("assigned_cpus", []) for r in resources]
    if (any(len(cpus) != 8 or any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != 8
            for cpus in cpu_assignments)
            or len(set().union(*(set(cpus) for cpus in cpu_assignments))) != 8*workers
            or {r.get("slot") for r in resources} != set(range(workers))):
        raise WorkflowFailed("canary slot CPU assignments must be complete and disjoint")
    values = {
        "display": [r.get("display") for r in resources],
        "network_namespace": [r.get("network_namespace") for r in resources],
        "case_runtime": [r.get("case_runtime") for r in resources],
        "case_output": [r.get("case_output") for r in resources],
        **{key: [r.get("private_paths", {}).get(key) for r in resources]
           for key in ("TMPDIR", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")},
        "process": [json.dumps(r.get("process", {}), sort_keys=True) for r in resources],
    }
    start = max(r["native_interval"]["start_monotonic"] for r in resources)
    end = min(r["native_interval"]["end_monotonic"] for r in resources)
    for key, group in values.items():
        # X server numbers and namespace inodes may safely be reused after an
        # earlier process has closed. Such a nonoverlapping wave cannot promote
        # parallelism; it must not be mislabeled as a simultaneous collision.
        simultaneous = key not in ("display", "network_namespace", "process") or end > start
        if any(not value for value in group) or (simultaneous and len(set(group)) != workers):
            raise WorkflowFailed("canary cases share a process, display, network namespace, output or private path")
    return dict(schema="hierarchy_block_parallel_qualification_20261102_v1", complete=True,
                requested_workers=workers, qualified=end > start, actual_overlap_seconds=max(end-start, 0.),
                resource_isolation_verified=True, task_success_rate_used=False,
                assigned_cpus=cpu_assignments,
                cases=[row["case_key"] for row in records], independent_resources=values,
                reason="actual_simulation_intervals_overlap" if end > start else "no_actual_overlap_keep_previously_proven_level")


def validate_training_exit(root, exit_path):
    """Adopt a reaped, source-bound training process; never infer an exit."""
    from .evaluation import checked
    from .freeze_builder import _verify_tree
    import torch
    root = Path(root).resolve()
    path = Path(exit_path)
    path = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    if not path.is_relative_to(root / RUNTIME_FAMILY / "training_process_v1"):
        raise WorkflowBlocked("only the new explicit external training process receipt may be adopted")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "hierarchy_block_external_training_actual_exit_20261102_v1"
            or receipt.get("actual_process_exit_observed") is not True
            or receipt.get("returncode") != 0 or receipt.get("actual_returncode") != 0
            or receipt.get("timed_out") is not False or receipt.get("administrative_stop") is not False):
        raise WorkflowBlocked("adoption needs observed actual exit zero without timeout or administrative stop")
    references = {name: checked(root, receipt.get(name)) for name in
                  ("training_status", "checkpoint", "launch", "spawned_process", "log")}
    if (references["training_status"] != root / ARTIFACT_FAMILY / "training_v1/status.json"
            or references["checkpoint"] != root / ARTIFACT_FAMILY / "training_v1/best.pt"):
        raise WorkflowBlocked("adoption must bind this run's exact training status and best checkpoint")
    status = json.loads(references["training_status"].read_text())
    payload = torch.load(references["checkpoint"], map_location="cpu", weights_only=True)
    metadata, identity = payload.get("metadata", {}), status.get("identity", {})
    if (status.get("version") != "hierarchy_block_training_20261102_v1"
            or status.get("complete") is not True or status.get("completed_epochs") != 120
            or status.get("TEST_read") is not False or not identity.get("code_sha256")
            or status.get("best_checkpoint_sha256") != receipt["checkpoint"]["sha256"]
            or payload.get("version") != status["version"] or metadata.get("training_complete") is not True
            or metadata.get("completed_epochs") != 120 or metadata.get("TEST_read") is not False
            or metadata.get("identity") != identity or not payload.get("model_state_dict")):
        raise WorkflowBlocked("completed source-identical 120-epoch checkpoint/status chain required")
    _verify_tree(root, identity)
    # Launch/spawn evidence is immutable and describes the actual waiting
    # parent; the completed status alone is insufficient for this route.
    launch = json.loads(references["launch"].read_text())
    spawned = json.loads(references["spawned_process"].read_text()).get("process", {})
    command = launch.get("command", [])
    try:
        target = command.index(DEFAULT_TRAINING_CLI)
    except (ValueError, AttributeError):
        raise WorkflowBlocked("external training launch did not run the fixed new training CLI")
    if (command != receipt.get("command")
            or command[target:] != [DEFAULT_TRAINING_CLI, "--device", "cuda:0", "--workers", "4"]
            or not all(key in spawned for key in ("pid", "start_ticks", "boot_id"))
            or status.get("process") != spawned or identity_alive(spawned)):
        raise WorkflowBlocked("adoption command or actually exited training process identity differs")
    if isinstance(launch.get("source"), dict):
        checked(root, launch["source"])
    elif isinstance(launch.get("source_sha256"), dict):
        _verify_tree(root, {"code_sha256": launch["source_sha256"]})
    elif isinstance(launch.get("launcher_source_sha256"), str):
        if sha256(root / "scripts/run_hierarchy_training_process_block_20261102.py") != launch["launcher_source_sha256"]:
            raise WorkflowBlocked("external training launcher source changed")
    else:
        raise WorkflowBlocked("external launch must bind its actual launcher source")
    return dict(training_exit=pointer(root, path), training_status=receipt["training_status"],
                checkpoint=receipt["checkpoint"], actual_process_exit_observed=True,
                actual_returncode=0, optimization_repeated=False,
                precision_qualified=status.get("precision_qualified", False))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def atomic_json(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x") as stream:
        json.dump(plain(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    partial.replace(path)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def identity_alive(identity):
    try:
        current = process_identity(identity["pid"])
        return (current["state"] not in ("Z", "X")
                and current["start_ticks"] == identity["start_ticks"]
                and current["boot_id"] == identity["boot_id"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def pointer(root, path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise WorkflowFailed("workflow artifact is outside the shared project")
    return dict(path=str(path.relative_to(root)), sha256=sha256(path))




class RuntimeBackend:
    """Production adapters; TEST imports/calls are deferred until the VAL gate."""
    def __init__(self, root, *, namespace, case_timeout=900, verified_test_manifest=None):
        self.root, self.namespace = Path(root).resolve(), namespace
        self.case_timeout = case_timeout
        self.verified_test_manifest = verified_test_manifest

    def source_files(self):
        return SOURCE_FILES

    def dependencies(self):
        missing = [name for name in SOURCE_FILES if not (self.root / name).is_file()]
        functions = {
            "our_mani/hierarchy_block_20261102/freeze_builder.py": ("build_freeze",),
            "our_mani/hierarchy_block_20261102/evaluation_test.py": (
                "prepare_test_plan", "cpu_preflight_test", "launch_test_case", "audit_test", "request_stop"),
        }
        for name, expected in functions.items():
            path = self.root / name
            if path.is_file():
                present = {node.name for node in ast.parse(path.read_text()).body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                missing.extend(name + ":" + method for method in expected if method not in present)
        return dict(ready=not missing, missing=missing,
                    scope="workflow_source_dependencies_only",
                    native_or_GPU_started=False, TEST_source_opened=False,
                    native_execution_qualification_granted=False,
                    model_or_source_readiness_audited=False,
                    required_pending_code_dependencies=missing,
                    required_pending_stages=["TRAIN200_offline_VAL40_training",
                                             "runtime_freeze", "Native_TRAIN8_release",
                                             "Native_VAL40_release_before_any_TEST_read", "Native_TEST80_audit"])

    def blocked_call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (FileNotFoundError, ImportError, AttributeError) as error:
            raise WorkflowBlocked("required runtime dependency unavailable: " + str(error)) from error
        except Exception as error:
            evaluation = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
            if isinstance(error, evaluation.EvaluationBlocked):
                raise WorkflowBlocked(str(error)) from error
            raise

    def training_command(self, mode, runtime):
        command = [str(Path(sys.executable).resolve()), "-B", "-u", DEFAULT_TRAINING_CLI]
        return command + (["--check"] if mode == "CPU" else ["--device", "cuda:0", "--workers", "4"])

    def adopt_training(self, exit_path):
        try:
            return validate_training_exit(self.root, exit_path)
        except (OSError, ValueError) as error:
            raise WorkflowBlocked("external training receipt cannot be adopted: " + str(error)) from error


    def build_freeze(self):
        try:
            module = importlib.import_module("our_mani.hierarchy_block_20261102.freeze_builder")
            function = getattr(module, "build_freeze")
        except (ImportError, AttributeError) as error:
            raise WorkflowBlocked("required completed runtime freeze builder is missing") from error
        return Path(self.blocked_call(function, self.root)).resolve(strict=True)

    def assert_freeze(self, freeze_path):
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        module.load_freeze(self.root, freeze_path)
        return pointer(self.root, freeze_path)

    def prepare(self, split, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation_test")
            return self.blocked_call(module.prepare_test_plan, self.root, freeze_path, prior_release,
                                     verified_manifest_path=self.verified_test_manifest)
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        plan_path = self.root / module.ARTIFACTS / self.namespace / split / "plan.json"
        return self.blocked_call(module.prepare_plan, self.root, split, freeze_path=freeze_path,
                                 plan_path=plan_path, train_release_path=prior_release)

    def preflight(self, split, plan_path, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation_test")
            return self.blocked_call(module.cpu_preflight_test, self.root, plan_path, freeze_path, prior_release)
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        return self.blocked_call(module.cpu_preflight, self.root, plan_path)

    def launch(self, split, plan_path, case_key, freeze_path, prior_release, *, resource_slot=0, parallel=False):
        administrative_stop_path = self.root / RUNTIME_FAMILY / self.namespace / "STOP"
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation_test")
            return self.blocked_call(module.launch_test_case, self.root, plan_path, case_key, freeze_path, prior_release,
                                     device="cpu", timeout=self.case_timeout, administrative_stop_path=administrative_stop_path,
                                     resource_slot=resource_slot, parallel=parallel)
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        return self.blocked_call(module.launch_case, self.root, plan_path, case_key, device="cpu", timeout=self.case_timeout,
                                 administrative_stop_path=administrative_stop_path, resource_slot=resource_slot, parallel=parallel)

    def case_resource_evidence(self, receipt):
        if receipt.get("terminal") is None:
            return {}
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        return json.loads(module.checked(self.root, receipt["terminal"]).read_text()).get("resources", {})

    def audit(self, split, plan_path, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation_test")
            release = self.blocked_call(module.audit_test, self.root, plan_path, freeze_path, prior_release)
            # Extension may expose its release pointer directly; otherwise use
            # its declared artifact namespace and audit result to resolve it.
            if isinstance(release, tuple):
                return release
            for name in ("release_path", "path"):
                if name in release:
                    path = Path(release[name]); path = path if path.is_absolute() else self.root / path
                    return release, path
            path = self.root / module.ARTIFACTS / "TEST_results.json"
            if path.is_file() and json.loads(path.read_text()) == release:
                return release, path
            raise WorkflowBlocked("TEST audit did not expose its actual saved release artifact")
        module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation")
        function = module.audit_training if split == "train" else module.audit_validation
        release = self.blocked_call(function, self.root, plan_path)
        name = "Native_TRAIN_release.json" if split == "train" else "Native_VAL_release.json"
        return release, self.root / module.ARTIFACTS / split / name

    def request_stop(self, split, reason):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_block_20261102.evaluation_test")
            return module.request_stop(self.root, reason)
        directory = self.root / RUNTIME_FAMILY / "evaluation_v1"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "STOP"
        try:
            with path.open("x") as stream:
                stream.write(reason + "\n")
        except FileExistsError:
            pass
        return path

    def native_exit_evidence(self, receipt):
        ref = receipt.get("launch")
        if not isinstance(ref, dict):
            return {}
        path = self.root / ref["path"]
        if sha256(path) != ref["sha256"]:
            raise WorkflowFailed("actual Native launch evidence changed")
        launch = json.loads(path.read_text())
        return dict(command=launch["command"], actual_returncode=receipt["returncode"],
                    actual_process_exit_observed=receipt["actual_process_exit_observed"],
                    launch=ref, spawned_process=receipt.get("spawned_process"),
                    timed_out=receipt.get("timed_out"))


class Workflow:
    def __init__(self, root, *, namespace="workflow_v1", backend=None,
                 command_timeout=28800, command_runner=None, adopt_training_exit=None, native_workers=1):
        if not re.fullmatch(r"workflow_[a-z0-9_]+", namespace):
            raise ValueError("fresh workflow_* namespace required")
        self.root, self.namespace = Path(root).resolve(), namespace
        self.runtime = self.root / RUNTIME_FAMILY / namespace
        self.artifacts = self.root / ARTIFACT_FAMILY / namespace
        self.backend = backend or RuntimeBackend(self.root, namespace=namespace)
        self.command_timeout = command_timeout
        self.adopt_training_exit = adopt_training_exit
        if native_workers not in (1, 2, 4):
            raise ValueError("only one, two or four bounded Native workers are supported")
        self.requested_native_workers, self.proven_native_workers = native_workers, 1
        self.parallel_qualifications = []
        self.command_runner = command_runner or self.run_subprocess
        self.stop_path = self.runtime / "STOP"
        self.stop_event, self.finished_event = threading.Event(), threading.Event()
        self.active_native, self.watcher_error = None, None
        self.freeze_path = self.freeze_ref = None
        self.precision_qualified = False
        self.releases, self.plans = {}, {}
        self.source_hashes, self.status = {}, {}
        self.journal_sequence = 0
        self.command_number = 0
        self.case_outcomes = {split: [] for split in ("train", "validation", "test")}

    def check_stop(self):
        if self.watcher_error is not None:
            raise WorkflowFailed("administrative STOP bridge failed: " + repr(self.watcher_error))
        if self.stop_event.is_set() or self.stop_path.exists():
            raise AdministrativeStop("new workflow namespace administrative STOP")

    def verify_sources(self):
        for name, digest in self.source_hashes.items():
            if sha256(self.root / name) != digest:
                raise WorkflowFailed("workflow/runtime source changed: " + name)

    def journal(self, event, **payload):
        self.journal_sequence += 1
        record = dict(sequence=self.journal_sequence, observed_utc=utc(), event=event, **plain(payload))
        with (self.runtime / "journal.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        return record

    def update_status(self, **values):
        self.status.update(plain(values), updated_utc=utc())
        atomic_json(self.runtime / "status.json", self.status)

    def stage(self, name, function):
        self.check_stop(); self.verify_sources()
        self.update_status(stage=name)
        self.journal("stage_started", stage=name)
        started = time.monotonic()
        try:
            result = function()
            self.verify_sources(); self.check_stop()
            self.journal("stage_completed", stage=name, wall_seconds=time.monotonic() - started, result=result)
            self.status.setdefault("completed_stages", []).append(name)
            self.update_status()
            return result
        except BaseException as error:
            self.journal("stage_terminated", stage=name, wall_seconds=time.monotonic() - started,
                         error_type=type(error).__name__, error=repr(error))
            raise

    @contextmanager
    def gpu_lock(self):
        path = self.root / GPU_LOCK
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WorkflowBlocked("new hierarchy GPU training resource lock is already held") from error
            self.journal("GPU_lock_acquired", path=str(path.relative_to(self.root)), owner=self.status["process"])
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run_subprocess(self, command, *, stage):
        self.check_stop()
        self.command_number += 1
        directory = self.runtime / "commands"
        directory.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", stage)
        base = directory / f"{self.command_number:04d}_{slug}"
        launch = dict(stage=stage, command=list(command), started_utc=utc(), parent=self.status["process"], automatic_retry=False)
        atomic_json(base.with_suffix(".launch.json"), launch)
        started, stopped, timed_out = time.monotonic(), False, False
        with base.with_suffix(".log").open("xb") as stream:
            process = subprocess.Popen(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), start_new_session=True)
            try:
                spawned = process_identity(process.pid)
            except FileNotFoundError:
                spawned = dict(pid=process.pid, start_ticks_unavailable="exited before initial process snapshot")
            atomic_json(base.with_suffix(".spawned.json"), dict(process=spawned, observed_utc=utc()))
            while process.poll() is None:
                stopped = self.stop_event.is_set() or self.stop_path.exists()
                timed_out = time.monotonic() - started > self.command_timeout
                if stopped or timed_out:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    break
                time.sleep(.5)
            returncode = process.wait()
        receipt = dict(stage=stage, command=list(command), actual_process_exit_observed=True,
                       actual_returncode=returncode, spawned_process=spawned,
                       wall_seconds=time.monotonic() - started, administrative_stop=stopped,
                       timed_out=timed_out, finished_utc=utc(), automatic_retry=False,
                       log=pointer(self.root, base.with_suffix(".log")))
        atomic_json(base.with_suffix(".exit.json"), receipt)
        self.journal("subprocess_actual_exit", **receipt)
        return receipt

    def train(self, mode):
        command = self.backend.training_command(mode, self.runtime)
        if mode == "GPU":
            with self.gpu_lock():
                receipt = self.command_runner(command, stage="training_GPU")
        else:
            receipt = self.command_runner(command, stage="training_CPU_check")
        if receipt.get("actual_process_exit_observed") is not True:
            raise WorkflowFailed("training subprocess actual exit was not observed")
        if receipt.get("administrative_stop"):
            raise AdministrativeStop("training subprocess administratively stopped")
        if receipt.get("actual_returncode") != 0 or receipt.get("timed_out"):
            raise WorkflowFailed("training subprocess failed; no automatic retry")
        return receipt

    def freeze(self):
        self.freeze_path = Path(self.backend.build_freeze()).resolve(strict=True)
        self.freeze_ref = self.backend.assert_freeze(self.freeze_path)
        if self.freeze_ref != pointer(self.root, self.freeze_path):
            raise WorkflowFailed("freeze builder/runtime artifact pointers differ")
        self.precision_qualified = json.loads(self.freeze_path.read_text()).get("precision_qualified", False)
        return self.freeze_ref

    def assert_same_freeze(self):
        if self.freeze_path is None or self.freeze_ref is None:
            raise WorkflowBlocked("trained immutable hierarchy freeze is still missing")
        if self.backend.assert_freeze(self.freeze_path) != self.freeze_ref:
            raise WorkflowFailed("source or model freeze changed between stages")

    def prepare(self, split):
        self.assert_same_freeze()
        prior = None
        if split == "validation":
            if "train" not in self.releases:
                raise WorkflowBlocked("Native TRAIN release is required before VAL")
            prior = self.releases["train"][1]
        elif split == "test":
            if "validation" not in self.releases:
                raise WorkflowBlocked("complete Native VAL release is required before any TEST reader")
            prior = self.releases["validation"][1]
        plan, path = self.backend.prepare(split, self.freeze_path, prior)
        expected = {"train": 8, "validation": 40, "test": 80}[split]
        ready = (plan.get("native_execution_ready") is True if split != "test" else
                 plan.get("complete_Native_VAL_observed_before_TEST_read") is True
                 and plan.get("actual_VAL_exits_revalidated") == 40
                 and plan.get("TEST_used_for_fitting_or_selection") is False)
        if (plan.get("expected_cases") != expected or len(plan.get("cases", [])) != expected
                or len({row["case_key"] for row in plan["cases"]}) != expected
                or plan.get("freeze") != self.freeze_ref or not ready):
            raise WorkflowBlocked("exact same-freeze ready case plan required for " + split)
        self.plans[split] = plan, Path(path), prior
        return dict(plan=pointer(self.root, path), expected_cases=expected, freeze=self.freeze_ref)

    def preflight(self, split):
        self.assert_same_freeze()
        _, path, prior = self.plans[split]
        return self.backend.preflight(split, path, self.freeze_path, prior)

    def _launch_worker(self, split, case, slot, parallel):
        self.assert_same_freeze()
        _, path, prior = self.plans[split]
        started = time.monotonic()
        receipt = self.backend.launch(split, path, case["case_key"], self.freeze_path, prior,
                                      resource_slot=slot, parallel=parallel)
        try:
            evidence = self.backend.native_exit_evidence(receipt)
        except BaseException as error:
            evidence = dict(verification_error=repr(error))
        return receipt, evidence, time.monotonic()-started

    def _record_case(self, split, case, receipt, evidence, elapsed):
        category = receipt.get("category")
        natural = ("verification_error" not in evidence and receipt.get("actual_process_exit_observed") is True
                   and receipt.get("case_complete") is True and receipt.get("returncode") == 0
                   and receipt.get("timed_out") is False
                   and category in ("NativeTrue", "NativeFalse", "NativeTerminalFailure"))
        outcome = category if natural or category == "AdministrativeSTOP" else "InfrastructureFailure"
        try:
            resource = self.backend.case_resource_evidence(receipt) if hasattr(self.backend, "case_resource_evidence") else {}
        except BaseException as error:
            resource = dict(verification_error=repr(error))
            if outcome != "AdministrativeSTOP":
                outcome, natural = "InfrastructureFailure", False
        entry = dict(case_key=case["case_key"], task=case["task"], raw_episode_id=case.get("raw_episode_id"),
                     receipt=receipt, outcome_category=outcome, subprocess=evidence, resources=resource,
                     wall_seconds=elapsed, wall_seconds_scope="owner launch including source verification and observed child wait")
        self.case_outcomes[split].append(entry)
        self.journal("Native_case_actual_exit", split=split, **entry)
        return entry, None if natural else AdministrativeStop("Native administrative STOP; actual exit preserved") if outcome == "AdministrativeSTOP" else WorkflowFailed("Native infrastructure/incomplete case; no retry")

    def native_batch(self, split, cases, workers):
        self.check_stop(); self.assert_same_freeze()
        entries, errors = [], []
        self.active_native = split
        try:
            # Every submitted case has a unique slot and immutable output. On
            # any failure admission closes at this wave; all actual waits are
            # still collected before resources are released or owner exits.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="block-native") as pool:
                pending = {pool.submit(self._launch_worker, split, case, slot, workers > 1): case
                           for slot, case in enumerate(cases)}
                for future in as_completed(pending):
                    case = pending[future]
                    try:
                        receipt, evidence, elapsed = future.result()
                        entry, error = self._record_case(split, case, receipt, evidence, elapsed)
                        entries.append(entry)
                        if error is not None:
                            errors.append(error)
                    except BaseException as error:
                        errors.append(error)
                        administrative = self.stop_path.exists() or self.stop_event.is_set()
                        receipt = dict(category="AdministrativeSTOP" if administrative else "InfrastructureFailure",
                                       actual_process_exit_observed=False, returncode=None, timed_out=False,
                                       case_complete=False, launcher_exception=repr(error))
                        entry, classified = self._record_case(split, case, receipt, {}, 0.)
                        entries.append(entry)
                        if isinstance(classified, AdministrativeStop):
                            errors.append(classified)
                        self.journal("Native_case_launch_or_receipt_exception", split=split,
                                     case_key=case["case_key"], error_type=type(error).__name__, error=repr(error), automatic_retry=False)
        finally:
            self.active_native = None
        if errors:
            administrative = next((error for error in errors if isinstance(error, AdministrativeStop)), None)
            raise administrative or errors[0]
        self.check_stop()
        order = {case["case_key"]: i for i, case in enumerate(cases)}
        return sorted(entries, key=lambda row: order[row["case_key"]])

    def launch_one(self, split, case):
        return self.native_batch(split, [case], 1)[0]

    def _qualify(self, records, workers):
        result = qualify_parallel(records, workers)
        path = self.artifacts / ("parallel_qualification_"+str(workers)+".json")
        atomic_json(path, result)
        self.parallel_qualifications.append(pointer(self.root, path))
        if result["qualified"]:
            self.proven_native_workers = workers
        self.update_status(proven_native_workers=self.proven_native_workers,
                           parallel_qualifications=self.parallel_qualifications)
        self.journal("parallel_qualification", result=result, evidence=pointer(self.root, path))
        return result

    def run_native(self, split):
        cases = list(self.plans[split][0]["cases"])
        if split == "train" and self.requested_native_workers > 1:
            first = [next(case for case in cases if case["task"] == task) for task in TASKS]
            used = {case["case_key"] for case in first}
            result = self._qualify(self.native_batch(split, first, 2), 2)
            remaining = [case for case in cases if case["case_key"] not in used]
            if result["qualified"] and self.requested_native_workers == 4:
                # Fixed next two sources per task, independent of all outcomes.
                wave = [case for task in TASKS for case in [r for r in remaining if r["task"] == task][:2]]
                if len(wave) != 4:
                    raise WorkflowBlocked("four untouched fixed TRAIN sources required for four-way canary")
                self._qualify(self.native_batch(split, wave, 4), 4)
                used.update(case["case_key"] for case in wave)
            cases = [case for case in cases if case["case_key"] not in used]
        workers = self.proven_native_workers
        for start in range(0, len(cases), workers):
            self.native_batch(split, cases[start:start+workers], workers)
        return dict(split=split, cases=len(self.case_outcomes[split]), proven_native_workers=workers,
                    qualification=self.parallel_qualifications, automatic_retry=False)

    def audit(self, split):
        self.assert_same_freeze()
        plan, path, prior = self.plans[split]
        release, release_path = self.backend.audit(split, path, self.freeze_path, prior)
        count = plan["expected_cases"]
        if (release.get("complete") is not True or release.get("freeze") != self.freeze_ref
                or release.get("expected_cases") != count
                or release.get("actual_process_exits_audited") != count
                or len(release.get("cases", [])) != count):
            raise WorkflowBlocked("complete actual-exit same-freeze release required for " + split)
        self.releases[split] = release, Path(release_path)
        return dict(release=pointer(self.root, release_path), expected_cases=count, freeze=self.freeze_ref)

    def _watch_stop(self):
        while not self.finished_event.wait(.5):
            if self.stop_path.exists() or self.stop_event.is_set():
                self.stop_event.set()
                split = self.active_native
                if split is not None:
                    try:
                        self.backend.request_stop(split, "workflow administrative STOP at " + utc())
                    except BaseException as error:
                        self.watcher_error = error
                return

    def _signal(self, signum, _frame):
        self.stop_event.set()
        try:
            with self.stop_path.open("x") as stream:
                stream.write(f"Workflow received signal {signum} at {utc()}\n")
        except FileExistsError:
            pass

    def summary(self):
        summaries = {}
        for split, rows in self.case_outcomes.items():
            categories = {name: sum(row["outcome_category"] == name for row in rows)
                          for name in ("NativeTrue", "NativeFalse", "NativeTerminalFailure", "InfrastructureFailure", "AdministrativeSTOP")}
            summaries[split] = dict(attempted=len(rows), outcomes=categories,
                per_task={task: {name: sum(row["task"] == task and row["outcome_category"] == name for row in rows)
                                 for name in categories} for task in TASKS},
                cases=rows, release=pointer(self.root, self.releases[split][1]) if split in self.releases else None)
            for task, counts in summaries[split]["per_task"].items():
                denominator = sum(counts[name] for name in ("NativeTrue", "NativeFalse", "NativeTerminalFailure"))
                counts["completed_native_cases"] = denominator
                counts["native_success_rate"] = counts["NativeTrue"] / denominator if denominator else None
        return dict(version=VERSION, complete=self.status.get("complete", False),
                    stage=self.status.get("stage"), process=self.status.get("process"),
                    estimated_native_workers=self.proven_native_workers, requested_native_workers=self.requested_native_workers,
                    parallel_qualifications=self.parallel_qualifications,
                    freeze=self.freeze_ref, source_sha256=self.source_hashes, splits=summaries,
                    precision_qualified=self.precision_qualified,
                    native_success_is_precision_qualification=False,
                    accuracy_target_met_is_not_required_to_disclose_experimental_results=True,
                    automatic_retry=False, administrative_stop_is_native_failure=False,
                    TEST_source_read_only_after_full_Native_VAL_release=True,
                    required_pending_stages=[] if "test" in self.releases else ["complete_same_freeze_TEST80_and_audit"],
                    finished_utc=utc())

    def run(self):
        if self.stop_path.exists():
            raise AdministrativeStop("new namespace STOP already exists; preserved without starting work")
        if self.runtime.exists() or self.artifacts.exists():
            raise WorkflowBlocked("fresh owner namespace required; prior evidence is never overwritten or resumed automatically")
        self.runtime.mkdir(parents=True); self.artifacts.mkdir(parents=True)
        with (self.runtime / "owner.lock").open("xb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_stat = os.fstat(lock.fileno())
            self.status = dict(version=VERSION, stage="starting", complete=False, created_utc=utc(),
                               process=process_identity(), namespace=self.namespace, automatic_retry=False,
                               completed_stages=[], exit_code=None, native_device="cpu", native_cases_parallel=self.requested_native_workers > 1,
                               requested_native_workers=self.requested_native_workers, proven_native_workers=1,
                               estimated_native_workers=self.requested_native_workers,
                               owner_lock=dict(path=str((self.runtime / "owner.lock").relative_to(self.root)),
                                               device=lock_stat.st_dev, inode=lock_stat.st_ino, fd=lock.fileno()))
            self.update_status()
            old_handlers = {}
            if threading.current_thread() is threading.main_thread():
                for number in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[number] = signal.signal(number, self._signal)
            watcher = threading.Thread(target=self._watch_stop, daemon=True)
            watcher.start()
            try:
                dependencies = self.backend.dependencies()
                self.journal("dependency_check", result=dependencies)
                if dependencies.get("ready") is not True:
                    raise WorkflowBlocked("required workflow dependencies missing: " + ", ".join(dependencies.get("missing", [])))
                self.source_hashes = {name: sha256(self.root / name) for name in self.backend.source_files()}
                self.update_status(source_sha256=self.source_hashes)
                self.stage("training_CPU_check", lambda: self.train("CPU"))
                if self.adopt_training_exit is None:
                    self.stage("training_GPU_TRAIN200_offline_VAL40", lambda: self.train("GPU"))
                else:
                    self.stage("adopt_observed_completed_training_exit", lambda: self.backend.adopt_training(self.adopt_training_exit))
                self.stage("build_and_verify_runtime_freeze", self.freeze)
                for split in ("train", "validation", "test"):
                    self.stage(split + "_prepare", lambda split=split: self.prepare(split))
                    self.stage(split + "_CPU_preflight", lambda split=split: self.preflight(split))
                    self.stage(split + "_Native_cases_and_resource_qualification", lambda split=split: self.run_native(split))
                    self.stage(split + "_actual_exit_audit", lambda split=split: self.audit(split))
                self.verify_sources(); self.assert_same_freeze()
                self.update_status(stage="complete", complete=True, exit_code=0, finished_utc=utc())
            except AdministrativeStop as error:
                self.update_status(stage="administrative_stop", complete=False, exit_code=3,
                                   error=repr(error), administrative_stop=True, finished_utc=utc())
            except WorkflowBlocked as error:
                self.update_status(stage="blocked", complete=False, exit_code=2,
                                   error=repr(error), required_dependency_missing=True, finished_utc=utc())
            except BaseException as error:
                self.update_status(stage="failed", complete=False, exit_code=1,
                                   error_type=type(error).__name__, error=repr(error), finished_utc=utc())
            finally:
                self.finished_event.set(); watcher.join(timeout=1.)
                for number, handler in old_handlers.items():
                    signal.signal(number, handler)
                atomic_json(self.artifacts / "summary.json", self.summary())
                self.journal("owner_terminal", status=self.status, summary=pointer(self.root, self.artifacts / "summary.json"))
            return self.status


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="CPU dependency check only; no fresh namespace or child processes")
    mode.add_argument("--run", action="store_true", help="explicitly start the authorized controlled workflow")
    command.add_argument("--namespace", default="workflow_v1")
    command.add_argument("--training-timeout", type=int, default=28800)
    command.add_argument("--case-timeout", type=int, default=900)
    command.add_argument("--verified-test-manifest", type=Path)
    command.add_argument("--native-workers", type=int, choices=(1,2,4), default=4)
    command.add_argument("--adopt-training-exit", type=Path,
                         help="adopt only an already reaped source-bound completed training process; no retry or resume")
    return command


def main(args=None):
    args = parser().parse_args(args)
    if args.training_timeout < 1 or args.case_timeout < 1:
        raise ValueError("positive subprocess wall-time bounds required")
    backend = RuntimeBackend(ROOT, namespace=args.namespace, case_timeout=args.case_timeout,
                             verified_test_manifest=args.verified_test_manifest)
    if args.check:
        result = backend.dependencies()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["ready"] else 2
    try:
        owner = Workflow(ROOT, namespace=args.namespace, backend=backend,
                         command_timeout=args.training_timeout, adopt_training_exit=args.adopt_training_exit, native_workers=args.native_workers)
        result = owner.run()
    except AdministrativeStop as error:
        result = dict(complete=False, stage="administrative_stop", exit_code=3, error=repr(error))
    except WorkflowBlocked as error:
        result = dict(complete=False, stage="blocked", exit_code=2, error=repr(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result["exit_code"]
