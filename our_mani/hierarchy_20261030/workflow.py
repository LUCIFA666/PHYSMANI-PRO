"""Fresh, serial owner for the explicitly authorized hierarchy experiment.

No work starts on import or during --check. The running owner waits for the
complete public-motion cohort, observes every subprocess exit, and admits TEST
only after a complete same-freeze Native VAL release. It never retries a case.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


VERSION = "hierarchy_controlled_workflow_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_FAMILY = Path("artifacts/framework/multitask_r10/hierarchy_20261030")
RUNTIME_FAMILY = Path("runtime/framework/hierarchy_20261030")
DEFAULT_MOTION = ARTIFACT_FAMILY / "motion_inputs_blind_v1/manifest.json"
GPU_LOCK = RUNTIME_FAMILY / "gpu_training.lock"
TASKS = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed",
         "insert_onto_rotating_peg", "insert_onto_rotating_peg_high_speed")
SOURCE_FILES = (
    "our_mani/hierarchy_20261030/workflow.py", "scripts/run_hierarchy_workflow_20261030.py",
    "our_mani/hierarchy_20261030/freeze_builder.py", "our_mani/hierarchy_20261030/evaluation.py",
    "our_mani/hierarchy_20261030/evaluation_test.py", "our_mani/hierarchy_20261030/actor.py",
    "our_mani/hierarchy_20261030/candidate_training.py", "our_mani/hierarchy_20261030/candidate_multibody.py",
    "scripts/train_hierarchy_candidates_20261030.py", "scripts/evaluate_hierarchy_case_20261030.py",
    "scripts/evaluate_hierarchy_test_20261030.py",
    "our_mani/hierarchy_20261030/tracking.py", "our_mani/hierarchy_20261030/segmentation.py",
    "our_mani/hierarchy_20261030/data.py", "our_mani/hierarchy_20261030/networks.py",
    "our_mani/hierarchy_20261030/handoff_training_v2.py", "our_mani/hierarchy_20261030/training.py",
)


class WorkflowBlocked(RuntimeError):
    pass


class WorkflowFailed(RuntimeError):
    pass


class AdministrativeStop(RuntimeError):
    pass


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


def motion_state(manifest, *, owner_alive):
    """No incomplete dead-owner snapshot can turn into an infinite wait."""
    if manifest.get("schema") != "hierarchy_public_motion_inputs_20261030_v1":
        raise WorkflowFailed("unexpected public motion manifest schema")
    if manifest.get("stage") in ("failed", "drained") or manifest.get("infrastructure_failed", 0):
        raise WorkflowFailed("public motion collection failed or stopped; no automatic replay")
    if manifest.get("complete") is True:
        cases = manifest.get("cases", [])
        identities = {(row["split"], row["task"], row["raw_episode_id"]) for row in cases}
        counts = {split: sum(row["split"] == split for row in cases) for split in ("train", "validation")}
        if (manifest.get("expected_cases") != 477 or len(cases) != 477 or len(identities) != 477
                or counts != {"train": 400, "validation": 77}
                or manifest.get("smoke_only") is not False
                or manifest.get("source_task_selects_frontend") is not False
                or manifest.get("TEST_read") is not False):
            raise WorkflowFailed("public motion cohort is incomplete, changed, or task-routed")
        if manifest.get("actual_all_worker_returns_observed") is not True:
            if owner_alive:
                return "waiting_final_motion_receipt"
            raise WorkflowFailed("motion owner exited without its complete worker-return receipt")
        return "complete"
    if not owner_alive:
        raise WorkflowFailed("public motion owner PID/start_ticks/boot no longer live while manifest is incomplete")
    return "waiting"


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
            "our_mani/hierarchy_20261030/freeze_builder.py": ("build_freeze",),
            "our_mani/hierarchy_20261030/evaluation_test.py": (
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
                    required_pending_stages=["complete_motion477", "candidate_CPU_and_GPU_training",
                                             "runtime_freeze", "Native_TRAIN16_release",
                                             "Native_VAL77_release_before_any_TEST_read", "Native_TEST160_audit"])

    def blocked_call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (FileNotFoundError, ImportError, AttributeError) as error:
            raise WorkflowBlocked("required runtime dependency unavailable: " + str(error)) from error
        except Exception as error:
            evaluation = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
            if isinstance(error, evaluation.EvaluationBlocked):
                raise WorkflowBlocked(str(error)) from error
            raise

    def candidate_command(self, mode, runtime):
        command = [str(Path(sys.executable).resolve()), "-B", "-u", "scripts/train_hierarchy_candidates_20261030.py",
                   "--readiness-out", str(runtime / "candidate_readiness.json")]
        return command + (["--dry-run"] if mode == "CPU" else ["--device", "cuda:0", "--workers", "4"])

    def build_freeze(self):
        try:
            module = importlib.import_module("our_mani.hierarchy_20261030.freeze_builder")
            function = getattr(module, "build_freeze")
        except (ImportError, AttributeError) as error:
            raise WorkflowBlocked("required completed runtime freeze builder is missing") from error
        return Path(self.blocked_call(function, self.root)).resolve(strict=True)

    def assert_freeze(self, freeze_path):
        module = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
        module.load_freeze(self.root, freeze_path)
        return pointer(self.root, freeze_path)

    def prepare(self, split, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_20261030.evaluation_test")
            return self.blocked_call(module.prepare_test_plan, self.root, freeze_path, prior_release,
                                     verified_manifest_path=self.verified_test_manifest)
        module = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
        plan_path = self.root / module.ARTIFACTS / self.namespace / split / "plan.json"
        return self.blocked_call(module.prepare_plan, self.root, split, freeze_path=freeze_path,
                                 plan_path=plan_path, train_release_path=prior_release)

    def preflight(self, split, plan_path, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_20261030.evaluation_test")
            return self.blocked_call(module.cpu_preflight_test, self.root, plan_path, freeze_path, prior_release)
        module = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
        return self.blocked_call(module.cpu_preflight, self.root, plan_path)

    def launch(self, split, plan_path, case_key, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_20261030.evaluation_test")
            return self.blocked_call(module.launch_test_case, self.root, plan_path, case_key, freeze_path, prior_release,
                                     device="cpu", timeout=self.case_timeout)
        module = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
        return self.blocked_call(module.launch_case, self.root, plan_path, case_key, device="cpu", timeout=self.case_timeout)

    def audit(self, split, plan_path, freeze_path, prior_release):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_20261030.evaluation_test")
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
        module = importlib.import_module("our_mani.hierarchy_20261030.evaluation")
        function = module.audit_training if split == "train" else module.audit_validation
        release = self.blocked_call(function, self.root, plan_path)
        name = "Native_TRAIN_release.json" if split == "train" else "Native_VAL_release.json"
        return release, self.root / module.ARTIFACTS / split / name

    def request_stop(self, split, reason):
        if split == "test":
            module = importlib.import_module("our_mani.hierarchy_20261030.evaluation_test")
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
    def __init__(self, root, *, namespace="workflow_v1", backend=None, poll_seconds=20.,
                 motion_manifest=DEFAULT_MOTION, command_timeout=7200, command_runner=None):
        if not re.fullmatch(r"workflow_[a-z0-9_]+", namespace):
            raise ValueError("fresh workflow_* namespace required")
        if poll_seconds != 20.:
            raise ValueError("the controlled motion polling interval is 20 seconds")
        self.root, self.namespace = Path(root).resolve(), namespace
        self.runtime = self.root / RUNTIME_FAMILY / namespace
        self.artifacts = self.root / ARTIFACT_FAMILY / namespace
        self.motion_path = (self.root / motion_manifest).resolve()
        self.backend = backend or RuntimeBackend(self.root, namespace=namespace)
        self.poll_seconds, self.command_timeout = poll_seconds, command_timeout
        self.command_runner = command_runner or self.run_subprocess
        self.stop_path = self.runtime / "STOP"
        self.stop_event, self.finished_event = threading.Event(), threading.Event()
        self.active_native, self.watcher_error = None, None
        self.freeze_path = self.freeze_ref = None
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

    def wait_for_motion(self):
        while True:
            self.check_stop(); self.verify_sources()
            if not self.motion_path.is_file():
                raise WorkflowBlocked("required motion owner manifest is missing; no blind infinite wait")
            manifest = json.loads(self.motion_path.read_text())
            state = motion_state(manifest, owner_alive=identity_alive(manifest.get("process", {})))
            self.journal("motion_poll", state=state, completed=len(manifest.get("cases", [])),
                         expected=manifest.get("expected_cases"), motion_owner=manifest.get("process"))
            if state == "complete":
                return dict(motion_manifest=pointer(self.root, self.motion_path), expected_cases=477)
            self.stop_event.wait(self.poll_seconds)

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

    def candidate(self, mode):
        command = self.backend.candidate_command(mode, self.runtime)
        if mode == "GPU":
            with self.gpu_lock():
                receipt = self.command_runner(command, stage="candidate_GPU")
        else:
            receipt = self.command_runner(command, stage="candidate_CPU_dry_run")
        if receipt.get("actual_process_exit_observed") is not True:
            raise WorkflowFailed("candidate subprocess actual exit was not observed")
        if receipt.get("administrative_stop"):
            raise AdministrativeStop("candidate subprocess administratively stopped")
        if receipt.get("actual_returncode") != 0 or receipt.get("timed_out"):
            raise WorkflowFailed("candidate subprocess failed; no automatic retry")
        return receipt

    def freeze(self):
        self.freeze_path = Path(self.backend.build_freeze()).resolve(strict=True)
        self.freeze_ref = self.backend.assert_freeze(self.freeze_path)
        if self.freeze_ref != pointer(self.root, self.freeze_path):
            raise WorkflowFailed("freeze builder/runtime artifact pointers differ")
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
        expected = {"train": 16, "validation": 77, "test": 160}[split]
        ready = (plan.get("native_execution_ready") is True if split != "test" else
                 plan.get("complete_Native_VAL_observed_before_TEST_read") is True
                 and plan.get("actual_VAL_exits_revalidated") == 77
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

    def launch_one(self, split, case):
        self.assert_same_freeze()
        _, path, prior = self.plans[split]
        started = time.monotonic()
        self.active_native = split
        try:
            receipt = self.backend.launch(split, path, case["case_key"], self.freeze_path, prior)
        finally:
            self.active_native = None
        evidence = self.backend.native_exit_evidence(receipt)
        category = receipt.get("category")
        natural = (receipt.get("actual_process_exit_observed") is True
                   and receipt.get("case_complete") is True and receipt.get("returncode") == 0
                   and receipt.get("timed_out") is False
                   and category in ("NativeTrue", "NativeFalse", "NativeTerminalFailure"))
        outcome = (category if natural else "AdministrativeSTOP"
                   if category == "AdministrativeSTOP" and receipt.get("actual_process_exit_observed") is True
                   else "InfrastructureFailure")
        entry = dict(case_key=case["case_key"], task=case["task"],
                     raw_episode_id=case.get("raw_episode_id"), receipt=receipt,
                     outcome_category=outcome, subprocess=evidence,
                     wall_seconds=time.monotonic() - started,
                     wall_seconds_scope="owner launch call including source verification; actual child exit separately recorded")
        self.case_outcomes[split].append(entry)
        self.journal("Native_case_actual_exit", split=split, **entry)
        if receipt.get("actual_process_exit_observed") is not True:
            raise WorkflowFailed("Native actual process exit missing")
        if receipt.get("category") == "AdministrativeSTOP":
            raise AdministrativeStop("Native case administrative STOP; no failure score")
        if (receipt.get("case_complete") is not True or receipt.get("returncode") != 0
                or receipt.get("timed_out") is not False
                or receipt.get("category") not in ("NativeTrue", "NativeFalse", "NativeTerminalFailure")):
            raise WorkflowFailed("Native infrastructure/incomplete case; owner stops without retry")
        return entry

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
        return dict(version=VERSION, complete=self.status.get("complete", False),
                    stage=self.status.get("stage"), process=self.status.get("process"),
                    estimated_native_workers=1,
                    freeze=self.freeze_ref, source_sha256=self.source_hashes, splits=summaries,
                    automatic_retry=False, administrative_stop_is_native_failure=False,
                    TEST_source_read_only_after_full_Native_VAL_release=True,
                    required_pending_stages=[] if "test" in self.releases else ["complete_same_freeze_TEST160_and_audit"],
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
                               completed_stages=[], exit_code=None, native_device="cpu", native_cases_parallel=False,
                               estimated_native_workers=1,
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
                self.stage("wait_complete_public_motion477", self.wait_for_motion)
                self.stage("candidate_CPU_dry_run", lambda: self.candidate("CPU"))
                self.stage("candidate_GPU_training", lambda: self.candidate("GPU"))
                self.stage("build_and_verify_runtime_freeze", self.freeze)
                for split in ("train", "validation", "test"):
                    self.stage(split + "_prepare", lambda split=split: self.prepare(split))
                    self.stage(split + "_CPU_preflight", lambda split=split: self.preflight(split))
                    plan, _, _ = self.plans[split]
                    for case in plan["cases"]:
                        self.stage(split + "_case_" + case["case_key"], lambda split=split, case=case: self.launch_one(split, case))
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
    command.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION)
    command.add_argument("--candidate-timeout", type=int, default=7200)
    command.add_argument("--case-timeout", type=int, default=900)
    command.add_argument("--verified-test-manifest", type=Path)
    return command


def main(args=None):
    args = parser().parse_args(args)
    if args.candidate_timeout < 1 or args.case_timeout < 1:
        raise ValueError("positive subprocess wall-time bounds required")
    backend = RuntimeBackend(ROOT, namespace=args.namespace, case_timeout=args.case_timeout,
                             verified_test_manifest=args.verified_test_manifest)
    if args.check:
        result = backend.dependencies()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["ready"] else 2
    try:
        owner = Workflow(ROOT, namespace=args.namespace, backend=backend,
                         motion_manifest=args.motion_manifest, command_timeout=args.candidate_timeout)
        result = owner.run()
    except AdministrativeStop as error:
        result = dict(complete=False, stage="administrative_stop", exit_code=3, error=repr(error))
    except WorkflowBlocked as error:
        result = dict(complete=False, stage="blocked", exit_code=2, error=repr(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result["exit_code"]
