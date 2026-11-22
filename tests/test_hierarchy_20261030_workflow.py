import json
from pathlib import Path
import tempfile
import unittest
import sys

from our_mani.hierarchy_20261030.workflow import (
    ARTIFACT_FAMILY, RUNTIME_FAMILY, TASKS, Workflow, WorkflowFailed,
    atomic_json, motion_state, pointer, process_identity,
)


class FakeBackend:
    def __init__(self, root, *, incomplete_val=False, stop_first=False):
        self.root = root
        self.incomplete_val, self.stop_first = incomplete_val, stop_first
        self.calls, self.plans, self.freeze_ref = [], {}, None
        (root / "source.py").write_text("# fixed mock dependency\n")

    def source_files(self):
        return ("source.py",)

    def dependencies(self):
        return dict(ready=True, missing=[])

    def candidate_command(self, mode, runtime):
        self.calls.append("candidate_" + mode)
        return ["unit-test-only", mode]

    def build_freeze(self):
        self.calls.append("freeze")
        path = self.root / "freeze.json"
        atomic_json(path, {"frozen": True})
        self.freeze_ref = pointer(self.root, path)
        return path

    def assert_freeze(self, path):
        return pointer(self.root, path)

    def prepare(self, split, freeze_path, prior_release):
        self.calls.append("prepare_" + split)
        if split == "test":
            self.calls.append("TEST_source_reader")
            if prior_release is None or not json.loads(prior_release.read_text())["complete"]:
                raise AssertionError("TEST source read before complete VAL release")
        counts = [4] * 4 if split == "train" else [20, 20, 18, 19] if split == "validation" else [40] * 4
        cases = [dict(case_key=f"{task}__episode{i}", task=task, raw_episode_id=f"episode{i}")
                 for task, count in zip(TASKS, counts) for i in range(count)]
        plan = dict(expected_cases=sum(counts), cases=cases, freeze=self.freeze_ref, native_execution_ready=True)
        if split == "test":
            plan.update(complete_Native_VAL_observed_before_TEST_read=True,
                        actual_VAL_exits_revalidated=77, TEST_used_for_fitting_or_selection=False)
        path = self.root / (split + "_plan.json")
        atomic_json(path, plan); self.plans[split] = plan
        return plan, path

    def preflight(self, split, *unused):
        self.calls.append("preflight_" + split)
        return dict(native_started=False)

    def launch(self, split, plan_path, case_key, *unused):
        self.calls.append(("case", split, case_key))
        if self.stop_first:
            return dict(category="AdministrativeSTOP", actual_process_exit_observed=True,
                        case_complete=False, returncode=2, timed_out=False)
        return dict(category="NativeFalse", actual_process_exit_observed=True,
                    case_complete=True, returncode=0, timed_out=False)

    def native_exit_evidence(self, receipt):
        return dict(command=["unit-test-only", "Native"], actual_returncode=receipt["returncode"])

    def audit(self, split, *unused):
        self.calls.append("audit_" + split)
        plan = self.plans[split]
        release = dict(complete=not (split == "validation" and self.incomplete_val),
                       freeze=self.freeze_ref, expected_cases=plan["expected_cases"],
                       actual_process_exits_audited=plan["expected_cases"], cases=plan["cases"])
        path = self.root / (split + "_release.json")
        atomic_json(path, release)
        return release, path

    def request_stop(self, split, reason):
        self.calls.append(("stop", split))


class WorkflowTests(unittest.TestCase):
    def build(self, root, **kwargs):
        backend = FakeBackend(root, **kwargs)
        owner = Workflow(root, backend=backend, command_runner=lambda *args, **unused: dict(
            actual_process_exit_observed=True, actual_returncode=0, administrative_stop=False, timed_out=False))
        owner.wait_for_motion = lambda: dict(expected_cases=477, unit_test_stub=True)
        return owner, backend

    def test_stage_order_and_nativefalse_continue_without_retry_or_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            result = owner.run()
            self.assertTrue(result["complete"])
            self.assertLess(backend.calls.index("candidate_CPU"), backend.calls.index("candidate_GPU"))
            self.assertLess(backend.calls.index("candidate_GPU"), backend.calls.index("freeze"))
            self.assertLess(backend.calls.index("audit_train"), backend.calls.index("prepare_validation"))
            self.assertLess(backend.calls.index("audit_validation"), backend.calls.index("TEST_source_reader"))
            cases = [row for row in backend.calls if isinstance(row, tuple) and row[0] == "case"]
            self.assertEqual(len(cases), 16 + 77 + 160)
            self.assertEqual(len(set(cases)), len(cases))
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v1/summary.json").read_text())
            self.assertEqual(summary["estimated_native_workers"], 1)
            self.assertEqual(summary["splits"]["validation"]["outcomes"]["NativeFalse"], 77)
            self.assertEqual(summary["splits"]["test"]["outcomes"]["NativeFalse"], 160)
            self.assertFalse(summary["automatic_retry"])

    def test_incomplete_val_release_never_calls_test_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend = self.build(Path(temporary), incomplete_val=True)
            result = owner.run()
            self.assertEqual(result["stage"], "blocked")
            self.assertNotIn("TEST_source_reader", backend.calls)
            self.assertNotIn("prepare_test", backend.calls)

    def test_administrative_stop_is_not_nativefalse_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root, stop_first=True)
            result = owner.run()
            self.assertEqual(result["stage"], "administrative_stop")
            cases = [row for row in backend.calls if isinstance(row, tuple) and row[0] == "case"]
            self.assertEqual(len(cases), 1)
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v1/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["outcomes"]["AdministrativeSTOP"], 1)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)
            self.assertNotIn("TEST_source_reader", backend.calls)

    def test_dead_incomplete_motion_owner_fails_instead_of_waiting(self):
        manifest = dict(schema="hierarchy_public_motion_inputs_20261030_v1", complete=False,
                        stage="running", infrastructure_failed=0)
        with self.assertRaises(WorkflowFailed):
            motion_state(manifest, owner_alive=False)
        self.assertEqual(motion_state(manifest, owner_alive=True), "waiting")
        failed = dict(manifest, stage="failed")
        with self.assertRaises(WorkflowFailed):
            motion_state(failed, owner_alive=True)

    def test_namespace_stop_prevents_following_gpu_and_native_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            def runner(command, **unused):
                owner.stop_path.write_text("explicit test administrative stop\n")
                return dict(actual_process_exit_observed=True, actual_returncode=0,
                            administrative_stop=False, timed_out=False)
            owner.command_runner = runner
            result = owner.run()
            self.assertEqual(result["stage"], "administrative_stop")
            self.assertNotIn("candidate_GPU", backend.calls)
            self.assertNotIn("prepare_train", backend.calls)
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v1/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["attempted"], 0)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)

    def test_actual_cpu_subprocess_exit_and_command_are_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner = Workflow(root)
            owner.runtime.mkdir(parents=True)
            owner.status = {"process": process_identity()}
            command = [sys.executable, "-c", "import sys; sys.exit(7)"]
            result = owner.run_subprocess(command, stage="CPU_receipt_test")
            self.assertEqual(result["actual_returncode"], 7)
            self.assertTrue(result["actual_process_exit_observed"])
            self.assertEqual(result["command"], command)
            self.assertGreaterEqual(result["wall_seconds"], 0.)
            self.assertFalse(result["administrative_stop"])

    def test_missing_dependency_blocks_before_wait_or_any_child_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend = self.build(Path(temporary))
            backend.dependencies = lambda: dict(ready=False, missing=["required_TEST_extension.py"])
            result = owner.run()
            self.assertEqual(result["stage"], "blocked")
            self.assertEqual(backend.calls, [])

    def test_nonzero_actual_case_exit_is_infrastructure_not_nativefalse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            backend.launch = lambda *args, **kwargs: dict(category="NativeFalse",
                actual_process_exit_observed=True, case_complete=False, returncode=1, timed_out=False)
            result = owner.run()
            self.assertEqual(result["stage"], "failed")
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v1/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["outcomes"]["InfrastructureFailure"], 1)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)


if __name__ == "__main__":
    unittest.main()
