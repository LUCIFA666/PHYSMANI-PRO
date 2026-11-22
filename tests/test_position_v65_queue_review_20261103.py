"""Independent V65 queue contracts. No Native/model/controller construction."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_position_v65_fixed_development_20261103.py"
spec = importlib.util.spec_from_file_location("v65_queue_review_subject", SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class QueueReview(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        plan = {"jobs": [], "all_four_run_regardless_of_task_outcome": True,
                "retry_or_case_replacement_based_on_outcome": False, "formal_TEST_read": False,
                "natural_witness_only": True, "active_new_probe_execution": False}
        for name in subject.EXPECTED:
            job = dict(id=name, scene="static" if name in ("00_val_static", "static_no_proxy") else "uniform",
                       config="config/" + name + ".json",
                       path="runtime/framework/local_mpc_20261102/public_candidate_reverification_v65_development_v1/" + name)
            dump(root / job["config"], {"id": name})
            plan["jobs"].append(job)
        return root, plan

    def closed(self, root, job, reason="budget_exhausted"):
        output = root / job["path"]
        output.mkdir(parents=True)
        dump(output.with_name(output.name + ".owner.actual_exit.json"),
             dict(actual_process_returncode=0, actual_wait_completed=True, source_unchanged=True, stop_reason=None,
                  case_id=job["id"], output=str(output.resolve())))
        dump(output / "worker_exit.json", dict(exit_code=0, native_closed=True, result_recorded=True))
        dump(output / "result.json", dict(complete=True, source_unchanged=True, canary_stop_reason=reason,
                                         public_lift_hold_observed=False, task_success=False))
        dump(output / "run_inputs.json", dict(source_sha256={}, config={"id": job["id"]}))

    def launch(self, root, plan, calls):
        def run(command, **kwargs):
            case = Path(command[command.index("--output") + 1]).name
            job = next(row for row in plan["jobs"] if row["id"] == case)
            calls.append(case)
            self.closed(root, job)
            return subprocess.CompletedProcess(command, 0)
        return run

    def test_task_false_budget_outcome_runs_all_four_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root, plan = self.fixture(temp)
            calls = []
            with patch.object(subject, "ROOT", root), patch.object(subject, "check_plan", return_value=(plan, [{"ready": True}] * 4)), \
                    patch.object(subject, "verify_sources", return_value={}), patch.object(subject.subprocess, "run", side_effect=self.launch(root, plan, calls)):
                code = subject.main(["--source-freeze", str(root / "freeze.json"), "--source-freeze-sha256", "0" * 64, "--execute"])
            self.assertEqual(code, 0)
            self.assertEqual(calls, list(subject.EXPECTED))
            ledger = root / "runtime/framework/local_mpc_20261102/public_candidate_reverification_v65_development_v1/serial_queue_v1.jsonl"
            events = [json.loads(line)["event"] for line in ledger.read_text().splitlines()]
            self.assertEqual(events, ["start"] + ["actual_closed"] * 4 + ["complete"])

    def test_queue_stop_after_first_closed_prevents_second_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root, plan = self.fixture(temp)
            calls = []
            original_audit = subject.audit_closed
            def audit(job, sources):
                value = original_audit(job, sources)
                (root / job["path"]).parent.joinpath("STOP").write_text("administrative stop\n")
                return value
            with patch.object(subject, "ROOT", root), patch.object(subject, "check_plan", return_value=(plan, [{"ready": True}] * 4)), \
                    patch.object(subject, "verify_sources", return_value={}), patch.object(subject, "audit_closed", side_effect=audit), \
                    patch.object(subject.subprocess, "run", side_effect=self.launch(root, plan, calls)):
                with self.assertRaisesRegex(ValueError, "subsequent case"):
                    subject.main(["--source-freeze", str(root / "freeze.json"), "--source-freeze-sha256", "0" * 64, "--execute"])
            self.assertEqual(calls, [subject.EXPECTED[0]])
            self.assertFalse((root / plan["jobs"][1]["path"]).exists())

    def test_admin_and_no_tick_result_never_count_as_clean_closed(self):
        for reason in ("administrative_stop", "no_actual_public_tick", "no_actual_tick", None):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                root, plan = self.fixture(temp)
                self.closed(root, plan["jobs"][0], reason)
                with patch.object(subject, "ROOT", root):
                    with self.assertRaisesRegex(ValueError, "clean closure"):
                        subject.audit_closed(plan["jobs"][0], {})

    def test_default_check_never_calls_child_launch(self):
        with patch.object(subject, "check_plan", return_value=({"jobs": []}, [{"ready": False}])), \
                patch.object(subject.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(subject.main([]), 0)
        run.assert_not_called()
        self.assertFalse(json.loads(stdout.getvalue())["Native_started"])

    def test_two_static_config_swap_is_rejected_before_any_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root, plan = self.fixture(temp)
            first, last = plan["jobs"][0], plan["jobs"][-1]
            first["config"], last["config"] = last["config"], first["config"]
            plan_path = root / "plan.json"
            dump(plan_path, plan)
            fake_worker = types.ModuleType("our_mani.local_mpc_20261103.position_v65_worker")
            fake_worker.check_config = lambda path, scene, **kwargs: dict(case_id=json.loads(Path(path).read_text())["id"], ready=True)
            with patch.object(subject, "ROOT", root), patch.object(subject, "PLAN", plan_path), \
                    patch.dict(sys.modules, {fake_worker.__name__: fake_worker}), patch.object(subject.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "case"):
                    subject.check_plan(types.SimpleNamespace(source_freeze=None, source_freeze_sha256=None))
            run.assert_not_called()

    def test_wrong_owner_case_or_output_and_boolean_returncodes_rejected(self):
        cases = (("owner", "case_id", "static_no_proxy"), ("owner", "output", "/wrong/case"),
                 ("owner", "actual_process_returncode", False), ("worker", "exit_code", False))
        for kind, field, value in cases:
            with self.subTest(kind=kind, field=field), tempfile.TemporaryDirectory() as temp:
                root, plan = self.fixture(temp)
                job = plan["jobs"][0]
                self.closed(root, job)
                output = root / job["path"]
                path = output.with_name(output.name + ".owner.actual_exit.json") if kind == "owner" else output / "worker_exit.json"
                metadata = json.loads(path.read_text()); metadata[field] = value; dump(path, metadata)
                with patch.object(subject, "ROOT", root):
                    with self.assertRaisesRegex(ValueError, "clean closure"):
                        subject.audit_closed(job, {})

    def test_actual_configs_preserve_v64_numerical_policy(self):
        new = json.loads((ROOT / "config/framework/P4_position_v65_development_canaries_v1/plan.json").read_text())
        old = json.loads((ROOT / "config/framework/P4_position_v64_development_canaries_v1/plan.json").read_text())
        expected_delta = {"schema", "information_setting", "single_change", "public_reverification_epoch_v65",
                          "v65_development_reference", "v65_preparation_only"}
        self.assertEqual([row["id"] for row in new["jobs"]], list(subject.EXPECTED))
        for a, b in zip(new["jobs"], old["jobs"]):
            self.assertEqual((a["id"], a["scene"]), (b["id"], b["scene"]))
            ca = json.loads((ROOT / a["config"]).read_text())
            cb = json.loads((ROOT / b["config"]).read_text())
            delta = {key for key in ca.keys() | cb.keys() if ca.get(key) != cb.get(key)}
            self.assertEqual(delta, expected_delta)
            self.assertEqual({key: value for key, value in ca.items() if key not in expected_delta},
                             {key: value for key, value in cb.items() if key not in expected_delta})


if __name__ == "__main__":
    unittest.main()
