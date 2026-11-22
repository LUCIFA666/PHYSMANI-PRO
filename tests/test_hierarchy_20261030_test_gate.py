"""Synthetic-only TEST gate checks; no actual TEST manifest or recording read."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.hierarchy_20261030 import evaluation as base
from our_mani.hierarchy_20261030 import evaluation_test as extension


class TestExtensionGateTests(unittest.TestCase):
    def test_missing_gate_rejects_before_any_test_file_is_resolved_or_read(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("forbidden pre-gate read")):
            with self.assertRaisesRegex(base.EvaluationBlocked, "before any TEST read"):
                extension.prepare_test_plan(Path("/tmp/UNOPENED_TEST_ROOT"), None, None,
                    original_manifest_path=Path("/UNOPENED_TEST/manifest.json"))

    def test_each_entry_validates_gate_before_opening_a_test_plan(self):
        with patch.object(extension, "validate_val_gate", side_effect=base.EvaluationBlocked("gate closed")) as gate:
            with patch.object(Path, "read_text", side_effect=AssertionError("TEST plan opened before gate")):
                for method in (extension.cpu_preflight_test, extension.audit_test):
                    with self.assertRaisesRegex(base.EvaluationBlocked, "gate closed"):
                        method(Path("/tmp/TEST_SYNTH"), Path("/UNOPENED_TEST/plan.json"), None, None)
                with self.assertRaisesRegex(base.EvaluationBlocked, "gate closed"):
                    extension.launch_test_case(Path("/tmp/TEST_SYNTH"), Path("/UNOPENED_TEST/plan.json"), "first", None, None)
            self.assertEqual(gate.call_count, 3)

    @staticmethod
    def release_fixture(root, complete=True):
        freeze_path = root / base.FAMILY_ROOT / "freeze.json"
        freeze_path.parent.mkdir(parents=True)
        freeze_path.write_text("{}")
        path = root / base.ARTIFACTS / "validation/Native_VAL_release.json"
        path.parent.mkdir(parents=True)
        release = {"schema": base.SCHEMA + "_Native_VAL_release", "split": "validation", "complete": complete,
            "expected_cases": 77, "actual_process_exits_audited": 77,
            "freeze": base.pointer(root, freeze_path), "cases": [{"case_key": str(i)} for i in range(77)],
            "offline_validation_is_native_release": False}
        path.write_text(json.dumps(release))
        return freeze_path, path, release

    def test_incomplete_or_different_freeze_val_release_blocks_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, release_path, release = self.release_fixture(root, complete=False)
            with patch.object(base, "load_freeze", return_value={"protocol": {"inference_device": "cpu"}}):
                with self.assertRaisesRegex(base.EvaluationBlocked, "same-freeze complete actual Native VAL77"):
                    extension.prepare_test_plan(root, freeze, release_path, original_manifest_path=Path("/UNOPENED_TEST/manifest.json"))
                release["complete"] = True
                release["freeze"]["sha256"] = "9" * 64
                release_path.write_text(json.dumps(release))
                with self.assertRaisesRegex(base.EvaluationBlocked, "same-freeze complete actual Native VAL77"):
                    extension.prepare_test_plan(root, freeze, release_path, original_manifest_path=Path("/UNOPENED_TEST/manifest.json"))
            self.assertFalse((root / extension.ARTIFACTS).exists())

    def test_all_77_actual_exits_are_checked_before_test_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_path, release_path, release = self.release_fixture(root)
            val_plan_path = release_path.with_name("plan.json")
            val_plan_path.write_text("{}")
            cases = [{"case_key": str(i), "task": "synthetic_only", "raw_episode_id": str(i)} for i in range(77)]
            plan = {"split": "validation", "expected_cases": 77, "freeze": base.pointer(root, freeze_path), "cases": cases}
            release["plan"] = base.pointer(root, val_plan_path)
            release["cases"] = []
            for index, case in enumerate(cases):
                runtime = root / base.RUNTIME / "validation/cases" / str(index)
                runtime.mkdir(parents=True)
                terminal_path = runtime / "terminal.json"
                terminal_path.write_text(json.dumps({"complete": True, "natural_completion": True,
                    "administrative_stop": False, "infrastructure_failed": False, "category": "NativeFalse",
                    "freeze": plan["freeze"], "plan": release["plan"], "case": case}))
                exit_path = runtime / "parent_exit.json"
                exit_path.write_text(json.dumps({"actual_process_exit_observed": True, "returncode": 0,
                    "timed_out": False, "case_complete": True, "category": "NativeFalse", "terminal": base.pointer(root, terminal_path)}))
                release["cases"].append({"case_key": str(index), "exit": base.pointer(root, exit_path), "terminal": base.pointer(root, terminal_path)})
            release_path.write_text(json.dumps(release))
            with patch.object(base, "load_freeze", return_value={"protocol": {"inference_device": "cpu"}}), patch.object(base, "verify_plan", return_value=(plan, {})):
                proof = extension.validate_val_gate(root, freeze_path, release_path)
                self.assertEqual(proof["actual_VAL_exits_revalidated"], 77)
                final = root / base.RUNTIME / "validation/cases/76/parent_exit.json"
                record = json.loads(final.read_text()); record["returncode"] = 1
                final.write_text(json.dumps(record))
                release["cases"][-1]["exit"] = base.pointer(root, final)
                release_path.write_text(json.dumps(release))
                with self.assertRaisesRegex(base.EvaluationBlocked, "incomplete, administrative"):
                    extension.prepare_test_plan(root, freeze_path, release_path,
                        original_manifest_path=Path("/UNOPENED_TEST/manifest.json"))

    def test_pure_roster_requires_original160_and_verified_19_files(self):
        original, mapping = [], []
        for task in base.TASKS:
            for index in range(40):
                row = {"task": task, "raw_episode_id": "episode" + str(index), "path": "/SYNTHETIC_UNOPENED/" + task + "/" + str(index)}
                original.append(row)
                mapping.append(dict(row, mapping_status="verified", source_required_files={name: {} for name in extension.OFFICIAL_FILES}))
        source = {"episodes": original}
        verified = {"split": "test", "mapping_status": "verified", "episodes": mapping}
        rows = extension.select_test_rows(source, verified)
        self.assertEqual(len(rows), 160)
        self.assertEqual(len(extension.OFFICIAL_FILES), 19)
        mapping[0]["source_required_files"].pop("target_state.pkl")
        with self.assertRaisesRegex(base.EvaluationBlocked, "19-file"):
            extension.select_test_rows(source, verified)


if __name__ == "__main__":
    unittest.main()
