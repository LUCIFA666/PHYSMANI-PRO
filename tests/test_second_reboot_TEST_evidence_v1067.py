"""CPU/temp checks for the new inherited-evidence and recovery partition."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_second_reboot_TEST_evidence_v1067_r10.py"
spec = importlib.util.spec_from_file_location("second_reboot_audit1067_test", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def inherited_fixture():
    ids = ["case_%04d" % i for i in range(1411)]
    rows = {key: {"native_success": bool(i % 2), "complete": True} for i, key in enumerate(ids)}
    evidence = [{"job_id": key, "native_success": row["native_success"],
                 "recoverable_complete": True, "errors": [], "files": {"case": {"SHA_matches": True}},
                 "evidence_runtime": "old_runtime"} for key, row in rows.items()]
    previous = {"schema": "r10_readonly_host_reboot_TEST_evidence_observation_v1054",
                "audit_complete": True, "recoverable_complete_count": 1411,
                "accepted_completed_job_ids": ids, "completed_evidence": evidence}
    control_pointer = {"path": "control.json", "sha256": "a" * 64}
    status = {"completed": deepcopy(rows), "resume_control": control_pointer}
    parent = {"completed": deepcopy(rows)}
    jobs = {key: {"imported_completed": True, "evidence_runtime": "old_runtime"} for key in ids}
    control = {"completed_job_ids": ids, "completed_key_sha256": audit.job_set_digest(ids)}
    lock = {"control": control_pointer}
    return previous, parent, status, jobs, control, lock


class InheritedEvidenceTests(unittest.TestCase):
    def test_retains_native_false_and_labels_old_file_observations(self):
        args = inherited_fixture()
        result = audit.inherited_evidence(*args)
        self.assertEqual(len(result), 1411)
        self.assertEqual(sum(r["native_success"] is False for r in result), 706)
        self.assertTrue(all(r["literal_parent_status_record_matches"] for r in result))
        self.assertTrue(all(r["evidence_inherited"] for r in result))
        self.assertTrue(all(r["physical_output_rechecked_this_audit"] is False for r in result))
        self.assertTrue(all(r["files_observations_are_from_pinned_v1054"] for r in result))
        self.assertNotIn("evidence_inherited", args[0]["completed_evidence"][0])

    def test_rejects_rewritten_parent_record(self):
        args = inherited_fixture()
        args[2]["completed"]["case_0000"]["extra"] = "changed"
        with self.assertRaisesRegex(ValueError, "rewritten"):
            audit.inherited_evidence(*args)

    def test_rejects_changed_inherited_native_outcome(self):
        args = inherited_fixture()
        args[0]["completed_evidence"][0]["native_success"] = True
        with self.assertRaisesRegex(ValueError, "Native evidence mismatch"):
            audit.inherited_evidence(*args)

    def test_rejects_int_in_place_of_native_boolean(self):
        args = inherited_fixture()
        args[0]["completed_evidence"][0]["native_success"] = 0
        with self.assertRaisesRegex(ValueError, "Native evidence mismatch"):
            audit.inherited_evidence(*args)

    def test_rejects_different_old_audit_set_and_lock(self):
        args = inherited_fixture()
        args[4]["completed_job_ids"] = args[4]["completed_job_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "keyset"):
            audit.inherited_evidence(*args)
        args = inherited_fixture()
        args[5]["control"] = {"path": "other.json", "sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "lock/control"):
            audit.inherited_evidence(*args)


class PartitionAndWriterTests(unittest.TestCase):
    def test_acceptance_is_evidence_based_and_keeps_native_false(self):
        jobs = {"a": {}, "b": {}, "c": {}, "unknown": {}}
        evidence = [{"job_id": "a", "recoverable_complete": True, "native_success": False},
                    {"job_id": "b", "recoverable_complete": True, "native_success": True},
                    {"job_id": "c", "recoverable_complete": False, "native_success": None}]
        result = audit.canonical_sets(jobs, evidence)
        self.assertEqual(result["accepted_completed_job_ids"], ["a", "b"])
        self.assertEqual(result["rejected_saved_completed_job_ids"], ["c"])
        self.assertEqual(result["remaining_job_ids"], ["c", "unknown"])
        self.assertEqual(result["saved_completed_job_ids"], ["a", "b", "c"])
        self.assertEqual(result["accepted_completed_job_ids_sha256"], audit.job_set_digest(["b", "a"]))

    def test_partition_rejects_duplicate_and_undeclared_job(self):
        jobs = {str(i): {} for i in range(1920)}
        status = {"completed": {"0": {}}, "active": ["1"], "checking": ["2"],
                  "ready_jobs": ["3"], "failed": {}, "owner_error": None, "expected": 1920}
        states, listed = audit.state_partition(status, jobs)
        self.assertEqual(len(listed), 4)
        status["active"] = ["0"]
        with self.assertRaisesRegex(ValueError, "partition"):
            audit.state_partition(status, jobs)
        status["active"] = ["undeclared"]
        with self.assertRaisesRegex(ValueError, "partition"):
            audit.state_partition(status, jobs)

    def test_report_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new_report.json"
            audit.write_exclusive_report(path, {"native_success": None})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_exclusive_report(path, {"native_success": False})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
