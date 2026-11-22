"""Scope and owned process-local reuse, without native imports or execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pointer=ROOT/'scripts/pilot_rigid_orientation_registration_r10.py'
spec=importlib.util.spec_from_file_location('five_task_rigid_CPU',pointer)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class FiveTaskPilotChecks(unittest.TestCase):
    def test_scope_is_exact_independently_evidenced_five_and_excludes_unverified_transfer(self):
        self.assertEqual(len(pilot.TASKS),5)
        self.assertNotIn('push_moving_button',pilot.TASKS)
        self.assertNotIn('remove_cups_from_rotating_frame_high_speed',pilot.TASKS)

    def test_wrong_case_budget_fails_before_collection_or_native_import(self):
        payload=dict(version=pilot.VERSION,split='train',expected_cases=14,maximum_CPU_workers=2,cases=[])
        payload['plan_sha256']=pilot.original.digest(payload)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json';path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'exact five'):pilot.verify(path)

    def test_original_loop_verifier_restored_after_owned_child_error(self):
        original_verify=pilot.original.verify_plan;original_version=pilot.original.VERSION
        plan=dict(geometry_filter_required='r10_shared_native_TRAIN_mesh_v2',plan_sha256='0'*64)
        def fail(*args):
            self.assertIs(pilot.original.verify_plan,pilot.verify)
            self.assertEqual(pilot.original.VERSION,pilot.VERSION)
            raise RuntimeError('CPU mock before simulator construction')
        with patch.object(pilot,'verify',return_value=plan),patch.object(pilot.original,'run_case',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'CPU mock'):
                pilot.run_case('never-read',pilot.TASKS[0],'never-created')
        self.assertIs(pilot.original.verify_plan,original_verify)
        self.assertEqual(pilot.original.VERSION,original_version)


if __name__=='__main__':unittest.main()
