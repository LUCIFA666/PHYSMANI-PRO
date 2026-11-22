"""CPU-only1220 CLI contracts; no physical runtime or actual roster access."""
from pathlib import Path
import unittest
from scripts import run_public_held_reboot_case_v1220_r10 as entry
from our_mani.multitask import public_held_reboot_case_v1220 as case

class RebootCaseEntryContracts(unittest.TestCase):
    def test_parser_preserves_explicit_case_and_resource_bindings(self):
        args=entry.parser().parse_args(['--graph','graph.json','--config','config.json',
            '--split','train','--task','task','--raw-episode-id','episode0',
            '--method','cv_geometry','--out','out','--runtime','runtime',
            '--resource-session','session.json','--resource-session-sha256','a'*64,
            '--resource-job-id','j0000','--resource-slot','0','--check'])
        self.assertTrue(args.check);self.assertEqual(args.resource_slot,0)
        self.assertEqual(args.resource_session_sha256,'a'*64)
        self.assertIsNone(args.resource_gpu_slot)

    def test_actual_entry_imports_new_case_without_changing_physical_consumer(self):
        self.assertIs(entry.case,case)
        self.assertEqual(case.WORKER,'scripts/run_public_held_reboot_case_v1220_r10.py')
        self.assertEqual(case.RUNTIME,'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
        self.assertEqual(case.GUARD_SOURCE,'our_mani/multitask/public_held_repaired_resource_guard_v1213.py')
        self.assertTrue(case.ARTIFACT_PREFIX.endswith('public_held_repaired_policy_v1206/'))

    def test_new_schema_cannot_impersonate_the_frozen1206_schema(self):
        from our_mani.multitask import public_held_repaired_case_v1206 as old
        self.assertNotEqual(case.SCHEMA,old.SCHEMA)
        self.assertNotEqual(case.CASE_SCHEMAS,old.CASE_SCHEMAS)
        self.assertEqual(case.PHYSICAL_SOURCE_OVERLAY,old.PHYSICAL_SOURCE_OVERLAY)

if __name__=='__main__':unittest.main()
