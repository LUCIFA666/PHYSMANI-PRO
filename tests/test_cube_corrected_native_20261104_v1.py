"""No-simulator worker admission/source/archive contracts."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

from our_mani.cube_corrected_native_20261104_v1 import contracts as c
from our_mani.cube_corrected_native_20261104_v1 import worker as w
from our_mani.local_mpc_20261102.records import archive_sources

PLAN=c.ROOT/'config/framework/cube_corrected_native_20261104_v1/plan.json'


class CorrectedNativeContracts(unittest.TestCase):
    def test_fixed_cases_and_original_control_values(self):
        plan=c.validate_plan(PLAN);reference=c.read(c.REFERENCE)
        self.assertEqual(plan['profiles_order'],['corrected_legacy','corrected_matched'])
        self.assertEqual([(x['id'],x['task'],x['seed'])for x in plan['cases']],list(c.ASSIGNMENTS))
        changed={'schema','task','seed','variation','simulation_seconds','dataset_use','formal_evaluation',
            'calibration_train_episode','candidate_transformer_checkpoint','candidate_transformer_checkpoint_sha256',
            'candidate_model_freeze_manifest','P4_collection_metadata','native_task_source_sha256','native_task_scene_sha256'}
        for profile in c.PROFILES:
            config=c.case_config(plan,plan['cases'][0],profile)
            for key,value in reference.items():
                if key not in changed:self.assertEqual(config[key],value,key)
            self.assertEqual(config['simulation_seconds'],8.)
            self.assertFalse(config['reobserve_enabled'])
        self.assertTrue(plan['preparation_only']);self.assertFalse(plan['execution_authorized'])

    def test_all_eight_default_checks_remain_nonadmitted_without_model_load(self):
        plan=c.validate_plan(PLAN)
        for profile in c.PROFILES:
            for case in plan['cases']:
                report=w.check(PLAN,case['id'],profile)
                self.assertFalse(report['ready']);self.assertFalse(report['Native_started'])
                self.assertEqual(report['NN_forward_calls'],0)
                self.assertFalse(report['strict_model_constructors_checked'])
                self.assertIn('Root_final_plan',report['missing'])
                self.assertIn('complete_fixed_profile_checkpoint_set',report['missing'])

    def test_execution_rejects_before_output_and_native_when_pending(self):
        with tempfile.TemporaryDirectory()as directory:
            output=Path(directory)/'must_not_exist'
            with self.assertRaises(ValueError):w.execute(PLAN,'ordinary_00','corrected_legacy',output,
                source_freeze=Path(directory)/'no_freeze',source_freeze_sha256='a'*64)
            self.assertFalse(output.exists())

    def test_repo_absolute_alias_normalizes_and_archive_stays_under_output(self):
        relative='our_mani/cube_corrected_native_20261104_v1/contracts.py'
        absolute=c.ROOT/relative;digest=c.sha(absolute)
        mapping=c.normalize_repo_sources({relative:digest,str(absolute):digest})
        self.assertEqual(mapping,{relative:digest})
        with tempfile.TemporaryDirectory()as directory:
            archive_sources(directory,mapping)
            archived=Path(directory)/'source_snapshot'/relative
            self.assertTrue(archived.is_file());self.assertEqual(c.sha(archived),digest)
            self.assertTrue(archived.resolve().is_relative_to(Path(directory).resolve()))

    def test_complete_code_map_relative_and_external_native_seven_separate(self):
        repo=c.code_sources_for_plan(PLAN);native=c.native_sources_for_plan(PLAN)
        self.assertEqual(len(native),7)
        self.assertTrue(all(not Path(p).is_absolute()and '..'not in Path(p).parts for p in repo))
        self.assertTrue(all(c.sha(c.ROOT/p)==h for p,h in repo.items()))
        self.assertTrue(all(Path(p).is_absolute()and not Path(p).is_relative_to(c.ROOT)for p in native))
        for p,h in native.items():
            with self.assertRaises(ValueError):c.normalize_repo_sources({p:h})
        self.assertIn('our_mani/cube_corrected_live_20261104_v1/matched_producer.py',repo)
        self.assertIn('our_mani/local_mpc_20261102/native.py',repo)

    def test_no_complete_freeze_until_matched_checkpoint_is_real(self):
        with self.assertRaises(ValueError):c.sources_for_plan(PLAN)
        with self.assertRaises(ValueError):c.case_config(c.validate_plan(PLAN),c.validate_plan(PLAN)['cases'][0],'corrected_matched_reobserve')
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit)as error:w.main(['--plan',str(PLAN),'--case-id','ordinary_00','--profile','corrected_matched_reobserve'])
        self.assertEqual(error.exception.code,2)

    def test_current_runtime_and_correct_global_pair_only(self):
        shell=(c.ROOT/'scripts/run_cube_corrected_native_20261104_v1.sh').read_text()
        self.assertIn('source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh',shell)
        self.assertIn('exec xvfb-run -a python -B',shell)
        self.assertIn('/runtime/framework/fixed_TRAIN104_geometry_native.lock',shell)
        self.assertIn('/runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock',shell)
        self.assertNotIn('run_frozen_system_environment',shell)
        self.assertNotIn('/runtime/framework/local_mpc_20261102/fixed_TRAIN104_geometry_native.lock',shell)


if __name__=='__main__':unittest.main()
