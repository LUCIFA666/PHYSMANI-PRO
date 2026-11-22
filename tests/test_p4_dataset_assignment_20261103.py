"""Reject split/parent/seed substitution before opening any episode labels."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('p4_dataset_compiler',Path(__file__).resolve().parents[1]/'scripts/compile_p4_public_dataset_20261103.py')
compiler=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(compiler)


class AssignmentTests(unittest.TestCase):
    def fixture(self,root):
        config=root/'config.json'
        config.write_text(json.dumps(dict(dataset_use='train',seed=7,task='pick_moving_target_on_the_table',
            calibration_train_episode='/recorded/train_package_compressed/task/data.h5',
            P4_collection_metadata=dict(parent_id='parentA',split='train',episode_id='episodeA',variant_id='base',
                                        parent_assigned_before_episode_generation=True))))
        return dict(config='config.json',config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    path='runtime/framework/local_mpc_20261102/caseA',parent_id='parentA',split='train',seed=7,
                    episode_id='episodeA',variant_id='base')

    def test_exact_assignment_is_bound_without_reading_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            with patch.object(compiler,'ROOT',root):
                case,path,cfg=compiler.validate_assignment(job,'train')
                self.assertFalse(case.exists());self.assertEqual(cfg['P4_collection_metadata']['parent_id'],'parentA')

    def test_relabelled_parent_seed_or_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            with patch.object(compiler,'ROOT',root):
                for key,value in [('parent_id','other'),('seed',8),('episode_id','another'),('variant_id','augmentation')]:
                    changed=deepcopy(job);changed[key]=value
                    with self.assertRaises(ValueError):compiler.validate_assignment(changed,'train')

    def test_train_cannot_be_passed_as_val_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            with patch.object(compiler,'ROOT',root):
                with self.assertRaises(ValueError):compiler.validate_assignment(job,'val')
                plan=root/'plan.json';plan.write_text(json.dumps({'jobs':[job]}))
                with self.assertRaises(ValueError):compiler.compile_dataset(plan,root/'test_output',expected_split='test')
                self.assertFalse((root/'test_output').exists())

    def test_changed_config_and_case_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            with patch.object(compiler,'ROOT',root):
                changed=deepcopy(job);changed['config_sha256']='0'*64
                with self.assertRaises(ValueError):compiler.validate_assignment(changed,'train')
                changed=deepcopy(job);changed['path']='some_other_experiment/caseA'
                with self.assertRaises(ValueError):compiler.validate_assignment(changed,'train')


if __name__=='__main__':unittest.main()
