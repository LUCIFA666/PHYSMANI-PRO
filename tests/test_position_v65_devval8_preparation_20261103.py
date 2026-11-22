"""CPU preparation contracts; no renderer, model forward or data decoding."""
import ast
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261103 import position_v65_worker as original
from our_mani.local_mpc_20261103 import position_v65_devval8_worker as subject

ROOT=subject.ROOT
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
queue=module('scripts/run_position_v65_full_devval8_20261103.py','v65_devval8_queue_under_test')
oldqueue=module('scripts/run_position_v65_fixed_development_20261103.py','v65_canary_queue_under_test')


class V65Full8Preparation(unittest.TestCase):
    def test_complete_same_original_parents_scenes_motion_and_config(self):
        plan=json.loads(subject.CANARY_PLAN.read_text());reference=json.loads(subject.REFERENCE_PLAN.read_text())
        self.assertEqual(tuple(row['id']for row in plan['jobs']),subject.CASE_IDS)
        self.assertEqual([row['seed']for row in plan['jobs']],list(range(261103100,261103108)))
        self.assertEqual([row['scene']for row in plan['jobs']],['static']*2+['uniform']*4+['speed_change']*2)
        for job,prior in zip(plan['jobs'],reference['jobs']):
            cfg=json.loads((ROOT/job['config']).read_text());base=json.loads((ROOT/prior['config']).read_text())
            self.assertEqual(subject._reference(cfg)[0]['id'],job['id'])
            self.assertEqual({k:v for k,v in cfg.items()if k not in subject.ALLOWED_CONFIG_CHANGES},
                {k:v for k,v in base.items()if k not in subject.ALLOWED_CONFIG_CHANGES})
            self.assertTrue(cfg['public_reverification_epoch_v65']);self.assertTrue(cfg['candidate_position_measurement_v64'])
            self.assertTrue(cfg['v65_preparation_only']);self.assertFalse((ROOT/job['path']).exists())
        self.assertFalse(plan['execution_ready']);self.assertFalse(plan['execution_authorized'])
        self.assertFalse(plan['public_hold_canary_gate']['passed']);self.assertTrue(plan['evaluation_only'])

    def test_default_all8_check_calls_no_native_or_model_and_creates_no_case(self):
        with patch.object(subject.original,'make_backend',side_effect=AssertionError('Native forbidden')),redirect_stdout(io.StringIO())as output:
            self.assertEqual(queue.main([]),0)
        result=json.loads(output.getvalue());self.assertFalse(result['ready']);self.assertFalse(result['Native_started'])
        self.assertEqual(len(result['cases']),8)
        for row in result['cases']:
            self.assertEqual(row['model_forward_calls'],0)
            self.assertEqual(row['missing_admission'],['final_source_freeze_and_external_SHA256','final_case_config'])

    def test_execute_refuses_preparation_before_any_native_or_mkdir(self):
        job=json.loads(subject.CANARY_PLAN.read_text())['jobs'][0]
        with patch.object(subject.original,'environment_proof',side_effect=AssertionError('Must not reach environment')),patch.object(subject.original,'make_backend',side_effect=AssertionError('Must not reach Native')):
            with self.assertRaisesRegex(ValueError,'execution is blocked'):
                subject.execute(ROOT/job['config'],job['scene'],ROOT/job['path'])
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):queue.main(['--execute'])
        self.assertFalse((ROOT/job['path']).exists())

    def test_physical_or_parent_substitution_rejected(self):
        first=json.loads(subject.CANARY_PLAN.read_text())['jobs'][0];cfg=json.loads((ROOT/first['config']).read_text())
        for key,value in [('physics_dt',.1),('seed',261103999),('initial_target_position',[.2,.2,.775]),('uniform_velocity',[.01,0,0])]:
            with self.subTest(key=key):
                changed=deepcopy(cfg);changed[key]=value
                with self.assertRaisesRegex(ValueError,'preserve original'):subject._reference(changed)
        cfg['v65_development_reference']['case_id']='uniform_002'
        with self.assertRaises(ValueError):subject._reference(cfg)

    def test_execute_and_main_are_exact_existing_v65_physical_code(self):
        for name in ('execute','main'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(subject,name))),include_attributes=False),
                ast.dump(ast.parse(inspect.getsource(getattr(original,name))),include_attributes=False))
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(queue.audit_closed)),include_attributes=False),
            ast.dump(ast.parse(inspect.getsource(oldqueue.audit_closed)),include_attributes=False))

    def test_host_shell_only_redirect_independent_worker_paths(self):
        for stem in ('launch_local_mpc_position_v65_20261103.py','run_local_mpc_position_v65_20261103.sh','run_local_mpc_position_v65_20261103.py'):
            old=(ROOT/'scripts'/stem).read_text();new=(ROOT/'scripts'/stem.replace('position_v65_','position_v65_devval8_')).read_text()
            for a,b in [('position_v65_devval8_worker','position_v65_worker'),('run_local_mpc_position_v65_devval8_20261103','run_local_mpc_position_v65_20261103'),('launch_local_mpc_position_v65_devval8_20261103','launch_local_mpc_position_v65_20261103')]:new=new.replace(a,b)
            self.assertEqual(new,old)

    def test_queue_failed_public_gate_ack_cannot_self_authorize(self):
        plan=json.loads(subject.CANARY_PLAN.read_text())
        reports=[dict(ready=True)]*8
        with patch.object(queue,'check_plan',return_value=(plan,reports)),patch.object(queue.subprocess,'run',side_effect=AssertionError('No spawn')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):queue.main(['--execute','--acknowledge-public-hold-canary-failed'])
        self.assertEqual(set(subject.OWN_FILES),{
            'our_mani/local_mpc_20261103/position_v65_devval8_worker.py',
            'scripts/run_local_mpc_position_v65_devval8_20261103.py',
            'scripts/launch_local_mpc_position_v65_devval8_20261103.py',
            'scripts/run_local_mpc_position_v65_devval8_20261103.sh',
            'scripts/run_position_v65_full_devval8_20261103.py'})


if __name__=='__main__':unittest.main()
