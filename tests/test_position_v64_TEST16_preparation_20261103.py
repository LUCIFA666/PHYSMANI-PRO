"""Reserved TEST metadata/source guards; no real TEST materialization or data."""
import ast
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261103 import position_v64_test16_protocol as protocol
from our_mani.local_mpc_20261103 import position_v64_test16_worker as worker
from our_mani.local_mpc_20261103 import position_v64_devval8_worker as baseline
from our_mani.local_mpc_control_20261103 import native_task_evaluation as judge
from scripts import prepare_position_v64_TEST16_evaluation_20261103 as prepare
from scripts import launch_local_mpc_position_v64_test16_20261103 as host
from scripts import run_position_v64_frozen_TEST16_evaluation_20261103 as queue


class ReservedTESTPreparation(unittest.TestCase):
    def test_original16_partition_exact_and_failed_gate_not_promoted(self):
        method,plan=protocol.metadata()
        self.assertEqual(plan['partition']['sha256'],'c917d28f7889258b8961a3e462bf511dd9819d1ee01df9f63951998b15c59234')
        self.assertEqual([j['seed']for j in plan['jobs']],list(range(261103500,261103516)))
        self.assertEqual([j['scene']for j in plan['jobs']].count('static'),2)
        self.assertEqual([j['scene']for j in plan['jobs']].count('uniform'),8)
        self.assertEqual([j['scene']for j in plan['jobs']].count('speed_change'),6)
        self.assertFalse(plan['public_hold_canary_gate']['passed'])
        self.assertFalse(method['public_hold_canary_gate']['original_V59_partition_gate_satisfied'])
        self.assertTrue(plan['evaluation_only']);self.assertFalse(plan['formal_pose_real_time_force_qualified'])
        self.assertFalse(protocol.CONFIG_DIRECTORY.exists())

    def test_four_actual_default_checks_do_not_spawn_or_materialize(self):
        with patch.object(host.subprocess,'Popen',side_effect=AssertionError('no child')),\
             patch.object(queue.subprocess,'run',side_effect=AssertionError('no Native')):
            for main in (prepare.main,worker.main,host.main,queue.main):
                stream=io.StringIO()
                with redirect_stdout(stream):self.assertEqual(main([]),0)
                report=json.loads(stream.getvalue());self.assertFalse(report['Native_started'])
                self.assertFalse(report['materialized']);self.assertEqual(report['model_forward_calls'],0)
        self.assertFalse(protocol.CONFIG_DIRECTORY.exists())

    def test_in_memory_recipes_preserve_controller_and_only_assign_fixed_motion(self):
        method,plan=protocol.metadata();base=protocol.load(protocol.checked(method['base_config']))
        fields=('physics_dt','control_dt','decision_ticks','horizon_steps','horizon_seconds',
            'cem_sequences_total','cem_iteration_sequences','gripper_velocity','joint_reference_acceleration',
            'joint_reference_jerk','active_visual_settings','public_grasp_settings','candidate_transformer_checkpoint',
            'candidate_transformer_checkpoint_sha256','position_refiner_checkpoint','position_refiner_checkpoint_sha256',
            'candidate_pi_symmetry_representative','task_e_source','calibration_train_episode')
        for job in plan['jobs']:
            value=protocol.expected_config(job);assignment=job['assignment']
            for field in fields:self.assertEqual(value.get(field),base.get(field))
            self.assertEqual(value['initial_target_position'][:2],assignment['initial_xy'])
            self.assertEqual(value['initial_target_position'][2],base['initial_target_position'][2])
            self.assertEqual(value['uniform_velocity'],assignment['initial_velocity_m_s'])
            self.assertEqual(value['changed_velocity'],assignment['changed_velocity_m_s'])
            self.assertEqual(value['seed'],assignment['seed'])
            self.assertEqual(value['P4_collection_metadata']['parent_id'],assignment['parent_id'])
            for key in ('dataset_use','data_split','evaluation_split'):self.assertEqual(value[key],'test')
        self.assertFalse(protocol.CONFIG_DIRECTORY.exists())

    def test_no_materialization_without_later_explicit_root_authorization(self):
        with TemporaryDirectory(dir=protocol.ROOT/'runtime/framework/local_mpc_20261102')as temp:
            path=Path(temp)/'not_authorization.json';path.write_text(json.dumps({'authorize_materialization':False}))
            with self.assertRaises(ValueError):protocol.materialize(path,protocol.sha(path))
        self.assertFalse(protocol.CONFIG_DIRECTORY.exists())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):prepare.main(['--materialize'])
            for main in (worker.main,host.main,queue.main):
                with self.assertRaises(SystemExit):main(['--execute'])
        self.assertFalse(protocol.CONFIG_DIRECTORY.exists())

    def test_frozen_physical_execute_and_same_renderer_shell_unmodified(self):
        def get(module):return next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute')
        self.assertEqual(ast.dump(get(worker),include_attributes=False),ast.dump(get(baseline),include_attributes=False))
        new=(protocol.ROOT/'scripts/run_local_mpc_position_v64_test16_20261103.sh').read_text().replace('local_mpc_position_v64_test16_20261103','local_mpc_position_v64_devval8_20261103')
        old=(protocol.ROOT/'scripts/run_local_mpc_position_v64_devval8_20261103.sh').read_text()
        self.assertEqual(new,old)
        self.assertIs(worker.original.make_backend,baseline.original.make_backend)
        self.assertIs(worker.original.make_executor,baseline.original.make_executor)

    def test_original_development_judge_TEST_guard_still_rejects(self):
        with self.assertRaises(ValueError):
            judge._reject_test_source(Path('/synthetic/test/only'),{'config':{'evaluation_split':'test'}})
        method,_=protocol.metadata();freeze=protocol.load(protocol.checked(method['validation_source_freeze']))
        self.assertEqual(len(freeze['files_sha256']),234)
        for relative,expected in freeze['files_sha256'].items():
            self.assertEqual(protocol.sha(protocol.ROOT/relative),expected)


if __name__=='__main__':unittest.main()
