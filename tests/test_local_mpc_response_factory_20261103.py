"""Response/action version-boundary tests; no Native or weight fitting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_control_20261103.coupled_response import FEATURE_ORDER,SCHEMA,CoupledWithFingerResponse
from our_mani.local_mpc_control_20261103.planner_factory import DiagnosticPlanner
from our_mani.local_mpc_20261102.terminal_mpc import TerminalBatchMPCPlanner
from our_mani.local_mpc_20261102 import records,worker
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture


class FactoryCaptured(RuntimeError):pass


class ResponseFactoryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)
        params=dict(schema=SCHEMA,feature_order=FEATURE_ORDER,fit_source=dict(parameter_fit_split='train'),
            normalization=dict(fit_split='train',mean=[0.]*42,scale=[1.]*42),
            hyperparameters=dict(ridge=1e-3,dt=.05,input_dimension=43,output_dimension=7),
            formal_TEST_used=False,coefficient_scaled=np.zeros((43,7)).tolist(),fixture='synthetic_CPU_only')
        self.params=self.root/'synthetic_B.json';self.params.write_text(json.dumps(params))
        self.sha=hashlib.sha256(self.params.read_bytes()).hexdigest()

    def original(self):
        joint,_,_=fixture();helper=joint.goal_helper
        raw=TerminalBatchMPCPlanner(joint.config,execution_model=joint.execution_model,action_adapter=helper.adapter)
        p=dict(controller='learned',checkpoint=dict(sha256='synthetic_CPU_only',neural_response_used=True),source_sha256={},
               diagnostic_only=True,precision_qualified=False,global_all_height_qualified=False,formal_qualification=False)
        return DiagnosticPlanner(raw,p),helper,p

    def test_three_variants_keep_limits_frequency_finger_and_unqualified_provenance(self):
        for controller,space in [('learned','joint'),('coupled','joint'),('coupled','pose')]:
            original,helper,p=self.original();execution=original.execution_model
            with patch('our_mani.local_mpc_control_20261103.response_factory.build_original',return_value=(original,helper,p)):
                out,same_helper,prov=build_planner(self.root,SimpleNamespace(),controller=controller,action_space=space,
                    parameter_path=self.params,expected_parameter_sha256=self.sha)
            self.assertIs(same_helper,helper);self.assertIs(out.execution_model.limits,execution.limits)
            self.assertIs(out.execution_model.reference_step,execution.reference_step)
            self.assertIs(out.execution_model.fk_batch,execution.fk_batch)
            self.assertIs(out.action_adapter,helper.adapter)
            self.assertTrue(prov['diagnostic_only']);self.assertFalse(prov['global_all_height_qualified'])
            self.assertFalse(prov['precision_qualified']);self.assertFalse(prov['gripper_or_contact_qualified'])
            if controller=='coupled':
                self.assertIsInstance(out.execution_model.transition_batch,CoupledWithFingerResponse)
                self.assertIs(out.execution_model.transition_batch.finger_response,execution.transition_batch)
                self.assertEqual(prov['coupled_parameters']['sha256'],self.sha)
            else:self.assertIs(out.execution_model,execution)
            self.assertEqual(type(out.planner).__name__,'JointGoalCEMPlanner' if space=='joint' else 'TerminalBatchMPCPlanner')

    def test_bad_controller_action_and_parameter_hash_rejected(self):
        with self.assertRaises(ValueError):build_planner(self.root,None,controller='unknown')
        with self.assertRaises(ValueError):build_planner(self.root,None,action_space='unknown')
        original,helper,p=self.original()
        with patch('our_mani.local_mpc_control_20261103.response_factory.build_original',return_value=(original,helper,p)):
            with self.assertRaisesRegex(ValueError,'SHA256'):
                build_planner(self.root,None,parameter_path=self.params,expected_parameter_sha256='0'*64)

    def test_worker_terminal_flag_forwarded_and_disabled_version_options_rejected_before_backend(self):
        executor=SimpleNamespace(limits=object(),config=SimpleNamespace(reference_frequency=5.))
        for enabled in (False,True):
            options=dict(use_joint_goal_planner=True,terminal_goal_cost=enabled,response_controller='learned')
            with patch('our_mani.local_mpc_control_20261103.planner_factory.build_planner',side_effect=FactoryCaptured) as call:
                with self.assertRaises(FactoryCaptured):worker.run_p1(None,executor,self.root,options,capture_rgbd=False)
            self.assertEqual(call.call_args.kwargs['terminal_config'],{'enabled':enabled})
            self.assertEqual(call.call_args.kwargs['reference_frequency'],5.)
        for options in ({'response_factory':'coupled_joint_20261103_v1'},{'action_space':'joint'}):
            with self.assertRaisesRegex(ValueError,'require use_joint_goal_planner'):
                worker.run_p1(None,executor,self.root,options,capture_rgbd=False)
        with self.assertRaisesRegex(ValueError,'unknown explicitly versioned'):
            worker.run_p1(None,executor,self.root,dict(use_joint_goal_planner=True,response_factory='bad'),capture_rgbd=False)

    def test_records_bind_version_sources_and_coupled_parameters_only_when_used(self):
        for relative in ('our_mani/local_mpc_control_20261103','our_mani/local_mpc_p2_20261103',
                         'our_mani/local_mpc_p2_20261103_v2','our_mani/local_mpc_20261102','config'):
            (self.root/relative).mkdir(parents=True,exist_ok=True)
        for name in ('__init__.py','joint_goals.py','planner_factory.py'):
            (self.root/'our_mani/local_mpc_control_20261103'/name).write_text('# synthetic CPU graph fixture\n')
        (self.root/'our_mani/local_mpc_p2_20261103/servo.py').write_text('# synthetic v1\n')
        (self.root/'our_mani/local_mpc_p2_20261103_v2/servo.py').write_text('# synthetic v2\n')
        path=self.root/'config/test.json'
        protected={
            'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt':
                '25a0e0cbd49e0e0c9e8a4671649f96a1bb565013011d728e88b2c2f3eaca0073',
            'runtime/framework/local_mpc_20261102/p2_fk_canary_v1/result.json':
                '2a3d301dbd0926760b74ba6b68fc1de85f448c402c215d0f6877e7059278600a',
            'artifacts/framework/local_mpc_20261102/P2_data_v2/manifest.json':
                'bf7187fcb47b7e1392d762915bc3a577b0d472be124c8a6ecc7dbe65bec2af1d',
            'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/source_archive.json':
                '6e004e03e3c3829a4504b5fb43b3200839721e3c190000ce7ba00f9a7bcdf4ff'}
        def fixture_sha(p):return protected.get(str(Path(p).relative_to(self.root)),'0'*64)
        for controller in ('learned','coupled'):
            path.write_text(json.dumps(dict(response_factory='coupled_joint_20261103_v1',response_controller=controller)))
            with patch.object(records,'ROOT',self.root),patch.object(records,'sha',side_effect=fixture_sha):
                graph=records.freeze_sources(path,'p1')
            self.assertIn('our_mani/local_mpc_control_20261103/response_factory.py',graph)
            self.assertIn('our_mani/local_mpc_control_20261103/joint_goal_cem.py',graph)
            self.assertIn('our_mani/local_mpc_p2_20261103_v2/servo.py',graph)
            parameter='artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json'
            self.assertEqual(parameter in graph,controller=='coupled')


if __name__=='__main__':unittest.main()
