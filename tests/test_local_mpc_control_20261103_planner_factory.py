"""Synthetic factory-boundary checks; no Native or optimization training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.planner_factory import build_planner,validate_node_state
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_p2_20261103.kinematics import SCHEMA
from our_mani.local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel,SimpleBatchExecutionModel


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state():
    return dict(tcp_position=np.array([.4,0.,.9]),tcp_quaternion=np.array([0.,0.,0.,1.]),tcp_velocity=np.zeros(3),
        object_position=np.array([.4,0.,.84]),object_quaternion=np.array([0.,0.,0.,1.]),object_linear_velocity=np.zeros(3),
        object_angular_velocity=np.zeros(3),opening=.5,finger_open_amounts=np.array([.2,.8]),held=False,time=0.,
        q=np.zeros(7),dq=np.zeros(7),requested_opening=1.,finger_positions=np.array([.008,.032]),finger_velocities=np.zeros(2),
        servo_target_positions=np.zeros(7),gripper_actuator_history=dict(previous_positions=[None,None]),
        actuator_state=dict(reference_valid=False,reference_positions=None,reference_velocities=None,
                            reference_accelerations=None,target_joints=None,gripper_target=1.))


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)
        pose=np.eye(4);pose[:3,3]=[.4,0.,.9]
        snapshot=dict(schema=SCHEMA,twist_order='v_omega',q_ref=[0.]*7,
                      space_screws_vw=np.tile([0.,0.,0.,0.,0.,1.],(7,1)).tolist(),
                      tcp_reference_matrix=pose.tolist(),Native_FK_validation_passed=True,
                      fixture='synthetic_CPU_only_not_Native_qualification')
        self.snapshot=self.root/'fk.json';self.snapshot.write_text(json.dumps(snapshot))
        self.validation=self.root/'validation.json';self.validation.write_text(json.dumps(dict(complete=True,passed=True,
            Native_controlled_physics_ticks=0,restoration_checked=True,maximum_TCP_position_error_m=0.,maximum_rotation_error_rad=0.)))
        self.checkpoint=self.root/'synthetic.pt';self.checkpoint.write_bytes(b'synthetic CPU boundary fixture; not model weights')
        self.limits=JointLimits([-2.]*7,[2.]*7,[2.]*7)
        self.evidence=dict(trained=True,training_complete=True,selected_epoch=1,full_known_reference_chain_qualified=False,
            identity=dict(fk=dict(snapshot_path=str(self.snapshot),snapshot_sha256=sha(self.snapshot),
                                  validation_path=str(self.validation),validation_sha256=sha(self.validation))))
        self.transition=SimpleNamespace(evidence=self.evidence,predict_execution=lambda *args,**kwargs:None)

    def factory(self,**kwargs):
        with patch('our_mani.local_mpc_p2_20261103_v2.servo_adapter.TorchServoTransition.from_checkpoint',return_value=self.transition):
            return build_planner(self.root,self.limits,checkpoint_path=self.checkpoint,expected_checkpoint_sha256=sha(self.checkpoint),**kwargs)

    def test_default_simple_and_learned_share_adapter_terminal_and_unqualified_flags(self):
        for mode,cls in [('simple',SimpleBatchExecutionModel),('learned',JointResidualFKExecutionModel)]:
            planner,helper,p=self.factory(controller=mode)
            self.assertIsInstance(planner.execution_model,cls)
            self.assertIs(planner.action_adapter,helper.adapter)
            self.assertTrue(planner.terminal_config.enabled)
            self.assertTrue(p['diagnostic_only']);self.assertFalse(p['formal_qualification'])
            self.assertFalse(p['global_all_height_qualified']);self.assertFalse(p['precision_qualified'])
            self.assertFalse(p['posthoc_native_height_subset_is_formal_qualification'])
            self.assertFalse(p['Native_started']);self.assertFalse(p['formal_TEST_used'])
            self.assertEqual(p['joint_goal_config']['position_tolerance_m'],.002)
            self.assertEqual(p['joint_goal_config']['angle_tolerance_rad'],.02)

    def test_all_actual_finger_and_executor_dimensions_preserved_without_mean_substitution(self):
        initial=state();before=deepcopy(initial);value=validate_node_state(initial)
        for key in ('finger_open_amounts','finger_positions','finger_velocities','servo_target_positions'):
            np.testing.assert_array_equal(value[key],initial[key])
        self.assertIn('gripper_actuator_history',value)
        np.testing.assert_array_equal(initial['q'],before['q'])
        del initial['finger_open_amounts']
        with self.assertRaisesRegex(ValueError,'finger'):validate_node_state(initial)
        initial=state();initial['opening']=.6
        with self.assertRaisesRegex(ValueError,'mean'):validate_node_state(initial)
        initial=state();del initial['actuator_state']
        with self.assertRaisesRegex(ValueError,'executor'):validate_node_state(initial)

    def test_bad_hash_and_invalid_fk_gate_before_ready(self):
        with self.assertRaisesRegex(ValueError,'SHA'):
            build_planner(self.root,self.limits,checkpoint_path=self.checkpoint,expected_checkpoint_sha256='0'*64)
        r=json.loads(self.validation.read_text());r['Native_controlled_physics_ticks']=1
        self.validation.write_text(json.dumps(r));self.evidence['identity']['fk']['validation_sha256']=sha(self.validation)
        with self.assertRaisesRegex(ValueError,'zero-step'):self.factory()

    def test_joint_model_reference_uses_declared_frequency_and_actual_state_seed(self):
        planner,_,p=self.factory(controller='learned',reference_frequency=4.)
        x,h=planner.execution_model.initialize_batch(state(),2)
        out=planner.execution_model.reference_step(x['q'],x['dq'],np.full((2,7),.1),h,
            planner.execution_model.limits,dt=.05)
        from our_mani.local_mpc_p2_20261103.reference import reference_step
        expected=reference_step(x['q'],x['dq'],np.full((2,7),.1),h,planner.execution_model.limits,dt=.05,frequency=4.)
        np.testing.assert_array_equal(out['q_ref'],expected['q_ref'])
        self.assertEqual(p['reference_frequency'],4.)


if __name__=='__main__':unittest.main()
