"""Synthetic CPU bridge checks; fixtures are not trained/Native evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from our_mani.local_mpc_p2_20261103 import servo
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from our_mani.local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel,validate_step_batch
from our_mani.local_mpc_p2_20261103.kinematics import SCHEMA,PoEKinematics
from our_mani.local_mpc_p2_20261103.reference import ReferenceLimits,reference_step
from our_mani.local_mpc_p2_20261103.servo_adapter import TorchServoTransition


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    tip=np.eye(4);tip[:3,3]=[.4,0.,.9]
    return dict(schema=SCHEMA,twist_order='v_omega',q_ref=[0.]*7,
                space_screws_vw=np.tile([0.,0.,0.,0.,0.,1.],(7,1)).tolist(),
                tcp_reference_matrix=tip.tolist(),Native_FK_validation_passed=True,
                fixture='synthetic_CPU_only_not_Native_qualification')


def model():
    return servo.JointServoResidual(snapshot(),residual_q_bound=[.01]*7,residual_dq_bound=[.1]*7)


class ServoAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(9)

    def test_vector_network_receives_two_actual_fingers_and_anchor_held(self):
        network=model();adapter=TorchServoTransition(network)
        x=dict(q=np.zeros((3,7)),dq=np.zeros((3,7)),finger_open_amounts=np.tile([.2,.8],(3,1)))
        refs=dict(q_ref=np.ones((3,7))*.1,dq_ref=np.zeros((3,7)),ddq_ref=np.zeros((3,7)),gripper_target=np.zeros(3))
        anchor=np.array([True,False,True])
        result=adapter.predict_execution(x,refs,None,.05,held_anchor=anchor)
        state=torch.tensor(np.c_[x['q'],x['dq'],x['finger_open_amounts'],anchor],dtype=torch.float32)
        reference=torch.tensor(np.c_[refs['q_ref'],refs['dq_ref'],refs['ddq_ref'],refs['gripper_target']],dtype=torch.float32)
        with torch.no_grad():expected=network(state,reference)
        np.testing.assert_array_equal(result['q'],expected['q'].numpy())
        np.testing.assert_array_equal(result['finger_open_amounts'],expected['opening'].numpy())
        self.assertFalse(np.array_equal(result['finger_open_amounts'][:,0],result['finger_open_amounts'][:,1]))
        self.assertFalse(adapter.evidence['trained'])
        with self.assertRaises(ValueError):adapter.predict_execution(dict(q=x['q'],dq=x['dq'],opening=np.ones(3)*.5),refs,None,.05,held_anchor=anchor)
        with self.assertRaises(ValueError):adapter.predict_execution(x,refs,None,.1,held_anchor=anchor)

    def test_two_tick_learned_response_carries_fingers_and_cost_mean_but_not_new_contact(self):
        class Observed(TorchServoTransition):
            def __init__(self,network):super().__init__(network);self.calls=[]
            def predict_execution(self,x,refs,latent,dt,*,held_anchor):
                result=super().predict_execution(x,refs,latent,dt,held_anchor=held_anchor)
                self.calls.append((deepcopy(x),held_anchor.copy(),deepcopy(result)))
                return result
        adapter=Observed(model());kinematics=PoEKinematics(snapshot())
        def fk(q):
            pose=kinematics(q)
            return dict(tcp_position=pose[:,:3,3],tcp_quaternion=matrix_quaternion(pose[:,:3,:3]))
        actual=dict(tcp_position=np.array([.4,0.,.9]),tcp_quaternion=np.array([0.,0.,0.,1.]),tcp_velocity=np.zeros(3),
                    object_position=np.array([.4,0.,.9]),object_quaternion=np.array([0.,0.,0.,1.]),object_linear_velocity=np.zeros(3),
                    object_angular_velocity=np.zeros(3),opening=.5,finger_open_amounts=np.array([.2,.8]),held=True,time=0.,
                    q=np.zeros(7),dq=np.zeros(7),requested_opening=0.,actuator_state={})
        wrapper=JointResidualFKExecutionModel(reference_step,ReferenceLimits([-2.]*7,[2.]*7,[2.]*7),adapter,fk)
        x,h=wrapper.initialize_batch(actual,3)
        action=dict(tcp_position=x['tcp_position'],tcp_quaternion=x['tcp_quaternion'],opening=np.ones(3),joint_goal=np.zeros((3,7)))
        result,hn,info=wrapper.step_batch(x,action,h,.1)
        validate_step_batch(x,action,result,hn,info,.1)
        self.assertEqual(len(adapter.calls),2)
        np.testing.assert_array_equal(adapter.calls[1][0]['finger_open_amounts'],adapter.calls[0][2]['finger_open_amounts'])
        self.assertTrue(adapter.calls[1][1].all())
        self.assertFalse(adapter.calls[1][0]['held'].any())
        self.assertFalse(result['held'].any())
        np.testing.assert_array_equal(result['opening'],result['finger_open_amounts'].mean(axis=-1))
        np.testing.assert_array_equal(x['finger_open_amounts'],np.tile([.2,.8],(3,1)))

    def test_checkpoint_complete_identity_strict_buffers_and_partial_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot_path=root/'synthetic_fk.json'
            snapshot_path.write_text(json.dumps(snapshot()))
            network=model()
            stats={key:getattr(network,key).tolist() for key in ('residual_q_bound','residual_dq_bound','feature_mean','feature_std')}
            contract=deepcopy(network.metadata);contract.update(trained=True,training_complete=True)
            payload=dict(state_dict=network.state_dict(),normalization=stats,model_contract=contract,
                         identity=dict(fk=dict(snapshot_path=str(snapshot_path),snapshot_sha256=sha(snapshot_path)),
                                       normalization=deepcopy(stats),source_sha256={servo.__file__:sha(servo.__file__)},formal_TEST_used=False),
                         training_complete=True,training_status=dict(status='complete',training_complete=True,epochs_requested=2,epochs_completed=2),
                         epoch=1,optimization_steps=1,formal_TEST_used=False,full_known_reference_chain_qualified=False)
            checkpoint=root/'synthetic.pt';torch.save(payload,checkpoint)
            loaded=TorchServoTransition.from_checkpoint(checkpoint,expected_sha256=sha(checkpoint))
            self.assertTrue(loaded.evidence['trained'])
            self.assertFalse(loaded.metadata['full_known_reference_chain_qualified'])
            payload['training_complete']=False;torch.save(payload,checkpoint)
            with self.assertRaisesRegex(ValueError,'complete'):TorchServoTransition.from_checkpoint(checkpoint)
            payload['training_complete']=True;payload['state_dict']['feature_mean']=torch.ones(39);torch.save(payload,checkpoint)
            with self.assertRaisesRegex(ValueError,'buffer'):TorchServoTransition.from_checkpoint(checkpoint)


if __name__=='__main__':unittest.main()
