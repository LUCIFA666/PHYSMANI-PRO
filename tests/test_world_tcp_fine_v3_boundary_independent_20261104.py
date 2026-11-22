"""Independent synthetic public geometry/cleanup/pending-intent contracts."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from our_mani.world_tcp_command_warmstart_20261104_v3.geometry import nominal_cell_geometry
from our_mani.world_tcp_command_warmstart_20261104_v3.features import FrozenFinePublicSceneMemoryV3
from our_mani.world_tcp_command_warmstart_20261104_v1.features import FrozenPublicSceneMemoryV1
from our_mani.world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output,select_generic_reference
from our_mani.generic_reference_mpc_20261104_v1.contracts import retained_gripper_command


class Boundary(unittest.TestCase):
    def test_all_current_valid_points_all_views_with_cameraZ_not_unit_range(self):
        torch.set_num_threads(2)
        depths=torch.full((1,2,4,256,256),float('nan'))
        K=torch.tensor([[-100.,0,128],[0,-200.,128],[0,0,1.]])[None,None].repeat(1,4,1,1)
        T=torch.eye(4)[None,None].repeat(1,4,1,1)
        T[0,1,:3,:3]=torch.tensor([[0.,-1,0],[1.,0,0],[0,0,1.]])
        T[0,:,0,3]=torch.arange(4,dtype=torch.float32)
        for camera in range(4):
            depths[0,1,camera,10,12]=1.+camera
            depths[0,1,camera,20,22]=2.+camera
            depths[0,1,camera,25,25]=float('inf')
            depths[0,1,camera,26,26]=.1 # exactnear excluded
        clips=torch.tensor([.1,10.])[None,None].repeat(1,4,1)
        present=torch.tensor([[False,True]])
        original=(K.clone(),T.clone())
        geometry,mask=nominal_cell_geometry(depths,K,T,clips,present)
        self.assertTrue(torch.equal(geometry[:,:256],torch.zeros_like(geometry[:,:256])))
        self.assertEqual(int(mask.sum()),256)
        for camera in range(4):
            expected=[]
            for u,v,z in ((12.5,10.5,1.+camera),(22.5,20.5,2.+camera)):
                cam=torch.tensor([(u-128)/-100*z,(v-128)/-200*z,z])
                expected.append(T[0,camera,:3,:3]@cam+T[0,camera,:3,3])
            cell=geometry[0,256+camera*64]
            torch.testing.assert_close(cell[:3],torch.stack(expected).mean(0),atol=1e-6,rtol=0)
            self.assertEqual(float(cell[3]),2./1024.)
            torch.testing.assert_close(cell[4:],torch.tensor([17.5/256,15.5/256]),atol=0,rtol=0)
        self.assertTrue(torch.equal(K,original[0]));self.assertTrue(torch.equal(T,original[1]))

    def test_prepool_hook_removed_if_same_single_encoder_call_fails(self):
        # Tiny tensor-only failure fixture, not a substituted qualified encoder.
        pool=nn.AdaptiveAvgPool2d((2,2))
        fixture=object.__new__(FrozenFinePublicSceneMemoryV3)
        fixture.model=SimpleNamespace(camera_encoder=nn.Sequential(pool))
        def fail_encoder(self,**kwargs):
            pool(torch.zeros(4,512,16,16))
            raise RuntimeError('synthetic_encoder_failure_after_readonly_capture')
        with patch.object(FrozenPublicSceneMemoryV1,'encode',fail_encoder):
            with self.assertRaisesRegex(RuntimeError,'synthetic_encoder_failure'):
                FrozenFinePublicSceneMemoryV3.encode(fixture,rgbd=torch.zeros(1,5,4,4,256,256),
                    robot_history=torch.zeros(1,5,15),lengths=torch.tensor([1]),
                    K=torch.eye(3)[None,None].repeat(1,4,1,1),camera_to_world=torch.eye(4)[None,None].repeat(1,4,1,1),
                    near_far=torch.tensor([.1,10.])[None,None].repeat(1,4,1),frame_indices=torch.tensor([[0,-1,-1,-1,-1]]))
        self.assertEqual(len(pool._forward_pre_hooks),0);self.assertEqual(len(pool._forward_hooks),0)

    def test_v3_raw_to_typedbank_to_frozen_generic_keeps_gripper_pending(self):
        raw=dict(world_tcp_positions=np.arange(24,dtype=float).reshape(1,8,3)/100,
            world_tcp_rotations=np.tile(np.eye(3),(1,8,1,1)),opening_intent=np.zeros((1,8)),
            opening_logits=np.full((1,8),-3.),proposal_logits=np.array([[0,0,0,0,0,0,2,1.]]),
            proposal_score_semantics='expert_command_demonstration_mixture_not_task_success',
            output_semantics='expert_command_intent_not_observed_state',handoff_tau_seconds=None,
            physical_handoff_qualified=False,calibrated_success_probability=False)
        bank=bank_from_v3_model_output(raw)
        raw['world_tcp_positions'].fill(100.) # immutable publicbank does not alias model buffers
        ref=select_generic_reference(bank,timestamp=12.3,policy_sha256='c'*64,e_sha256='d'*64,
            selection_rule='highest_demo_likelihood')
        self.assertEqual(ref.selected_index,6);self.assertAlmostEqual(ref.position[0],.18)
        self.assertEqual(ref.opening_intent,0.)
        self.assertEqual(ref.opening_intent_condition,'after_arm_sequence_unresolved_not_immediate_actuator_command')
        self.assertEqual(retained_gripper_command({'actuator_state':{'gripper_target':1.}}),1.)
        with self.assertRaises(ValueError):bank_from_v3_model_output(dict(raw,Native_held=False))
        with self.assertRaises(ValueError):bank_from_v3_model_output(dict(raw,handoff_tau_seconds=.5))
        with self.assertRaises(ValueError):bank_from_v3_model_output(dict(raw,calibrated_success_probability=True))


if __name__=='__main__':unittest.main()
