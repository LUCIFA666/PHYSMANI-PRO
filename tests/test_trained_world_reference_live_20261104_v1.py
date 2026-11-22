"""Synthetic public input/history contracts; no trained NN/model construction."""
import unittest
import numpy as np
from our_mani.trained_world_reference_live_20261104_v1.history import PublicHistoryV1
from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation
from our_mani.world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output,select_generic_reference
from our_mani.generic_reference_mpc_20261104_v1.contracts import WorldTCPReference,retained_gripper_command


def packet(time):
    rgb=np.zeros((4,256,256,3),np.uint8);rgb[...,1]=255;rgb[...,2]=128
    return pack_public_observation(timestamp=time,rgb=rgb,depth_m=np.full((4,256,256),1.234,np.float32),
        K=np.tile(np.diag([-200.,-200.,1.]),(4,1,1)),T=np.tile(np.eye(4),(4,1,1)),
        camera_clips=np.tile([.01,4.],(4,1)),q=np.arange(7)/10,tcp_pose=[.1,.2,.3,0,0,0,1])


class PublicInputs(unittest.TestCase):
    def test_exact_rgb_depth_calibration_and_missing_previous_padding(self):
        history=PublicHistoryV1();p=packet(1.25);history.append(p,frame_index=47);a=history.arrays()
        np.testing.assert_array_equal(a['rgbd'][0,0,:,0],-np.ones((4,256,256),np.float32))
        np.testing.assert_array_equal(a['rgbd'][0,0,:,1],np.ones((4,256,256),np.float32))
        np.testing.assert_array_equal(a['rgbd'][0,0,:,3],p['depth_m'])
        self.assertEqual(a['rgbd'][0,0,0,2,0,0],np.float32(128)/np.float32(127.5)-np.float32(1))
        self.assertEqual(a['lengths'].tolist(),[1]);self.assertEqual(a['frame_indices'].tolist(),[[47,-1,-1,-1,-1]])
        self.assertTrue(np.all(a['rgbd'][:,1:]==0));self.assertEqual(float(a['robot_history'][0,0,-1]),0.)
        np.testing.assert_array_equal(a['K'][0],p['K']);np.testing.assert_array_equal(a['camera_to_world'][0],p['T'])
        np.testing.assert_allclose(a['near_far'][0],p['camera_clips'],atol=1e-9,rtol=1e-7)

    def test_actual_history_rollover_consecutive_unique_and_no_fake_dt(self):
        history=PublicHistoryV1();times=[1.,1.05,1.16,1.2,1.33,1.4]
        for index,t in enumerate(times):history.append(packet(t),frame_index=47+index)
        self.assertEqual(history.arrays()['frame_indices'].tolist(),[[48,49,50,51,52]])
        self.assertEqual(history.trace()['actual_public_timestamps'],times[1:])
        self.assertFalse(history.trace()['model_uses_physical_dt_channel'])
        with self.assertRaises(ValueError):history.append(packet(1.4),frame_index=53)
        with self.assertRaises(ValueError):history.append(packet(1.5),frame_index=54)
        self.assertEqual(history.arrays()['frame_indices'].tolist(),[[48,49,50,51,52]])

    def test_missing_clips_private_keys_or_changed_history_calibration_refused(self):
        history=PublicHistoryV1();p=packet(0.);bad=dict(p);bad.pop('camera_clips')
        with self.assertRaises(ValueError):history.append(bad,frame_index=0)
        with self.assertRaises(ValueError):history.append(dict(p,Native_held=False),frame_index=0)
        history.append(p,frame_index=0);changed=packet(.05);changed['K']=changed['K'].copy();changed['K'][0,0,0]-=.01
        with self.assertRaises(ValueError):history.append(changed,frame_index=1)
        self.assertEqual(history.arrays()['lengths'].tolist(),[1])
        history.clear();history.append(changed,frame_index=1)
        self.assertEqual(history.arrays()['lengths'].tolist(),[1])

    def test_existing_generic_reference_ABI_keeps_future_close_pending(self):
        raw=dict(world_tcp_positions=np.zeros((1,8,3)),world_tcp_rotations=np.tile(np.eye(3),(1,8,1,1)),
            opening_intent=np.zeros((1,8)),opening_logits=np.full((1,8),-1.),proposal_logits=np.arange(8,dtype=float)[None],
            output_semantics='expert_command_intent_not_observed_state',handoff_tau_seconds=None,
            physical_handoff_qualified=False,calibrated_success_probability=False,
            proposal_score_semantics='expert_command_demonstration_mixture_not_task_success')
        ref=select_generic_reference(bank_from_v3_model_output(raw),timestamp=1.25,policy_sha256='a'*64,e_sha256='b'*64,
            selection_rule='highest_demo_likelihood')
        self.assertIs(type(ref),WorldTCPReference);self.assertEqual(ref.selected_index,7)
        self.assertIn('not_immediate_actuator_command',ref.opening_intent_condition)
        self.assertEqual(retained_gripper_command({'actuator_state':dict(gripper_target=1.)}),1.)


if __name__=='__main__':unittest.main()
