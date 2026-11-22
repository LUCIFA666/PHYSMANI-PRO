"""Public replay projection only; never construct a forecaster or load weights."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.cube_candidate_teacher_20261104_v3.replay import _public_frame,normalize_private_labels,predict_public_episode

class ReplayProjection(unittest.TestCase):
    def test_public_geometry_and_robot_whitelist(self):
        packet=dict(timestamp=.2,Native_labels_in_input=False,close_requested=False,target=dict(available=True,position_world=[0,0,.8],confidence=.9),robot=dict(tcp_pose=[0,0,1,0,1,0,0],q=[0]*7,dq=[0]*7,finger_open_amounts=[1,1],private_native_pose='must not flow'))
        archived=dict(object_observation=dict(timestamp=.2,pose=dict(position=[0,0,.8]),points_world=[[0,0,.8],[.01,0,.8],[0,.01,.8]]),forecast='old data must not be used')
        robot,obj,_=_public_frame(packet,archived);self.assertEqual(set(robot),{'timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts'})
        self.assertTrue(np.array_equal(obj.pose.rotation,np.eye(3)));self.assertEqual(obj.confidence,.9)
        unknown=deepcopy(packet);unknown['target']=dict(available=False,position_world=None,confidence=None)
        self.assertIsNone(_public_frame(unknown,{})[1])

    def test_fake_provider_rejected_before_data_or_model(self):
        with self.assertRaises(TypeError):predict_public_episode([],[],object(),episode_id='none',expected_forecast_sha256='a'*64)

    def test_private_adapter_retains_true_clock_and_never_native_body_rotation(self):
        row=dict(parent_id='p',episode_id='e',role='fit',reset_boundary_id='e',physics_tick=0,actual_relative_physics_timestamp=0.,physics_time_absolute=12.,task_time_absolute=4.,held=False,native_success=False,requested_opening=1.,target_position_world=[1,2,3],tcp_pose_world_xyzw=[0,0,1,0,1,0,0],target_quaternion_world_xyzw='ignored')
        initial,ticks=normalize_private_labels([row],parent_id='p',episode_id='e',role='fit');self.assertEqual(initial['physics_time'],12.);self.assertNotIn('object_quaternion',initial);self.assertEqual(ticks,[])

if __name__=='__main__':unittest.main()
