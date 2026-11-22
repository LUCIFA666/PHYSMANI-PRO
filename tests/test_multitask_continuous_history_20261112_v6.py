from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.multitask_executor_cycle_20261112_v6.runtime import PublicContinuousHistoryPolicy
from our_mani.multitask_executor_cycle_20261112_v6.backend import TaskExecutionBackend,PreviousBackend
from our_mani.trained_world_reference_live_20261104_v1.history import PublicHistoryV1
from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation
from our_mani.multitask_executor_cycle_20261112_v6.common import ROOT

def packet(timestamp):
    return pack_public_observation(timestamp=timestamp,rgb=np.zeros((4,256,256,3),np.uint8),
        depth_m=np.ones((4,256,256),np.float32),K=np.tile(np.eye(3),(4,1,1)),
        T=np.tile(np.eye(4),(4,1,1)),camera_clips=np.tile([.01,3.],(4,1)),
        q=np.zeros(7),tcp_pose=[0,0,1,0,0,0,1])

class ContinuousHistoryContracts(unittest.TestCase):
    def policy(self):
        p=PublicContinuousHistoryPolicy.__new__(PublicContinuousHistoryPolicy)
        p.history=PublicHistoryV1();p.observed_frames=0;p.last_observation_time=None
        return p

    def test_physical_prefix_contains_last_five_actual50ms_frames(self):
        p=self.policy()
        for i in range(9):p.observe(packet(i*.05),i)
        self.assertEqual(p.history.trace()['source_frame_indices'],[4,5,6,7,8])
        self.assertTrue(np.allclose(p.history.trace()['observed_frame_dt'],.05))
        self.assertEqual(p.observed_frames,9)

    def test_sparse_action_endpoints_and_duplicate_frames_are_rejected(self):
        for timestamp,index in ((.3,1),(.05,2),(0.,1)):
            p=self.policy();p.observe(packet(0.),0)
            with self.assertRaises(ValueError):p.observe(packet(timestamp),index)
            self.assertEqual(p.observed_frames,1)

    def test_backend_captures_once_per_physics_tick_and_reuses_current_packet(self):
        value=TaskExecutionBackend.__new__(TaskExecutionBackend)
        value._public_cache=None;value._ticks=0
        now=[0.];value.clock=lambda:now[0]
        captured=[];value.on_public_frame=lambda p,i:captured.append((p['timestamp'],i))
        def tick(phase):now[0]+=.05;value._ticks+=1
        with (patch.object(PreviousBackend,'public_sample',side_effect=lambda:packet(now[0]))as sample,
              patch.object(PreviousBackend,'tick',side_effect=tick)):
            first=value.public_sample();self.assertIs(first,value.public_sample())
            value.tick('fixture')
            value.public_sample()
            self.assertEqual(sample.call_count,2)
        self.assertEqual(captured,[(0.,0),(.05,1)])

    def test_current_native_launcher_exposes_GPU_for_frozen_inference_only(self):
        script=(ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v6.sh').read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES="0"',script)
        self.assertIn('multitask_executor_cycle_20261112_v6.native_worker',script)

if __name__=='__main__':unittest.main()
