"""v4.1 real observer + runner dispatch CPU contracts, no Native/fit."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionEvidence
from our_mani.local_mpc_control_20261103.public_grasp_v41 import (
    ActiveVisualPublicBridge,ObservedPublicGraspV41Backend,run_public_grasp_v41)
from tests.test_local_mpc_active_visual_attachment_20261103 import sample,h
from tests.test_local_mpc_public_grasp_20261103 import RawBackend
from tests.test_local_mpc_public_pregrasp_20261103 import Executor


class V41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.wrapper=ActiveVisualPublicBridge()
    def setUp(self):self.wrapper.reset()
    def tracked(self,obs):
        target,_=sample(obs['timestamp']);return target,{'backend':'explicit_synthetic_CPU_fixture'}

    def test_observer_records_recovery_on_each_real50ms_without_extra_sensor_or_physics(self):
        raw=RawBackend(budget=6)
        with TemporaryDirectory() as directory:
            observer=ObservedPublicGraspV41Backend(raw,Path(directory)/'public',public_bridge=self.wrapper,save_frames=False)
            with patch.object(self.wrapper.tracker,'observe',side_effect=self.tracked) as spy,\
                 patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
                observer.sample();observer.tick('one');observer.tick('two')
            self.assertEqual(observer.recovery.last_timestamp,.1)
            self.assertEqual(raw.ticks,2);self.assertEqual(raw.captures,3);self.assertEqual(spy.call_count,3)
            self.assertFalse(observer.receipt()['recovery']['historical_confirmation_is_current_held'])

    def test_runner_first_dispatch_with_confirmed_history_unknown_keeps_closed_request_and_actual_q(self):
        raw=RawBackend(budget=6);raw.requested=0.
        with TemporaryDirectory() as directory:
            observer=ObservedPublicGraspV41Backend(raw,Path(directory)/'public',public_bridge=self.wrapper,save_frames=False)
            executor=Executor(observer);executor.h=h(0.,target=0.)
            with patch.object(self.wrapper.tracker,'observe',side_effect=self.tracked),\
                 patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})):
                observer.sample()
                # Explicit synthetic history fixture, never hidden Native-held:
                # active current state is still unknown and no lift is planned.
                robot=observer.latest_robot;proof=PublicMotionEvidence('attachment_confirmed',0.,
                    ('rgbd_relative_pose_tracking','public_robot_pose'),Pose([0.,0.,-.025]))
                observer.recovery.observe(dict(timestamp=0.,mode='ATTACHED_PUBLIC',available=True,controlled_mode_available=True,
                    attachment_estimate=dict(timestamp=0.,protocol='active_vertical_visual_comotion_v4',attempt_id=1,
                    attempt_active=True,mode='ATTACHED_PUBLIC',evidence=proof)),robot)
                with patch('our_mani.local_mpc_control_20261103.public_grasp_v41.build_response_planner',
                    return_value=(SimpleNamespace(),object(),{})):
                    report=run_public_grasp_v41(observer,executor,directory,dict(public_grasp_active_visual_v41=True,simulation_seconds=.3))
            self.assertEqual(report['cycles'],3);self.assertFalse(report['public_lift_hold_observed'])
            self.assertTrue(all(c['opening']==0. for c in executor.commands))
            self.assertTrue(all('POST_CONFIRMED_UNKNOWN_HOLD' in c['phase'] for c in executor.commands))
            self.assertTrue(all(np.array_equal(c['q'],np.zeros(7)) for c in executor.commands))
            self.assertFalse(report['Native_success_exit_used']);self.assertFalse(report['recovery_final']['historical_confirmation_is_current_held'])


if __name__=='__main__':unittest.main()
