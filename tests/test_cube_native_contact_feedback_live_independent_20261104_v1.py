"""Synthetic state-machine contracts only; no model, checkpoint, or Native."""
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.contracts import Pose
from our_mani.cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
from our_mani.cube_native_contact_feedback_live_20261104_v1.coordinator import NativeContactCoordinatorV1
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors


def robot():
    return PublicRobotSensors(timestamp=1.,tcp_position=[.2,0.,1.],tcp_quaternion=[0.,0.,0.,1.],
                              q=np.zeros(7),dq=np.zeros(7),finger_open_amounts=[.4,.4])


def coordinator(held):
    # No fake qualified model or planner is constructed. These branches must
    # resolve before any model/geometry access on missing current vision.
    value=NativeContactCoordinatorV1.__new__(NativeContactCoordinatorV1)
    value.backend=SimpleNamespace(latest_contact=dict(timestamp=1.,held=held),
        latest=dict(free_forecast_available=False,object_observation=None),
        latest_carry=dict(estimated_pose=None),goal_pose=Pose([.3,0.,1.]))
    value.phase=value.warm=value.candidates=None
    value.mode='UNAVAILABLE';value.mode_events=[];value._last_held=False;value.carry_stage=None
    return value


class NativeFeedbackBoundaryTests(unittest.TestCase):
    def test_original_success_first_true_latches_without_a_visual_or_hold_gate(self):
        clock=[.5]
        synthetic=SimpleNamespace(_last_native_result={'success':False},
            native_task_success_latched=False,native_success_first_time=None,clock=lambda:clock[0])
        latch=NativeContactCubeBackendV1._latch_original_task_success
        self.assertFalse(latch(synthetic));self.assertFalse(synthetic.native_task_success_latched)
        synthetic._last_native_result['success']=True
        self.assertTrue(latch(synthetic));self.assertTrue(synthetic.native_task_success_latched)
        self.assertEqual(synthetic.native_success_first_time,.5)
        clock[0]=.55;synthetic._last_native_result['success']=False
        self.assertFalse(latch(synthetic))  # current native value, not the latched judge
        self.assertTrue(synthetic.native_task_success_latched)
        self.assertEqual(synthetic.native_success_first_time,.5)

    def test_actual_membership_with_visual_unknown_preserves_closed_request(self):
        value=coordinator(True)
        result=value.decide(robot(),dict(gripper_target=0.,gripper_done=True))
        self.assertEqual(result['mode'],'NATIVE_HELD_OFFSET_UNKNOWN')
        self.assertIsNone(result['action']);self.assertIs(result['keep_closed'],True)
        self.assertNotIn('public_attachment_confirmed',result)

    def test_only_completed_actual_nonmembership_close_can_reopen(self):
        value=coordinator(False)
        result=value.decide(robot(),dict(gripper_target=0.,gripper_done=False))
        self.assertIsNone(result['action']);self.assertIs(result['keep_closed'],True)
        self.assertNotIn('direct_membership_miss_open',result)
        value=coordinator(False)
        result=value.decide(robot(),dict(gripper_target=0.,gripper_done=True))
        self.assertIs(result['direct_membership_miss_open'],True)
        self.assertEqual(result['action']['opening'],1.)
        self.assertIs(result['visual_proof_failure_used'],False)
        np.testing.assert_array_equal(result['action']['joint_goal'],robot().q)


if __name__=='__main__':unittest.main()
