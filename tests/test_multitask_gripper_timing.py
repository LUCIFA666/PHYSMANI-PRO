"""Measured arrival, not a planned arrival timestamp, triggers one close edge."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask.interaction import GripperTimingConfig,InteractionController,continuation_fingerprint
from tests.test_multitask_interaction import synthetic_library,candidates_config,mpc_config,state


class PublicGripperTimingTests(unittest.TestCase):
    def controller(self,phase='acquire'):
        # Isolate the command edge from native grasp/contact machinery. The
        # actual controller invokes this same gate only after valid IK/MPC.
        controller=InteractionController.__new__(InteractionController)
        controller.gripper_timing_config=GripperTimingConfig(.02,.10)
        controller.gripper_timing_events=[];controller.last_diagnostics={}
        controller._gripper_goal_token=controller._close_edge_issued_at=None
        controller.attempts=SimpleNamespace(records=[{}])
        controller.goal=SimpleNamespace(candidate_id='first_actual_candidate',selected_at=0.,
            template=replace(synthetic_library().templates[0],phase=phase))
        return controller

    def test_tracking_lag_keeps_open_then_one_close_is_latched(self):
        controller=self.controller();target=Pose(np.zeros(3))
        self.assertEqual(controller._gripper_opening(state(.5,robot_z=.17),0.,target),1.)
        self.assertEqual(controller._gripper_opening(state(.55,robot_z=.01),0.,target),0.)
        # Subsequent motion while fingers close cannot produce alternating
        # open/close commands or an attach retry in the already closed state.
        self.assertEqual(controller._gripper_opening(state(.6,robot_z=.17),0.,target),0.)
        reasons=[row['reason'] for row in controller.gripper_timing_events]
        self.assertEqual(reasons.count('observed_final_grasp_SE3_reached_issue_one_close_edge'),1)

    def test_orientation_is_measured_and_new_empty_attempt_requires_open(self):
        controller=self.controller();target=Pose(np.zeros(3),Rotation.from_euler('z',.2).as_matrix())
        self.assertEqual(controller._gripper_opening(state(.5,robot_z=0.),0.,target),1.)
        current=state(.6,robot_z=0.)
        current=replace(current,robot=replace(current.robot,gripper_open=False,contact=False))
        controller.goal.candidate_id='new_actual_candidate'
        self.assertEqual(controller._gripper_opening(current,0.,Pose(np.zeros(3))),1.)
        opened=state(.65,robot_z=0.)
        self.assertEqual(controller._gripper_opening(opened,0.,Pose(np.zeros(3))),0.)

    def test_release_opening_is_never_delayed_by_acquire_guard(self):
        controller=self.controller('release');current=state(.5,robot_z=1.)
        current=replace(current,robot=replace(current.robot,gripper_open=False,contact=True))
        self.assertEqual(controller._gripper_opening(current,1.,None),1.)

    def test_gripper_threshold_changes_continuation_identity(self):
        library=synthetic_library();candidate=candidates_config();mpc=mpc_config()
        first=continuation_fingerprint(library,candidate,mpc,GripperTimingConfig(.02,.10))
        second=continuation_fingerprint(library,candidate,mpc,GripperTimingConfig(.03,.10))
        self.assertNotEqual(first,second)


if __name__=='__main__': unittest.main()
