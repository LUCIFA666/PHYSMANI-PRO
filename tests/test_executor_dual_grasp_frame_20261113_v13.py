import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from our_mani.executor_dual_grasp_frame_20261113_v13.pose_executor import grasp_frames,PoseGoalExecutor,RawTimedExecutor
from our_mani.executor_grasp_frame_20261113_v1.grasp_frame import apply_delta

class DualFrameTest(unittest.TestCase):
    def test_waiting_for_fingers_does_not_repeat_a_two_centimetre_contact_stroke(self):
        p=np.array([.3,.2,.8]);q=[1,0,0,0];actual=np.r_[p,q]
        contact,transport=grasp_frames(p,q,[0,0,.02],actual)
        during,_,_=apply_delta(p,q,np.zeros(6),contact)
        after,_,_=apply_delta([.5,.1,1.],q,np.zeros(6),transport)
        np.testing.assert_allclose(during,p)
        np.testing.assert_allclose(after,[.5,.1,.98])

    def test_raised_approach_does_not_qualify_a_grasp(self):
        executor=PoseGoalExecutor.__new__(PoseGoalExecutor)
        executor.backend=SimpleNamespace(clock=lambda:1.,capture_window=lambda:dict(graspables=True,detected=True),
            arm=SimpleNamespace(get_tip=lambda:SimpleNamespace(get_pose=lambda:[.3,.2,1.,0,0,0,1])))
        executor.capture_lookahead_seconds=0.;executor._active_action=SimpleNamespace(timestamp=1.)
        executor.compensate_grasp_frame=True
        executor.executor=SimpleNamespace(config=SimpleNamespace(clock_tolerance=1e-5))
        executor.capture_tolerances=(.003,.03)
        with patch.object(RawTimedExecutor,'_sample_target',return_value=(
                np.array([.3,.2,.8]),[0,0,0,1],np.zeros(6),dict(valid=True))):
            receipt=executor._capture_receipt()
        self.assertFalse(receipt['allowed']);self.assertAlmostEqual(receipt['grasp_position_error_m'],.2)

    def test_fresh_goal_while_fingers_close_keeps_contact_frame_until_stage_changes(self):
        executor=PoseGoalExecutor.__new__(PoseGoalExecutor)
        p=np.array([.3,.2,.8]);q=np.array([0.,0.,0.,1.])
        contact,transport=grasp_frames(p,q,[0.,0.,.02],np.r_[p,q])
        executor.compensate_grasp_frame=True
        executor.backend=SimpleNamespace(gripper=SimpleNamespace(get_grasped_objects=lambda:['held']))
        executor.grasp_delta=None;executor.contact_delta=None
        executor.pending_grasp_delta=transport;executor.pending_contact_delta=contact
        executor.captured_segment_id='close';executor.active_segment_id='close'
        executor.captured_action_time=1.;executor._active_action=SimpleNamespace(timestamp=1.1)
        executor.approach=None;executor.placement_alignment=None
        # A new timestamp does not mean that the physical closing phase ended.
        with patch.object(RawTimedExecutor,'_sample_target',return_value=(p.copy(),q.copy(),np.zeros(6),dict(valid=True))):
            result=executor._sample_target()
            np.testing.assert_allclose(result[0],p)
            self.assertEqual(result[3]['applied_grasp_reference'],'contact')
            executor.active_segment_id='lift'
            result=executor._sample_target()
            np.testing.assert_allclose(result[0],p-[0.,0.,.02])
            self.assertEqual(result[3]['applied_grasp_reference'],'nominal_transport')

if __name__=='__main__':unittest.main()
