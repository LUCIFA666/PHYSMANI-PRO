import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from our_mani.executor_placement_alignment_20261113_v5.alignment import HeldPlacementAlignment
from our_mani.executor_placement_alignment_20261113_v5.pose_executor import PoseGoalExecutor,UnloadExecutor

class HeldAlignmentTest(unittest.TestCase):
    def test_descent_waits_above_destination_until_orientation_and_lateral_alignment(self):
        gate=HeldPlacementAlignment();q=[0,0,0,1];v=np.zeros(6)
        gate.sample([.3,.2,1.],q,v,[.3,.2,1.,*q],0.,held=True,raw_goal_z=1.)
        goal=np.array([.3,.2,.9])
        p,_,r=gate.sample(goal,q,v,[.32,.2,.98,1,0,0,0],.1,held=True,raw_goal_z=.9)
        self.assertEqual(r['phase'],'align_above');self.assertGreaterEqual(p[2],1.)
        np.testing.assert_allclose(goal,[.3,.2,.9])
        gate.sample(goal,q,v,[.3,.2,1.,*q],.2,held=True,raw_goal_z=.9)
        p,_,r=gate.sample(goal,q,v,[.3,.2,1.,*q],.3,held=True,raw_goal_z=.9)
        self.assertEqual(r['phase'],'descend')
        before=p[2];p,twist,r=gate.sample(goal,q,v,[.3,.2,before,*q],.35,held=True,raw_goal_z=.9)
        self.assertLessEqual(before-p[2],.002+1e-9);self.assertGreaterEqual(twist[2],-.04-1e-9)
        repeat,_,_=gate.sample(goal,q,v,[.3,.2,before,*q],.35,held=True,raw_goal_z=.9)
        np.testing.assert_allclose(repeat,p)

    def test_loss_of_alignment_retracts_and_unheld_state_clears_history(self):
        gate=HeldPlacementAlignment();q=[0,0,0,1];v=np.zeros(6)
        gate.sample([0,0,1.],q,v,[0,0,1.,*q],0.,held=True,raw_goal_z=1.)
        gate.sample([0,0,.9],q,v,[0,0,1.,*q],.1,held=True,raw_goal_z=.9)
        gate.sample([0,0,.9],q,v,[0,0,1.,*q],.3,held=True,raw_goal_z=.9)
        p,_,r=gate.sample([0,0,.9],q,v,[.03,0,.96,*q],.4,held=True,raw_goal_z=.9)
        self.assertEqual(r['phase'],'align_above');self.assertGreaterEqual(p[2],1.)
        p,_,r=gate.sample([0,0,.9],q,v,[0,0,1.,*q],.5,held=False,raw_goal_z=.9)
        self.assertFalse(r['active']);self.assertIsNone(gate.peak_raw)
        np.testing.assert_allclose(p,[0,0,.9])

    def test_reaching_raised_control_waypoint_does_not_release_object(self):
        executor=PoseGoalExecutor.__new__(PoseGoalExecutor)
        executor.backend=SimpleNamespace(clock=lambda:1.,arm=SimpleNamespace(
            get_tip=lambda:SimpleNamespace(get_pose=lambda:[0,0,1.,0,0,0,1])))
        executor._active_action=SimpleNamespace(timestamp=1.)
        executor.release_position_tolerance=.005;executor.release_angle_tolerance=.03
        with patch.object(UnloadExecutor,'_sample_target',return_value=(
                np.array([0,0,.9]),[0,0,0,1],np.zeros(6),dict(valid=True))):
            receipt=executor._release_receipt()
        self.assertFalse(receipt['allowed'])
        self.assertAlmostEqual(receipt['measured_TCP_position_error_m'],.1)

if __name__=='__main__':unittest.main()
