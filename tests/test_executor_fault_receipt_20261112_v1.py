import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
from our_mani.executor_cartesian_tracking_20261112_v1.pose_executor import PoseGoalExecutor,PhysicalExecutor,WorldPoseGoal
from our_mani.executor_cartesian_tracking_20261112_v1.phases import HeldDescent

class FaultReceiptTests(unittest.TestCase):
    def test_post_step_hardware_fault_preserves_executed_ticks(self):
        dispatch=object.__new__(PoseGoalExecutor);clock=SimpleNamespace(now=0.)
        dispatch.backend=SimpleNamespace(clock=lambda:clock.now,gripper=SimpleNamespace(get_grasped_objects=lambda:[]))
        dispatch.executor=SimpleNamespace(config=SimpleNamespace(clock_tolerance=2e-5))
        dispatch._goal_history=deque(maxlen=5);dispatch.capture_lookahead_seconds=.5
        dispatch._fault=None
        dispatch.held_descent=HeldDescent()
        def fail_after_steps(*args,**kwargs):
            clock.now=.1
            dispatch.last_dispatch_receipt=dict(start_clock=0.,end_clock=.1,executed_ticks=2,execution=[{'tick':1},{'tick':2}])
            dispatch._fault='Robot body FK disagrees with Native geometry: diagnostic'
            raise RuntimeError(dispatch._fault)
        with patch.object(PhysicalExecutor,'execute',side_effect=fail_after_steps):
            receipt=dispatch.execute(WorldPoseGoal(0.,[.2,0,.8],[0,1,0,0],0.))
        self.assertEqual(receipt['status'],'hardware_model_diverged')
        self.assertEqual(receipt['executed_ticks'],2)
        self.assertEqual(len(receipt['execution']),2)
        self.assertTrue(receipt['dispatcher_remains_faulted'])
        self.assertIsNotNone(dispatch._fault)

if __name__=='__main__':unittest.main()
