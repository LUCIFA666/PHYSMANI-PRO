import unittest
from tests.test_executor_protocol_20261113_v2 import Driver,POSE
from our_mani.execution_protocol_20261113_v2 import ExecutionRequest
from our_mani.executor_gripper_goal_refresh_20261113_v24.bridge import ProtocolExecutor


class MovingGripperEventTest(unittest.TestCase):
    def make(self):
        driver=Driver();driver.executor.gripper_done=True
        bridge=ProtocolExecutor(driver)
        bridge.execute(ExecutionRequest([*POSE[:7],1.],1.,'timed_sample',.5,'place','gripper'))
        driver.backend.clock=lambda:1.1
        return driver,bridge

    def test_unissued_opening_does_not_block_fresh_same_stage_target(self):
        driver,bridge=self.make();marker=driver.executor.reference_marker
        goal=[.35,.12,.95,*POSE[3:7],0.]
        result=bridge.execute(ExecutionRequest(goal,1.1,'timed_sample',.25,'place'))
        latest=driver.executed[-1][0]
        self.assertEqual(len(driver.executed),2)
        self.assertEqual(latest.pose.position,tuple(goal[:3]))
        self.assertEqual(latest.pose.timestamp,1.1)
        self.assertEqual(latest.lookahead_seconds,.25)
        self.assertEqual(latest.pose.opening,1.)
        self.assertTrue(result['pending_gripper_pose_refresh']['latest_pose_accepted'])
        self.assertIs(marker,driver.executor.reference_marker)

    def test_inflight_opening_refreshes_pose_without_reversing_fingers(self):
        driver,bridge=self.make()
        driver.executor.gripper_target=1.;driver.executor.gripper_done=False
        bridge.execute(ExecutionRequest(POSE,1.1,'timed_sample',.5,'place'))
        self.assertEqual(len(driver.executed),2)
        self.assertEqual(driver.executed[-1][0].pose.opening,1.)

    def test_different_stage_still_waits_for_physical_completion(self):
        driver,bridge=self.make()
        result=bridge.execute(ExecutionRequest(POSE,1.1,'timed_sample',0.,'retreat'))
        self.assertEqual(len(driver.executed),1)
        self.assertFalse(result['gripper_completion_ack']['transition_accepted'])
        self.assertEqual(driver.active_segment_id,'place')

    def test_completed_opening_allows_the_next_close_intent(self):
        driver,bridge=self.make()
        driver.executor.gripper_target=1.;driver.executor.gripper_done=True
        result=bridge.execute(ExecutionRequest(POSE,1.1,'timed_sample',0.,'place'))
        self.assertEqual(driver.executed[-1][0].pose.opening,0.)
        self.assertTrue(result['gripper_completion_ack']['transition_accepted'])


if __name__=='__main__':unittest.main()
