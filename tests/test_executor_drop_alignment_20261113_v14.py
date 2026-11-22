import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from our_mani.executor_drop_alignment_20261113_v14.pose_executor import drop_alignment,PoseGoalExecutor,RawTimedExecutor
from our_mani.execution_protocol_20261113_v2 import ExecutionRequest,ProtocolExecutor
from tests.test_executor_protocol_20261113_v2 import Driver,POSE

class DropAlignmentTest(unittest.TestCase):
    def test_vertical_release_clearance_does_not_mask_lateral_error(self):
        self.assertFalse(drop_alignment([.004,0,.027],.001,True))
        self.assertTrue(drop_alignment([.001,0,.027],.001,True))
        self.assertFalse(drop_alignment([0,0,.027],.03,True))
        self.assertFalse(drop_alignment([0,0,.027],.001,False))
        self.assertFalse(drop_alignment([0,0,.05],.001,True))

    def test_release_checks_terminal_pose_and_preserves_measured_height(self):
        driver=PoseGoalExecutor.__new__(PoseGoalExecutor)
        driver.release_policy='aligned_drop';driver.grasp_delta=None
        driver.backend=SimpleNamespace(arm=SimpleNamespace(get_tip=lambda:SimpleNamespace(get_pose=lambda:[.301,.2,.827,0,0,0,1])))
        with patch.object(RawTimedExecutor,'_sample_target',return_value=(np.array([.3,.2,.8]),[0,0,0,1],np.zeros(6),dict(valid=True))):
            receipt=driver._release_receipt()
        self.assertTrue(receipt['allowed'])
        self.assertAlmostEqual(receipt['measured_TCP_position_error_xyz_m'][2],.027)
        self.assertTrue(receipt['Native_task_success_predicate_unchanged'])

    def test_new_release_contract_requires_capable_driver_and_roundtrips(self):
        packet=ExecutionRequest([*POSE[:7],1.],1.,'timed_sample',.5,'release',release_policy='aligned_drop')
        self.assertEqual(ExecutionRequest.decode(packet.to_wire()),packet)
        driver=Driver();driver.release_policy='terminal_pose';bridge=ProtocolExecutor(driver)
        with self.assertRaises(ValueError):bridge.execute(packet)
        driver.supports_aligned_drop=True;bridge.execute(packet)
        self.assertEqual(driver.release_policy,'aligned_drop')

    def test_explicit_continuous_opening_allows_retreat_before_fingers_finish(self):
        driver=Driver();bridge=ProtocolExecutor(driver)
        driver.executor.gripper_target=1.;driver.executor.gripper_done=False
        opened=[*POSE[:7],1.]
        bridge.execute(ExecutionRequest(opened,1.,'timed_sample',.5,'release'))
        result=bridge.execute(ExecutionRequest(opened,1.,'timed_sample',0.,'retreat'))
        self.assertTrue(result['execution_protocol']['request_accepted'])
        self.assertEqual(driver.active_segment_id,'retreat')
        self.assertFalse(driver.executor.gripper_done)
        self.assertEqual(driver.executor.gripper_target,1.)

if __name__=='__main__':unittest.main()
