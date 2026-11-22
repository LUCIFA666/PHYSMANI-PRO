import unittest
from our_mani.executor_contract_repair_20261112_v5.gripper_schedule import GripperSchedule

class ScheduleTests(unittest.TestCase):
    def test_replanning_every_two_ticks_does_not_starve_close(self):
        s=GripperSchedule()
        issued=[]
        for tick in range(10):
            if tick%2==0:s.request(0.,1.,tick)
            op=s.due(tick)
            if op is not None:issued.append((tick,op));s.acknowledge(op)
        self.assertEqual(issued,[(9,0.)])

    def test_cancel_unissued_request_when_braking_after_rejection(self):
        s=GripperSchedule();s.request(0.,1.,0);s.cancel()
        self.assertIsNone(s.due(20))
        s.request(0.,1.,20)
        self.assertIsNone(s.due(28));self.assertEqual(s.due(29),0.)

    def test_reversal_to_current_opening_cancels_pending_close(self):
        s=GripperSchedule();s.request(0.,1.,0);s.request(1.,1.,4)
        self.assertIsNone(s.due(20))

    def test_existing_motor_request_does_not_restart_or_reissue(self):
        s=GripperSchedule();s.request(0.,1.,0);op=s.due(9);s.acknowledge(op)
        for tick in range(10,100):
            s.request(0.,0.,tick);self.assertIsNone(s.due(tick))

if __name__=='__main__':unittest.main()
