import unittest
from our_mani.executor_capture_repair_20261112_v1.gripper_commitment import GripperCommitment
from our_mani.executor_contract_repair_20261112_v5.gripper_schedule import GripperSchedule

class CommitmentTests(unittest.TestCase):
    def test_one_update_release_pulse_survives_ten_physical_tick_delay(self):
        commitment=GripperCommitment();schedule=GripperSchedule()
        for tick in range(10):
            if tick%2==0:
                request=1. if tick==0 else 0.
                accepted=commitment.resolve(request,actual_target=0.,done=True,held=True)
                schedule.request(accepted,0.,tick)
            self.assertEqual(schedule.due(tick),1. if tick==9 else None)

    def test_opening_completes_before_next_close_is_accepted(self):
        c=GripperCommitment();c.resolve(1.,actual_target=0.,done=True,held=True)
        self.assertEqual(c.resolve(0.,actual_target=1.,done=False,held=False),1.)
        self.assertEqual(c.resolve(0.,actual_target=1.,done=True,held=False),0.)

    def test_braking_cancels_unissued_release_but_preserves_inflight_opening(self):
        c=GripperCommitment();c.resolve(1.,actual_target=0.,done=True,held=True)
        c.brake(actual_target=0.,done=True);self.assertFalse(c.release_latched)
        c.resolve(1.,actual_target=0.,done=True,held=True)
        c.brake(actual_target=1.,done=False);self.assertTrue(c.release_latched)

if __name__=='__main__':unittest.main()
