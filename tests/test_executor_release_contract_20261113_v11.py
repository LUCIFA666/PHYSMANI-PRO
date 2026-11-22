import unittest
from our_mani.executor_release_contract_20261113_v11.pose_executor import release_decision

class ReleaseContractTest(unittest.TestCase):
    def receipt(self,p):return dict(allowed=False,reason='old_5mm_gate',motion_compensation=dict(valid=True),
        measured_TCP_position_error_m=p,measured_TCP_rotation_error_rad=.0003)

    def test_source_position_window_can_release_where_old_gate_refused(self):
        source=self.receipt(.0275);r=release_decision(source,'position_window',.03)
        self.assertTrue(r['allowed']);self.assertFalse(r['task_success_claimed'])
        self.assertFalse(source['allowed'])
        self.assertFalse(release_decision(self.receipt(.04),'position_window',.03)['allowed'])

    def test_explicit_intent_is_an_actuator_command_not_a_task_success(self):
        r=release_decision(self.receipt(.1),'caller_intent',.03)
        self.assertTrue(r['allowed']);self.assertFalse(r['task_success_claimed'])
        self.assertFalse(release_decision(self.receipt(.0275),'terminal_pose',.03)['allowed'])

if __name__=='__main__':unittest.main()
