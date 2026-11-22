import math
import unittest
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend


class NativeCompletionFixture:
    """Models the observed original cache-reset/restart contract; no physics."""
    def __init__(self):self.calls=[];self.t=0.;self.finished=False;self.velocity=0.;self.current=None;self.stop=False
    def clock(self):return self.t
    def terminal(self):return self.stop
    def actuate_gripper(self,target):
        self.calls.append(target)
        if self.current!=target:self.current=target;self.finished=False
        if self.finished:
            self.finished=False;self.velocity=-.2 if target==0 else .2
            return False  # A new call after done restarts the original command.
        if self.velocity==0.:
            self.velocity=-.2 if target==0 else .2
            return False
        self.finished=True;self.velocity=0.
        return True


class CompletedGripperTests(unittest.TestCase):
    def create(self):raw=NativeCompletionFixture();return raw,CompletedGripperBackend(raw)
    def test_completed_blocked_close_does_not_restart_at_each_50ms_tick(self):
        raw,adapter=self.create()
        self.assertFalse(adapter.actuate_gripper(0.));self.assertTrue(adapter.actuate_gripper(0.))
        for _ in range(8):self.assertTrue(adapter.actuate_gripper(0.))
        self.assertEqual(raw.calls,[0.,0.]);self.assertEqual(raw.velocity,0.);self.assertEqual(raw.t,0.)
        self.assertEqual(adapter.gripper_completion_receipt()['completed_same_target_ticks_not_restarted'],8)
    def test_changed_request_resumes_original_opening_and_can_close_again(self):
        raw,adapter=self.create()
        for target in [0.,1.,0.]:
            self.assertFalse(adapter.actuate_gripper(target));self.assertTrue(adapter.actuate_gripper(target))
        self.assertEqual(raw.calls,[0.,0.,1.,1.,0.,0.]);self.assertEqual(adapter.gripper_completion_receipt()['requested_target_changes'],3)
    def test_native_binary_mapping_preserved(self):
        raw,adapter=self.create();adapter.actuate_gripper(.1);adapter.actuate_gripper(.2);adapter.actuate_gripper(.3)
        self.assertEqual(raw.calls,[0.,0.])
        adapter.actuate_gripper(.8);self.assertEqual(raw.calls[-1],1.)
    def test_done_is_not_contact_or_attachment_evidence(self):
        raw,adapter=self.create();adapter.actuate_gripper(0.);adapter.actuate_gripper(0.)
        receipt=adapter.gripper_completion_receipt()
        self.assertFalse(receipt['completion_implies_held']);self.assertFalse(receipt['completion_implies_contact'])
        self.assertNotIn('held',receipt);self.assertEqual(receipt['physics_steps_added'],0)
    def test_interrupted_close_reversal_is_not_falsely_latched_as_open_complete(self):
        raw,adapter=self.create();self.assertFalse(adapter.actuate_gripper(0.))
        self.assertFalse(adapter.actuate_gripper(1.))
        self.assertEqual(raw.calls,[0.,1.,1.]);self.assertEqual(raw.velocity,.2)
        self.assertEqual(raw.t,0.)
        self.assertTrue(adapter.actuate_gripper(1.));self.assertEqual(raw.velocity,0.)
        self.assertEqual(adapter.gripper_completion_receipt()['unfinished_target_changes_rearmed_without_physics'],1)
    def test_hidden_time_advance_latches_fault_and_prevents_further_commands(self):
        raw,adapter=self.create()
        def hidden(target):raw.t+=.05;return False
        raw.actuate_gripper=hidden
        with self.assertRaisesRegex(RuntimeError,'physical time'):adapter.actuate_gripper(0.)
        self.assertTrue(adapter.terminal())
        with self.assertRaisesRegex(RuntimeError,'terminal'):adapter.actuate_gripper(0.)
    def test_invalid_requests_and_terminal_backend_rejected(self):
        raw,adapter=self.create()
        for value in [math.nan,math.inf,-.1,1.1]:
            with self.assertRaises(ValueError):adapter.actuate_gripper(value)
        raw.stop=True
        with self.assertRaises(RuntimeError):adapter.actuate_gripper(0.)
        self.assertEqual(raw.calls,[])


if __name__=='__main__':unittest.main()
