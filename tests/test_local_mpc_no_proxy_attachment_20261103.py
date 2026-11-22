import unittest
from types import SimpleNamespace
from our_mani.local_mpc_control_20261103.no_proxy_attachment import NoProxyAttachmentBackend


class RawFixture:
    def __init__(self):
        self.calls=0;self.motor_calls=[];self.t=0.;self.fail=False
        self.gripper=SimpleNamespace(grasp=self.grasp)
    def grasp(self,*args,**kwargs):self.calls+=1;return True
    def clock(self):return self.t
    def actuate_gripper(self,opening):
        if opening==0:self.gripper.grasp(object())
        self.motor_calls.append(opening)
        if self.fail:raise ValueError('injected original motor failure')
        return True


class NoProxyTests(unittest.TestCase):
    def test_only_attachment_attempt_suppressed_motor_and_time_preserved(self):
        raw=RawFixture();wrapper=NoProxyAttachmentBackend(raw);original=raw.gripper.grasp
        self.assertTrue(wrapper.actuate_gripper(0.))
        self.assertEqual(raw.calls,0);self.assertEqual(raw.motor_calls,[0.]);self.assertEqual(raw.t,0.)
        self.assertIs(raw.gripper.grasp,original);self.assertEqual(wrapper.suppressed,1)
        self.assertFalse(wrapper.no_proxy_attachment_receipt()['positive_success_score_eligible'])
    def test_original_method_restored_after_motor_failure(self):
        raw=RawFixture();wrapper=NoProxyAttachmentBackend(raw);original=raw.gripper.grasp;raw.fail=True
        with self.assertRaises(ValueError):wrapper.actuate_gripper(0.)
        self.assertIs(raw.gripper.grasp,original)
        raw.gripper.grasp(object());self.assertEqual(raw.calls,1)
    def test_opening_has_no_suppressed_attempt(self):
        raw=RawFixture();wrapper=NoProxyAttachmentBackend(raw);wrapper.actuate_gripper(1.)
        self.assertEqual(wrapper.suppressed,0);self.assertEqual(raw.motor_calls,[1.])
    def test_class_method_lookup_is_restored_without_persistent_instance_shadow(self):
        class Gripper:
            def grasp(self,*args):return True
        raw=RawFixture();raw.gripper=Gripper();wrapper=NoProxyAttachmentBackend(raw)
        self.assertNotIn('grasp',vars(raw.gripper));wrapper.actuate_gripper(0.)
        self.assertNotIn('grasp',vars(raw.gripper));self.assertTrue(raw.gripper.grasp(object()))


if __name__=='__main__':unittest.main()
