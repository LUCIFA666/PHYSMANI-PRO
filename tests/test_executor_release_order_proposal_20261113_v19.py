import unittest
from types import SimpleNamespace
from unittest.mock import patch
from our_mani.executor_release_order_proposal_20261113_v19.backend import TaskExecutionBackend,OriginalBackend

class ReleaseOrderProposalTest(unittest.TestCase):
    def driver(self):
        calls=[];held=['object'];done=[False]
        def release():calls.append('release');held.clear()
        driver=TaskExecutionBackend.__new__(TaskExecutionBackend)
        driver.release_order='open_fingers_then_native_release';driver.requested_opening=0.;driver.gripper_velocity=.2
        driver.release_order_events=[];driver.clock=lambda:1.
        driver.gripper=SimpleNamespace(actuate=lambda action,velocity:(calls.append('actuate')or done[0]),
            get_grasped_objects=lambda:held,release=release,get_open_amount=lambda:[1.,1.])
        driver._can_command=lambda:None;driver._clock_tuple=lambda:(1.,1.,0)
        driver._assert_unchanged=lambda before,label:self.assertEqual(before,driver._clock_tuple())
        return driver,calls,held,done

    def test_default_order_delegates_without_reinterpreting_the_command(self):
        driver,calls,_,_=self.driver();driver.release_order='native_detach_before_open'
        with patch.object(OriginalBackend,'actuate_gripper',return_value=True)as original:
            self.assertTrue(driver.actuate_gripper(1.));original.assert_called_once_with(1.)
        self.assertEqual(calls,[])

    def test_proposed_order_waits_for_fingers_before_original_release(self):
        driver,calls,held,done=self.driver()
        self.assertFalse(driver.actuate_gripper(1.));self.assertTrue(held);self.assertEqual(calls,['actuate'])
        done[0]=True;self.assertTrue(driver.actuate_gripper(1.))
        self.assertEqual(calls,['actuate','actuate','release']);self.assertFalse(held)
        driver.actuate_gripper(1.);self.assertEqual(calls.count('release'),1)

    def test_closing_still_uses_the_original_grasp_api_path(self):
        driver,calls,_,_=self.driver()
        with patch.object(OriginalBackend,'actuate_gripper',return_value=False)as original:
            self.assertFalse(driver.actuate_gripper(0.));original.assert_called_once_with(0.)
        self.assertEqual(calls,[])

    def test_invalid_order_fails_before_any_native_launch(self):
        with self.assertRaises(ValueError):TaskExecutionBackend(release_order='unsupported')

    def test_native_done_without_finger_clearance_cannot_detach(self):
        driver,calls,held,done=self.driver();done[0]=True
        driver.gripper.get_open_amount=lambda:[.65,.65]
        self.assertFalse(driver.actuate_gripper(1.));self.assertTrue(held)
        self.assertNotIn('release',calls)
        self.assertTrue(driver.last_release_order_receipt['reversal_rearmed_without_physics'])
        driver.gripper.get_open_amount=lambda:[1.,1.]
        self.assertTrue(driver.actuate_gripper(1.));self.assertFalse(held)

    def test_native_order_records_early_detachment_without_relabeling_it_done(self):
        driver,calls,held,done=self.driver();driver.release_order='native_detach_before_open'
        def original(opening):held.clear();return False
        driver.gripper.get_open_amount=lambda:[.65,.65]
        with patch.object(OriginalBackend,'actuate_gripper',side_effect=original):
            self.assertFalse(driver.actuate_gripper(1.))
        self.assertTrue(driver.last_release_order_receipt['Native_release_called'])
        self.assertFalse(driver.last_release_order_receipt['physical_opening_completed'])

if __name__=='__main__':unittest.main()
