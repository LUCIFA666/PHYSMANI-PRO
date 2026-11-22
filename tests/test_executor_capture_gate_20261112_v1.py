import unittest
from our_mani.executor_capture_repair_20261112_v1.capture import qualify

class CaptureGateTests(unittest.TestCase):
    def test_arrival_without_native_detection_does_not_close(self):
        allowed,_=qualify(graspables=True,detected=False,stationary=True,position_error=0.,rotation_error=0.)
        self.assertFalse(allowed)
    def test_stationary_detection_with_wrong_grasp_transform_waits(self):
        allowed,_=qualify(graspables=True,detected=True,stationary=True,position_error=.02,rotation_error=0.)
        self.assertFalse(allowed)
    def test_moving_capture_does_not_require_arrival_at_a_future_waypoint(self):
        allowed,_=qualify(graspables=True,detected=True,stationary=False,position_error=.1,rotation_error=0.)
        self.assertTrue(allowed)
    def test_non_grasp_operation_keeps_explicit_close_intent(self):
        allowed,_=qualify(graspables=False,detected=False,stationary=True,position_error=1.,rotation_error=1.)
        self.assertTrue(allowed)

if __name__=='__main__':unittest.main()
