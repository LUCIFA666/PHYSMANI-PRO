import unittest
from our_mani.executor_capture_precision_20261113_v2.pose_executor import precise_capture

class CapturePrecisionTest(unittest.TestCase):
    def test_capture_deviation_is_not_hidden_by_proximity(self):
        receipt=dict(allowed=True,reason='native_capture',grasp_position_error_m=.0031,grasp_rotation_error_rad=.092)
        self.assertTrue(precise_capture(receipt,.02,.1)['allowed'])
        self.assertFalse(precise_capture(receipt,.003,.03)['allowed'])
        self.assertTrue(receipt['allowed'])

    def test_small_error_cannot_override_native_or_motion_rejection(self):
        receipt=dict(allowed=False,reason='native_not_detected',measured_TCP_position_error_m=.0001,measured_TCP_rotation_error_rad=.001)
        self.assertFalse(precise_capture(receipt,.003,.03)['allowed'])
        receipt['allowed']=True;receipt['measured_TCP_position_error_m']=float('nan')
        self.assertFalse(precise_capture(receipt,.003,.03)['allowed'])

if __name__=='__main__':unittest.main()
