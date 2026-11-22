import unittest
import numpy as np
from our_mani.executor_contact_frame_20261113_v3.pose_executor import contact_and_reference
from our_mani.executor_grasp_frame_20261113_v1.grasp_frame import capture_delta,apply_delta

class ContactFrameTest(unittest.TestCase):
    def test_deliberate_contact_stroke_is_not_capture_error(self):
        contact=np.array([.3,.1,.90]);reference,errors=contact_and_reference(contact,[0,0,.02],contact)
        np.testing.assert_allclose(reference,[.3,.1,.92]);self.assertEqual(errors['contact_position_error_m'],0.)
        self.assertAlmostEqual(errors['nominal_reference_position_error_m'],.02)
        delta=capture_delta(reference,[0,0,0,1],[*contact,0,0,0,1])
        np.testing.assert_allclose(delta[:3,3],[0,0,-.02])
        np.testing.assert_allclose(contact,[.3,.1,.9])

    def test_offset_affects_object_transport_even_with_exact_contact(self):
        reference,_=contact_and_reference([.3,.1,.9],[0,0,.02],[.3,.1,.9])
        delta=capture_delta(reference,[0,0,0,1],[.3,.1,.9,0,0,0,1])
        p,q,v=apply_delta([.4,.2,1.],[0,0,0,1],np.zeros(6),delta)
        np.testing.assert_allclose(p,[.4,.2,.98])

if __name__=='__main__':unittest.main()
