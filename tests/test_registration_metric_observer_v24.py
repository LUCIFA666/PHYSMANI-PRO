from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.registration_metric_observer_v24 import RegistrationMetricObserver


class Tests(unittest.TestCase):
    def test_original_result_identity_is_preserved_and_labels_are_metric_only(self):
        result=dict(pose=Pose(np.array([0.,0.,1.])),rotation_ambiguous=True,valid=True,rms_m=.001,per_role={})
        tracker=SimpleNamespace(_current_timestamp=0.,_registration=lambda *args:result)
        env=SimpleNamespace(ledger=SimpleNamespace(ticks=0),physics_time=lambda:0.,task=SimpleNamespace(t=0.))
        obj=SimpleNamespace(get_pose=lambda:[0.,0.,1.,0.,0.,0.,1.])
        observer=RegistrationMetricObserver(SimpleNamespace(tracker=tracker),env,metric_objects={'rubbish':obj})
        self.assertIs(tracker._registration({'rubbish':Pose(np.zeros(3))},{},None),result)
        self.assertTrue(result['rotation_ambiguous']);self.assertEqual(observer.events[0]['orientation_error_rad'],0.)
        self.assertFalse(observer.receipt()['native_labels_supplied_to_actor'])

    def test_missing_native_metric_does_not_change_registration(self):
        result=dict(pose=Pose(np.zeros(3)),rotation_ambiguous=False,valid=True,rms_m=.001,per_role={})
        tracker=SimpleNamespace(_current_timestamp=0.,_registration=lambda *args:result)
        env=SimpleNamespace(ledger=SimpleNamespace(ticks=0),physics_time=lambda:0.,task=SimpleNamespace(t=0.))
        obj=SimpleNamespace(get_pose=lambda:(_ for _ in ()).throw(ValueError('unavailable')))
        observer=RegistrationMetricObserver(SimpleNamespace(tracker=tracker),env,metric_objects={'rubbish':obj})
        self.assertIs(tracker._registration({'rubbish':Pose(np.zeros(3))},{},None),result)
        self.assertIn('native_metric_unavailable',observer.events[0])


if __name__=='__main__':unittest.main()
