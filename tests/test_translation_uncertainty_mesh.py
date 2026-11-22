from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
import test_multitask_cup_feature_adapter as actual
from our_mani.multitask.collision_geometry import collision_geometry_for_orientation
from our_mani.multitask.translation_uncertainty_mesh import expanded_cube
from our_mani.multitask.perception import MultiBodySE3Tracker,RegistrationConfig
from our_mani.multitask.cup_translation_uncertainty import install


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):actual.ActualCupFeatureAdapterTests.setUpClass()

    def test_expanded_native_float_cube_contains_all_previous_vertices_plus_translation_uncertainty(self):
        points=np.array([[-.04,-.06,-.07],[.04,.06,.07],[0,0,0.]])
        original=collision_geometry_for_orientation(points,np.array([[0,1,2]]),orientation_observability='none')
        v,f,r=expanded_cube(original.receipt,original.receipt['registered_mesh_sha256'],.024)
        for shift in ((.024,0,0),(-.024,0,0),(0,.024,0),(0,0,-.024)):
            self.assertTrue(np.all(original.vertices+shift>=v.min(0)))
            self.assertTrue(np.all(original.vertices+shift<=v.max(0)))
        self.assertGreater(r['registered_cube_half_extent_m'],original.receipt['registered_cube_half_extent_m'])
        self.assertFalse(r['native_pose_or_future_state_used'])

    def test_multiple_current_translation_hypotheses_keep_full_rotation_unknown(self):
        helper=actual.ActualCupFeatureAdapterTests();template=next(t for t in helper.templates if t.role_id=='cup0')
        tracker=MultiBodySE3Tracker((template,),config=RegistrationConfig(free_space_weight=2.))
        actor=NS(tracker=tracker,controller=NS(feedback_step_calls=0),metrics_report=lambda:{})
        install(actor,install_geometry=False)
        tracker.observe(helper.observation(0),modes={'cup0':'external'})
        with patch.object(tracker,'_cup_feature_registration',return_value=None):
            states,metadata=tracker.observe(helper.observation(1),modes={'cup0':'external'})
        self.assertTrue(actor.metrics_report()['cup_translation_uncertainty']['events'][-1]['accepted'])
        self.assertFalse(metadata['bodies']['cup0']['orientation_observable'])
        self.assertTrue(states[0].visible)


if __name__=='__main__':unittest.main()
