from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import test_multitask_cup_feature_adapter as actual
from our_mani.multitask.perception import MultiBodySE3Tracker,RegistrationConfig
from our_mani.multitask.cup_translation_fallback import install


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):actual.ActualCupFeatureAdapterTests.setUpClass()

    def fixture(self):
        helper=actual.ActualCupFeatureAdapterTests();template=next(t for t in helper.templates if t.role_id=='cup0')
        tracker=MultiBodySE3Tracker((template,),config=RegistrationConfig(free_space_weight=2.))
        actor=NS(tracker=tracker,controller=NS(feedback_step_calls=0),metrics_report=lambda:{})
        install(actor);tracker.observe(helper.observation(0),modes={'cup0':'external'})
        return helper,tracker,actor,template

    def test_actual_saved_TRAIN_RGBD_can_recover_only_current_translation_without_promoting_rotation(self):
        helper,tracker,actor,template=self.fixture()
        with patch.object(tracker,'_cup_feature_registration',return_value=None):
            states,metadata=tracker.observe(helper.observation(1),modes={'cup0':'external'})
        self.assertIs(tracker.templates['cup0'],template)
        event=actor.metrics_report()['cup_translation_fallback']['events'][-1]
        self.assertTrue(event['accepted'],event)
        self.assertTrue(states[0].visible)
        self.assertFalse(metadata['bodies']['cup0']['orientation_observable'])
        self.assertEqual(states[0].timestamp,float(helper.times[1]))
        self.assertTrue(event['original_current_depth_gate_passed'])

    def test_missing_current_depth_cannot_be_replaced_by_previous_pose(self):
        helper,tracker,actor,template=self.fixture()
        with patch.object(tracker,'_cup_feature_registration',return_value=None):
            states,metadata=tracker.observe(helper.observation(1,missing_depth=True),modes={'cup0':'external'})
        self.assertIs(tracker.templates['cup0'],template)
        self.assertFalse(metadata['bodies']['cup0']['registration_valid'])
        self.assertFalse(states[0].visible)
        self.assertFalse(actor.metrics_report()['cup_translation_fallback']['events'][-1]['accepted'])


if __name__=='__main__':unittest.main()
