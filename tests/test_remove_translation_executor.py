from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import test_multitask_cup_feature_adapter as actual
from our_mani.multitask.perception import MultiBodySE3Tracker,RegistrationConfig
from our_mani.multitask.remove_translation_executor import RemoveTranslationBackend
from our_mani.multitask.remove_continuous_executor import RemoveContinuousBackend


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):actual.ActualCupFeatureAdapterTests.setUpClass()

    def test_real_backend_installs_fallback_on_the_actual_current_RGBD_tracker(self):
        helper=actual.ActualCupFeatureAdapterTests();template=next(t for t in helper.templates if t.role_id=='cup0')
        tracker=MultiBodySE3Tracker((template,),config=RegistrationConfig(free_space_weight=2.))
        actor=NS(tracker=tracker,controller=NS(feedback_step_calls=0),metrics_report=lambda:{})
        with patch.object(RemoveContinuousBackend,'actor',return_value=(actor,{})):
            RemoveTranslationBackend('remove_cups_from_rotating_frame').actor({},None,None)
        tracker.observe(helper.observation(0),modes={'cup0':'external'})
        with patch.object(tracker,'_cup_feature_registration',return_value=None):
            _,metadata=tracker.observe(helper.observation(1),modes={'cup0':'external'})
        self.assertTrue(actor.metrics_report()['cup_translation_fallback']['events'][-1]['accepted'])
        self.assertFalse(metadata['bodies']['cup0']['orientation_observable'])
        self.assertIs(tracker.templates['cup0'],template)


if __name__=='__main__':unittest.main()
