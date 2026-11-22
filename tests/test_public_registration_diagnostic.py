from types import SimpleNamespace as NS
import unittest
from our_mani.multitask.public_registration_diagnostic import install


class Tests(unittest.TestCase):
    def test_observer_returns_identical_public_values_and_records_without_mutation(self):
        states=[NS(timestamp=.5)];metadata=dict(bodies={'cup2':dict(observed_point_count=37,registration_valid=False)})
        tracker=NS(observe=lambda *a,**kw:(states,metadata));actor=NS(tracker=tracker,
            controller=NS(feedback_step_calls=0),metrics_report=lambda:{'original':True})
        install(actor);actual=tracker.observe('existing observation')
        self.assertIs(actual[0],states);self.assertIs(actual[1],metadata)
        r=actor.metrics_report()['public_registration_diagnostic'];self.assertEqual(r['events'][0]['bodies']['cup2']['observed_point_count'],37)
        self.assertFalse(r['policy_commands_changed']);self.assertEqual(r['extra_sensor_or_physics_queries'],0)
        r['events'][0]['bodies']['cup2']['observed_point_count']=0
        self.assertEqual(metadata['bodies']['cup2']['observed_point_count'],37)


if __name__=='__main__':unittest.main()
