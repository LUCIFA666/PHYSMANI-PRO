import unittest
import numpy as np

from our_mani.local_mpc_20261102.candidates import GeometricCandidates
from our_mani.local_mpc_20261102.records import HoldMetric


def state(t=0., held=False, z=.8):
    return dict(time=t, held=held, object_position=[0, 0, z],
                object_quaternion=[0, 0, 0, 1], tcp_position=[0, 0, z],
                tcp_quaternion=[0, 0, 0, 1])


class CandidateAndHoldTests(unittest.TestCase):
    def test_deadlines_are_committed_until_explicit_expiry(self):
        generator = GeometricCandidates()
        a = generator.update(state())
        b = generator.update(state(.7))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(len({tuple(c['object_tcp_quaternion']) for c in a}), 4)
        c = generator.update(state(1.21))
        self.assertNotEqual(a[0]['id'], c[0]['id'])
        self.assertEqual(generator.events[-1]['reason'], 'all_contact_deadlines_missed')

    def test_closed_command_cannot_pass_hold_and_lift(self):
        metric = HoldMetric()
        for t in np.arange(0, 1, .1):
            metric.update(state(t))
        self.assertFalse(metric.passed)
        metric.update(state(1, held=True))
        metric.update(state(1.6, held=True))
        self.assertFalse(metric.passed)
        metric.update(state(1.7, held=True, z=.85))
        self.assertFalse(metric.passed)
        metric.update(state(2.2, held=True, z=.85))
        self.assertTrue(metric.passed)
        metric.update(state(2.3, held=False, z=.8))
        self.assertEqual(metric.drops, 1)

    def test_previous_lift_does_not_qualify_a_later_low_grasp(self):
        metric = HoldMetric()
        metric.update(state())
        metric.update(state(.1, held=True, z=.86))
        metric.update(state(.2, held=False))
        metric.update(state(.3, held=True))
        metric.update(state(1., held=True))
        self.assertFalse(metric.passed)

