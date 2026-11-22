"""Public mode association and conservative press segmentation, CPU only."""
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask.mode_observer import (PublicModeObserver, AttachmentModeObserver,
                                             PublicCommandSegments, PublicModeError)


def robot(contact):
    return RobotState(0., Pose(np.zeros(3)), np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7), contact=contact)


def body(name, x=0., visible=True):
    return ObjectState(name, 0., Pose(np.array([x, 0., 0.])), np.ones((3, 3)), np.zeros(3), np.zeros(3), visible=visible)


class ModeTests(unittest.TestCase):
    def test_ambiguous_contact_invalidates_then_known_attachment_releases(self):
        assert AttachmentModeObserver is PublicModeObserver
        observer = PublicModeObserver(('a', 'b'), ('a', 'b'), .2)
        modes, invalid = observer.observe([body('a'), body('b')], robot(True))
        assert invalid == {'a', 'b'} and set(modes.values()) == {'external'}
        modes, invalid = observer.observe([body('a'), body('b', visible=False)], robot(True))
        assert modes['a'] == 'attached' and not invalid
        modes, invalid = observer.observe([], robot(False))
        assert modes['a'] == 'released' and observer.held is None
        assert observer.observe([], robot(False))[0]['a'] == 'released'

    def test_far_or_invisible_body_does_not_supply_privileged_attachment_id(self):
        observer = PublicModeObserver(('a',), ('a',), .2)
        _, invalid = observer.observe([body('a', .3)], robot(True))
        assert invalid == {'a'}
        _, invalid = observer.observe([body('a', visible=False)], robot(True))
        assert invalid == {'a'} and observer.held is None

    def test_press_is_latched_through_later_nonpress_commands(self):
        segment = PublicCommandSegments(('button', 'plate', 'wrap'))
        np.testing.assert_array_equal(segment.issue('joint_trajectory'), [True, True, True])
        np.testing.assert_array_equal(segment.issue('press'), [True, False, True])
        for kind in ('move', 'joint_trajectory', 'hold', 'no_action_terminal'):
            np.testing.assert_array_equal(segment.issue(kind), [True, False, True])
        assert segment.non_exogenous == {'plate'}
        with self.assertRaises(PublicModeError): segment.issue('native_joint_condition')


if __name__ == '__main__':
    unittest.main()
