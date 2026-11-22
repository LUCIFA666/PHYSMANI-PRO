from types import MethodType
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.button_author_transit_v358 import BoundedApproach as Previous
from our_mani.multitask.button_reference_transit_v386 import BoundedApproach
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager


class Tests(unittest.TestCase):
    def fixture(self, cls):
        actor, model, robot = fixture.Tests().fixture()
        m = actor.continuous_reference_manager
        # Exercise the actual button86 stream implementation rather than37.
        m.check_stream = MethodType(ContinuousReferenceManager.check_stream, m)
        m.approach_transit = cls(m, {'maximum_TCP_step_m': .2})
        return actor, model, robot

    def test_previous6_prefix_is_rejected_by_actual4_stream_and_new4_prefix_dispatches(self):
        actor, _, robot = self.fixture(Previous)
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, 9.5))
        self.assertEqual(actor.continuous_reference_manager.reference_events[-1]['reference_gate_reason'],
                         'trajectory_joint_acceleration_limit')
        actor, _, robot = self.fixture(BoundedApproach)
        m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        self.assertTrue(m.pending['reference_gate_passed'])
        self.assertTrue(m.reference_reserve_gate.check_joints(robot, m.plan.q1, m.plan.v1, .5).feasible)
        self.assertGreater(next(e for e in m.approach_transit.events if e['event'] == 'partial_approach_staged')['nominal_duration'], .5)

    def test_dynamic_collision_remains_a_rejection(self):
        actor, model, robot = self.fixture(BoundedApproach)
        model.dynamic_block = True
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, 9.5))
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_remaining_segment_uses_the_reference_gate_instead_of_only_actual6(self):
        actor, model, robot = self.fixture(BoundedApproach)
        m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        m.reference_reserve_gate.model.acceleration_limits = np.full(7, .1)
        self.assertFalse(m.approach_transit._remaining_proof(robot, m.current_state, m.current_forecast))
        np.testing.assert_allclose(model.acceleration_limits, 6.)

    def test_stale_or_late_prefix_never_bypasses_original_budget(self):
        for stale in (True, False):
            actor, _, robot = self.fixture(BoundedApproach)
            m = actor.continuous_reference_manager
            if stale:
                m.current_forecast.metadata['observed_until'] = 0.
            self.assertFalse(m.approach_transit.try_approach(robot, 9.5 if stale else .05))
            self.assertIsNone(m.pending)


if __name__ == '__main__':
    unittest.main()
