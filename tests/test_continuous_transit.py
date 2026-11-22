from dataclasses import replace
from types import MethodType
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager
from our_mani.multitask.continuous_transit import ContinuousTransit


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = fixture.Tests().fixture()
        m = actor.continuous_reference_manager
        m.check_stream = MethodType(ContinuousReferenceManager.check_stream, m)
        m.approach_transit = ContinuousTransit(m, {'maximum_TCP_step_m': .2},
            phases=('approach',))
        return actor, model, robot

    def advance(self, manager, model, robot, time, *, lag=0.):
        q, v = manager.plan.at(time)
        actual_q = q.copy(); actual_q[0] -= lag
        robot = replace(robot, timestamp=time, joints=actual_q,
            joint_velocities=v, pose=model.forward(actual_q))
        manager.last_confirmed.update(time=time, joints=q.tolist(), nominal_velocity=v.tolist())
        manager.pending = None
        manager.current_state = replace(manager.current_state, robot=robot,
            bodies={key: replace(body, timestamp=time)
                    for key, body in manager.current_state.bodies.items()})
        manager.current_forecast.metadata['observed_until'] = time
        manager.forecast_observed_until = time
        return robot

    def test_moving_boundary_is_replanned_without_a_zero_velocity_hold(self):
        actor, model, robot = self.fixture(); m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        old = m.plan; deadline = m.approach_transit.active['deadline']
        robot = self.advance(m, model, robot, .7, lag=.001)
        q, v = old.at(.7)
        self.assertGreater(np.linalg.norm(v), .01)
        self.assertTrue(m.approach_transit.try_approach(robot, 9.3))
        self.assertAlmostEqual(m.plan.start, .7)
        np.testing.assert_allclose(m.plan.q0, q)
        np.testing.assert_allclose(m.plan.v0, v)
        self.assertEqual(m.approach_transit.active['deadline'], deadline)
        self.assertTrue(m.pending['reference_gate_passed'])
        self.assertIsNone(actor.controller.goal)

    def test_collision_during_replan_cannot_resume_old_reference(self):
        actor, model, robot = self.fixture(); m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        robot = self.advance(m, model, robot, .7)
        model.dynamic_block = True
        self.assertFalse(m.approach_transit.try_approach(robot, 9.3))
        self.assertIsNone(m.pending)
        self.assertTrue(m.invalidated)
        self.assertIsNone(m.plan)

    def test_stale_forecast_and_global_budget_reject_a_renewal(self):
        for stale in (True, False):
            actor, model, robot = self.fixture(); m = actor.continuous_reference_manager
            self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
            robot = self.advance(m, model, robot, .7)
            if stale:
                m.current_forecast.metadata['observed_until'] = .4
            self.assertFalse(m.approach_transit.try_approach(robot, 9.3 if stale else .05))
            self.assertIsNone(m.pending)

    def test_constraint_drift_fails_before_a_reference_is_proposed(self):
        actor, _, robot = self.fixture(); m = actor.continuous_reference_manager
        m.reference_reserve_gate.model.acceleration_limits = np.full(7, 6.)
        with self.assertRaisesRegex(ValueError, 'authority changed'):
            m.approach_transit.try_approach(robot, 9.5)
        self.assertIsNone(m.pending)


if __name__ == '__main__':
    unittest.main()
