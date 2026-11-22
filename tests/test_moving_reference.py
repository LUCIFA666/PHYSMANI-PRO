from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.interaction import CURRENT
from our_mani.multitask.moving_reference import install_moving_reference


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = fixture.Tests().fixture()
        m, c = actor.continuous_reference_manager, actor.controller
        c.library.templates = tuple(replace(t, phase='press') for t in c.library.for_phase('approach'))
        c.library.phase_order = ('press',)
        c.phase_index = 0
        return actor, model, robot

    def test_actual_dual_gate_preserves_nonzero_endpoint_velocity(self):
        actor, model, robot = self.fixture(); m = actor.continuous_reference_manager
        install_moving_reference(actor, phases=('press',))
        v = np.zeros(7); v[0] = .1
        q = robot.joints.copy(); q[0] += .025
        check = m.dual_gate.check_joints(robot, q, v, .5)
        self.assertTrue(check.feasible, check.reason)
        np.testing.assert_allclose(check.joint_velocities, v)
        self.assertTrue(check.evidence['continuous_reference_proof']['moving_target_velocity_retained'])
        np.testing.assert_allclose(model.acceleration_limits, 6.)

    def test_dynamic_collision_still_rejects_tracking(self):
        actor, model, robot = self.fixture(); m = actor.continuous_reference_manager
        install_moving_reference(actor, phases=('press',))
        model.dynamic_block = True
        check = m.dual_gate.check_joints(robot, robot.joints, np.zeros(7), .5)
        self.assertFalse(check.feasible)

    def test_failed_roll_does_not_turn_into_stale_endpoint_hold(self):
        actor, _, robot = self.fixture(); m, c = actor.continuous_reference_manager, actor.controller
        install_moving_reference(actor, phases=('press',))
        c.goal = SimpleNamespace(phase='press', template=c.library.templates[0],
            candidate_id='current', selected_at=.5, phase_budget=9.5)
        m.plan = None
        m.roll_current_reference = lambda _robot: False
        check = m.check_stream(robot, .05)
        self.assertFalse(check.feasible)
        self.assertEqual(check.reason, 'moving_reference_requires_fresh_feasible_roll')
        self.assertIsNone(m.pending)

    def test_offsets_are_preserved_and_attachment_is_not_external_motion(self):
        actor, _, _ = self.fixture()
        before = actor.controller.library.templates[0].positions.copy()
        install_moving_reference(actor, phases=('press',))
        template = actor.controller.library.templates[0]
        self.assertEqual(template.reference_time_basis, CURRENT)
        np.testing.assert_array_equal(template.positions, before)
        actor, _, _ = self.fixture()
        actor.controller.library.templates = tuple(replace(t, reference_mode='attached')
            for t in actor.controller.library.templates)
        with self.assertRaisesRegex(ValueError, 'external'):
            install_moving_reference(actor, phases=('press',))


if __name__ == '__main__':
    unittest.main()
