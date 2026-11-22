from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
import unittest
import numpy as np
import test_button_intent_actor_v74 as original_factory
import test_button_segment_forecast_v155 as transit_fixture
from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.multitask.interaction import PolicyState, _world_target
from our_mani.multitask import button_intent_actor_v91 as actual_actor
from our_mani.multitask import button_author_native_v359 as policy
from our_mani.multitask.button_author_transit_v358 import BoundedApproach
from our_mani.multitask.reference_stop_guard_v274 import binding as stop_binding


class Tests(unittest.TestCase):
    def setUp(self):
        self.helper = original_factory.Tests()
        self.helper.setUp()
        self.helper.env.robot_model.acceleration_limits = np.full(7, 6.)
        self.addCleanup(self.helper.tearDown)

    def actor(self):
        with patch.object(original_factory, 'binding', actual_actor.binding), patch.object(
                original_factory, 'create_actor', actual_actor.create_actor):
            actor = self.helper.actor()
        task = 'push_moving_button'
        protocol = {'button_author_v359': {task: policy.binding(original_factory.ROOT, {}, task)},
                    'reference_stop_guard_v274': {task: stop_binding(original_factory.ROOT, {}, task)}}
        policy.install(actor, protocol, original_factory.ROOT, task)
        return actor

    def state(self, actor, offset, phase='approach'):
        c = actor.controller
        template = c.library.for_phase(phase)[0]
        body = ObjectState('button', 1., Pose([.2, 0., .8]), np.tile([.2, 0., .8], (3, 1)), np.zeros(3), np.zeros(3))
        target = _world_target(body.pose, np.zeros(3), np.zeros(3), template, 1.)[0]
        pose = Pose(target.position + [offset, 0., 0.], target.rotation)
        robot = RobotState(1., pose, np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7), gripper_open=True, contact=False)
        state = PolicyState(robot, {'button': body}, {'bodies': {'button': dict(
            directly_visible=True, registration_valid=True, motion_mode='external')}}, {'button': ('button',)})
        c.goal = SimpleNamespace(phase=phase, template=template, reference_id='button')
        return state, target

    def test_actual91_factory_retains_author30cm_stage_guard_and_native_press_ownership(self):
        actor = self.actor()
        state, target = self.state(actor, .29)
        self.assertTrue(actor.controller._phase_done(state, target, public_geometry=True))
        actor.controller._complete_phase(state)
        self.assertEqual(actor.controller.phase_index, 1)
        state, target = self.state(actor, 0., 'press')
        self.assertFalse(actor.controller._phase_done(state, target, public_geometry=True))
        self.assertFalse(actor.controller.library.metadata['public_press_completion_granted'])

    def test_original_author_distance_guard_cannot_transition_early(self):
        actor = self.actor()
        state, target = self.state(actor, .301)
        self.assertFalse(actor.controller._phase_done(state, target, public_geometry=True))
        self.assertIsInstance(actor.continuous_reference_manager.approach_transit, BoundedApproach)
        self.assertIn('reference_stop_guard_v274', actor.metrics_report())
        np.testing.assert_allclose(actor.continuous_reference_manager.native_gate.model.acceleration_limits, 6.)
        np.testing.assert_allclose(actor.continuous_reference_manager.reference_reserve_gate.model.acceleration_limits, 4.)

    def test_press_prefix_requires_both_original_motion_proofs_and_current_horizon(self):
        actor, model, robot = transit_fixture.Tests().fixture(.05)
        c, m = actor.controller, actor.continuous_reference_manager
        c.library.templates = tuple(replace(t, phase='press') for t in c.library.for_phase('approach'))
        c.library.phase_order = ('press',)
        c.phase_index = 0
        m.approach_transit = BoundedApproach(m, {'maximum_TCP_step_m': .2})
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        event = next(e for e in m.approach_transit.events if e['event'] == 'partial_approach_staged')
        self.assertTrue(event['actual_and_reference_full_gate_feasible'])
        self.assertEqual(event['phase'], 'press')
        self.assertAlmostEqual(event['predicted_reference_time'], robot.timestamp + .55)
        self.assertIsNone(c.goal)

    def test_dynamic_collision_or_global_budget_blocks_button_prefix(self):
        for blocked, remaining in [(True, 9.5), (False, .05)]:
            actor, model, robot = transit_fixture.Tests().fixture()
            model.dynamic_block = blocked
            m = actor.continuous_reference_manager
            m.approach_transit = BoundedApproach(m, {'maximum_TCP_step_m': .2})
            self.assertFalse(m.approach_transit.try_approach(robot, remaining))
            self.assertIsNone(m.pending)


if __name__ == '__main__':
    unittest.main()
