import unittest
import test_bounded_transport_v159 as prior
from our_mani.multitask.bounded_alignment_v352 import BoundedApproach
from our_mani.multitask import high_peg_alignment_native_v353 as policy


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = prior.Tests().fixture()
        c, m = actor.controller, actor.continuous_reference_manager
        c.phase_index = c.library.phase_order.index('align')
        m.approach_transit = BoundedApproach(m, {'maximum_TCP_step_m': .2})
        return actor, model, robot

    def test_align_can_stage_actual_and_reference_checked_partial_transit(self):
        actor, model, robot = self.fixture()
        m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        self.assertEqual(m.approach_transit.active['phase'], 'align')
        event = next(e for e in m.approach_transit.events if e['event'] == 'partial_approach_staged')
        self.assertTrue(event['actual_and_reference_full_gate_feasible'])
        self.assertGreater(len(model.collision_calls), 0)
        self.assertIsNone(actor.controller.goal)
        self.assertFalse(m.approach_transit.metrics()['stage_completion_claimed'])

    def test_dynamic_collision_cannot_be_waived_to_align(self):
        actor, model, robot = self.fixture()
        model.dynamic_block = True
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, 9.5))
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_alignment_cannot_spend_reserved_suffix_or_extend10s(self):
        actor, model, robot = self.fixture()
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, .5))
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_install_preserves_stream_stop_hook_and_requires_fresh_controller(self):
        actor, _, _ = self.fixture()
        m = actor.continuous_reference_manager
        m.approach_transit.normalized['maximum_TCP_step_m'] = .2
        task = 'insert_onto_rotating_peg_high_speed'
        root = prior.ROOT
        protocol = {'high_peg_alignment_v353': {task: policy.binding(root, {}, task)}}
        hook = m.check_stream
        policy.install(actor, protocol, root, task)
        self.assertEqual(m.check_stream, hook)
        self.assertIsInstance(m.approach_transit, BoundedApproach)
        actor.controller.feedback_step_calls += 1
        with self.assertRaises(ValueError):
            policy.install(actor, protocol, root, task)


if __name__ == '__main__':
    unittest.main()
