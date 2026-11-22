from copy import deepcopy
from dataclasses import replace
from types import MethodType
from pathlib import Path
import json
import unittest
import numpy as np
import test_bounded_transport_v159 as prior
import test_button_segment_forecast_v155 as button_fixture
from our_mani.contracts import Pose, ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary
from our_mani.multitask.bounded_task_transit import BoundedApproach
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = prior.Tests().fixture()
        c, m = actor.controller, actor.continuous_reference_manager
        config = json.loads((prior.ROOT/'artifacts/framework/multitask_r10/high_peg_retarget_v374_setup/configuration.json').read_text())
        pointer = config['learned_artifacts']['insert_onto_rotating_peg_high_speed']['interaction']
        c.library = c.generator.library = PrimarySequenceLibrary.load(prior.ROOT/pointer['path'])
        c.phase_index = c.library.phase_order.index('insert')
        body = ObjectState('peg0', robot.timestamp, Pose([1.5, 0., .7]), np.tile([1.5, 0., .7], (3, 1)), np.zeros(3), np.zeros(3))
        state = replace(m.current_state, bodies={'peg0': body}, role_bindings={'selected_peg': ('peg0',), 'ring': ('ring',)},
            perception_metadata={'bodies': {'peg0': dict(directly_visible=True, registration_valid=True,
                orientation_observable=True, motion_mode='external')}})
        forecast = ConstantVelocityDynamics().predict([body], 1.1, .05)
        forecast.metadata['observed_until'] = robot.timestamp
        m.current_state, m.current_forecast = state, forecast
        m.approach_transit = BoundedApproach(m, {'maximum_TCP_step_m': .2}, phases=('insert',))
        return actor, model, robot

    def test_actual_fitted_insert_can_advance_without_faking_candidate_or_completion(self):
        actor, model, robot = self.fixture()
        c, m = actor.controller, actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        self.assertEqual(m.approach_transit.active['phase'], 'insert')
        self.assertIsNone(c.goal)
        self.assertTrue(m.pending['reference_gate_passed'])
        self.assertFalse(m.approach_transit.metrics()['stage_completion_claimed'])
        self.assertGreater(len(model.collision_calls), 0)

    def test_collision_or_global_budget_prevents_insert_prefix(self):
        for collision, remaining in [(True, 9.5), (False, .05)]:
            actor, model, robot = self.fixture()
            model.dynamic_block = collision
            self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, remaining))
            self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_unconfigured_phase_and_stale_forecast_cannot_get_a_new_path(self):
        actor, _, robot = self.fixture()
        m = actor.continuous_reference_manager
        m.current_forecast.metadata['observed_until'] = 0.
        self.assertFalse(m.approach_transit.try_approach(robot, 9.5))
        m.current_forecast.metadata['observed_until'] = robot.timestamp
        actor.controller.phase_index = actor.controller.library.phase_order.index('align')
        self.assertFalse(m.approach_transit.try_approach(robot, 9.5))

    def test_actual_button4_stream_and_full_prefix_use_the_same_reference_gate(self):
        actor, model, robot = button_fixture.Tests().fixture()
        m = actor.continuous_reference_manager
        m.check_stream = MethodType(ContinuousReferenceManager.check_stream, m)
        m.approach_transit = BoundedApproach(m, {'maximum_TCP_step_m': .2}, phases=('approach',))
        self.assertIs(m.approach_transit.reference_gate, m.reference_reserve_gate)
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        self.assertTrue(m.reference_reserve_gate.check_joints(robot, m.plan.q1, m.plan.v1, .5).feasible)
        np.testing.assert_allclose(model.acceleration_limits, 6.)


if __name__ == '__main__':
    unittest.main()
