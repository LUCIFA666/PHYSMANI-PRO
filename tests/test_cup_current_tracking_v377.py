from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_bounded_transport_v159 as prior
from our_mani.contracts import Pose, ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.interaction import InteractionCandidate, _world_target
from our_mani.multitask.candidate_success import FEATURE_NAMES
from our_mani.multitask import cup_current_tracking_v377 as policy


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = prior.Tests().fixture()
        c, m = actor.controller, actor.continuous_reference_manager
        task = 'place_cups_on_rotating_frame'
        protocol = {'learned_artifacts': {task: {'interaction': {'path': 'CPU_actual_fitted_library'}}}}
        protocol['cup_current_tracking_v377'] = {task: policy.binding(prior.ROOT, protocol, task)}
        policy.install(actor, protocol, prior.ROOT, task)
        template = c.library.for_phase('transport')[0]
        body = ObjectState('spoke0', .5, Pose([.2, 0., .7]), np.tile([.2, 0., .7], (3, 1)),
            np.array([.1, 0., 0.]), np.zeros(3))
        target = _world_target(body.pose, np.zeros(3), np.zeros(3), template, 1.)[0]
        q = model.inverse(Pose(target.position + [.05, 0., 0.], target.rotation), robot.joints)
        robot = replace(robot, joints=q, joint_velocities=np.zeros(7), pose=model.forward(q))
        state = replace(m.current_state, robot=robot, bodies={'spoke0': body})
        forecast = ConstantVelocityDynamics().predict([body], 1.1, .05)
        forecast.metadata['observed_until'] = .5
        m.current_state, m.current_forecast, m.forecast_observed_until = state, forecast, .5
        checked = m.native_gate.check_joints(robot, q, np.zeros(7), .3)
        c.goal = InteractionCandidate('CPU_transport', 'transport', c.phase_index, 'spoke0', ('cup0',),
            template, 0., .3, target, np.zeros(3), np.zeros(3), body.pose, np.zeros(3), np.zeros(3),
            body, checked, 0., template.duration_seconds, 2., .85, True, {})
        c.attempts.begin('transport', 'CPU_transport', 0., np.zeros(len(FEATURE_NAMES)))
        m.plan = JointTrajectory(.3, .5, q, np.zeros(7), q, np.zeros(7))
        m.plan_kind, m.plan_candidate_id, m.invalidated = 'braking', 'CPU_completed_brake', False
        m.last_confirmed.update(time=.5, joints=q.tolist(), nominal_velocity=np.zeros(7).tolist())
        return actor, model, robot

    def test_completed_brake_renews_real_fitted_transport_with_original_dual_proofs(self):
        actor, model, robot = self.fixture()
        c, m = actor.controller, actor.continuous_reference_manager
        old, records = c.goal, deepcopy(c.attempts.to_report())
        proof = m.check_stream(robot, .05)
        self.assertTrue(proof.feasible, proof)
        self.assertEqual(m.plan_kind, 'current_acquire_rolling')
        self.assertGreater(m.plan.end, robot.timestamp)
        self.assertIs(c.goal, old)
        self.assertEqual(c.attempts.to_report(), records)
        self.assertTrue(m.rolling_events[-1]['actual_and_reference_full_gate_feasible'])
        self.assertGreater(len(model.collision_calls), 0)

    def test_fresh_dynamic_collision_rejects_renewal(self):
        actor, model, robot = self.fixture()
        model.dynamic_block = True
        result = actor.continuous_reference_manager.check_stream(robot, .05)
        self.assertFalse(result.feasible)
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_stale_forecast_or_expired_original_phase_cannot_be_renewed(self):
        for stale in (True, False):
            actor, _, robot = self.fixture()
            m = actor.continuous_reference_manager
            if stale:
                m.current_forecast.metadata['observed_until'] = 0.
            else:
                actor.controller.goal = replace(actor.controller.goal, phase_budget=.49)
            self.assertFalse(m.check_stream(robot, .05).feasible)
            self.assertIsNone(m.pending)

    def test_no_renewal_before_original_stream_endpoint(self):
        actor, _, robot = self.fixture()
        m = actor.continuous_reference_manager
        m.plan = replace(m.plan, end=.6)
        m.check_stream(robot, .05)
        self.assertFalse(actor.metrics_report()['cup_current_tracking_v377']['events'])


if __name__ == '__main__':
    unittest.main()
