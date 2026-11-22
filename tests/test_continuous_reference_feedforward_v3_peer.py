"""Independent v3 integration boundaries; robots/physics are CPU fixtures.

The actual manager, dual reachability, rolling reference, close gate and
dispatcher run unchanged. This is not native tracking or task-success proof.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.continuous_direct_counted_setpoint_v3 import CountedDirectDispatcher
from tests import test_continuous_reference_feedforward_v3 as owner_fixtures
from tests.test_direct_counted_setpoint import Environment


def rolling_fixture():
    return owner_fixtures.ContinuousTests().rolling_fixture()


def propose(manager, measured, opening=1.):
    checked = manager.check_stream(measured, .05)
    assert checked is not None and checked.feasible, checked
    manager.pending_dispatch = manager.pending
    certificate = deepcopy(manager.pending)
    certificate.update(issued=False, dispatch_authorized=True)
    return dict(command='joint_trajectory', phase='acquire', opening=opening,
                public_command_kind='joint_trajectory', joints=checked.joints.copy(),
                joint_velocities=checked.joint_velocities.copy(),
                feedforward_execution=certificate)


def environment(measured):
    class CountedEnvironment(Environment):
        @property
        def time(self):
            return self.ledger.ticks * self.ledger.dt
    env = CountedEnvironment()
    env.ledger.ticks = round(measured.timestamp / .05)
    env.arm.q = measured.joints.copy()
    env.arm.target = measured.joints.copy()
    env.arm.velocity = measured.joint_velocities.copy()
    # GateModel's toy rotation-vector joints span +/-10; use the same fixture
    # limits at dispatch rather than clipping a valid fixture orientation.
    env.robot_model = SimpleNamespace(lower=np.full(7, -10.), upper=np.full(7, 10.),
                                     velocity_limits=np.ones(7), acceleration_limits=np.full(7, 6.))
    return env


class ContinuousReferenceV3IndependentTests(unittest.TestCase):
    def test_learned_close_retains_rotation_gate_and_actual_dispatch_confirmation(self):
        actor, _, measured = rolling_fixture()
        controller = actor.controller
        manager = actor.continuous_reference_manager
        template = controller.goal.template
        final = Pose(np.array([.1, .2, .7]), template.at(1.)[0].rotation)
        close = replace(measured, pose=Pose(final.position + [.015, 0, 0], final.rotation))
        bad_angle = replace(close, pose=Pose(close.pose.position,
                            Rotation.from_rotvec([.11, 0, 0]).as_matrix() @ final.rotation))
        self.assertEqual(controller._gripper_opening(SimpleNamespace(robot=bad_angle), 1., final), 1.)
        self.assertIsNone(controller._close_edge_issued_at)
        self.assertEqual(controller._gripper_opening(SimpleNamespace(robot=close), 1., final), 0.)
        event = controller.gripper_timing_events[-1]
        command = propose(manager, measured, opening=0.)
        self.assertFalse(command['feedforward_execution']['issued'])
        self.assertNotIn('actual_opening_request_issued', event)
        before = deepcopy(manager.last_confirmed)
        env = environment(measured)
        dispatcher = CountedDirectDispatcher()
        dispatcher.bind_manager(manager)
        result = dispatcher.dispatch(env, command)
        self.assertEqual(result['persistent_reference_execution']['schema'],
                         'r10_continuous_reference_direct_counted_setpoint_v3')
        self.assertTrue(event['actual_opening_request_issued'])
        self.assertEqual(event['actual_opening_request'], 0.)
        self.assertEqual(env.ledger.grasp_requests, 1)
        self.assertEqual(manager.last_confirmed['time'], env.time)
        self.assertGreater(manager.last_confirmed['time'], before['time'])
        self.assertTrue(manager.reference_events[-1]['issued'])
        self.assertEqual(controller._gripper_opening(SimpleNamespace(robot=close), 1., final), 0.)
        self.assertEqual(controller._close_edge_issued_at, measured.timestamp)
        self.assertEqual(controller.gripper_timing_events[-1]['reason'],
                         'one_close_edge_already_issued_this_attempt')

    def test_nonzero_stream_endpoint_renews_from_issued_boundary_without_new_goal_epoch(self):
        actor, _, measured = rolling_fixture()
        controller = actor.controller
        manager = actor.continuous_reference_manager
        first = propose(manager, measured)
        env = environment(measured)
        dispatcher = CountedDirectDispatcher()
        dispatcher.bind_manager(manager)
        dispatcher.dispatch(env, first)
        plan = manager.plan
        self.assertGreater(np.linalg.norm(plan.v1), 0.)
        goal = controller.goal
        identity = (goal.candidate_id, goal.selected_at, goal.phase_budget, goal.suffix_budget)
        end = plan.end
        # This fixture supplies an already issued endpoint, not an inferred
        # motor derivative; the renewed proof must keep that exact boundary.
        manager.last_confirmed = dict(time=end, joints=plan.q1.tolist(),
                                      nominal_velocity=plan.v1.tolist(), command='joint_trajectory')
        endpoint = replace(measured, timestamp=end, joints=plan.q1.copy(), joint_velocities=plan.v1.copy())
        state = manager.current_state
        pose, velocity, angular = manager.current_forecast.at('ball', end)
        old_body = state.bodies['ball']
        body = replace(old_body, timestamp=end, pose=pose,
                       points_world=old_body.points_world + pose.position - old_body.pose.position,
                       linear_velocity=velocity, angular_velocity=angular)
        manager.current_state = replace(state, robot=endpoint, bodies={'ball': body})
        manager.current_forecast = ConstantVelocityDynamics().predict([body], 1.1, .05)
        manager.current_forecast.metadata['observed_until'] = end
        manager.forecast_observed_until = end
        checked = manager.check_stream(endpoint, .05)
        self.assertIsNotNone(checked)
        self.assertTrue(checked.feasible, checked.reason)
        self.assertIs(controller.goal, goal)
        self.assertEqual((goal.candidate_id, goal.selected_at, goal.phase_budget, goal.suffix_budget), identity)
        np.testing.assert_allclose(manager.plan.q0, plan.q1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(manager.plan.v0, plan.v1, rtol=0, atol=1e-12)
        self.assertLessEqual(manager.plan.end, goal.selected_at + goal.phase_budget + 1e-8)
        self.assertFalse(manager.pending['issued'])
        self.assertTrue(manager.rolling_events[-1]['actual_and_reference_full_gate_feasible'])

    def test_stale_forecast_and_deadline_do_not_commit_a_renewal_or_reset_epoch(self):
        for failure in ('stale', 'deadline'):
            with self.subTest(failure=failure):
                actor, _, measured = rolling_fixture()
                manager = actor.continuous_reference_manager
                goal = actor.controller.goal
                old_plan = manager.plan
                before = deepcopy(manager.last_confirmed)
                if failure == 'stale':
                    manager.current_forecast.metadata['observed_until'] = 0.
                else:
                    goal.phase_budget = measured.timestamp + .049
                epoch = goal.selected_at
                self.assertFalse(manager.roll_current_reference(measured))
                self.assertIs(manager.plan, old_plan)
                self.assertEqual(manager.last_confirmed, before)
                self.assertEqual(goal.selected_at, epoch)
                self.assertFalse(manager.rolling_events[-1]['committed'])
                self.assertEqual(manager.confirmed_reissues_by_phase_index, {})


if __name__ == '__main__':
    unittest.main()
