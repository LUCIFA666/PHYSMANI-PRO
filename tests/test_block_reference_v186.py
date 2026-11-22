from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.block_reference_v186 import SmoothBlockEnvironment, public_decision_scope


class Model:
    name = 'linear_native_fixture'
    def __init__(self):
        self.lower = np.full(7, -10.)
        self.upper = np.full(7, 10.)
        self.velocity_limits = np.full(7, 2.)
        self.acceleration_limits = np.full(7, 6.)
        self.collision_query = None
        self.block = False
    def inverse(self, pose, seed): return np.r_[pose.position, np.zeros(4)]
    def forward(self, q): return Pose(np.asarray(q)[:3])
    def jacobian(self, q): return np.eye(6, 7)
    def collision(self, q, offset): return self.block


class Native:
    def __init__(self):
        self.q = np.zeros(7)
        self.target = self.q.copy()
        self.ledger = NS(ticks=0, dt=.05)
        self.robot_model = Model()
        self._requested_opening = 1.
        self.arm = NS(get_joint_positions=lambda:self.q.copy(), get_joint_target_positions=lambda:self.target.copy())
        self.actions = 0
        self.map_service = NS(check=lambda pose:False)
    @property
    def time(self): return self.ledger.ticks*.05
    def observe(self):
        return {}, RobotState(self.time, Pose(self.q[:3]), np.zeros(3), np.zeros(3), self.q.copy(), np.zeros(7), gripper_open=True, contact=False)
    def command_joints(self, q, *, opening=None, phase=None):
        self.target = np.array(q)
        self.q += .9*(self.target-self.q)
        self.ledger.ticks += 1
        self.actions += 1
        if opening is not None: self._requested_opening = opening
        return True


class Tests(unittest.TestCase):
    def fixture(self):
        env = Native()
        b = SmoothBlockEnvironment(env)
        b.hold_tick(); b.hold_tick()
        return env, b

    def decision(self, env, b):
        _, robot = b.observe()
        if b.last_window is not None: robot = replace(robot, joint_velocities=b.last_window.copy())
        body = ObjectState('target', robot.timestamp, Pose([.5, 0., 0.]),
            np.array([[.5,0.,0.],[.51,0.,0.],[.5,.01,0.]]), np.zeros(3), np.zeros(3))
        pose = Pose(robot.pose.position+[.015, 0., 0.])
        decision = dict(command='local_capture', pose=pose, opening=1.)
        b.supplied_decision(robot, body, decision, ConstantVelocityDynamics())
        return pose

    def test_rolling_reference_avoids_original_local_jump_and_retains_native_six(self):
        env, b = self.fixture()
        for _ in range(30): b.advance(self.decision(env,b), opening=1.)
        self.assertEqual(env.actions, 32)
        self.assertEqual(len(b.events), 32)
        self.assertTrue(b.receipt()['all_stage_actual_window_checks_passed'])
        self.assertGreater(env.q[0], .1)
        issued = [e for e in b.reference_events if e['event']=='local_reference_actually_dispatched']
        self.assertEqual(len(issued),30)
        self.assertTrue(all(e['evidence']['actual_gate']=='feasible_sampled_trajectory' for e in issued))
        np.testing.assert_array_equal(env.robot_model.acceleration_limits, np.full(7,6.))

    def test_original_collision_rejection_never_dispatches_a_new_moving_reference(self):
        env,b = self.fixture()
        env.robot_model.block = True
        b.advance(self.decision(env,b))
        np.testing.assert_array_equal(env.target, np.zeros(7))
        self.assertEqual(env.actions, 3)
        self.assertFalse(any(e['event']=='local_reference_actually_dispatched' for e in b.reference_events))

    def test_actuator_error_after_counted_tick_is_not_caught_and_dispatched_twice(self):
        env,b = self.fixture()
        original = env.command_joints
        def invalid_after_tick(*args,**kwargs):
            original(*args,**kwargs)
            raise ValueError('backend failed after actual tick')
        env.command_joints = invalid_after_tick
        with self.assertRaisesRegex(ValueError,'after actual tick'):
            b.advance(self.decision(env,b))
        self.assertEqual(env.actions,3)

    def test_unforwarded_or_stale_decision_cannot_reach_local_actuator(self):
        env,b = self.fixture()
        with self.assertRaises(ValueError): b.advance(Pose([0.,0.,0.]))
        pose = self.decision(env,b)
        env.ledger.ticks += 1
        with self.assertRaises(ValueError): b.advance(pose)
        self.assertEqual(env.actions,2)

    def test_decision_scope_restores_original_constructor_and_preserves_policy_output(self):
        env,b = self.fixture()
        _,robot=b.observe()
        body=NS()
        decision=dict(command='hold')
        original=lambda *args,**kwargs:NS(step=lambda *args:decision,dynamics='original-model')
        evaluator=NS(CandidateInterception=original)
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception()
            self.assertIs(workflow.step(robot,body,10.),decision)
            self.assertIs(b.public_body,body)
            self.assertEqual(b.local_dynamics,'original-model')
        self.assertIs(evaluator.CandidateInterception,original)


if __name__ == '__main__': unittest.main()
