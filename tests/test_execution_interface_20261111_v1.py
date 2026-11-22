"""Execution boundary tests: commands, actual timing, failure and Native feedback."""
from copy import deepcopy
import unittest

import numpy as np

from our_mani.execution_interface_20261111_v1 import (
    ExecutionConfig, JointGoal, JointTrajectory, PhysManiExecutor,
)
from our_mani.local_mpc_20261102.executor import (
    ExecutionFailure, JointLimits, PoseTarget, ShortHorizonExecutor,
)
from our_mani.local_mpc_p2_20261103.reference import reference_step


class Backend:
    def __init__(self, lag=.35, budget=100):
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.target = self.q.copy()
        self.time = 0.
        self.budget = budget
        self.lag = lag
        self.writes = []
        self.gripper_calls = []
        self.gripper_done = False
        self.ticks = 0
        self.success_at = None
        self.held = False
        self.tick_error = None
        self.hidden_gripper_step = False
        self.ik_value = [.1, .2]

    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.dq.copy()
    def clock(self): return self.time
    def remaining_ticks(self): return self.budget - self.ticks
    def terminal(self): return self.success_at is not None and self.ticks >= self.success_at
    def solve_ik(self, position, quaternion): return self.ik_value

    def set_joint_target_positions(self, q):
        self.target = np.asarray(q).copy()
        self.writes.append((self.time, tuple(q)))

    def actuate_gripper(self, opening):
        self.gripper_calls.append((self.time, opening))
        if self.hidden_gripper_step:
            self.time += .05
        return self.gripper_done

    def tick(self, phase):
        if self.tick_error == "before":
            raise RuntimeError("before physical step")
        q = self.q + self.lag * (self.target - self.q)
        self.dq = (q - self.q) / .05
        self.q = q
        self.time += .05
        self.ticks += 1
        if self.tick_error == "after":
            raise RuntimeError("after physical step")

    def read_contact_feedback(self):
        return dict(timestamp=self.time, held=self.held)

    def native_task_judge_receipt(self):
        return dict(native_task_success_latched=self.terminal(), first_success_time=self.time if self.terminal() else None)


def make(backend=None, **options):
    backend = backend or Backend()
    limits = JointLimits([-2., -2.], [2., 2.], [2., 1.5])
    return backend, PhysManiExecutor(backend, limits, **options)


class ExecutionBoundaryTests(unittest.TestCase):
    def test_existing_goal_controller_matches_frozen_executor_under_lag_and_retargets(self):
        old_backend, new_backend = Backend(), Backend()
        limits = JointLimits([-2., -2.], [2., 2.], [2., 1.5])
        old, new = ShortHorizonExecutor(old_backend, limits), PhysManiExecutor(new_backend, limits)
        for goal in ([.6, -.3], [.6, -.3], [.6, -.3], [-.2, .1], [-.2, .1], [.1, .3]):
            before = new.state()
            a = old.execute_joint_target(goal, opening=0.)
            b = new.execute_joint_target(goal, opening=0.)
            np.testing.assert_allclose([r["reference_positions"] for r in a.ticks],
                                       [r["reference_positions"] for r in b.ticks], atol=1e-12, rtol=0)
            np.testing.assert_allclose(old_backend.q, new_backend.q, atol=1e-12, rtol=0)
            self.assertEqual(a.reference_reseeded, b.reference_reseeded)
            self.assertEqual(b.unexpected_physics_ticks, 0)
            self.assertFalse(before.get("faulted", False))

    def test_trajectory_points_reach_native_command_unchanged_and_feedback_is_actual(self):
        backend, executor = make()
        points = ((.005, -.002), (.010, -.004))
        receipt = executor.execute_command(JointTrajectory(0., points, (.05, .1), opening=0.))
        self.assertEqual([q for _, q in backend.writes], list(points))
        self.assertEqual(receipt.command_mode, "joint_trajectory")
        self.assertFalse(receipt.trajectory_reshaped)
        self.assertFalse(receipt.velocity_reference_actuated)
        self.assertNotEqual(receipt.feedback["q"], points[-1])
        self.assertEqual(receipt.feedback["timestamp"], .1)
        self.assertGreater(receipt.ticks[-1]["maximum_tracking_error_rad"], 0.)

    def test_same_numbers_have_explicitly_different_goal_and_reference_semantics(self):
        a, goal_executor = make()
        b, trajectory_executor = make()
        goal_executor.execute_command(JointGoal([.005, 0.], ticks=1))
        trajectory_executor.execute_command(JointTrajectory(0., [[.005, 0.]], [.05]))
        self.assertNotEqual(a.writes[0][1], b.writes[0][1])
        self.assertEqual(b.writes[0][1], (.005, 0.))

    def test_invalid_later_trajectory_point_is_rejected_before_any_write(self):
        backend, executor = make()
        with self.assertRaises(ExecutionFailure):
            executor.execute_command(JointTrajectory(0., [[.002, 0.], [.5, 0.]], [.05, .1], opening=0.))
        self.assertFalse(backend.writes)
        self.assertFalse(backend.gripper_calls)
        self.assertEqual(backend.ticks, 0)
        self.assertFalse(executor.state()["faulted"])
        self.assertEqual(executor.gripper_target, 1.)
        executor.execute_command(JointTrajectory(0., [[.002, 0.]], [.05]))

    def test_stale_or_future_trajectory_is_not_retimed(self):
        backend, executor = make()
        executor.execute_joint_target([.1, 0.], ticks=1)
        for at in (0., .1):
            with self.assertRaisesRegex(ValueError, "timestamp"):
                executor.execute_command(JointTrajectory(at, [[.005, 0.]], [.05]))
        self.assertEqual(backend.ticks, 1)
        self.assertEqual(len(backend.writes), 1)

    def test_explicit_joint_order_must_match(self):
        backend, executor = make(joint_names=("j1", "j2"))
        with self.assertRaisesRegex(ValueError, "joint order"):
            executor.execute_command(JointGoal([.1, 0.], joint_names=("j2", "j1")))
        self.assertFalse(backend.writes)
        executor.execute_command(JointGoal([.1, 0.], ticks=1, joint_names=("j1", "j2")))

    def test_single_tick_and_one_remaining_tick_do_not_overrun(self):
        backend, executor = make(Backend(budget=2))
        a = executor.execute_joint_target([.2, 0.], ticks=1)
        b = executor.execute_joint_target([.2, 0.])
        self.assertEqual(a.executed_ticks, 1)
        self.assertEqual((b.executed_ticks, b.truncated_reason), (1, "budget"))
        self.assertEqual(backend.ticks, 2)
        c = executor.execute_joint_target([.3, 0.], opening=0.)
        self.assertEqual(c.executed_ticks, 0)
        self.assertEqual(executor.gripper_target, 1.)

    def test_success_after_first_physics_tick_stops_second_tick(self):
        backend, executor = make()
        backend.success_at = 1
        receipt = executor.execute_joint_target([.2, 0.], opening=0.)
        self.assertEqual(receipt.executed_ticks, 1)
        self.assertEqual(receipt.truncated_reason, "native_task_success")
        self.assertTrue(receipt.feedback["native_task_success"])
        self.assertEqual(len(backend.writes), 1)

    def test_membership_and_completed_gripper_are_not_task_success(self):
        backend, executor = make()
        backend.held = backend.gripper_done = True
        receipt = executor.execute_joint_target([.1, 0.], opening=0.)
        self.assertEqual(receipt.executed_ticks, 2)
        self.assertTrue(receipt.feedback["held"])
        self.assertTrue(receipt.feedback["gripper_done"])
        self.assertFalse(receipt.feedback["native_task_success"])

    def test_completed_gripper_is_not_restarted_for_same_request(self):
        backend, executor = make()
        backend.gripper_done = True
        executor.execute_joint_target([.1, 0.], opening=0.)
        executor.execute_joint_target([.1, 0.], opening=0.)
        self.assertEqual(len(backend.gripper_calls), 1)
        executor.execute_joint_target([.1, 0.], opening=1., ticks=1)
        self.assertEqual(len(backend.gripper_calls), 2)

    def test_failure_after_physical_step_records_unconfirmed_step_and_faults(self):
        backend, executor = make()
        backend.tick_error = "after"
        with self.assertRaises(ExecutionFailure) as error:
            executor.execute_joint_target([.1, 0.])
        receipt = error.exception.receipt
        self.assertEqual(receipt.tick_calls_attempted, 1)
        self.assertEqual(receipt.tick_calls_completed, 0)
        self.assertEqual(receipt.physical_ticks_observed, 1)
        self.assertEqual(receipt.unexpected_physics_ticks, 1)
        self.assertTrue(executor.state()["faulted"])
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            executor.execute_joint_target([.1, 0.])
        self.assertEqual(backend.ticks, 1)

    def test_hidden_step_in_gripper_is_detected_before_counted_tick(self):
        backend, executor = make()
        backend.hidden_gripper_step = True
        with self.assertRaises(ExecutionFailure) as error:
            executor.execute_joint_target([.1, 0.], opening=0.)
        self.assertEqual(error.exception.receipt.tick_calls_attempted, 0)
        self.assertEqual(error.exception.receipt.physical_ticks_observed, 1)

    def test_external_step_invalidates_persistent_references(self):
        backend, executor = make()
        executor.execute_joint_target([.1, 0.], ticks=1)
        backend.tick("outside")
        with self.assertRaisesRegex(Exception, "outside this executor"):
            executor.execute_joint_target([.1, 0.])

    def test_velocity_reference_requires_explicit_nonactuated_role(self):
        backend, executor = make()
        action = dict(joint_goal=[.1, 0.], opening=0., tcp_velocity_reference=[.4, 0., 0.],
                      angular_velocity_reference=[0., 0., .2])
        with self.assertRaisesRegex(ValueError, "objective_only"):
            executor.execute_planner_action(action)
        self.assertFalse(backend.writes)
        result = executor.execute_planner_action(action, velocity_reference_role="objective_only")
        self.assertFalse(result.planner_contract["velocity_reference_actuated"])
        self.assertEqual(result.planner_contract["objective_references"]["tcp_velocity_reference"], (.4, 0., 0.))

    def test_unsupported_motor_commands_are_not_silently_dropped(self):
        backend, executor = make()
        for key in ("joint_velocity", "joint_goal_velocity", "dq", "torque"):
            with self.assertRaisesRegex(ValueError, "does not actuate"):
                executor.execute_planner_action(dict(joint_goal=[.1, 0.], opening=1., **{key: [0., 0.]}))
        for dt in (.2, float("nan")):
            with self.assertRaisesRegex(ValueError, "duration"):
                executor.execute_planner_action(dict(joint_goal=[.1, 0.], opening=1., dt=dt))
        self.assertFalse(backend.writes)

    def test_mode_switch_reseeds_goal_kernel_from_measured_state(self):
        backend, executor = make()
        executor.execute_command(JointTrajectory(0., [[.005, 0.], [.010, 0.]], [.05, .1]))
        actual_q, actual_dq = backend.q.copy(), backend.dq.copy()
        expected = reference_step(actual_q, actual_dq, [.2, 0.], {}, executor.limits)
        result = executor.execute_joint_target([.2, 0.], ticks=1)
        np.testing.assert_allclose(result.ticks[0]["reference_positions"], expected["q_ref"][0], atol=1e-12)
        self.assertTrue(result.reference_reseeded)

    def test_exported_goal_snapshot_is_reusable_by_mpc_reference_model(self):
        backend, executor = make()
        executor.execute_joint_target([.2, -.1])
        state = deepcopy(executor.state())
        expected = reference_step(backend.q, backend.dq, [.2, -.1], state, executor.limits)
        result = executor.execute_joint_target([.2, -.1], ticks=1)
        np.testing.assert_allclose(result.ticks[0]["reference_positions"], expected["q_ref"][0], atol=1e-12)

    def test_nonbinary_gripper_and_bad_sample_times_are_rejected(self):
        for value in (.3, float("nan"), "CLOSE"):
            with self.assertRaises(ValueError): JointGoal([0., 0.], opening=value)
        for times in ([0.], [.1], [.05, .11]):
            with self.assertRaises(ValueError): JointTrajectory(0., [[0., 0.]] * len(times), times)
        with self.assertRaises(ValueError): JointGoal([0., 0.], ticks=True)

    def test_pose_ik_failure_is_clear_and_does_not_step(self):
        backend, executor = make()
        backend.ik_value = None
        with self.assertRaisesRegex(ValueError, "IK failed"):
            executor.execute(PoseTarget([0., 0., 0.], [0., 0., 0., 1.]))
        self.assertFalse(backend.writes)
        self.assertEqual(backend.ticks, 0)

    def test_feedback_reads_do_not_advance_physics(self):
        backend, executor = make()
        feedback = executor.read_feedback()
        self.assertEqual(feedback["timestamp"], 0.)
        self.assertFalse(feedback["native_task_success"])
        self.assertEqual(backend.ticks, 0)

    def test_config_supports_single_tick_without_changing_physics_dt(self):
        backend, executor = make(config=ExecutionConfig(control_ticks=1))
        executor.execute_joint_target([.1, 0.])
        self.assertEqual(backend.ticks, 1)
        with self.assertRaises(ValueError): ExecutionConfig(physics_dt=.1)


if __name__ == "__main__":
    unittest.main()
