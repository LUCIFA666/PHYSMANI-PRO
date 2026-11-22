import unittest

from our_mani.local_mpc_20261102.executor import (
    ClockInvariantError, ExecutionConfig, ExecutionFailure, JointLimits,
    PoseTarget, ShortHorizonExecutor,
)


class FakeBackend:
    def __init__(self, *, remaining=20, lag=1.0):
        self.q = [0.0, 0.0]
        self.v = [0.0, 0.0]
        self.target = list(self.q)
        self.time = 0.0
        self.remaining = remaining
        self.lag = lag
        self.events = []
        self.opening = 1.0
        self.grip_progress = 0
        self.ik_calls = 0
        self.fail_tick = None
        self.fail_after_step = False
        self.terminal_at_tick = None
        self.hidden_step_in_gripper = False

    def get_joint_positions(self):
        return self.q[:]

    def get_joint_velocities(self):
        return self.v[:]

    def solve_ik(self, position, quaternion):
        self.ik_calls += 1
        return position[:2]

    def set_joint_target_positions(self, positions):
        self.target = list(positions)
        self.events.append(("arm", self.time, tuple(positions)))

    def actuate_gripper(self, opening):
        self.events.append(("gripper", self.time, opening))
        if opening != self.opening:
            self.opening, self.grip_progress = opening, 0
        self.grip_progress += 1
        if self.hidden_step_in_gripper:
            self.time += 0.05
        return self.grip_progress >= 4

    def tick(self, phase):
        self.events.append(("tick", self.time, phase))
        if self.fail_tick == len([event for event in self.events if event[0] == "tick"]):
            raise RuntimeError("injected tick failure")
        next_q = [q + self.lag * (target - q) for q, target in zip(self.q, self.target)]
        self.v = [(after - before) / 0.05 for after, before in zip(next_q, self.q)]
        self.q = next_q
        self.time += 0.05
        self.remaining -= 1
        if self.fail_after_step:
            raise RuntimeError("injected failure after physical step")

    def clock(self):
        return self.time

    def remaining_ticks(self):
        return self.remaining

    def terminal(self):
        return self.terminal_at_tick is not None and round(self.time / 0.05) >= self.terminal_at_tick


def make_executor(backend=None, **limit_options):
    backend = backend or FakeBackend()
    limits = JointLimits([-2.0, -2.0], [2.0, 2.0], [2.0, 1.5],
        provenance={"native_source": "fake native model"}, **limit_options)
    return backend, ShortHorizonExecutor(backend, limits)


class ShortHorizonExecutorTests(unittest.TestCase):
    def test_two_ticks_arm_and_gripper_share_clock_and_no_hidden_steps(self):
        backend, executor = make_executor()
        receipt = executor.execute(PoseTarget([0.5, 0.2, 0.0], [0, 0, 0, 1], 0))
        assert receipt.executed_ticks == receipt.tick_calls_completed == 2
        self.assertAlmostEqual(receipt.executed_duration, 0.1)
        assert [event[0] for event in backend.events] == ["arm", "gripper", "tick"] * 2
        self.assertEqual([event[1] for event in backend.events], [0, 0, 0, 0.05, 0.05, 0.05])
        assert receipt.ik_wall_seconds >= 0
        assert receipt.planning_wall_seconds >= receipt.ik_wall_seconds
        assert receipt.limits["controller_limits_are_native_physical_parameters"] is False
        assert receipt.limits["native_joint_velocity_limits"] == [2.0, 1.5]


    def test_same_target_persists_reference_and_gripper_across_cycles(self):
        backend, executor = make_executor()
        target = PoseTarget([0.5, 0.2, 0], [0, 0, 0, 1], 0)
        first = executor.execute(target)
        last_reference = first.ticks[-1]["reference_positions"]
        # Deliberate lag proves an unchanged target continues reference rather than
        # silently replacing it with the measured state.
        backend.q = [value * 0.6 for value in backend.q]
        second = executor.execute(target)
        assert backend.ik_calls == 1
        assert second.reference_reseeded is False
        assert second.ticks[0]["reference_positions"] != second.initial_measured_positions
        assert second.ticks[0]["reference_positions"][0] > last_reference[0]
        assert second.gripper_target == 0
        assert first.gripper_done is False and second.gripper_done is True
        assert len([event for event in backend.events if event[0] == "tick"]) == 4


    def test_replacement_uses_actual_position_and_nonzero_velocity(self):
        backend, executor = make_executor()
        executor.execute_joint_target([0.8, 0.3])
        backend.q, backend.v = [0.07, 0.02], [0.4, -0.1]
        receipt = executor.execute_joint_target([-0.4, -0.2], opening=0)
        assert receipt.reference_reseeded is True
        assert receipt.initial_measured_positions == (0.07, 0.02)
        assert receipt.initial_measured_velocities == (0.4, -0.1)
        first = receipt.ticks[0]
        for index in range(2):
            acceleration = first["reference_accelerations"][index]
            backend_value = receipt.initial_measured_velocities[index] + acceleration * 0.05
            self.assertAlmostEqual(first["reference_velocities"][index], backend_value)
            self.assertAlmostEqual(first["reference_positions"][index],
                receipt.initial_measured_positions[index] + receipt.initial_measured_velocities[index] * 0.05
                + 0.5 * acceleration * 0.05 ** 2)
            assert abs(backend_value) <= executor.limits.effective_velocity[index]


    def test_one_remaining_tick_returns_partial_cycle(self):
        backend, executor = make_executor(FakeBackend(remaining=1))
        receipt = executor.execute_joint_target([0.5, 0.2], opening=0)
        assert receipt.status == "truncated" and receipt.truncated_reason == "budget"
        assert receipt.executed_ticks == receipt.tick_calls_completed == 1
        self.assertAlmostEqual(receipt.executed_duration, 0.05)
        assert backend.grip_progress == 1


    def test_tick_exception_returns_reviewable_partial_receipt_without_extra_steps(self):
        backend, executor = make_executor()
        backend.fail_tick = 2
        with self.assertRaises(ExecutionFailure) as failure:
            executor.execute_joint_target([0.5, 0.2], opening=0)
        receipt = failure.exception.receipt
        assert receipt is executor.last_receipt and receipt.status == "failed"
        assert receipt.tick_calls_attempted == 2 and receipt.executed_ticks == 1
        self.assertAlmostEqual(receipt.executed_duration, 0.05)
        assert "injected tick failure" in receipt.error
        self.assertAlmostEqual(backend.time, 0.05)


    def test_gripper_hidden_step_is_detected_before_counted_tick(self):
        backend, executor = make_executor()
        backend.hidden_step_in_gripper = True
        with self.assertRaises(ExecutionFailure) as failure:
            executor.execute_joint_target([0.5, 0.2], opening=0)
        assert isinstance(failure.exception.__cause__, ClockInvariantError)
        assert failure.exception.receipt.tick_calls_attempted == 0
        self.assertAlmostEqual(failure.exception.receipt.executed_duration, 0.05)
        assert failure.exception.receipt.physical_ticks_observed == 1
        assert failure.exception.receipt.unexpected_physics_ticks == 1


    def test_failure_after_step_reports_observed_physics_and_reseeds_next_command(self):
        backend, executor = make_executor()
        backend.fail_after_step = True
        with self.assertRaises(ExecutionFailure) as failure:
            executor.execute_joint_target([0.5, 0.2])
        receipt = failure.exception.receipt
        assert receipt.executed_ticks == receipt.tick_calls_completed == 0
        assert receipt.physical_ticks_observed == receipt.unexpected_physics_ticks == 1
        backend.fail_after_step = False
        retry = executor.execute_joint_target([0.5, 0.2])
        assert retry.reference_reseeded is True


    def test_thirteen_tick_budget_ends_with_one_tick_without_overrun(self):
        backend, executor = make_executor(FakeBackend(remaining=13))
        receipts = []
        while backend.remaining:
            receipts.append(executor.execute_joint_target([0.0, 0.0], opening=0))
        assert [receipt.executed_ticks for receipt in receipts] == [2] * 6 + [1]
        assert sum(receipt.physical_ticks_observed for receipt in receipts) == 13
        assert all(receipt.unexpected_physics_ticks == 0 for receipt in receipts)
        self.assertAlmostEqual(backend.time, 0.65)
        zero = executor.execute_joint_target([0.0, 0.0])
        assert zero.executed_ticks == 0 and zero.truncated_reason == "budget"
        self.assertAlmostEqual(backend.time, 0.65)


    def test_terminal_after_first_tick_returns_without_advancing_again(self):
        backend, executor = make_executor()
        backend.terminal_at_tick = 1
        receipt = executor.execute_joint_target([0.5, 0.2])
        assert receipt.executed_ticks == 1 and receipt.truncated_reason == "terminal"
        self.assertAlmostEqual(receipt.executed_duration, 0.05)


    def test_reference_derivatives_respect_explicit_controller_limits(self):
        backend, executor = make_executor(acceleration=3.0, jerk=10.0)
        rows = []
        for _ in range(4):
            rows.extend(executor.execute_joint_target([1.0, -1.0]).ticks)
        previous_acceleration = [0.0, 0.0]
        for row in rows:
            for index, acceleration in enumerate(row["reference_accelerations"]):
                assert abs(acceleration) <= 3.0 + 1e-9
                assert abs(acceleration - previous_acceleration[index]) <= 10.0 * 0.05 + 1e-9
                assert abs(row["reference_velocities"][index]) <= executor.limits.effective_velocity[index] + 1e-9
            previous_acceleration = row["reference_accelerations"]


    def test_tracking_error_is_measured_not_reported_as_reference_accuracy(self):
        backend, executor = make_executor(FakeBackend(lag=0.2))
        receipt = executor.execute_joint_target([0.8, 0.3])
        assert receipt.ticks[-1]["maximum_tracking_error_rad"] > 0
        assert receipt.ticks[-1]["measured_positions_after"] != receipt.ticks[-1]["reference_positions"]


    def test_protocol_and_invalid_goal_validation_do_not_advance_simulation(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(physics_dt=0.1)
        with self.assertRaises(ValueError):
            ExecutionConfig(control_ticks=10)
        backend, executor = make_executor()
        with self.assertRaises(ValueError):
            executor.execute_joint_target([3, 0])
        assert backend.time == 0 and not backend.events


if __name__ == "__main__":
    unittest.main()
