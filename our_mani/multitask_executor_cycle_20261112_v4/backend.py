"""Restore robot query state on both successful and failed native path calls."""
import numpy as np
from ..multitask_executor_cycle_20261112_v1.backend import TaskExecutionBackend as PreviousBackend


class RestoringArmQuery:
    def __init__(self, arm, clock):
        self._arm, self._clock = arm, clock
        self.query_count = self.restored_changed_q_queries = 0
        self.last_query_receipt = None
        self.receipt_sink = None

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def get_path(self, *args, **kwargs):
        before = self._clock()
        arm = self._arm
        actual = np.asarray(arm.get_joint_positions(), float)
        velocity = np.asarray(arm.get_joint_velocities(), float)
        servo = np.asarray(arm.get_joint_target_positions(), float)
        ik = np.asarray(arm._ik_target.get_pose(), float)
        if callable(getattr(arm, 'get_joint_intervals', None)):
            _, intervals = arm.get_joint_intervals()
            intervals = np.asarray(intervals, float)
            lower, upper = intervals[:,0], intervals[:,0]+intervals[:,1]
            if np.any(actual < lower) or np.any(actual > upper):
                raise RuntimeError('Counted servo recovery required before a kinematic query outside native bounds')
        succeeded = False
        try:
            result = arm.get_path(*args, **kwargs)
            succeeded = True
            return result
        finally:
            changed = not np.allclose(arm.get_joint_positions(), actual, atol=1e-6, rtol=0)
            # RRT/IK uses temporary joint configurations. Restoring those is a
            # zero-time query rollback, never an action, reset or teleport to a
            # requested goal. Keep dynamics enabled and preserve cached dq.
            if changed:
                arm.set_joint_positions(actual.tolist(), disable_dynamics=False)
            arm._ik_target.set_pose(ik.tolist())
            arm.set_joint_target_positions(servo.tolist())
            errors = dict(q=float(np.max(np.abs(np.asarray(arm.get_joint_positions())-actual))),
                          dq=float(np.max(np.abs(np.asarray(arm.get_joint_velocities())-velocity))),
                          servo=float(np.max(np.abs(np.asarray(arm.get_joint_target_positions())-servo))))
            self.query_count += 1
            self.restored_changed_q_queries += int(changed)
            self.last_query_receipt = dict(query=self.query_count, native_path_returned=succeeded,
                                           temporary_q_changed=changed, restoration_errors=errors,
                                           disable_dynamics=False, physics_steps=0)
            if self.receipt_sink is not None:
                self.receipt_sink(self.last_query_receipt)
            if self._clock() != before or errors['q'] > 1e-6 or errors['dq'] > 1e-9 or errors['servo'] > 1e-6:
                raise RuntimeError('Native arm query could not restore robot state without physics')


class TaskExecutionBackend(PreviousBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = RestoringArmQuery(self.arm, self._clock_tuple)
        self.provenance.update(path_queries_atomic_on_success_and_exception=True,
                               zero_time_restore_is_not_executed_goal=True)
