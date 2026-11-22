"""Original all-task backend with explicit first-success/held/readback channels.

Task private state never enters the public RGBD/robot policy packet. Membership
is diagnostic/control feedback only, not success. Every task has its own native
graspable-object dispatch inherited unchanged from the original backend.
"""
from copy import deepcopy
import numpy as np

from ..task_time_limits_20261111_v1.feedback_backend import ExtendedTimeFeedbackBackend


class TaskExecutionBackend(ExtendedTimeFeedbackBackend):
    def __init__(self, *args, **kwargs):
        self.native_task_success_latched = False
        self.native_success_first_time = None
        self.servo_readback_max_error_rad = 0.
        self.servo_writes = 0
        super().__init__(*args, **kwargs)
        self._latch()
        self.provenance.update(original_task_success_only=True,
                               first_success_stops_at_actual50ms=True,
                               binary_Native_membership_outside_NN_only=True)

    def _latch(self):
        current = self.metric_receipt()['native_current']
        if current['success'] and not self.native_task_success_latched:
            self.native_task_success_latched = True
            self.native_success_first_time = current['physical_timestamp']

    def terminal(self):
        return self.native_task_success_latched or super().terminal()

    def tick(self, phase='multitask_execution'):
        super().tick(phase)
        self._latch()

    def read_contact_feedback(self):
        before = self.clock()
        held = bool(self.gripper.get_grasped_objects())
        if self.clock() != before:
            raise RuntimeError('Membership query advanced physics')
        return dict(timestamp=before, held=held, Native_object_pose_used=False,
                    source='original_gripper_membership', held_alone_is_task_success=False)

    def native_task_judge_receipt(self):
        return dict(native_task_success_latched=self.native_task_success_latched,
                    first_success_time=self.native_success_first_time,
                    original_native_metric=deepcopy(self.metric_receipt()),
                    episode_success_source='original_PhysMani_Task.success_only',
                    visual_success_gate=False)

    def set_joint_target_positions(self, values):
        expected = np.asarray(values, dtype=float)
        super().set_joint_target_positions(values)
        error = float(np.max(np.abs(np.asarray(self.arm.get_joint_target_positions()) - expected)))
        self.servo_readback_max_error_rad = max(self.servo_readback_max_error_rad, error)
        self.servo_writes += 1
        if error > 1e-6:
            raise RuntimeError('Native POSITION servo did not retain the exact supplied reference')
