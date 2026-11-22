"""Original Cube physics/executor plus explicit contact and task judge channels."""
from ..cube_v72_live_20261104_v1.backend import CubeTaskNativeBackendV1


class NativeContactCubeBackendV1(CubeTaskNativeBackendV1):
    def __init__(self,*args,**kwargs):
        self.native_task_success_latched=False;self.native_success_first_time=None
        super().__init__(*args,**kwargs)
        self._latch_original_task_success()

    def _latch_original_task_success(self):
        success=bool(self._last_native_result['success'])
        if success and not self.native_task_success_latched:
            self.native_task_success_latched=True;self.native_success_first_time=self.clock()
        return success

    def tick(self,phase):
        state=super().tick(phase)
        # Original task.success was sampled by the original committed tick.
        self._latch_original_task_success()
        return state

    def terminal(self):
        return self.native_task_success_latched or super().terminal()

    def read_contact_feedback(self):
        before=self.clock()
        held=self.task.target_block in self.gripper.get_grasped_objects()
        if self.clock()!=before:raise RuntimeError('Membership query advanced physics')
        return dict(timestamp=before,held=bool(held),Native_contact_feedback_used=True,
            source='original_simulated_gripper_target_membership',force_sensor_contact=False,
            Native_object_pose_used=False,Native_future_state_used=False)

    def native_task_judge_receipt(self):
        return dict(native_task_success_latched=self.native_task_success_latched,
            first_success_time=self.native_success_first_time,original_native_result=dict(self._last_native_result),
            episode_success_source='original_PhysMani_Task.success_only',
            held_alone_is_task_success=False,visual_probe_hold_or_goal_is_success_gate=False,
            stop_at_first_original_success_actual50ms=True)
