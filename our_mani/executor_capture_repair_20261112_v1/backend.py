from ..executor_task_goals_20261112_v1.backend import TaskExecutionBackend as PreviousBackend

class TaskExecutionBackend(PreviousBackend):
    def capture_window(self):
        before=self._clock_tuple()
        objects=self.task.get_graspable_objects()
        detected=any(self.gripper._proximity_sensor.is_detected(o)for o in objects)
        if self._clock_tuple()!=before:raise RuntimeError('Capture sensor query stepped physics')
        return dict(graspables=bool(objects),detected=bool(detected),
            source='unchanged_original_gripper_grasp_proximity_precondition',
            task_object_pose_read=False,NN_input=False,task_success=False,
            physical_hardware_sensor_equivalence_claimed=False)
