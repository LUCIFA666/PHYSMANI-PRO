import math
from ..executor_grasp_frame_20261113_v1.pose_executor import PoseGoalExecutor as GraspExecutor,WorldPoseGoal

def precise_capture(receipt,position_tolerance,rotation_tolerance):
    position=receipt.get('grasp_position_error_m',receipt.get('measured_TCP_position_error_m'))
    rotation=receipt.get('grasp_rotation_error_rad',receipt.get('measured_TCP_rotation_error_rad'))
    allowed=bool(receipt['allowed']and position is not None and rotation is not None
                 and math.isfinite(position)and math.isfinite(rotation)
                 and position<=position_tolerance and rotation<=rotation_tolerance)
    return dict(receipt,allowed=allowed,reason=receipt['reason']if allowed else'wait_for_explicit_capture_precision',
        explicit_capture_position_tolerance_m=position_tolerance,
        explicit_capture_rotation_tolerance_rad=rotation_tolerance,
        original_native_precondition_preserved=True)

class PoseGoalExecutor(GraspExecutor):
    def __init__(self,*args,capture_position_tolerance=None,capture_rotation_tolerance=None,**kwargs):
        super().__init__(*args,**kwargs)
        if (capture_position_tolerance is None)!=(capture_rotation_tolerance is None):
            raise ValueError('Both capture tolerances must be explicit')
        if capture_position_tolerance is not None and not(
                math.isfinite(capture_position_tolerance)and math.isfinite(capture_rotation_tolerance)
                and capture_position_tolerance>0 and capture_rotation_tolerance>0):
            raise ValueError('Positive finite capture tolerances required')
        self.capture_tolerances=(capture_position_tolerance,capture_rotation_tolerance)

    def _capture_receipt(self):
        result=super()._capture_receipt()
        if self.capture_tolerances[0]is None:return result
        result=precise_capture(result,*self.capture_tolerances)
        if not result['allowed']and self.compensate_grasp_frame:
            self.pending_grasp_delta=None;self.capture_stamp=None;self.capture_pose=None
        return result
