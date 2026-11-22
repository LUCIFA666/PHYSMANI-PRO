import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_capture_precision_20261113_v2.pose_executor import PoseGoalExecutor as PrecisionExecutor,WorldPoseGoal
from ..executor_grasp_frame_20261113_v1.grasp_frame import capture_delta

def contact_and_reference(contact_position,reference_offset,actual_position):
    contact=np.asarray(contact_position,float);reference=contact+np.asarray(reference_offset,float)
    actual=np.asarray(actual_position,float)
    return reference,dict(contact_position_error_m=float(np.linalg.norm(actual-contact)),
        nominal_reference_position_error_m=float(np.linalg.norm(actual-reference)),
        distinct_contact_and_nominal_reference_frames=True)

class PoseGoalExecutor(PrecisionExecutor):
    def _capture_receipt(self):
        if not self.compensate_grasp_frame:return super()._capture_receipt()
        position,quaternion,twist,motion=self._sample_target()
        native=self.backend.capture_window()
        elapsed=self.backend.clock()-self._active_action.timestamp
        fresh_current=(self.capture_lookahead_seconds==0. and abs(elapsed)<self.executor.config.clock_tolerance)
        actual=np.asarray(self.backend.arm.get_tip().get_pose())
        nominal,errors=contact_and_reference(position,self.grasp_reference_world_offset,actual[:3])
        angle=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
        p_tol,r_tol=self.capture_tolerances
        if p_tol is None:p_tol,r_tol=.03,.5
        quality=errors['contact_position_error_m']<=p_tol and angle<=r_tol
        allowed=(not native['graspables'])or(native['detected']and(motion['valid']or fresh_current)and quality)
        if allowed and native['graspables']:
            self.pending_grasp_delta=capture_delta(nominal,quaternion,actual)
            self.capture_stamp=self._active_action.timestamp;self.capture_pose=actual.copy()
        else:
            self.pending_grasp_delta=None;self.capture_stamp=None;self.capture_pose=None
        return dict(allowed=allowed,reason='precise_contact_and_recorded_nominal_grasp'if allowed else'wait_for_precise_contact_pose',
            native=native,motion_compensation=motion,fresh_current_sample_used=fresh_current,
            candidate_nominal_to_actual_TCP=None if self.pending_grasp_delta is None else self.pending_grasp_delta.tolist(),
            grasp_position_error_m=errors['contact_position_error_m'],grasp_rotation_error_rad=angle,
            contact_and_reference_errors=errors,
            explicit_capture_position_tolerance_m=p_tol,explicit_capture_rotation_tolerance_rad=r_tol,
            physical_capture_and_held_feedback_required_before_correction=True,
            Native_object_pose_used=False,clock=self.backend.clock())
