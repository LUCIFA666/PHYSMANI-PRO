"""Explicit anisotropic alignment for a release above a moving receptacle."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_dual_grasp_frame_20261113_v13.pose_executor import PoseGoalExecutor as ContactExecutor,WorldPoseGoal,RawTimedExecutor
from ..executor_grasp_frame_20261113_v1.grasp_frame import apply_delta

def drop_alignment(error,angle,valid,*,lateral=.0015,rotation=.02,height=.04):
    error=np.asarray(error,float)
    return bool(valid and np.isfinite(error).all() and np.isfinite(angle)
        and np.linalg.norm(error[:2])<=lateral and abs(error[2])<=height and angle<=rotation)

class PoseGoalExecutor(ContactExecutor):
    supports_aligned_drop=True
    def __init__(self,*args,release_policy='terminal_pose',**kwargs):
        super().__init__(*args,release_policy='terminal_pose'if release_policy=='aligned_drop'else release_policy,**kwargs)
        self.release_policy=release_policy

    def _release_receipt(self):
        if self.release_policy!='aligned_drop':return super()._release_receipt()
        p,q,twist,motion=RawTimedExecutor._sample_target(self)
        if self.grasp_delta is not None:p,q,twist=apply_delta(p,q,twist,self.grasp_delta)
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        error=measured[:3]-p
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(q)).magnitude())
        allowed=drop_alignment(error,angle,motion['valid'])
        return dict(allowed=allowed,reason='explicit_aligned_drop'if allowed else'wait_for_lateral_and_angular_drop_alignment',
            explicit_release_policy='aligned_drop',measured_TCP_position_error_m=float(np.linalg.norm(error)),
            measured_TCP_position_error_xyz_m=error.tolist(),measured_TCP_rotation_error_rad=angle,
            motion_compensation=motion,lateral_tolerance_m=.0015,rotation_tolerance_rad=.02,height_tolerance_m=.04,
            intermediate_waypoint_does_not_qualify_release=True,Native_object_pose_used=False,
            task_success_claimed=False,Native_task_success_predicate_unchanged=True)
