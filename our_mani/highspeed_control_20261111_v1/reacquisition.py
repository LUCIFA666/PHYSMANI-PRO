"""Clear persistent public-camera occlusion using the observed start posture."""
import numpy as np
from scipy.spatial.transform import Rotation
from .tracking_v4 import reference_window
from .coordinator_v8 import PublicMotionCompleteTrackingCoordinator


def reacquisition_reference(planner,state,home):
    r,v,a=planner._history(state);dt=.05
    lo,hi=reference_window(r,v,a,planner.limits,margin=planner.config.joint_limit_margin)
    if np.any(lo>hi+1e-10):raise ValueError('No bounded reacquisition reference')
    error=np.asarray(home)-r
    desired=np.sign(error)*np.minimum(3.*np.abs(error),np.sqrt(2*np.asarray(planner.limits.acceleration)*np.abs(error)))
    ref=np.minimum(hi,np.maximum(lo,r+desired*dt))
    pose=planner.kinematics(ref)
    return dict(joint_goal=ref,tcp_position=pose[:3,3],tcp_quaternion=Rotation.from_matrix(pose[:3,:3]).as_quat(),
        joint_goal_feasible=True,execution_semantics='sampled_joint_reference',opening=1.,
        reacquisition_target='measured_initial_robot_posture',Native_object_state_used=False)


class ReacquiringPublicTrackingCoordinator(PublicMotionCompleteTrackingCoordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observed_home=None;self.missing_since=None;self.ever_ready=False

    def decide(self,robot,h):
        if self.observed_home is None:self.observed_home=robot.q.copy()
        result=super().decide(robot,h)
        ready=bool(self.backend.latest.get('free_forecast_available'))
        if ready:self.ever_ready=True;self.missing_since=None
        elif self.ever_ready and not self.backend.latest_contact['held']:
            if self.missing_since is None:self.missing_since=robot.timestamp
            if (robot.timestamp-self.missing_since>=.2-1e-9 and h['gripper_target']>=.5
                    and result.get('action') is None):
                state=dict(time=robot.timestamp,q=robot.q,dq=robot.dq,actuator_state=h)
                action=reacquisition_reference(self.reference_planner,state,self.observed_home)
                result.update(mode='PUBLIC_RGBD_REACQUISITION',state=state,action=action,
                    reason='persistent_public_measurement_loss_clear_camera_using_initial_robot_posture',
                    public_measurement_missing_seconds=robot.timestamp-self.missing_since,
                    missing_object_pose_fabricated=False,original_collision_guard_required=True)
        return result
