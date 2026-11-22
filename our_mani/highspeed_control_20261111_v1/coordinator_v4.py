"""Full-task tracking with a viable braking envelope and reference fallback."""
from .coordinator_v3 import CompleteTaskTrackingCoordinator
from .tracking_v4 import StoppableTrackingReferencePlanner,StoppableTrackingConfig


class StoppableCompleteTrackingCoordinator(CompleteTaskTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        # New stopping/posture keys belong to the new reference implementation.
        base_keys={'position_gain','rotation_gain','maximum_tcp_speed','maximum_tcp_acceleration',
            'maximum_angular_speed','maximum_angular_acceleration','rotation_length_scale','regularization','feedforward_scale'}
        super().__init__(*args,tracking_config={k:v for k,v in (tracking_config or {}).items() if k in base_keys},**kwargs)
        old=self.reference_planner
        self.reference_planner=StoppableTrackingReferencePlanner(old.kinematics,old.joint_response,old.limits,
            StoppableTrackingConfig(**(tracking_config or {})))

    def brake_action(self,robot,h,opening):
        state=dict(time=robot.timestamp,q=robot.q,dq=robot.dq,actuator_state=h)
        action=self.reference_planner.brake(state)
        action['opening']=opening
        return state,action
