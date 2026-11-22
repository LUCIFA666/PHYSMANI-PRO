import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_reference_unload_20261113_v4.pose_executor import PoseGoalExecutor as UnloadExecutor,WorldPoseGoal
from .alignment import HeldPlacementAlignment

class PoseGoalExecutor(UnloadExecutor):
    def __init__(self,*args,held_placement_alignment=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.placement_alignment=HeldPlacementAlignment()if held_placement_alignment else None

    def _sample_target(self):
        p,q,twist,motion=super()._sample_target()
        if self.placement_alignment is None:return p,q,twist,motion
        p,twist,diagnostic=self.placement_alignment.sample(p,q,twist,
            np.asarray(self.backend.arm.get_tip().get_pose()),self.backend.clock(),
            held=bool(self.backend.gripper.get_grasped_objects()),raw_goal_z=self._active_action.position[2])
        return p,q,twist,dict(motion,held_placement_alignment=diagnostic)

    def _release_receipt(self):
        # Release is checked against the terminal destination, never a raised
        # intermediate control waypoint that happens to be reached precisely.
        p,q,twist,motion=super()._sample_target()
        measured=np.asarray(self.backend.arm.get_tip().get_pose())
        position=float(np.linalg.norm(measured[:3]-p))
        angle=float((Rotation.from_quat(measured[3:]).inv()*Rotation.from_quat(q)).magnitude())
        allowed=motion['valid']and position<=self.release_position_tolerance and angle<=self.release_angle_tolerance
        return dict(allowed=allowed,reason='terminal_release_pose'if allowed else'wait_for_terminal_release_pose',
            measured_TCP_position_error_m=position,measured_TCP_rotation_error_rad=angle,
            motion_compensation=motion,intermediate_waypoint_does_not_qualify_release=True,
            current_physical_tick_time=self.backend.clock(),goal_observation_time=self._active_action.timestamp)
