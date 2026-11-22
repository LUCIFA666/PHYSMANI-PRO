"""Robot-only approach waypoints; source world goals remain unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation

class AlignedApproach:
    def __init__(self,clearance=.20):
        self.clearance=float(clearance);self.phase='hover';self.aligned_since=None

    def sample(self,position,quaternion,actual,clock,*,held):
        position=np.asarray(position,float);actual=np.asarray(actual,float)
        if held:
            self.phase='hover';self.aligned_since=None
            return position.copy(),dict(applied=False,reason='held_object_uses_original_timed_target')
        lateral=float(np.linalg.norm(position[:2]-actual[:2]))
        angle=float((Rotation.from_quat(quaternion)*Rotation.from_quat(actual[3:]).inv()).magnitude())
        if self.phase=='descend'and(lateral>.05 or angle>.4):
            self.phase='hover';self.aligned_since=None
        if self.phase=='hover':
            if lateral<=.02 and angle<=.20:
                if self.aligned_since is None:self.aligned_since=clock
                if clock-self.aligned_since>=.15:self.phase='descend'
            else:self.aligned_since=None
        waypoint=position.copy()
        if self.phase=='hover':waypoint[2]+=self.clearance
        return waypoint,dict(applied=self.phase=='hover',phase=self.phase,
            clearance_m=self.clearance,lateral_error_m=lateral,rotation_error_rad=angle,
            source_world_goal_changed=False,derived_robot_waypoint=True,task_body_state_used=False)

class TrackingRecovery:
    """Unload a stalled servo with a new bounded reference, without reseeding."""
    def __init__(self):self.target=None;self.since=None

    def update(self,reference,actual,clock):
        reference=np.asarray(reference,float);actual=np.asarray(actual,float)
        error=float(np.max(abs(reference-actual)))
        if self.target is None:
            if error>.15:
                if self.since is None:self.since=clock
                if clock-self.since>=.15:self.target=actual.copy()
            else:self.since=None
        elif error<.025:
            self.target=None;self.since=None
        return None if self.target is None else self.target.copy(),dict(active=self.target is not None,
            maximum_tracking_error_rad=error,trigger_after_seconds=.15,trigger_tracking_error_rad=.15,
            explicit_recovery_reference_goal=None if self.target is None else self.target.tolist(),
            reference_reseeded=False,actual_state_clipped=False,source_world_goal_changed=False)
