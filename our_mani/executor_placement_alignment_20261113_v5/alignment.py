import numpy as np
from scipy.spatial.transform import Rotation

class HeldPlacementAlignment:
    """Align in free space before lowering a held tool; robot/goals only."""
    def __init__(self,clearance=.08,descent_speed=.04):
        self.clearance=clearance;self.speed=descent_speed;self.reset()

    def reset(self):
        self.peak_raw=None;self.peak_derived=None;self.phase='inactive';self.since=None
        self.last_time=None;self.commanded_z=None;self.last_z_rate=0.

    def sample(self,position,quaternion,twist,actual,clock,*,held,raw_goal_z):
        position=np.asarray(position,float);actual=np.asarray(actual,float);twist=np.asarray(twist,float).copy()
        if not held:
            self.reset()
            return position.copy(),twist,dict(active=False,phase='unheld')
        self.peak_raw=raw_goal_z if self.peak_raw is None else max(self.peak_raw,raw_goal_z)
        self.peak_derived=position[2]if self.peak_derived is None else max(self.peak_derived,position[2])
        if self.phase=='inactive'and self.peak_raw-raw_goal_z>.02:self.phase='align_above'
        if self.phase=='inactive':return position.copy(),twist,dict(active=False,phase='held_transfer')
        lateral=float(np.linalg.norm(position[:2]-actual[:2]))
        angle=float((Rotation.from_quat(quaternion)*Rotation.from_quat(actual[3:]).inv()).magnitude())
        hover=max(self.peak_derived,position[2]+self.clearance)
        if self.phase=='descend'and(lateral>.01 or angle>.12):
            self.phase='align_above';self.since=None
        if self.phase=='align_above':
            if lateral<=.003 and angle<=.05 and actual[2]>=hover-.01:
                if self.since is None:self.since=clock
                if clock-self.since>=.10-1e-6:
                    self.phase='descend';self.commanded_z=max(actual[2],position[2])
            else:self.since=None
        old_z=self.commanded_z
        dt=0. if self.last_time is None else max(0.,clock-self.last_time)
        if self.phase=='align_above':self.commanded_z=hover;self.last_z_rate=0.
        else:
            self.commanded_z=max(position[2],self.commanded_z-self.speed*dt)
            if dt>1e-8 and old_z is not None:self.last_z_rate=(self.commanded_z-old_z)/dt
        waypoint=position.copy();waypoint[2]=self.commanded_z
        twist[2]=self.last_z_rate;self.last_time=clock
        return waypoint,twist,dict(active=True,phase=self.phase,hover_z=hover,
            commanded_waypoint_z=self.commanded_z,terminal_destination_z=float(position[2]),
            lateral_error_m=lateral,rotation_error_rad=angle,descent_speed_cap_m_s=self.speed,
            terminal_destination_changed=False,task_body_state_used=False,
            waypoint_is_not_release_completion_certificate=True)
