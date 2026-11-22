"""Advance a grasp approach in Cartesian space using actual elapsed time."""
import math
import numpy as np
from scipy.spatial.transform import Rotation

class GuidedApproach:
    def __init__(self,clearance=.20,speed=.08):
        if not math.isfinite(speed)or not 0<speed<=.2:raise ValueError('Positive Cartesian guide speed at most .2 m/s required')
        self.clearance=clearance;self.speed=speed;self.reset()

    def reset(self):
        self.phase='hover';self.aligned_since=None;self.last_time=None
        self.commanded_z=None;self.hover_height=None;self.vertical_velocity=0.;self.active_target_z=None

    def sample(self,position,quaternion,actual,clock,*,held):
        position=np.asarray(position,float);actual=np.asarray(actual,float)
        if held:
            self.reset();return position.copy(),dict(applied=False,phase='held',Native_object_pose_used=False)
        dt=0. if self.last_time is None else clock-self.last_time
        if dt<0:raise ValueError('Monotonic actual physical clock required')
        lateral=float(np.linalg.norm(position[:2]-actual[:2]))
        angle=float((Rotation.from_quat(quaternion)*Rotation.from_quat(actual[3:]).inv()).magnitude())
        if self.commanded_z is None:self.commanded_z=float(actual[2])
        if self.hover_height is None:self.hover_height=float(position[2])+self.clearance
        if self.phase=='descend'and(lateral>.05 or angle>.4):
            self.phase='hover';self.aligned_since=None
            self.hover_height=float(position[2])+self.clearance
        if self.phase=='hover':
            self.hover_height=float(position[2])+self.clearance
            if lateral<=.02 and angle<=.20 and actual[2]>=self.hover_height-.01:
                if self.aligned_since is None:self.aligned_since=clock
                if clock-self.aligned_since>=.15-1e-6:self.phase='descend'
            else:self.aligned_since=None
        target_z=self.hover_height if self.phase=='hover'else float(position[2])
        self.active_target_z=target_z
        if dt>1e-9:
            change=float(np.clip(target_z-self.commanded_z,-self.speed*dt,self.speed*dt))
            self.commanded_z+=change;self.vertical_velocity=change/dt
        waypoint=position.copy();waypoint[2]=self.commanded_z;self.last_time=clock
        return waypoint,dict(applied=True,phase=self.phase,lateral_error_m=lateral,rotation_error_rad=angle,
            commanded_waypoint_z=self.commanded_z,terminal_grasp_goal_z=float(position[2]),
            Cartesian_vertical_speed_cap_m_s=self.speed,derived_vertical_velocity_m_s=self.vertical_velocity,
            source_world_goal_changed=False,terminal_grasp_gate_unchanged=True,Native_object_pose_used=False)
