"""Retreat after an actual release; never reactivate a grasp descent mid-opening."""
import numpy as np
from ..executor_control_lead_20261113_v9.lead import lead_pose

class ReleaseRetreat:
    def __init__(self,clearance=.06,speed=.15):
        self.clearance=clearance;self.speed=speed;self.was_held=False;self.active=False

    def sample(self,p,q,twist,actual,clock,*,held,opening,done):
        p=np.asarray(p,float);q=np.asarray(q,float);twist=np.asarray(twist,float);actual=np.asarray(actual,float)
        if self.was_held and not held and opening==1.:
            self.active=True;self.start=clock;self.anchor=actual.copy()
            # Preserve the moving target's spatial velocity at the measured TCP.
            release_p,release_twist=self.last_held_motion
            angular=release_twist[3:];spatial_linear=release_twist[:3]-np.cross(angular,release_p)
            self.release_twist=np.r_[spatial_linear+np.cross(angular,actual[:3]),angular]
        self.was_held=held
        if held:self.active=False;self.last_held_motion=(p.copy(),twist.copy())
        if not self.active:return p,q,twist,dict(active=False)
        elapsed=max(0.,clock-self.start)
        if done is True and actual[2]>=self.anchor[2]+self.clearance-.003:
            self.active=False
            return p,q,twist,dict(active=False,completed=True,actual_clearance_m=float(actual[2]-self.anchor[2]))
        waypoint,_,velocity=lead_pose(self.anchor[:3],self.anchor[3:],self.release_twist,elapsed)
        rise=min(self.clearance,self.speed*elapsed)
        waypoint[2]=self.anchor[2]+rise
        velocity=np.r_[velocity[:2],self.speed if rise<self.clearance else 0.,np.zeros(3)]
        return waypoint,self.anchor[3:].copy(),velocity,dict(active=True,phase='physical_release_then_upward_clearance',
            elapsed_seconds=elapsed,required_clearance_m=self.clearance,actual_clearance_m=float(actual[2]-self.anchor[2]),
            physical_gripper_done=done,derived_waypoint_is_not_task_success=True,
            source='measured_robot_TCP_release_edge_and_accepted_goal_motion',Native_object_pose_used=False)
