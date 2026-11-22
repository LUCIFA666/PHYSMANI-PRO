"""Match measured target velocity before a slow moving-grasp descent."""
from dataclasses import dataclass
import numpy as np
from ..contracts import Pose
from ..highspeed_control_20261111_v1.hybrid_tracking import HybridTrackingConfig
from .coordinator_v2 import CandidateConsistentPerceptionCoordinator,CandidateConsistentTrackingPlanner


class ControlledApproachReference:
    """Desired TCP waypoint offset, separate from observed/predicted object data."""
    def __init__(self,forecast,now,clearance,direction,descent_speed):
        self.forecast,self.now,self.clearance=forecast,float(now),float(clearance)
        self.direction=np.asarray(direction);self.descent_speed=float(descent_speed)
        self.times=forecast.times
        self.metadata=dict(source='desired_controlled_TCP_approach_offset',is_object_motion_forecast=False,
            used_as_NN_observation=False,Native_object_state_used=False)

    def at(self,object_id,time):
        pose,v,w=self.forecast.at(object_id,time)
        clearance=max(0.,self.clearance-self.descent_speed*max(0.,time-self.now))
        direction=pose.rotation@self.direction;offset=direction*clearance
        rate=-self.descent_speed if clearance>0 else 0.
        return Pose(pose.position+offset,pose.rotation),v+np.cross(w,offset)+direction*rate,w


class VelocityMatchedApproachPlanner(CandidateConsistentTrackingPlanner):
    def plan(self,state,candidate,forecast,*,pregrasp):
        phase=state.get('phase_state',{})
        if (not pregrasp and hasattr(forecast,'times') and phase.get('phase')=='align'
                and state.get('held') is False):
            pose,_,_=forecast.at('target',state['time'])
            direction=pose.rotation@candidate['approach_direction']
            grasp=pose.apply(candidate['object_tcp_position'])
            clearance=float(np.clip((state['tcp_position']-grasp)@direction,0.,candidate['pregrasp_offset']))
            metrics=phase.get('metrics',{})
            matched=(metrics.get('relative_speed',float('inf'))<=candidate['allowed_relative_speed']
                and metrics.get('angle',float('inf'))<=.2
                and state['opening']>=candidate['required_opening']-.05 and not phase.get('reopening',False))
            speed=.07 if matched else 0.
            controlled=ControlledApproachReference(forecast,state['time'],clearance,candidate['approach_direction'],speed)
            result=super().plan(state,candidate,controlled,pregrasp=False)
            result.update(controlled_approach_reference=controlled.metadata,actual_clearance_m=clearance,
                measured_velocity_matched_for_descent=matched,desired_descent_speed_mps=speed,
                original_capture_gates_unchanged=True)
            return result
        return super().plan(state,candidate,forecast,pregrasp=pregrasp)


class VelocityMatchedPerceptionCoordinator(CandidateConsistentPerceptionCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        config={'maximum_tcp_speed':1.1,**(tracking_config or {})}
        super().__init__(*args,tracking_config=config,**kwargs)
        old=self.reference_planner
        self.reference_planner=VelocityMatchedApproachPlanner(old.kinematics,old.joint_response,old.limits,
            config=HybridTrackingConfig(**config))
