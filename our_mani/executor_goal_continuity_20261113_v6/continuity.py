from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_timed_tasks_20261112_v1.trajectory import predict

class GoalContinuity:
    def __init__(self,position_tolerance=.02,rotation_tolerance=.15):
        self.position_tolerance=position_tolerance;self.rotation_tolerance=rotation_tolerance
        self.history=deque(maxlen=5);self.segment=None;self.lookahead=None

    def observe(self,action,segment):
        pose=action.pose;row=(pose.timestamp,np.asarray(pose.position,float),np.asarray(pose.quaternion_xyzw,float))
        if segment!=self.segment or self.lookahead!=action.lookahead_seconds:
            self.history.clear();self.segment=segment;self.lookahead=action.lookahead_seconds
        diagnostic=dict(accepted=True,segment_id=segment,reason='initial_or_consistent_target',
            synthetic_observations_added=False,task_body_state_used=False)
        if len(self.history)>=3:
            dt=pose.timestamp-self.history[-1][0]
            if dt<=0:raise ValueError('Increasing actual goal observation timestamps required')
            p,q,motion=predict(list(self.history),-dt)
            position=float(np.linalg.norm(row[1]-p))
            angle=float((Rotation.from_quat(row[2])*Rotation.from_quat(q).inv()).magnitude())
            diagnostic.update(position_prediction_residual_m=position,rotation_prediction_residual_rad=angle,
                goal_motion_model=motion)
            if motion['valid']and(position>self.position_tolerance or angle>self.rotation_tolerance):
                diagnostic.update(accepted=False,reason='same_segment_target_discontinuity',
                    last_accepted_observation_timestamp=self.history[-1][0])
                return False,diagnostic
        self.history.append(row)
        return True,diagnostic

def endpoint_errors(actual,position,quaternion):
    actual=np.asarray(actual,float)
    p=float(np.linalg.norm(actual[:3]-position))
    r=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(quaternion)).magnitude())
    return p,r
