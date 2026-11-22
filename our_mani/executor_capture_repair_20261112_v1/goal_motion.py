"""Infer a current capture pose from timed future goals, without body state."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import logm,expm

def current_pose(history,lookahead):
    if not np.isfinite(lookahead)or lookahead<0:raise ValueError('Declared nonnegative lookahead required')
    # Timestamp is the observation time; goal position is at its declared
    # lookahead. Distinctness is essential at nonuniform/repeated updates.
    unique={float(t):(np.asarray(p),np.asarray(q))for t,p,q in history}
    times=np.asarray(sorted(unique));poses=[unique[t]for t in times]
    p,q=poses[-1]
    if len(times)<3 or times[-1]-times[0]<.15:
        return p.copy(),q.copy(),dict(valid=False,reason='insufficient_timed_goal_history')
    def matrix(pose):
        transform=np.eye(4);transform[:3,3]=pose[0]
        transform[:3,:3]=Rotation.from_quat(pose[1]).as_matrix()
        return transform
    transforms=[matrix(pose)for pose in poses];duration=times[-1]-times[0]
    twist=logm(transforms[-1]@np.linalg.inv(transforms[0]))/duration
    if np.max(abs(twist.imag))>1e-6:
        return p.copy(),q.copy(),dict(valid=False,reason='nonreal_rigid_motion_logarithm')
    twist=twist.real
    predictions=[expm((t-times[-1])*twist)@transforms[-1]for t in times]
    residual=float(max(np.linalg.norm(a[:3,3]-b[:3,3])for a,b in zip(predictions,transforms)))
    angle_residual=float(max((Rotation.from_matrix(a[:3,:3]).inv()*Rotation.from_matrix(b[:3,:3])).magnitude()
        for a,b in zip(predictions,transforms)))
    velocity=(twist@transforms[-1])[:3,3]
    angular_speed=float(np.linalg.norm([twist[2,1],twist[0,2],twist[1,0]]))
    x=times-times[-1];design=np.stack((np.ones(len(x)),x),axis=1)
    positions=np.stack([v[0]for v in poses]);linear_fit=np.linalg.lstsq(design,positions,rcond=None)[0]
    linear_residual=float(np.max(np.linalg.norm(design@linear_fit-positions,axis=1)))
    current=expm(-lookahead*twist)@transforms[-1]
    model='constant_world_rigid_body_twist'
    # A freely translating object can rotate independently about its centre.
    # That motion is not one constant spatial twist. Select it when the
    # observed positions support constant Cartesian velocity more closely.
    if linear_residual<max(1e-8,residual*.8):
        velocity=linear_fit[1]
        current[:3,3]=p-lookahead*velocity
        residual=linear_residual;model='independent_cartesian_and_angular_velocity'
    speed=float(np.linalg.norm(velocity))
    valid=residual<=.02 and angle_residual<=.05 and speed<=2. and angular_speed<=6.
    return current[:3,3],Rotation.from_matrix(current[:3,:3]).as_quat(),dict(valid=valid,goal_speed_m_s=speed,
        goal_angular_speed_rad_s=angular_speed,position_fit_max_residual_m=residual,
        rotation_fit_max_residual_rad=angle_residual,model=model,
        declared_lookahead_seconds=lookahead,task_object_state_read=False,
        source='accepted_world_goal_positions_quaternions_and_observation_timestamps')
