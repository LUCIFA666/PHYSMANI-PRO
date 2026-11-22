import numpy as np
from scipy.spatial.transform import Rotation

def desired_velocity(fk,actual,state,position,quaternion,feedforward,limits,*,gain=8.,mode='actual',near_contact=False,joint_avoidance='centre'):
    if mode not in ('actual','bounded_bias','compliant_insert'):raise ValueError('Explicit feedback mode required')
    transform=fk(actual);reference=fk(state.position)
    control_q=actual if mode=='actual'else state.position
    jac=fk.geometric_jacobian(control_q)
    raw_error=np.r_[np.asarray(position)-transform[:3,3],
        (Rotation.from_quat(quaternion)*Rotation.from_matrix(transform[:3,:3]).inv()).as_rotvec()]
    error=np.r_[np.asarray(position)-transform[:3,3],
        (Rotation.from_quat(quaternion)*Rotation.from_matrix(transform[:3,:3]).inv()).as_rotvec()]
    if mode!='actual':
        def bounded(vector,maximum):return vector*min(1.,maximum/max(float(np.linalg.norm(vector)),1e-12))
        position_bias=bounded(reference[:3,3]-transform[:3,3],.002)
        angle_bias=bounded((Rotation.from_matrix(reference[:3,:3])*Rotation.from_matrix(transform[:3,:3]).inv()).as_rotvec(),.005)
        if mode=='compliant_insert'and near_contact:
            difference=reference[:3,3]-transform[:3,3]
            position_bias=np.r_[bounded(difference[:2],.015),np.clip(difference[2],-.002,.002)]
            angle_bias=bounded((Rotation.from_matrix(reference[:3,:3])*Rotation.from_matrix(transform[:3,:3]).inv()).as_rotvec(),.05)
        compensated=Rotation.from_rotvec(angle_bias)*Rotation.from_quat(quaternion)
        error=np.r_[np.asarray(position)+position_bias-reference[:3,3],
            (compensated*Rotation.from_matrix(reference[:3,:3]).inv()).as_rotvec()]
    weights=np.diag([10.,10.,10.,1.,1.,1.]);weighted=weights@jac
    inverse=weighted.T@np.linalg.solve(weighted@weighted.T+.03**2*np.eye(6),np.eye(6))
    centre=(np.asarray(limits.lower)+np.asarray(limits.upper))/2
    if joint_avoidance=='barrier':
        low_distance=np.maximum(actual-np.asarray(limits.lower),.05)
        high_distance=np.maximum(np.asarray(limits.upper)-actual,.05)
        avoidance=np.clip(.3*(1/low_distance**2-1/high_distance**2),-1.5,1.5)
    elif joint_avoidance=='centre':avoidance=.1*(centre-actual)
    else:raise ValueError('Explicit joint avoidance profile required')
    null=(np.eye(len(actual))-inverse@weighted)@avoidance
    correction=gain*error
    if mode=='compliant_insert'and near_contact:
        correction[:2]=bounded(correction[:2],.04)
        correction[2]=np.clip(correction[2],-.025,.025)
        correction[3:]=bounded(correction[3:],.4)
        # Keep following the moving peg while aligning above it. Do not
        # drive down against an edge before lateral/angular alignment.
        if correction[2]<0 and(np.linalg.norm(raw_error[:2])>.002 or np.linalg.norm(raw_error[3:])>.03):
            correction[2]=0.
    velocity=inverse@(weights@(np.asarray(feedforward)+correction))+null
    progress=float(np.clip((.15-float(np.max(abs(state.position-actual))))/.10,0.,1.))
    return progress*velocity,dict(source='measured_world_TCP_error_and_timed_goal_motion',
        actual_position_error_m=float(np.linalg.norm(raw_error[:3])),actual_rotation_error_rad=float(np.linalg.norm(raw_error[3:])),
        feedback_mode=mode,maximum_contact_pose_bias_m=.002 if mode=='bounded_bias'else None,
        near_contact=near_contact,vertical_correction_speed_cap_m_s=.025 if mode=='compliant_insert'and near_contact else None,
        actual_feedback_gain=gain,tracking_progress_scale=progress,feedforward_twist=np.asarray(feedforward).tolist(),
        desired_Cartesian_twist=(progress*(np.asarray(feedforward)+correction)).tolist(),
        geometric_jacobian=jac.tolist(),preferred_null_velocity=(progress*null).tolist(),
        Native_task_object_state_used=False,Native_IK_branch_used_as_motion_direction=False)
