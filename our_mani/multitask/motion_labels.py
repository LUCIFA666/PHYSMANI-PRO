"""Offline GT-pose backward world twist supervision, never actor history.

Raw native reported velocities remain separate unchanged evidence. Derivatives
are right-endpoint interval averages, matching the causal tracker convention;
they are not claimed exact continuous instantaneous velocities at impacts.
"""
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA='r10_GT_pose_backward_world_twist_v1'
RAW_LINEAR='native_reported_linear_velocities'
RAW_ANGULAR='native_reported_angular_velocities'


class MotionLabelError(ValueError):pass


def backward_pose_supervision(arrays,*,expected_dt=.05,clock_tolerance=1e-6):
    if expected_dt<=0 or clock_tolerance<=0 or not np.isfinite([expected_dt,clock_tolerance]).all():
        raise MotionLabelError('finite positive declared physical interval and tolerance required')
    if RAW_LINEAR in arrays or RAW_ANGULAR in arrays:
        raise MotionLabelError('refusing to derive twice or replace existing native reported evidence')
    t=np.asarray(arrays['times'],float);ids=np.asarray(arrays['body_ids'])
    p=np.asarray(arrays['target_positions'],float);r=np.asarray(arrays['target_rotations'],float)
    n,b=len(t),len(ids)
    if (t.shape!=(n,) or n<1 or b<1 or len(set(map(str,ids)))!=b or p.shape!=(n,b,3)
            or r.shape!=(n,b,3,3) or not np.isfinite(t).all()):
        raise MotionLabelError('stable per-episode body IDs and GT pose/time axes required')
    masks={}
    for name in ('native_pose_valid','mode_valid','dynamics_segment_valid','target_valid'):
        x=np.asarray(arrays[name])
        if x.shape!=(n,b) or x.dtype!=np.bool_:raise MotionLabelError('explicit boolean body mask required: '+name)
        masks[name]=x
    modes=np.asarray(arrays['modes'])
    if modes.shape!=(n,b) or not np.isin(modes,(0,1,2)).all():raise MotionLabelError('known public body motion modes required')
    for name in ('target_linear_velocities','target_angular_velocities'):
        if np.asarray(arrays[name]).shape!=(n,b,3):raise MotionLabelError('raw native velocity shape differs')
    finite=np.isfinite(p).all(-1)&np.isfinite(r).all((-1,-2))
    safe_r=np.where(finite[...,None,None],r,np.eye(3))
    so3=(np.max(np.abs(np.swapaxes(safe_r,-1,-2)@safe_r-np.eye(3)),axis=(-1,-2))<=1e-4)&(
        np.abs(np.linalg.det(safe_r)-1.)<=1e-4)
    pose=masks['native_pose_valid']&finite&so3
    valid=np.zeros((n,b),bool);linear=np.zeros((n,b,3));angular=np.zeros((n,b,3))
    intervals=np.diff(t);clock=np.isfinite(intervals)&(intervals>0)&(np.abs(intervals-expected_dt)<=clock_tolerance)
    if n>1:
        valid[1:]=(pose[:-1]&pose[1:]&masks['target_valid'][:-1]&masks['target_valid'][1:]
            &masks['mode_valid'][:-1]&masks['mode_valid'][1:]&masks['dynamics_segment_valid'][:-1]
            &masks['dynamics_segment_valid'][1:]&(modes[:-1]==modes[1:])&clock[:,None])
        safe_dt=np.where(clock,intervals,expected_dt)
        with np.errstate(invalid='ignore'):
            differences=np.diff(p,axis=0)/safe_dt[:,None,None]
        linear[1:]=np.where(valid[1:,:,None],differences,0.)
        pair=np.argwhere(valid[1:])
        if len(pair):
            i,j=pair[:,0],pair[:,1]
            increments=r[i+1,j]@np.swapaxes(r[i,j],-1,-2)
            angular[i+1,j]=Rotation.from_matrix(increments).as_rotvec()/intervals[i,None]
    result=dict(arrays)
    result[RAW_LINEAR]=np.array(arrays['target_linear_velocities'],copy=True)
    result[RAW_ANGULAR]=np.array(arrays['target_angular_velocities'],copy=True)
    result['target_pose_valid_before_twist_mask']=masks['target_valid'].copy()
    result['target_twist_valid']=valid
    result['target_linear_velocities']=linear
    result['target_angular_velocities']=angular
    # Current loader/fitter use one future mask. Conservatively drop boundary
    # pose labels as well; preserve their independently valid pose evidence.
    result['target_valid']=masks['target_valid']&valid
    metadata=dict(schema=SCHEMA,source='same_body_same_mode_contiguous_valid_GT_pose_backward_difference',
        timing='right endpoint world displacement/log(Rcurrent @ Rprevious.T) over measured 50ms interval',
        expected_dt=float(expected_dt),clock_tolerance=float(clock_tolerance),
        first_frame_twist_valid=False,invalid_intervals=int((~clock).sum()),
        native_pose_marked_valid_but_nonfinite_or_non_SO3=int((masks['native_pose_valid']&~(finite&so3)).sum()),
        valid_twist_samples=int(valid.sum()),total_body_samples=int(n*b),
        raw_native_velocity_arrays=dict(linear=RAW_LINEAR,angular=RAW_ANGULAR),
        native_reported_velocity_time_alignment='v5 TRAIN scripted bodies showed one-frame lag; raw values retained, no global time shift applied',
        target_valid_rule='previous target pose/public mode/exogenous segment valid at both endpoints, same body and mode, declared contiguous interval; unified target_valid additionally intersects target_twist_valid',
        history_fields_modified=False,raw_native_velocity_modified=False,
        instantaneous_velocity_at_impact_claimed=False,position_boundary_samples_conservatively_masked=True)
    return result,metadata
