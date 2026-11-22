"""D research pure function: current visible near-axis Cube face coordinates.

Only current metric depth/raw mask, actual pixel coordinates and camera K/T
enter this function. No GT face, Cube pose, robot, previous centre or forecast
argument exists. The world-axis-aligned 50 mm Cube is an explicit prior; this
is not an arbitrary-object or general 6D pose estimator.
"""
import numpy as np

PROTOCOL_SHA256='efb13a6181f3d5bb8ecb3502232a637361212282af2de71fe88e16e9e0cc09ad'
HALF_SIDE_M=.025
PLANARITY_M=.00025
AXIS_NORMAL_TOLERANCE_DEGREES=5.
PIXEL_CENTRE_OFFSET=.5
DV,DU=np.indices((3,3))-1
DESIGN=np.c_[DU.ravel(),DV.ravel(),np.ones(9)]
LS=np.linalg.pinv(DESIGN)
AXIS_COSINE_MIN=np.cos(np.deg2rad(AXIS_NORMAL_TOLERANCE_DEGREES))


def point_face_proposals(depth,raw_mask,pixels_uv,K,T,classic):
    """One diagnostic row per original eroded point, with no point dropping.

    ``pixels_uv`` are the original integer pixel indices. Erosion/crop and
    all frame-level support/scope checks are the caller's unchanged contract.
    ``accepted`` alone identifies usable coordinate proposals; rejected rows
    retain reasons and any intermediate diagnostics that were computable.
    """
    depth,raw_mask,pixels_uv,K,T=map(np.asarray,(depth,raw_mask,pixels_uv,K,T))
    if type(classic) is not bool:raise ValueError('Explicit per-camera renderer bool required')
    if (depth.ndim!=2 or depth.dtype.kind!='f' or raw_mask.shape!=depth.shape or raw_mask.dtype!=np.bool_
        or pixels_uv.ndim!=2 or pixels_uv.shape[1]!=2 or not np.issubdtype(pixels_uv.dtype,np.integer)):
        raise ValueError('Current floating metricdepth, bool rawmask and integer[N,2] UV required')
    if (K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all()
        or abs(np.linalg.det(K))<1e-12 or not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6)
        or not np.isclose(np.linalg.det(T[:3,:3]),1.,rtol=0,atol=1e-6)):
        raise ValueError('Finite invertible signedK and proper public camera-to-world T required')
    n=len(pixels_uv)
    out=dict(accepted=np.zeros(n,bool),depth_corrected=np.full(n,np.nan),
        coefficients=np.full((n,3),np.nan),normals_world=np.full((n,3),np.nan),
        axis=np.full(n,-1,np.int8),sign=np.zeros(n,np.int8),coordinate_proposal=np.full(n,np.nan),
        point_world=np.full((n,3),np.nan),metric_residual_max=np.full(n,np.nan),
        normal_axis_angle_degrees=np.full(n,np.nan),
        reasons=np.full(n,'missing_or_invalid_current_raw_Cube_3x3',dtype=object))
    u,v=pixels_uv.T;h,w=depth.shape
    bounded=(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
    ids=np.flatnonzero(bounded)
    if not len(ids):return out
    x=u[ids,None]+DU.ravel();y=v[ids,None]+DV.ravel()
    z=depth[y,x].astype(np.float64)
    support=raw_mask[y,x].all(1)&np.isfinite(z).all(1)&(z>0).all(1)
    ids,z=ids[support],z[support]
    if not len(ids):return out
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        coeff=(1/z)@LS.T
        qfit=coeff@DESIGN.T
        good_fit=np.isfinite(qfit).all(1)&(qfit>0).all(1)&np.isfinite(coeff).all(1)
        fit_depth=np.full_like(qfit,np.nan);fit_depth[good_fit]=1/qfit[good_fit]
        residual=np.max(np.abs(z-fit_depth),axis=1)
        driver_term=.5*np.max(np.abs(coeff[:,:2]),axis=1) if classic else np.zeros(len(ids))
        q_point=1/z[:,4]+driver_term
        good_q=np.isfinite(q_point)&(q_point>0)
    out['coefficients'][ids]=coeff;out['metric_residual_max'][ids]=residual
    out['reasons'][ids]='nonplanar_current_metric_depth_stencil'
    out['reasons'][ids[~good_fit|~good_q]]='nonpositive_or_nonfinite_inverse_depth_fit'
    planar=good_fit&good_q&np.isfinite(residual)&(residual<=PLANARITY_M)
    if not planar.any():return out
    local_ids=ids[planar];c=coeff[planar];term=driver_term[planar]
    corrected=1/q_point[planar]
    # For point coordinates use the ORIGINAL centre depth, with only the
    # prescribed driver correction. The fitted intercept never replaces it.
    pixels=np.c_[u[local_ids]+PIXEL_CENTRE_OFFSET,v[local_ids]+PIXEL_CENTRE_OFFSET,np.ones(len(local_ids))]
    rays=pixels@np.linalg.inv(K).T
    points=(rays*corrected[:,None])@T[:3,:3].T+T[:3,3]
    # The corrected LS intercept is used ONLY for the plane normal. Convert
    # the local affine fit to global pixel-centre coordinates before K.T/R.
    c_normal=c[:,2]+term
    plane_h=np.c_[c[:,:2],c_normal-c[:,0]*pixels[:,0]-c[:,1]*pixels[:,1]]
    normals=(plane_h@K)@T[:3,:3].T
    with np.errstate(over='ignore',invalid='ignore'):
        lengths=np.linalg.norm(normals,axis=1)
    finite=np.isfinite(points).all(1)&np.isfinite(normals).all(1)&np.isfinite(lengths)&(lengths>0)
    out['depth_corrected'][local_ids]=corrected;out['point_world'][local_ids]=points
    out['reasons'][local_ids]='nonfinite_or_degenerate_current_plane_normal'
    if not finite.any():return out
    fids=local_ids[finite];unit=normals[finite]/lengths[finite,None];p=points[finite]
    axis=np.abs(unit).argmax(1);alignment=np.abs(unit[np.arange(len(unit)),axis])
    out['normals_world'][fids]=unit;out['axis'][fids]=axis
    out['normal_axis_angle_degrees'][fids]=np.rad2deg(np.arccos(np.clip(alignment,0.,1.)))
    out['reasons'][fids]='normal_not_within_fixed_five_degrees_of_world_axis'
    aligned=alignment>=AXIS_COSINE_MIN
    if not aligned.any():return out
    aids=fids[aligned];j=axis[aligned];p=p[aligned]
    side=np.sign(T[:3,3][j]-p[np.arange(len(p)),j])
    proposal=p[np.arange(len(p)),j]-side*HALF_SIDE_M
    good=np.isfinite(side)&(side!=0)&np.isfinite(proposal)
    out['reasons'][aids]='degenerate_or_nonfinite_public_camera_face_sign'
    final=aids[good]
    out['sign'][final]=side[good].astype(np.int8)
    out['coordinate_proposal'][final]=proposal[good]
    out['accepted'][final]=True
    out['reasons'][final]='accepted_current_visible_known_cube_face_coordinate'
    return out
