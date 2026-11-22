"""Nominal raw-depth surface XYZ/valid fraction/UV, no body/face assignment.

Classic depth retains renderer bias here: these are learned-input coordinates,
not certified true hit points or a corrected pure Cube surface measurement.
"""
import numpy as np


def cell_geometry(depth,K,camera_to_world,near_far,*,grid=8):
    depth=np.asarray(depth);K=np.asarray(K,float);T=np.asarray(camera_to_world,float)
    clips=np.asarray(near_far,float)
    if (depth.shape!=(256,256)or K.shape!=(3,3)or T.shape!=(4,4)or clips.shape!=(2,)
            or grid!=8 or not np.isfinite(K).all()or not np.isfinite(T).all()
            or not np.isfinite(clips).all()or not 0<clips[0]<clips[1]
            or abs(np.linalg.det(K))<1e-8 or not np.allclose(T[3],[0,0,0,1],atol=1e-8,rtol=0)
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-6,rtol=0)):
        raise ValueError('Actual256 raw metricZ, public signedK/T/clip and fixed8grid required')
    v,u=np.mgrid[:256,:256];pixels=np.stack((u+.5,v+.5,np.ones_like(u)),axis=-1)
    rays=pixels@np.linalg.inv(K).T
    if not np.allclose(rays[...,2],1.,atol=1e-7,rtol=0):
        raise ValueError('Public metric camera-Z intrinsics convention required')
    valid=np.isfinite(depth)&(depth>clips[0])&(depth<clips[1])
    safe=np.where(valid,depth,0.)
    nominal=(rays*safe[...,None])@T[:3,:3].T+T[:3,3]
    output=np.zeros((8,8,6),np.float32);counts=np.zeros((8,8),np.int64)
    for y in range(8):
        for x in range(8):
            sl=(slice(y*32,(y+1)*32),slice(x*32,(x+1)*32));m=valid[sl];n=int(m.sum())
            counts[y,x]=n;output[y,x,3]=n/1024.
            if n:
                output[y,x,:3]=nominal[sl][m].mean(0)
                output[y,x,4:]=pixels[sl][m,:2].mean(0)/256.
    return dict(geometry=output,valid_depth_counts=counts,
        fields=['nominal_surface_worldX','nominal_surface_worldY','nominal_surface_worldZ',
                'valid_fraction','valid_pixel_center_mean_u_over256','valid_pixel_center_mean_v_over256'],
        point_convention='K_inverse@[u+.5,v+.5,1]*raw_cameraZ',
        object_or_face_ownership=False,renderer_pure_plane_correction_applied=False,
        true_surface_bound_or_pose_measurement_qualified=False,
        zero_XYZ_when_no_depth_is_unknown=True)


def two_frame_metadata(lengths):
    lengths=np.asarray(lengths)
    if lengths.ndim!=1 or lengths.dtype.kind not in 'iu'or np.any((lengths<1)|(lengths>5)):
        raise ValueError('Actual prefix lengths1..5 required')
    previous=np.where(lengths>=2,lengths-2,-1)
    current=lengths-1
    return dict(previous_prefix_slot=previous.tolist(),current_prefix_slot=current.tolist(),
        previous_frame_present=(lengths>=2).tolist(),
        duplicate_current_as_previous_allowed=False,physical_dt_inferred=False)
