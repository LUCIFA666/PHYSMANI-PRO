"""Unqualified dual public crop preparation for near+transition research.

The original64crop is reused exactly. Wide features sample a fixed128pixel
Sphere-centred window at stride2, preserving actual RGB/depth pixel pairing.
There is no GT/robot/forecast ROI, interpolation, per-frame target recrop,
surface-normal inference or claim that the window certifies a20cm domain.
"""
import numpy as np
from ..near_goal_segmentation_20261103.data import (
    make_near_goal_crops,validate_public,FEATURE_NAMES)

VERSION='public_Sphere_dual64_stride1_stride2_crops_20261103_v1'
PROPOSED_SCOPE_RADIUS_M=.20
WIDE_STRIDE=2
WIDE_INPUT_WINDOW_PIXELS=128
TRAINING_OR_SCOPE_QUALIFIED=False


def make_dual_goal_crops(inputs):
    validate_public(inputs)
    near=make_near_goal_crops(inputs)
    features=np.zeros((4,9,64,64),np.float32)
    rgb=np.zeros((4,64,64,3),np.uint8);depth=np.zeros((4,64,64),np.float32)
    valid_crop=np.zeros((4,64,64),bool);valid_depth=valid_crop.copy()
    origins=np.zeros((4,2),np.int64);projection=np.zeros(4,bool)
    source_uv=np.zeros((4,64,64,2),np.int64)
    yy,xx=np.indices((64,64));sphere=np.asarray(inputs['public_sphere_center'],float)
    for camera in range(4):
        K=np.asarray(inputs['K'][camera],float);T=np.asarray(inputs['T'][camera],float)
        sc=(sphere-T[:3,3])@T[:3,:3]
        if sc[2]<=0:continue
        uv=(K@(sc/sc[2]))[:2]
        origin=np.floor(uv+.5).astype(np.int64)-64
        origins[camera]=origin;projection[camera]=True
        x=origin[0]+WIDE_STRIDE*xx;y=origin[1]+WIDE_STRIDE*yy
        source_uv[camera]=np.stack((x,y),axis=-1)
        valid=(x>=0)&(x<256)&(y>=0)&(y<256);valid_crop[camera]=valid
        rgb[camera,valid]=inputs['rgb'][camera,y[valid],x[valid]]
        depth[camera,valid]=inputs['depth_m'][camera,y[valid],x[valid]]
        dv=valid&np.isfinite(depth[camera])&(depth[camera]>0);valid_depth[camera]=dv
        pixels=np.stack((x,y,np.ones_like(x)),axis=-1)
        rays=pixels@np.linalg.inv(K).T
        points=rays*np.where(dv,depth[camera],0)[...,None]
        features[camera,:3]=np.moveaxis(np.where(valid[...,None],rgb[camera]/255.-.5,0),-1,0)
        features[camera,3]=np.where(dv,np.log1p(np.where(dv,depth[camera],0)),0)
        features[camera,4:7]=np.moveaxis(np.where(dv[...,None],(points-sc)/.1,0),-1,0)
        features[camera,7]=dv;features[camera,8]=valid
    wide=dict(features=features,rgb=rgb,depth_m=depth,valid_crop=valid_crop,valid_depth=valid_depth,
        origins_xy=origins,source_pixels_uv=source_uv,projection_available=projection,
        stride=WIDE_STRIDE,input_window_pixels=WIDE_INPUT_WINDOW_PIXELS)
    return dict(version=VERSION,near=near,wide=wide,feature_names=FEATURE_NAMES,
        target_or_GT_dependent_crop=False,scope_qualified=False,
        proposed_scope_radius_m=PROPOSED_SCOPE_RADIUS_M,
        feature_ray_convention='unchanged_original_integer_ray_feature_convention_not_a_measured_surface_certificate')


def public_model_arrays(inputs):
    crops=make_dual_goal_crops(inputs)
    return dict(features=crops['near']['features'],wide_features=crops['wide']['features'],
        K=np.asarray(inputs['K'],np.float32).copy(),
        camera_rotations=np.asarray(inputs['T'],np.float32)[:,:3,:3].copy(),
        initial_sphere=np.asarray(inputs['public_sphere_center'],np.float32).copy())
