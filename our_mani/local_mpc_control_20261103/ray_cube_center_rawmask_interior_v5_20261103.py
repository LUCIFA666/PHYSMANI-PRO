"""Offline V5: erode the original current-RGB mask, then remove goal shell.

Only the positive-ray support order changes from V4. A sphere-cut edge is not
an original RGB/depth antialias boundary and receives no second erosion.
Raw RGBD, generic measured depth rays, 13-start cost/solver/config and every
UNKNOWN/observability gate remain unchanged. No production integration.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from ..benchmark_perception import BoxPrior
from . import ray_cube_center_prototype_20261103 as original
from . import ray_cube_center_darkred_prototype_20261103 as v4
from .goal_aware_perception import VisualGoalSphere,filter_observed_sphere_surface,_validate_public

V4_SHA256='32e93e9b19d0d6a0c5bececbeefa28d87d07a959862ea87fce45051646d2fb0a'


class RawMaskInteriorDarkRedCubeEstimator(v4.DarkRedInteriorRayCubeCenterEstimator):
    def __init__(self,config=None):
        if hashlib.sha256(Path(v4.__file__).read_bytes()).hexdigest()!=V4_SHA256:
            raise ValueError('unchanged V4 method/config required')
        super().__init__(config)

    def current_bundle(self,observation,seed_centre,goal_sphere=None):
        _validate_public(observation);c=self.config
        seed=original._finite(seed_centre,(3,),'same-frame or previous real measured initializer')
        prior=BoxPrior(minimum_red=100,red_ratio=3.5);filtered=observation;trace=None
        if goal_sphere is not None:
            if type(goal_sphere) is not VisualGoalSphere:raise TypeError('public visual sphere fit only')
            if goal_sphere.observed_at>observation['timestamp']:raise ValueError('future sphere fit forbidden')
            filtered,trace=filter_observed_sphere_surface(observation,goal_sphere,prior)
        all_rows=[];counts={}
        for camera in sorted(observation['observation/images']):
            image=observation['observation/images'][camera].astype(float)
            filtered_image=filtered['observation/images'][camera].astype(float)
            calibration=observation['observation/calibrations'][camera]
            depth=np.asarray(observation['observation/depths'][camera],float)*calibration.depth_scale
            valid=np.isfinite(depth)&(depth>0);ys,xs=np.nonzero(valid)
            uv=np.c_[xs,ys,np.ones(len(xs))];camera_rays=uv@np.linalg.inv(calibration.intrinsics).T
            directions=camera_rays@calibration.camera_to_world.rotation.T
            origin=calibration.camera_to_world.position
            points=origin+directions*depth[ys,xs,None]
            raw_red=(image[...,0]>=100)&(image[...,0]>3.5*image[...,1])&(image[...,0]>3.5*image[...,2])
            raw_interior=binary_erosion(raw_red,structure=np.ones((3,3),bool),iterations=1,border_value=0)
            shell_survivors=(filtered_image[...,0]>=100)&(filtered_image[...,0]>3.5*filtered_image[...,1])&(filtered_image[...,0]>3.5*filtered_image[...,2])
            eroded=raw_interior&shell_survivors
            selected=np.flatnonzero(eroded[ys,xs]&(np.linalg.norm(points-seed,axis=1)<=c.ray_roi_radius_m))
            if len(selected)>c.target_pixels_per_view:selected=selected[np.linspace(0,len(selected)-1,c.target_pixels_per_view,dtype=int)]
            local=calibration.camera_to_world.inverse_apply(seed);context=np.empty(0,int)
            if local[2]>0:
                projected=calibration.intrinsics@(local/local[2]);u,v=projected[:2]
                radius=max(abs(calibration.intrinsics[0,0]),abs(calibration.intrinsics[1,1]))*c.ray_roi_radius_m/local[2]
                context=np.flatnonzero((np.abs(xs-u)<=radius)&(np.abs(ys-v)<=radius))
                if len(context)>c.context_pixels_per_view:context=context[np.linspace(0,len(context)-1,c.context_pixels_per_view,dtype=int)]
            indices=np.r_[selected,context];targetmask=np.r_[np.ones(len(selected),bool),np.zeros(len(context),bool)]
            all_rows.append((np.broadcast_to(origin,(len(indices),3)),directions[indices],depth[ys[indices],xs[indices]],
                np.full(len(indices),camera),targetmask))
            counts[camera]=dict(target_rays=len(selected),context_depth_rays=len(context),
                observed_R100_Ratio35_pixels=int(raw_red.sum()),raw_interior_pixels=int(raw_interior.sum()),
                red_survivors_after_shell=int(shell_survivors.sum()),interior_pixels=int(eroded.sum()),
                raw_interior_removed_by_shell=int((raw_interior&~shell_survivors).sum()))
        arrays=[np.concatenate([row[index] for row in all_rows]) for index in range(5)]
        bundle=original.CurrentRayBundle(float(observation['timestamp']),*arrays)
        return bundle,dict(per_camera=counts,public_sphere_filter=trace,
            positive_target_support=dict(schema='fixed_R100_ratio35_3x3_on_raw_mask_before_public_sphere_shell_v5',
                minimum_R=100,red_ratio=3.5,erosion_kernel=[3,3],erosion_iterations=1,erosion_before_sphere_removal=True,
                artificial_sphere_cut_edge_eroded_again=False,
                same_colour_predicate_used_for_sphere_shell=True,public_sphere_geometry_fit_unchanged=True,
                source_RGBD_modified=False,generic_depth_and_occluder_rays_unchanged=True,
                original_fit_cost_thresholds_and_multistarts_unchanged=True,
                proposed_using_completed_development_diagnostic=True,blind_validation_claimed=False,
                threshold_search=False,GT_used_for_current_pixel_selection=False))
