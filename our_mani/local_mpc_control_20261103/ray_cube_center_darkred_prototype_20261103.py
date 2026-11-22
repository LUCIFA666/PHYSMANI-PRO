"""V4 development-only colour hypothesis: minimumR100 once, ratio3.5 fixed.

This follows a completed development diagnostic, not blind validation. The
original V1/V2/V3 source/predictions remain unchanged. Only current_bundle is
copied to expose the hardcoded colour prior; all cost/solver/geometry gates
and13multistarts are inherited from the frozen original method unchanged.
Current measured depth and generic free/occluded rays are not modified.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from ..benchmark_perception import BoxPrior
from . import ray_cube_center_prototype_20261103 as original
from .goal_aware_perception import VisualGoalSphere,filter_observed_sphere_surface,_validate_public

ORIGINAL_SHA256='cd03b7b2f4736b8d55e69ae1ea7304f36a9049b6788f197ddff9ef8c470d5b3d'


class DarkRedInteriorRayCubeCenterEstimator(original.RayCubeCenterEstimator):
    minimum_red=100
    red_ratio=3.5

    def __init__(self,config=None):
        if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()!=ORIGINAL_SHA256:
            raise ValueError('unchanged original cost/config/solver required')
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
            image=filtered['observation/images'][camera].astype(float)
            calibration=observation['observation/calibrations'][camera]
            depth=np.asarray(observation['observation/depths'][camera],float)*calibration.depth_scale
            valid=np.isfinite(depth)&(depth>0);ys,xs=np.nonzero(valid)
            uv=np.c_[xs,ys,np.ones(len(xs))];camera_rays=uv@np.linalg.inv(calibration.intrinsics).T
            directions=camera_rays@calibration.camera_to_world.rotation.T
            origin=calibration.camera_to_world.position
            points=origin+directions*depth[ys,xs,None]
            raw_red=(image[...,0]>=100)&(image[...,0]>3.5*image[...,1])&(image[...,0]>3.5*image[...,2])
            eroded=binary_erosion(raw_red,structure=np.ones((3,3),bool),iterations=1,border_value=0)
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
                observed_R100_Ratio35_pixels=int(raw_red.sum()),interior_pixels=int(eroded.sum()))
        arrays=[np.concatenate([row[index] for row in all_rows]) for index in range(5)]
        bundle=original.CurrentRayBundle(float(observation['timestamp']),*arrays)
        return bundle,dict(per_camera=counts,public_sphere_filter=trace,
            positive_target_support=dict(schema='fixed_R100_ratio35_3x3_once_after_public_sphere_shell_v4',
                minimum_R=100,red_ratio=3.5,erosion_kernel=[3,3],erosion_iterations=1,
                same_colour_predicate_used_for_sphere_shell=True,public_sphere_geometry_fit_unchanged=True,
                source_RGBD_modified=False,generic_depth_and_occluder_rays_unchanged=True,
                original_fit_cost_thresholds_and_multistarts_unchanged=True,
                proposed_using_completed_development_diagnostic=True,blind_validation_claimed=False,
                threshold_search=False,GT_used_for_current_pixel_selection=False))
