"""V3 offline wrapper: one fixed3x3 erosion of positive RGB target support.

The original V1/V2 module, fit/cost, all thresholds,13multistarts, physical
rays and current-depth context are unchanged. Only the red target mask in a
private RGB copy loses its one-pixel boundary, after the same public sphere
shell subtraction. This is no production tracker or attachment estimator.
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


class InteriorRayCubeCenterEstimator(original.RayCubeCenterEstimator):
    def __init__(self,config=None):
        path=Path(original.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ORIGINAL_SHA256:
            raise ValueError('unchanged frozen original cube-ray method required')
        super().__init__(config)

    def current_bundle(self,observation,seed_centre,goal_sphere=None):
        _validate_public(observation);prior=BoxPrior();filtered=observation;trace=None
        if goal_sphere is not None:
            if type(goal_sphere) is not VisualGoalSphere:raise TypeError('public visual sphere fit only')
            if goal_sphere.observed_at>observation['timestamp']:raise ValueError('future sphere fit forbidden')
            filtered,trace=filter_observed_sphere_surface(observation,goal_sphere,prior)
        images={};counts={}
        for camera,value in filtered['observation/images'].items():
            image=np.asarray(value);floating=image.astype(float)
            red=(floating[...,0]>=prior.minimum_red)&(floating[...,0]>prior.red_ratio*floating[...,1])&(floating[...,0]>prior.red_ratio*floating[...,2])
            interior=binary_erosion(red,structure=np.ones((3,3),bool),iterations=1,border_value=0)
            copied=image.copy();copied[red&~interior]=0;images[camera]=copied
            counts[camera]=dict(original_positive_red_pixels=int(red.sum()),
                interior_positive_red_pixels=int(interior.sum()),removed_boundary_red_pixels=int((red&~interior).sum()))
        private=dict(filtered);private['observation/images']=images
        # No second sphere filtering or erosion. Generic depth-ray selection
        # in the inherited method does not inspect RGB and is exactly retained.
        bundle,metadata=super().current_bundle(private,seed_centre,goal_sphere=None)
        metadata.update(public_sphere_filter=trace,
            positive_target_support=dict(schema='fixed_3x3_binary_erosion_once_after_public_sphere_shell_v3',
                per_camera=counts,erosion_kernel=[3,3],erosion_iterations=1,
                generic_depth_and_occluder_rays_unchanged=True,source_RGBD_modified=False,
                original_fit_cost_thresholds_and_multistarts_unchanged=True,
                support_change_authorized_from_fixed_TRAIN_boundary_counterexample=True,
                GT_used_for_current_pixel_selection=False))
        return bundle,metadata
