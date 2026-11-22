"""Public static-goal sphere-shell removal before the unchanged RedBoxTracker.

Only initial RGBD fits the visual sphere. RedGoalMarker returns an offset goal
frame; subtract its own reported TRAIN offset to recover the fitted sphere
center. This module never accepts Native goal/object poses, held or success.
Only red pixels whose metric world depth lies near that observed sphere's
SURFACE are suppressed in a private RGB copy. It never clips the whole ball.
A target surface overlapping that shell is intrinsically ambiguous and may
also lose pixels; absence remains unknown under unchanged tracker thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import json
from pathlib import Path

import numpy as np

from ..benchmark_perception import RedBoxTracker
from ..contracts import Pose,array
from ..goal_perception import RedGoalMarker
from ..perception import CameraCalibration
from .public_motion_bridge import PUBLIC_OBSERVATION_KEYS,CAMERAS,GOAL_CALIBRATION


@dataclass(frozen=True)
class GoalSphereFilterConfig:
    minimum_surface_tolerance_m:float=.004
    maximum_surface_tolerance_m:float=.006
    public_fit_residual_multiplier:float=3.
    public_calibration_margin_m:float=.001
    def __post_init__(self):
        if (not all(np.isfinite(x) and x>0 for x in asdict(self).values())
                or self.minimum_surface_tolerance_m>self.maximum_surface_tolerance_m):
            raise ValueError('positive ordered declared sphere-surface uncertainty bounds required')


@dataclass(frozen=True)
class VisualGoalSphere:
    center:object
    radius_m:float
    observed_at:float
    median_fit_residual_m:float
    surface_tolerance_m:float
    marker_to_goal_offset:object
    def __post_init__(self):
        object.__setattr__(self,'center',array(self.center,(3,)))
        object.__setattr__(self,'marker_to_goal_offset',array(self.marker_to_goal_offset,(3,)))
        if (not all(np.isfinite(x) for x in (self.radius_m,self.observed_at,self.median_fit_residual_m,self.surface_tolerance_m))
                or not .025<=self.radius_m<=.055 or self.observed_at<0 or self.median_fit_residual_m<0 or self.surface_tolerance_m<=0):
            raise ValueError('finite public visual sphere fit required')


def sphere_from_public_goal_fit(goal_pose,metadata,timestamp,config=None):
    config=GoalSphereFilterConfig() if config is None else config
    if (not isinstance(goal_pose,Pose) or not isinstance(metadata,dict)
            or metadata.get('simulator_pose_used') is not False
            or metadata.get('backend')!='rgbd_red_goal_sphere_train_calibrated_frame'):
        raise ValueError('frozen public RedGoalMarker fit metadata required; Native goal is forbidden')
    offset=array(metadata['marker_to_goal_m'],(3,));residual=float(metadata['median_fit_residual_m'])
    if not np.isfinite(residual) or residual<0:raise ValueError('public sphere residual required')
    tolerance=max(config.minimum_surface_tolerance_m,
                  config.public_fit_residual_multiplier*residual+config.public_calibration_margin_m)
    if tolerance>config.maximum_surface_tolerance_m:
        raise ValueError('initial public sphere fit uncertainty exceeds declared shell bound; filtering unavailable')
    return VisualGoalSphere(goal_pose.position-offset,float(metadata['sphere_radius_m']),float(timestamp),
                            residual,tolerance,offset)


def sphere_surface_mask(world_points,sphere):
    points=np.asarray(world_points,dtype=float)
    if not isinstance(sphere,VisualGoalSphere) or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('finite current public depth points and observed visual sphere required')
    return np.abs(np.linalg.norm(points-sphere.center,axis=-1)-sphere.radius_m)<=sphere.surface_tolerance_m


def _validate_public(observation):
    if not isinstance(observation,dict) or set(observation)!=PUBLIC_OBSERVATION_KEYS:
        raise ValueError('only current public RGBD/calibration/timestamp accepted')
    if not np.isfinite(observation['timestamp']) or observation['timestamp']<0:raise ValueError('real public timestamp required')
    for key in ('observation/images','observation/depths','observation/calibrations'):
        if not isinstance(observation[key],dict) or set(observation[key])!=CAMERAS:
            raise ValueError('the original four public camera views are required')
    for camera in CAMERAS:
        rgb=np.asarray(observation['observation/images'][camera]);depth=np.asarray(observation['observation/depths'][camera])
        if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or depth.shape!=rgb.shape[:2]
                or not isinstance(observation['observation/calibrations'][camera],CameraCalibration)):
            raise ValueError('aligned original RGB/depth and signed calibration required')


def filter_observed_sphere_surface(observation,sphere,prior):
    """Return a private RGB copy; original images, depth and calibration stay exact."""
    _validate_public(observation)
    images={};per_camera={}
    for camera,rgb_value in observation['observation/images'].items():
        rgb=np.asarray(rgb_value);image=rgb.astype(float)
        red=(image[...,0]>=prior.minimum_red)&(image[...,0]>prior.red_ratio*image[...,1])&(image[...,0]>prior.red_ratio*image[...,2])
        depth=np.asarray(observation['observation/depths'][camera]);valid=red&np.isfinite(depth)&(depth>0)
        points=observation['observation/calibrations'][camera].backproject(depth,red)
        suppress=sphere_surface_mask(points,sphere)
        ys,xs=np.nonzero(valid)
        if len(ys)!=len(points):raise ValueError('public red-depth backprojection alignment changed')
        copied=rgb.copy();copied[ys[suppress],xs[suppress]]=0
        images[camera]=copied
        per_camera[camera]=dict(source_red_pixels=int(red.sum()),valid_red_depth_pixels=len(points),
            removed_surface_red_pixels=int(suppress.sum()),remaining_valid_red_depth_pixels=int((~suppress).sum()))
    filtered={'timestamp':observation['timestamp'],'observation/images':images,
              'observation/depths':observation['observation/depths'],
              'observation/calibrations':observation['observation/calibrations']}
    return filtered,dict(per_camera=per_camera,removed_red_pixels_total=sum(r['removed_surface_red_pixels'] for r in per_camera.values()),
        sphere_center=sphere.center.tolist(),sphere_radius_m=sphere.radius_m,surface_tolerance_m=sphere.surface_tolerance_m,
        initial_public_fit_residual_m=sphere.median_fit_residual_m,initial_public_observation_time=sphere.observed_at,
        source_RGBD_modified=False,depth_or_calibration_modified=False,whole_sphere_volume_cut=False,
        target_surface_overlap_with_sphere_shell_may_remove_target_pixels=True,
        tolerance_is_development_spatial_shell_not_calibrated_statistical_covariance=True,
        static_visual_sphere_assumed=True,Native_goal_or_object_pose_used=False)


class GoalAwareRedBoxTracker(RedBoxTracker):
    """Same box prior/scoring/confidence/height gate; only remove public goal shell.

    observe(public_observation)->(ObjectState, metadata), reset() clears causal
    box history while keeping the static visual sphere, reset_episode() refits
    from that episode's first real frame. If initial sphere fit is unavailable,
    explicitly fall back to unchanged unfiltered tracking; never use labels.
    """
    def __init__(self,*,tracker=None,goal_marker=None,goal_calibration=GOAL_CALIBRATION,config=None):
        self.tracker=RedBoxTracker() if tracker is None else tracker
        self.marker=RedGoalMarker(json.loads(Path(goal_calibration).read_text())['offset_train_calibrated_m']) if goal_marker is None else goal_marker
        self.config=GoalSphereFilterConfig() if config is None else config
        if not isinstance(self.config,GoalSphereFilterConfig):raise TypeError('GoalSphereFilterConfig required')
        self.prior=self.tracker.prior
        self._sphere=None;self._initial_fit_attempted=False;self._initial_fit_error=None;self.last_timestamp=None
        self.last_filter_metadata=None

    @property
    def sphere(self):return self._sphere
    @property
    def history(self):return self.tracker.history
    @property
    def last_velocity(self):return self.tracker.last_velocity
    @last_velocity.setter
    def last_velocity(self,value):self.tracker.last_velocity=np.asarray(value,dtype=float)

    def reset(self):
        self.tracker.history.clear();self.tracker.last_velocity=np.zeros(3)

    def reset_episode(self):
        self.reset();self._sphere=None;self._initial_fit_attempted=False;self._initial_fit_error=None;self.last_timestamp=None
        self.last_filter_metadata=None

    def initialize_from_public(self,observation):
        _validate_public(observation)
        if self._initial_fit_attempted:raise ValueError('static sphere fit attempted only on initial public frame')
        self._initial_fit_attempted=True
        try:
            goal,metadata=self.marker.observe(observation)
            self._sphere=sphere_from_public_goal_fit(goal,metadata,observation['timestamp'],self.config)
        except (ValueError,RuntimeError) as error:
            self._initial_fit_error=type(error).__name__+': '+str(error)
        return self._sphere

    def observe(self,observation):
        _validate_public(observation);now=float(observation['timestamp'])
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('unique causal public frames required; no repeated/interpolated observations')
        self.last_timestamp=now
        if not self._initial_fit_attempted:self.initialize_from_public(observation)
        if self._sphere is None:
            filtered=observation;filter_metadata=dict(filter_available=False,reason=self._initial_fit_error,
                Native_goal_or_object_pose_used=False,source_RGBD_modified=False)
        else:
            filtered,filter_metadata=filter_observed_sphere_surface(observation,self._sphere,self.prior)
            filter_metadata['filter_available']=True
        self.last_filter_metadata=filter_metadata
        state,metadata=self.tracker.observe(filtered)
        return state,dict(metadata,goal_aware_static_sphere_filter=filter_metadata,
            confidence_formula_unchanged=True,table_initialization_height_gate_unchanged=True,
            box_prior_unchanged=True,world_rotation_still_unobserved_identity_prior=True)
