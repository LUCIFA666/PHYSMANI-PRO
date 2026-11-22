"""Optional joint SE3 refinement using current public point clouds and depth.

This independent CPU prototype does not change the tracker. Candidate seeds
are supplied by its caller and never generated from native/GT poses. Both
optimization and candidate selection use the same bounded m² objective.
"""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..contracts import Pose
from ..perception import CameraCalibration
from .depth_pose_consistency import FreeSpaceConfig


SCHEMA = 'r10_current_depth_joint_SE3_refinement_prototype_v1'


@dataclass(frozen=True)
class JointRefinementConfig:
    max_nfev: int = 30
    maximum_correspondence_m: float = .06
    free_space: FreeSpaceConfig = field(default_factory=FreeSpaceConfig)
    translation_scale_m: float = .02
    rotation_scale_rad: float = .25

    def __post_init__(self):
        if (type(self.max_nfev) is not int or self.max_nfev < 1
                or not isinstance(self.free_space, FreeSpaceConfig)
                or any(type(value) not in (int,float) or not np.isfinite(value) or value <= 0
                    for value in (self.maximum_correspondence_m, self.translation_scale_m, self.rotation_scale_rad))):
            raise ValueError('finite positive refinement scales/correspondence and integer evaluation budget required')


def _pieces(value, label):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(label+' requires named local CAD/world observation arrays')
    result = {}
    for name, points in value.items():
        points = np.asarray(points, dtype=float)
        if not isinstance(name, str) or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(label+' requires named finite Nx3 point arrays')
        result[name] = points.copy()
    return result


def _bilinear_free_space(surface_world, views, config):
    """Fixed-length residual slots, continuous depth within each image cell.

    Pixel-nearest front-surface ownership remains piecewise constant; bilinear
    measured depth supplies xy gradients inside cells. All four interpolation
    support pixels must be valid. Occluded/outside/invalid slots remain zero.
    """
    residuals, counts = [], dict(evaluated_pixels=0, contradictory_pixels=0,
        occluded_pixels=0, invalid_depth_pixels=0, outside_frustum_samples=0,
        behind_camera_samples=0, front_surface_pixels=0)
    for view in views.values():
        calibration, depth = view[:2]
        points = calibration.camera_to_world.inverse_apply(surface_world)
        residual = np.zeros(len(points))
        positive = points[:,2] > 1e-8
        indices = np.flatnonzero(positive)
        z = points[positive,2]
        projected = points[positive] @ calibration.intrinsics.T
        uv = projected[:,:2] / projected[:,2:]
        h,w = depth.shape
        inside = (np.isfinite(uv).all(1) & (uv[:,0]>=0) & (uv[:,0]<=w-1)
            & (uv[:,1]>=0) & (uv[:,1]<=h-1))
        indices,uv,z = indices[inside],uv[inside],z[inside]
        nearest = np.rint(uv).astype(np.int64)
        pixels = nearest[:,1]*w+nearest[:,0]
        order = np.lexsort((z,pixels))
        unique = np.r_[True,pixels[order][1:]!=pixels[order][:-1]] if len(order) else np.zeros(0,bool)
        chosen = order[unique]
        indices,uv,z = indices[chosen],uv[chosen],z[chosen]
        x0,y0 = np.floor(uv).astype(np.int64).T if len(uv) else (np.zeros(0,int),np.zeros(0,int))
        x1,y1 = np.minimum(x0+1,w-1),np.minimum(y0+1,h-1)
        sample = np.stack((depth[y0,x0],depth[y0,x1],depth[y1,x0],depth[y1,x1]),axis=1)*calibration.depth_scale
        valid = np.isfinite(sample).all(1)&(sample>0).all(1)
        dx,dy = uv[:,0]-x0,uv[:,1]-y0
        weights = np.c_[(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy]
        measured = np.zeros(len(uv))
        measured[valid] = np.sum(sample[valid]*weights[valid],axis=1)
        difference = measured[valid]-z[valid]
        residual[indices[valid]] = np.clip(difference-config.tolerance_m,0.,config.maximum_gap_m)
        residuals.append(residual)
        counts['evaluated_pixels'] += int(valid.sum())
        counts['contradictory_pixels'] += int((difference>config.tolerance_m).sum())
        counts['occluded_pixels'] += int((difference < -config.tolerance_m).sum())
        counts['invalid_depth_pixels'] += int((~valid).sum())
        counts['outside_frustum_samples'] += int((~inside).sum())
        counts['behind_camera_samples'] += int((~positive).sum())
        counts['front_surface_pixels'] += len(chosen)
    return (np.concatenate(residuals)*np.sqrt(config.weight/max(1,counts['evaluated_pixels'])),counts)


class JointCurrentDepthObjective:
    """Reusable objective with no target pose, environment, or future input."""
    def __init__(self,surface_pieces,observed_pieces_world,views,*,config=JointRefinementConfig()):
        self.surface = _pieces(surface_pieces,'CAD')
        observed = _pieces(observed_pieces_world,'current observed points')
        if not isinstance(config,JointRefinementConfig) or set(observed)-set(self.surface):
            raise ValueError('same named CAD/observation components and explicit config required')
        self.config=config
        self.observed={name:points for name,points in observed.items() if len(points)>=3 and len(self.surface[name])>=3}
        if not self.observed or not any(len(points) for points in self.surface.values()):
            raise ValueError('at least one observed component with three points required')
        self.trees={name:cKDTree(self.surface[name]) for name in self.observed}
        self.all_surface=np.concatenate(list(self.surface.values()))
        if not isinstance(views,Mapping) or not views:
            raise ValueError('current synchronized public calibration/depth views required')
        self.views={}
        for name,view in views.items():
            if not isinstance(view,(tuple,list)) or len(view) not in (2,3) or not isinstance(view[0],CameraCalibration):
                raise ValueError('view is public CameraCalibration/current depth/optional ignored current mask')
            depth=np.asarray(view[1],dtype=float)
            if depth.ndim!=2 or min(depth.shape,default=0)<2:
                raise ValueError('bilinear depth requires current image at least2x2')
            self.views[name]=(view[0],depth.copy())

    def evaluate(self,pose):
        observation=[]
        for name,points in self.observed.items():
            distances=self.trees[name].query(pose.inverse_apply(points))[0]
            observation.append(np.minimum(distances,self.config.maximum_correspondence_m)/np.sqrt(len(self.observed)*len(points)))
        observation=np.concatenate(observation)
        free,counts=_bilinear_free_space(pose.apply(self.all_surface),self.views,self.config.free_space)
        obs=float(observation@observation);cost=float(free@free)
        return np.r_[observation,free],dict(observed_mse_m2=obs,current_depth_free_space_mse_m2=cost,
            joint_cost_m2=obs+cost,current_depth_counts=counts,depth_sampling='bilinear current depth, all four support pixels valid',
            maximum_free_space_cost_m2=self.config.free_space.weight*self.config.free_space.maximum_gap_m**2)


def refine_current_depth_pose(surface_pieces,observed_pieces_world,initial_pose,views,
                              *,config=JointRefinementConfig()):
    """Refine one supplied candidate; translation+world-left-rotvec, no bounds.

    The 30-evaluation default is per candidate. Trust-region numerical scales
    are fixed in config; neither seeds nor bounds come from GT or outcomes.
    Real visibility and nearest-correspondence boundaries are nonsmooth, so
    a terminated local fit is not a global optimum or observability proof.
    """
    if not isinstance(initial_pose,Pose):raise ValueError('supplied causal candidate Pose required')
    start=perf_counter();objective=JointCurrentDepthObjective(surface_pieces,observed_pieces_world,views,config=config)
    _,initial=objective.evaluate(initial_pose);calls=0
    def pose_at(parameters):
        return Pose(initial_pose.position+parameters[:3],Rotation.from_rotvec(parameters[3:]).as_matrix()@initial_pose.rotation)
    def residual(parameters):
        nonlocal calls
        calls+=1
        return objective.evaluate(pose_at(parameters))[0]
    result=least_squares(residual,np.zeros(6),max_nfev=config.max_nfev,method='trf',jac='2-point',
        x_scale=np.r_[np.full(3,config.translation_scale_m),np.full(3,config.rotation_scale_rad)])
    refined=pose_at(result.x);_,final=objective.evaluate(refined)
    # Retain the unchanged seed if a terminated finite fit raises the exact
    # joint objective. This is objective-based, never GT/outcome selection.
    accepted=bool(np.isfinite(result.x).all() and final['joint_cost_m2']<=initial['joint_cost_m2']+1e-15)
    return dict(schema=SCHEMA,pose=refined if accepted else initial_pose,initial=initial,
        final=final if accepted else initial,raw_optimizer_final=final,accepted=accepted,
        nfev=int(result.nfev),njev=int(result.njev or 0),residual_calls=calls,status=int(result.status),
        optimizer_success=bool(result.success),message=str(result.message),runtime_seconds=perf_counter()-start,
        translation_step_m=float(np.linalg.norm(result.x[:3])),rotation_step_rad=float(np.linalg.norm(result.x[3:])),
        max_nfev=config.max_nfev,GT_native_future_or_outcome_used=False,pose_observability_proven=False)


def refine_pose_candidates(surface_pieces,observed_pieces_world,candidate_poses,views,
                            *,config=JointRefinementConfig()):
    """Same joint objective for refinement and selection over fixed candidates."""
    candidates=tuple(candidate_poses)
    if not candidates:raise ValueError('supplied nonempty causal candidate set required')
    records=[refine_current_depth_pose(surface_pieces,observed_pieces_world,pose,views,config=config) for pose in candidates]
    selected=min(range(len(records)),key=lambda index:records[index]['final']['joint_cost_m2'])
    return dict(schema=SCHEMA,pose=records[selected]['pose'],selected_index=selected,candidates=records,
        candidate_count=len(records),total_runtime_seconds=sum(row['runtime_seconds'] for row in records),
        selection_cost_m2=records[selected]['final']['joint_cost_m2'],GT_native_future_or_outcome_used=False)
