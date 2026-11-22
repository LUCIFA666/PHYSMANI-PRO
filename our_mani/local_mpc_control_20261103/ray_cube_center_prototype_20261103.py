"""Offline current-RGBD cube-centre prototype; no controller integration.

Known side length is 50mm. Rotation is a nuisance variable, not an observed
orientation output. A surface point alone cannot distinguish a cube's front
and back faces. Measured first-hit depth and neighbouring free prefixes are
therefore included. Missing/degenerate/ambiguous support returns UNKNOWN.
No robot, TCP, GRU, Native object state, task label or future frame is accepted.
Uncertainty below is numerical sensitivity/multistart spread, NOT calibrated
statistical coverage or a grasp certificate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..benchmark_perception import BoxPrior
from .goal_aware_perception import VisualGoalSphere, filter_observed_sphere_surface, _validate_public


@dataclass(frozen=True)
class CubeRayConfig:
    edge_m: float = .05
    noise_scale_m: float = .002
    free_prefix_allowance_m: float = .002
    ray_roi_radius_m: float = .085
    centre_search_halfwidth_m: float = .075
    target_pixels_per_view: int = 128
    context_pixels_per_view: int = 256
    minimum_target_pixels: int = 24
    minimum_target_pixels_per_view: int = 8
    minimum_views: int = 2
    maximum_surface_p95_m: float = .005
    minimum_target_hit_fraction: float = .90
    maximum_free_violation_fraction: float = .05
    free_violation_threshold_m: float = .004
    near_optimal_objective_delta: float = .25
    maximum_near_optimal_centre_spread_m: float = .010
    maximum_numerical_axis_uncertainty_m: float = .006
    nuisance_rotation_seed_rad: float = .35
    max_nfev: int = 100

    def __post_init__(self):
        if self.edge_m != .05: raise ValueError('this declared prototype is a known50mm cube only')
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0: raise ValueError('positive finite declared config required: '+key)
        if self.minimum_views < 2 or self.minimum_target_pixels < 12:
            raise ValueError('current multi-view depth support cannot be disabled')
        if self.centre_search_halfwidth_m < self.edge_m:
            raise ValueError('search must include the50mm mirror, not lock to initializer')
        for key in ('target_pixels_per_view','context_pixels_per_view','minimum_target_pixels',
                    'minimum_target_pixels_per_view','minimum_views','max_nfev'):
            if type(getattr(self,key)) is not int: raise ValueError('integer config required: '+key)


def _finite(value, shape, name):
    value=np.asarray(value,dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all(): raise ValueError('finite '+name+str(shape)+' required')
    return value


def _proper_rotation(value):
    value=_finite(value,(3,3),'rotation')
    if (np.max(np.abs(value.T@value-np.eye(3)))>1e-6
            or abs(np.linalg.det(value)-1.)>1e-6):
        raise ValueError('proper SO3 nuisance rotation required')
    return value


def cube_sdf(points, centre, rotation, edge_m=.05):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all(): raise ValueError('points[N,3] required')
    centre=_finite(centre,(3,),'centre');rotation=_proper_rotation(rotation)
    if not np.isfinite(edge_m) or edge_m<=0:raise ValueError('positive finite cube side required')
    local=(points-centre)@rotation
    q=np.abs(local)-edge_m/2
    return np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.)


def ray_box_first_hit(origins, directions, centre, rotation, edge_m=.05):
    """Slabs with t in camera-Z metres: directions are NOT normalized.

    Truly parallel outside slabs are misses, not divided by a fake epsilon.
    A nearer measured occluder is handled by the caller's one-sided free
    prefix loss, never by declaring the object absent behind that occluder.
    """
    origins=np.asarray(origins,float);directions=np.asarray(directions,float)
    if (origins.ndim!=2 or origins.shape[1]!=3 or directions.shape!=origins.shape
            or not np.isfinite(origins).all() or not np.isfinite(directions).all()
            or np.any(np.linalg.norm(directions,axis=1)<1e-12)):
        raise ValueError('finite nonzero camera rays[N,3] required')
    centre=_finite(centre,(3,),'centre');rotation=_proper_rotation(rotation)
    if not np.isfinite(edge_m) or edge_m<=0:raise ValueError('positive finite cube side required')
    o=(origins-centre)@rotation;d=directions@rotation;h=edge_m/2
    parallel=np.abs(d)<1e-12;outside=parallel&(np.abs(o)>h)
    a=np.divide(-h-o,d,out=np.zeros_like(o),where=~parallel)
    b=np.divide(h-o,d,out=np.zeros_like(o),where=~parallel)
    lo=np.where(parallel,-np.inf,np.minimum(a,b));hi=np.where(parallel,np.inf,np.maximum(a,b))
    entry=np.max(lo,axis=1);exit=np.min(hi,axis=1)
    hit=(~np.any(outside,axis=1))&(exit>=np.maximum(entry,0.))&(exit>0.)
    return np.where(hit,np.maximum(entry,0.),np.inf),np.where(hit,exit,-np.inf),hit


@dataclass(frozen=True)
class CurrentRayBundle:
    timestamp: float
    origins: object
    directions: object
    depth: object
    view_ids: object
    is_target: object

    def __post_init__(self):
        if isinstance(self.timestamp,(bool,np.bool_)) or not np.isfinite(self.timestamp) or self.timestamp<0:
            raise ValueError('one current real timestamp required')
        o=np.asarray(self.origins,float);n=len(o)
        for key,shape in (('origins',(n,3)),('directions',(n,3)),('depth',(n,))):
            value=_finite(getattr(self,key),shape,key).copy();value.setflags(write=False);object.__setattr__(self,key,value)
        if np.any(self.depth<=0) or np.any(np.linalg.norm(self.directions,axis=1)<1e-12):
            raise ValueError('valid real depth and nonzero signed camera rays required')
        view=np.asarray(self.view_ids);mask=np.asarray(self.is_target)
        if view.shape!=(n,) or mask.shape!=(n,) or mask.dtype!=np.bool_:
            raise ValueError('view identities and explicit measured red-target mask required')
        for key,value in (('view_ids',view.copy()),('is_target',mask.copy())):
            value.setflags(write=False);object.__setattr__(self,key,value)

    @property
    def points(self): return self.origins+self.depth[:,None]*self.directions


class RayCubeCenterEstimator:
    def __init__(self, config=None): self.config=config or CubeRayConfig()

    def current_bundle(self, observation, seed_centre, goal_sphere=None):
        _validate_public(observation);c=self.config;seed=_finite(seed_centre,(3,),'same-frame or previous real measured initializer')
        prior=BoxPrior();filtered=observation;trace=None
        if goal_sphere is not None:
            if type(goal_sphere) is not VisualGoalSphere: raise TypeError('public visual sphere fit only')
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
            red=(image[ys,xs,0]>=prior.minimum_red)&(image[ys,xs,0]>prior.red_ratio*image[ys,xs,1])&(image[ys,xs,0]>prior.red_ratio*image[ys,xs,2])
            selected=np.flatnonzero(red&(np.linalg.norm(points-seed,axis=1)<=c.ray_roi_radius_m))
            if len(selected)>c.target_pixels_per_view:selected=selected[np.linspace(0,len(selected)-1,c.target_pixels_per_view,dtype=int)]
            local=calibration.camera_to_world.inverse_apply(seed)
            context=np.empty(0,int)
            if local[2]>0:
                projected=calibration.intrinsics@(local/local[2]);u,v=projected[:2]
                radius=max(abs(calibration.intrinsics[0,0]),abs(calibration.intrinsics[1,1]))*c.ray_roi_radius_m/local[2]
                context=np.flatnonzero((np.abs(xs-u)<=radius)&(np.abs(ys-v)<=radius))
                if len(context)>c.context_pixels_per_view:context=context[np.linspace(0,len(context)-1,c.context_pixels_per_view,dtype=int)]
            indices=np.r_[selected,context];targetmask=np.r_[np.ones(len(selected),bool),np.zeros(len(context),bool)]
            all_rows.append((np.broadcast_to(origin,(len(indices),3)),directions[indices],depth[ys[indices],xs[indices]],
                             np.full(len(indices),camera),targetmask))
            counts[camera]=dict(target_rays=len(selected),context_depth_rays=len(context))
        arrays=[np.concatenate([row[index] for row in all_rows]) for index in range(5)]
        return CurrentRayBundle(float(observation['timestamp']),*arrays),dict(per_camera=counts,public_sphere_filter=trace)

    @staticmethod
    def _weights(views):
        unique,inverse,counts=np.unique(views,return_inverse=True,return_counts=True)
        return 1/np.sqrt(len(unique)*counts[inverse])

    def _residual(self, x, bundle):
        c=self.config;centre=x[:3];rotation=Rotation.from_rotvec(x[3:]).as_matrix();target=bundle.is_target
        entry,_,hit=ray_box_first_hit(bundle.origins,bundle.directions,centre,rotation,c.edge_m)
        norms=np.linalg.norm(bundle.directions,axis=1)
        sdf=cube_sdf(bundle.points[target],centre,rotation,c.edge_m)
        first=np.where(hit[target],(entry[target]-bundle.depth[target])*norms[target],np.abs(sdf)+.005)
        free=np.where(hit,np.maximum(0.,(bundle.depth-entry)*norms-c.free_prefix_allowance_m),0.)
        wt=self._weights(bundle.view_ids[target]);wf=self._weights(bundle.view_ids)
        return np.r_[wt*sdf/c.noise_scale_m,wt*first/c.noise_scale_m,wf*free/c.noise_scale_m]

    def fit(self, observation, seed_centre, goal_sphere=None):
        started=time.perf_counter();bundle,metadata=self.current_bundle(observation,seed_centre,goal_sphere)
        result=self.fit_bundle(bundle,seed_centre);result['input_geometry']=metadata
        result['wall_seconds']=time.perf_counter()-started;return result

    def fit_bundle(self,bundle,seed_centre):
        if type(bundle) is not CurrentRayBundle: raise TypeError('current measured ray bundle only')
        started=time.perf_counter();c=self.config;seed=_finite(seed_centre,(3,),'public measurement initializer')
        target=bundle.is_target;views,counts=np.unique(bundle.view_ids[target],return_counts=True)
        base=dict(timestamp=bundle.timestamp,available=False,centre=None,uncertainty=None,
            config=asdict(c),mode='UNKNOWN',rotation_observed=False,rotation_nuisance_only=True,
            prediction_or_robot_used_as_observation=False,GT_inputs_used=False,future_frames_used=False,
            statistical_uncertainty_calibrated=False,production_integrated=False,
            support=dict(target_rays=int(target.sum()),views={str(v):int(n) for v,n in zip(views,counts)}))
        if target.sum()<c.minimum_target_pixels or np.sum(counts>=c.minimum_target_pixels_per_view)<c.minimum_views:
            return dict(base,reasons=['insufficient_current_multi_view_target_depth'],wall_seconds=time.perf_counter()-started)
        starts=[np.r_[seed,np.zeros(3)]]
        for axis in range(3):
            for sign in (-1.,1.):
                offset=np.eye(3)[axis]*sign*c.edge_m;starts.append(np.r_[seed+offset,np.zeros(3)])
                rotation=np.eye(3)[axis]*sign*c.nuisance_rotation_seed_rad;starts.append(np.r_[seed,rotation])
        lower=np.r_[seed-c.centre_search_halfwidth_m,[-np.pi]*3];upper=np.r_[seed+c.centre_search_halfwidth_m,[np.pi]*3]
        fits=[]
        for start in starts:
            fit=least_squares(self._residual,start,args=(bundle,),bounds=(lower,upper),
                max_nfev=c.max_nfev,loss='soft_l1',f_scale=1/np.sqrt(target.sum()),
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            if np.isfinite(fit.x).all() and np.isfinite(fit.cost):fits.append(fit)
        if not fits:return dict(base,reasons=['no_finite_fit'],wall_seconds=time.perf_counter()-started)
        fits.sort(key=lambda f:f.cost);best=fits[0];centre=best.x[:3];rotation=Rotation.from_rotvec(best.x[3:]).as_matrix()
        entry,_,hit=ray_box_first_hit(bundle.origins,bundle.directions,centre,rotation,c.edge_m)
        norms=np.linalg.norm(bundle.directions,axis=1)
        distances=np.where(hit[target],np.abs(entry[target]-bundle.depth[target])*norms[target],np.inf)
        penetration=np.where(hit,np.maximum(0.,(bundle.depth-entry)*norms),0.)
        surface_p95=float(np.quantile(distances,.95));hit_fraction=float(hit[target].mean())
        free_fraction=float(np.mean(penetration>c.free_violation_threshold_m))
        near=[fit for fit in fits if 2*fit.cost<=2*best.cost+c.near_optimal_objective_delta]
        centres=np.array([fit.x[:3] for fit in near]);spread=float(np.max(np.linalg.norm(centres[:,None]-centres[None,:],axis=-1)))
        jac=best.jac;jc,jr=jac[:,:3],jac[:,3:];marginal=jc-jr@np.linalg.pinv(jr)@jc
        singular=np.linalg.svd(marginal,compute_uv=False);rank=int(np.sum(singular>max(singular[0]*1e-5,1e-8)))
        uncertainty=np.full(3,np.inf) if rank<3 else np.sqrt(np.maximum(np.diag(np.linalg.pinv(marginal.T@marginal)),0.))
        reasons=[]
        if hit_fraction<c.minimum_target_hit_fraction:reasons.append('target_rays_do_not_intersect_candidate')
        if not np.isfinite(surface_p95) or surface_p95>c.maximum_surface_p95_m:reasons.append('current_first_surface_depth_inconsistent')
        if free_fraction>c.maximum_free_violation_fraction:reasons.append('candidate_occludes_measured_free_space')
        if spread>c.maximum_near_optimal_centre_spread_m:reasons.append('near_optimal_mirror_or_centre_ambiguity')
        if rank<3 or np.max(uncertainty)>c.maximum_numerical_axis_uncertainty_m:reasons.append('centre_unobservable_with_rotation_nuisance')
        if np.any(np.minimum(centre-lower[:3],upper[:3]-centre)<.001):reasons.append('search_boundary_not_current_observation_proof')
        diagnostics=dict(surface_depth_p95_m=surface_p95 if np.isfinite(surface_p95) else None,
            target_first_hit_fraction=hit_fraction,free_space_violation_fraction=free_fraction,
            robust_objective=float(2*best.cost),finite_multistarts=len(fits),near_optimal_starts=len(near),
            near_optimal_centre_spread_m=spread,near_optimal_axis_ranges_m=np.ptp(centres,axis=0).tolist(),
            position_marginal_rank=rank,position_marginal_singular_values=singular.tolist(),
            numerical_axis_sensitivity_m=[float(v) if np.isfinite(v) else None for v in uncertainty],
            optimizer_success=bool(best.success),total_nfev=sum(fit.nfev for fit in fits),
            nuisance_rotation_matrix=rotation.tolist(),
            best_centre_is_diagnostic_only_if_unavailable=centre.tolist(),
            hypothesis_count_is_not_exhaustive_uniqueness_proof=True)
        accepted=not reasons
        return dict(base,available=accepted,mode='CURRENT_RGBD_CENTRE' if accepted else 'UNKNOWN',
            centre=centre.tolist() if accepted else None,reasons=reasons,
            uncertainty=dict(numerical_axis_sensitivity_m=diagnostics['numerical_axis_sensitivity_m'],
                near_optimal_axis_ranges_m=diagnostics['near_optimal_axis_ranges_m'],
                near_optimal_centre_spread_m=spread,statistical_coverage_calibrated=False),
            diagnostics=diagnostics,wall_seconds=time.perf_counter()-started)
