"""One fixed, current-observation cup primitive prototype, never a policy gate.

The cylinder fit has an unsigned axis. The outside handle supplies a radial
direction; the open mouth versus closed bottom is scored only after these
features reduce the pose to a finite candidate set. There is no GT argument,
temporal pose seed, native handle, task schedule, or generic ICP iteration.
"""
from dataclasses import dataclass,asdict
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.perception import rigid_fit
from our_mani.multitask.depth_pose_consistency import FreeSpaceConfig,current_depth_pose_cost

VERSION='r10_fixed_current_RGBD_cup_cylinder_handle_mouth_prototype_v1'


@dataclass(frozen=True)
class FeatureConfig:
    voxel_m:float=.002
    maximum_points:int=768
    minimum_points:int=48
    normal_neighbours:int=12
    normal_pairs:int=40
    normal_pair_seed:int=86121
    cylinder_seed_refinements:int=8
    cylinder_max_nfev:int=45
    cylinder_loss_scale_m:float=.002
    cylinder_trim_fraction:float=.80
    axial_quantile:float=.01
    handle_outside_margin_m:float=.004
    minimum_handle_points:int=6
    handle_histogram_bins:int=36
    handle_peak_count:int=4
    handle_cluster_half_angle_degrees:float=20.
    handle_peak_separation_degrees:float=45.
    handle_distance_cap_m:float=.020
    handle_score_weight:float=2.
    observation_distance_cap_m:float=.025
    free_space_weight:float=2.
    ambiguity_cost_gap_m2:float=.000004
    cylinder_readiness_rms_m:float=.005


def unit(vector):
    vector=np.asarray(vector,float);norm=np.linalg.norm(vector)
    if not np.isfinite(norm) or norm<1e-10:raise ValueError('nondegenerate finite direction required')
    return vector/norm


def basis(axis):
    axis=unit(axis);reference=np.eye(3)[int(np.argmin(np.abs(axis)))]
    first=unit(np.cross(axis,reference));second=np.cross(axis,first)
    return first,second


def sample_cloud(points,config):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite current Nx3 cloud required')
    if len(points)==0:return points
    cells=np.floor(points/config.voxel_m).astype(np.int64)
    _,indices=np.unique(cells,axis=0,return_index=True)
    points=points[np.sort(indices)]
    if len(points)>config.maximum_points:
        points=points[np.linspace(0,len(points)-1,config.maximum_points,dtype=int)]
    return points


@dataclass(frozen=True)
class CupCADFeatures:
    canonical_surface:np.ndarray
    canonical_handle_surface:np.ndarray
    cylinder_centre_canonical:np.ndarray
    outer_radius_m:float
    inner_radius_m:float
    half_height_m:float
    canonical_to_role:Pose
    receipt:dict


def features_from_frozen_cad(canonical_asset,asset,role):
    """Corresponding TRAIN CAD vertices move feature axes into each root frame.

    Annotation is explicitly local CAD: canonical cup axis +Z, handle +Y,
    cylinder axis through (0,-15.85mm), outer/inner radii 32.8/29.3mm.
    No native or world-space orientation participates in this construction.
    """
    source=canonical_asset['bodies']['cup0'];body=asset['bodies'][role]
    original=np.asarray(source['collision_vertices_body'],float)[:916]
    current=np.asarray(body['collision_vertices_body'],float)[:916]
    if original.shape!=(916,3) or current.shape!=(916,3):raise ValueError('fixed actual mug physical vertex inventory changed')
    transform=rigid_fit(original,current)
    error=float(np.linalg.norm(transform.apply(original)-current,axis=1).max())
    if error>2e-6:raise ValueError('TRAIN mug CAD vertex correspondence exceeds fixed 2um check')
    components=[component for component in body['components'] if component['name']=='mug_visual'+role[-1]]
    if len(components)!=1:raise ValueError('one explicitly visible mug component required')
    surface=transform.inverse_apply(np.asarray(components[0]['points_body'],float))
    minimum,maximum=float(surface[:,2].min()),float(surface[:,2].max())
    centre=np.array([0.,-.01585,(minimum+maximum)/2])
    radial=np.linalg.norm(surface[:,:2]-centre[:2],axis=1)
    handle=surface[radial>.0328+.004]
    if len(handle)<6:raise ValueError('frozen CAD lacks the explicit outside-cylinder handle feature')
    receipt=dict(feature_source='TRAIN local CAD annotation plus corresponding TRAIN vertex geometry only',
        canonical_asset_internal_sha256=canonical_asset['asset_sha256'],asset_internal_sha256=asset['asset_sha256'],
        role=role,local_vertex_correspondence_max_error_m=error,corresponding_vertices=916,
        canonical_axis=[0.,0.,1.],canonical_handle_radial=[0.,1.,0.],
        cylinder_centre_canonical=centre.tolist(),outer_radius_m=.0328,inner_radius_m=.0293,
        half_height_m=(maximum-minimum)/2,canonical_surface_points=len(surface),
        canonical_outside_handle_points=len(handle),GT_or_world_pose_used=False)
    return CupCADFeatures(surface,handle,centre,.0328,.0293,(maximum-minimum)/2,transform,receipt)


def _cylinder_distances(points,centre,axis,model):
    local=points-centre;axial=local@axis
    radial=np.linalg.norm(local-axial[:,None]*axis,axis=1)
    axial_overflow=np.maximum(np.abs(axial)-model.half_height_m,0.)
    outer=np.hypot(radial-model.outer_radius_m,axial_overflow)
    inner=np.hypot(radial-model.inner_radius_m,axial_overflow)
    cap=np.hypot(np.abs(axial)-model.half_height_m,np.maximum(radial-model.outer_radius_m,0.))
    return np.minimum(np.minimum(inner,outer),cap)


def _trimmed_mse(values,fraction):
    values=np.asarray(values,float);count=max(1,int(math.ceil(len(values)*fraction)))
    selected=np.partition(values*values,count-1)[:count]
    return float(selected.mean())


def _circle_seed(points,axis,model,config):
    first,second=basis(axis);uv=np.c_[points@first,points@second]
    # Algebraic seed uses only the current projected observations.
    M=np.c_[2*uv,np.ones(len(uv))];y=np.sum(uv*uv,axis=1)
    centre=np.linalg.lstsq(M,y,rcond=None)[0][:2]
    if not np.isfinite(centre).all():centre=np.mean(uv,axis=0)
    limit=float(np.linalg.norm(np.ptp(points,axis=0)))
    if np.linalg.norm(centre-uv.mean(0))>limit:centre=uv.mean(0)
    def residual(x):
        radius=np.linalg.norm(uv-x,axis=1)
        outer=radius-model.outer_radius_m;inner=radius-model.inner_radius_m
        return np.where(np.abs(outer)<np.abs(inner),outer,inner)
    fit=least_squares(residual,centre,loss='soft_l1',f_scale=config.cylinder_loss_scale_m,max_nfev=20)
    axial=points@axis;lo,hi=np.quantile(axial,[config.axial_quantile,1-config.axial_quantile])
    return first*fit.x[0]+second*fit.x[1]+axis*(lo+hi)/2


def fit_unsigned_cylinder(points,model,config=FeatureConfig()):
    points=sample_cloud(points,config)
    if len(points)<config.minimum_points:return None
    _,Q=np.linalg.eigh(np.cov(points.T));seeds=[Q[:,i] for i in range(3)]
    tree=cKDTree(points);_,indices=tree.query(points,k=min(config.normal_neighbours,len(points)))
    neighbourhood=points[indices];means=neighbourhood.mean(axis=1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',neighbourhood-means,neighbourhood-means)
    _,normals=np.linalg.eigh(cov);normals=normals[:,:,0]
    rng=np.random.default_rng(config.normal_pair_seed)
    for one,two in rng.integers(0,len(points),size=(config.normal_pairs,2)):
        candidate=np.cross(normals[one],normals[two])
        if np.linalg.norm(candidate)>.15:seeds.append(unit(candidate))
    unique=[]
    for candidate in seeds:
        if not any(abs(candidate@previous)>.995 for previous in unique):unique.append(candidate)
    ranked=[]
    for axis in unique:
        centre=_circle_seed(points,axis,model,config)
        distances=_cylinder_distances(points,centre,axis,model)
        ranked.append((_trimmed_mse(distances,config.cylinder_trim_fraction),centre,axis))
    fitted=[]
    for initial_cost,initial_centre,initial_axis in sorted(ranked,key=lambda row:row[0])[:config.cylinder_seed_refinements]:
        first,second=basis(initial_axis)
        def unpack(parameters):
            return parameters[:3],unit(initial_axis+first*parameters[3]+second*parameters[4])
        def residual(parameters):
            centre,axis=unpack(parameters)
            return _cylinder_distances(points,centre,axis,model)
        fit=least_squares(residual,np.r_[initial_centre,0.,0.],loss='soft_l1',
            f_scale=config.cylinder_loss_scale_m,max_nfev=config.cylinder_max_nfev,
            bounds=(np.r_[points.mean(0)-.2,-4.,-4.],np.r_[points.mean(0)+.2,4.,4.]))
        centre,axis=unpack(fit.x);distances=residual(fit.x)
        fitted.append(dict(centre=centre,axis=axis,cost_m2=_trimmed_mse(distances,config.cylinder_trim_fraction),
            rms_m=float(np.sqrt(_trimmed_mse(distances,config.cylinder_trim_fraction))),
            nfev=int(fit.nfev),optimizer_success=bool(fit.success),initial_cost_m2=initial_cost))
    if not fitted:return None
    result=min(fitted,key=lambda row:row['cost_m2'])
    return {**result,'points':points,'initial_axis_candidates':len(unique),'refined_axis_candidates':len(fitted)}


def handle_radial_candidates(points,centre,axis,model,config=FeatureConfig()):
    local=points-centre;radial=local-(local@axis)[:,None]*axis
    lengths=np.linalg.norm(radial,axis=1)
    keep=lengths>model.outer_radius_m+config.handle_outside_margin_m
    outside=points[keep];vectors=radial[keep];lengths=lengths[keep]
    if len(outside)<config.minimum_handle_points:return [],outside
    directions=vectors/lengths[:,None];first,second=basis(axis)
    angles=np.arctan2(directions@second,directions@first)
    histogram,edges=np.histogram(angles,bins=config.handle_histogram_bins,range=(-np.pi,np.pi))
    smooth=histogram+np.roll(histogram,1)+np.roll(histogram,-1)
    centres=(edges[:-1]+edges[1:])/2;candidates=[]
    difference=lambda x,y:np.abs(np.arctan2(np.sin(x-y),np.cos(x-y)))
    for index in np.argsort(-smooth,kind='stable'):
        theta=centres[index]
        if any(difference(theta,old['angle_rad'])<np.deg2rad(config.handle_peak_separation_degrees) for old in candidates):continue
        cluster=difference(angles,theta)<=np.deg2rad(config.handle_cluster_half_angle_degrees)
        if int(cluster.sum())<config.minimum_handle_points:continue
        direction=unit(directions[cluster].mean(0));direction=unit(direction-axis*(direction@axis))
        candidates.append(dict(direction=direction,angle_rad=float(np.arctan2(direction@second,direction@first)),
            support_points=int(cluster.sum()),resultant_length=float(np.linalg.norm(directions[cluster].mean(0)))))
        if len(candidates)>=config.handle_peak_count:break
    return candidates,outside


def estimate_cup_feature_pose(points_world,model,views,*,reference_time=None,config=FeatureConfig()):
    """Pure numeric current RGB-D/CAD pose proposal; no production authorization."""
    fit=fit_unsigned_cylinder(points_world,model,config)
    base=dict(version=VERSION,policy_authorized=False,full_orientation_promoted=False,
        GT_or_native_pose_used=False,current_frame_only=True,configuration=asdict(config),
        CAD_feature_receipt=model.receipt)
    if fit is None:return {**base,'pose_xyzw':None,'status':'insufficient_current_points','ambiguous':True,'candidate_count':0}
    centre,axis,points=fit['centre'],fit['axis'],fit['points']
    directions,outside=handle_radial_candidates(points,centre,axis,model,config)
    diagnostic={key:(value.tolist() if isinstance(value,np.ndarray) else value)
        for key,value in fit.items() if key!='points'}
    if not directions:return {**base,'pose_xyzw':None,'status':'outside_handle_not_resolved','ambiguous':True,
        'candidate_count':0,'cylinder':diagnostic,'outside_handle_points':len(outside)}
    candidates=[]
    free_config=FreeSpaceConfig()
    for handle in directions:
        for polarity in (1.,-1.):
            z=axis*polarity;y=unit(handle['direction']-z*(handle['direction']@z));x=unit(np.cross(y,z))
            rotation=np.column_stack((x,y,z));canonical_pose=Pose(centre-rotation@model.cylinder_centre_canonical,rotation)
            surface=canonical_pose.apply(model.canonical_surface);handle_surface=canonical_pose.apply(model.canonical_handle_surface)
            distances=cKDTree(surface).query(points)[0]
            observation_cost=float(np.minimum(distances,config.observation_distance_cap_m).__pow__(2).mean())
            handle_distances=cKDTree(handle_surface).query(outside)[0]
            handle_cost=float(np.minimum(handle_distances,config.handle_distance_cap_m).__pow__(2).mean())
            depth=(current_depth_pose_cost({'cup_surface':model.canonical_surface},canonical_pose,views,
                config=free_config,reference_time=reference_time,
                view_timestamps={name:reference_time for name in views} if reference_time is not None else None) if views else None)
            depth_cost=0. if depth is None else float(depth['unweighted_cost_m2'])
            total=observation_cost+config.handle_score_weight*handle_cost+config.free_space_weight*depth_cost
            role_rotation=rotation@model.canonical_to_role.rotation.T
            role_position=canonical_pose.position-role_rotation@model.canonical_to_role.position
            role_pose=Pose(role_position,role_rotation)
            candidates.append(dict(pose_xyzw=role_pose.xyzw().tolist(),canonical_pose_xyzw=canonical_pose.xyzw().tolist(),
                cost_m2=total,observation_cost_m2=observation_cost,handle_cost_m2=handle_cost,
                current_depth_pose_cost=depth,axis_polarity=int(polarity),
                handle_support_points=handle['support_points'],handle_direction_world=y.tolist()))
    candidates.sort(key=lambda row:row['cost_m2']);best=candidates[0]
    gap=float(candidates[1]['cost_m2']-best['cost_m2']) if len(candidates)>1 else None
    ambiguous=(gap is None or gap<config.ambiguity_cost_gap_m2 or fit['rms_m']>config.cylinder_readiness_rms_m)
    return {**base,'pose_xyzw':best['pose_xyzw'],'status':'ambiguous_pose_proposal' if ambiguous else 'feature_pose_proposal',
        'ambiguous':bool(ambiguous),'candidate_count':len(candidates),'candidate_cost_gap_m2':gap,
        'current_point_count':len(points),'outside_handle_points':len(outside),'cylinder':diagnostic,
        'candidates':candidates,'readiness_is_not_accuracy_certification':True}
