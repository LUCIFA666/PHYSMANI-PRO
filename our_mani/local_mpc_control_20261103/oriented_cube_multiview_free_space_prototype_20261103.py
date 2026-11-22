"""All-four-view negative evidence for frozen nuisance-R Cube research.

No point, normal, frame, mask or positive face support is estimated here.
Every view retains its actual identity/renderer/time. Classic raw depth is
never treated as GL3 depth. All constraints are intersected in fixed order;
inconsistent evidence is reported, never repaired by selecting a view.
"""
from copy import deepcopy
import numpy as np

from .axis_general_free_space_prototype_20261103 import whole_pixel_face_footprints
from ..near_goal_cube_geometry_20261103_v1.free_space import subtract_open_forbidden, _open_union

VERSION = 'frozen_oriented_cube_all_four_current_views_research_20261103_v1'
VIEW_IDS = ('front', '8', '16', '36')
RENDERERS = ('GL3', 'CLASSIC', 'CLASSIC', 'CLASSIC')
HALF_SIDE = .025
MARGIN = .00025
MAXIMUM_RADIUS = .005
ROUNDS = 2
DV, DU = np.indices((3, 3))-1
DESIGN = np.c_[DU.ravel(), DV.ravel(), np.ones(9)]
LS = np.linalg.pinv(DESIGN)


def observed_depth_lower_bound(depth, renderer):
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind != 'f' or renderer not in ('GL3', 'CLASSIC'):
        raise ValueError('Actual floating camera-Z image and explicit renderer required')
    lower = np.full(depth.shape, np.nan)
    valid = np.zeros(depth.shape, bool)
    receipt = dict(renderer=renderer, pixels=int(depth.size), valid_raw_pixels=int((np.isfinite(depth)&(depth > 0)).sum()),
        reliable_lower_depth_pixels=0, maximum_metric_plane_residual_m=None,
        observed_depth_is_lower_bound_conditional_not_certified=True)
    if renderer == 'GL3':
        valid = np.isfinite(depth)&(depth > 0)
        lower[valid] = depth[valid]
        receipt.update(reliable_lower_depth_pixels=int(valid.sum()),
            method='original_GL3_raw_depth_centre_unchanged', planar_stencil_required=False)
        return dict(lower_depth=lower, valid=valid, receipt=receipt)
    receipt.update(method='Classic_current_3x3_corrected_four_corner_min_minus_actual_plane_residual',
        planar_stencil_required=True, maximum_admitted_metric_plane_residual_m=MARGIN,
        polygon_offset_inverse_depth_coefficient=.5, observed_pixel_corners_relative_to_centre=[[-.5,-.5],[.5,-.5],[-.5,.5],[.5,.5]],
        complete_valid_stencils=0, admitted_planar_stencils=0,
        boundary_or_invalid_or_nonplanar_pixels_no_negative_evidence=True)
    h, w = depth.shape
    if h < 3 or w < 3:
        return dict(lower_depth=lower, valid=valid, receipt=receipt)
    v, u = np.indices((h-2, w-2)); v=v.ravel()+1; u=u.ravel()+1
    z = depth[v[:, None]+DV.ravel(), u[:, None]+DU.ravel()].astype(np.float64)
    observed = np.isfinite(z).all(1)&(z > 0).all(1)
    ids = np.flatnonzero(observed)
    receipt['complete_valid_stencils'] = len(ids)
    if not len(ids):
        return dict(lower_depth=lower, valid=valid, receipt=receipt)
    z = z[ids]
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        coefficient = (1/z)@LS.T
        qfit = coefficient@DESIGN.T
        fitted_valid = np.isfinite(qfit).all(1)&(qfit > 0).all(1)&np.isfinite(coefficient).all(1)
        fitted = np.full_like(qfit, np.nan); fitted[fitted_valid] = 1/qfit[fitted_valid]
        residual = np.max(np.abs(z-fitted), axis=1)
        correction = .5*np.max(np.abs(coefficient[:, :2]), axis=1)
        centre_q = 1/z[:, 4]+correction
        corner_q = (centre_q[:, None]+coefficient[:, 0, None]*np.array([-.5,.5,-.5,.5])
            +coefficient[:, 1, None]*np.array([-.5,-.5,.5,.5]))
        corner_valid = np.isfinite(corner_q).all(1)&(corner_q > 0).all(1)
        candidate = 1/np.max(corner_q, axis=1)-residual
    admitted = (fitted_valid&corner_valid&np.isfinite(residual)&(residual <= MARGIN)
        &np.isfinite(candidate)&(candidate > 0))
    chosen = ids[admitted]
    valid[v[chosen], u[chosen]] = True
    lower[v[chosen], u[chosen]] = candidate[admitted]
    receipt.update(reliable_lower_depth_pixels=len(chosen), admitted_planar_stencils=len(chosen),
        maximum_metric_plane_residual_m=None if not len(chosen) else float(residual[admitted].max()))
    return dict(lower_depth=lower, valid=valid, receipt=receipt)


def _validate_views(packets):
    if not isinstance(packets, (list, tuple)) or len(packets) != 4:
        raise ValueError('Complete ordered four actual views, no selected subset')
    times=[]
    for packet, ident, renderer in zip(packets, VIEW_IDS, RENDERERS):
        if (not isinstance(packet, dict) or set(packet) != {'view_id','timestamp','renderer','depth','K','T'}
                or packet['view_id'] != ident or packet['renderer'] != renderer):
            raise ValueError('Actual ordered view id and renderer cannot be relabelled')
        now=packet['timestamp']
        if isinstance(now, (bool,np.bool_)) or not isinstance(now, (int,float,np.integer,np.floating)) or not np.isfinite(now) or now < 0:
            raise ValueError('Actual finite numeric frame timestamp required')
        times.append(float(now))
        depth,K,T = map(np.asarray,(packet['depth'],packet['K'],packet['T']))
        if (depth.ndim != 2 or depth.dtype.kind != 'f' or K.shape != (3,3) or T.shape != (4,4)
                or not np.isfinite(K).all() or not np.isfinite(T).all() or abs(np.linalg.det(K)) < 1e-12
                or not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-10)
                or not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-10)
                or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6)
                or not np.isclose(np.linalg.det(T[:3,:3]),1.,rtol=0,atol=1e-6)):
            raise ValueError('Current metric depth and finite proper signed K/T required')
    if any(now != times[0] for now in times):
        raise ValueError('All four views must share the actual recorded frame time')


def _length(components):
    return float(sum(hi-lo for lo,hi in components))


def _refine_mode(packets, observed, mode):
    support=deepcopy(mode['support']); rotation=np.asarray(mode['nuisance_rotation'],float)
    reference=np.asarray(mode['reference_point'],float)
    if (rotation.shape != (3,3) or reference.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(reference).all() or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-8)
            or not np.isclose(np.linalg.det(rotation),1.,rtol=0,atol=1e-8)):
        raise ValueError('Unchanged proper frozen nuisance frame required')
    counts=support.get('axis_counts',[])
    if len(counts) != 3 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Three original current face sample counts required')
    seen=np.asarray(counts)>=3
    bounds=np.asarray(support.get('intervals'),float)
    result=deepcopy(support)
    trace=dict(version=VERSION, view_order=list(VIEW_IDS), rounds=[], contradictions=[],
        all_four_views_considered=True, point_or_normal_frame_reestimated=False,
        no_new_positive_points=True, physical_or_statistical_certificate=False,
        known_axis_aligned_only_in_frozen_nuisance_frame=True,
        view_contributions=[dict(view_id=p['view_id'],renderer=p['renderer'],timestamp=float(p['timestamp']),
            K=np.asarray(p['K']).tolist(),camera_to_world=np.asarray(p['T']).tolist(),
            observed_depth_profile=deepcopy(observed[p['view_id']]['receipt']),
            anchor_admission=[],constraints=0,removed_interval_length_m=[0.,0.,0.])for p in packets])
    result.update(centre=None, diagnostic_centre_only=None, support_available=False, joint_radius=None,
        reasons=[], feasible_components=[[],[],[]], multiview_trace=trace)
    if bounds.shape != (3,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0] > bounds[:,1]):
        result['reasons']=['nonfinite_or_empty_original_oriented_support']; return result
    components=[[[float(lo),float(hi)]]for lo,hi in bounds]
    trace['initial_components']=deepcopy(components)
    anchors={}
    for view_index, packet in enumerate(packets):
        ident=packet['view_id']; T=np.asarray(packet['T']); local_T=T.copy()
        local_T[:3,:3]=rotation.T@T[:3,:3]; local_T[:3,3]=rotation.T@(T[:3,3]-reference)
        anchors[ident]={}
        for b in range(3):
            if not seen[b]:
                continue
            origin=float(local_T[b,3])
            side=1 if origin > bounds[b,1]+HALF_SIDE else -1 if origin < bounds[b,0]-HALF_SIDE else 0
            trace['view_contributions'][view_index]['anchor_admission'].append(dict(axis=b,side=side,
                original_face_samples=counts[b],usable=bool(side)))
            if not side:
                continue
            data=whole_pixel_face_footprints(packet['depth'],packet['K'],local_T,b,bounds[b]+side*HALF_SIDE,side)
            lower=observed[ident]['lower_depth'].ravel(); valid=observed[ident]['valid'].ravel()
            data['observed_lower_depth']=lower
            data['free']=data['valid_geometry']&valid&(lower > data['max_expected_depth']+MARGIN)
            anchors[ident][b]=data
    # All four observations enter each round, including evidence that removes
    # an entire interval. No success return or favourable-view choice exists.
    for round_index in range(ROUNDS):
        for a in range(3):
            if seen[a]:
                continue
            for b in range(3):
                if not seen[b] or a == b:
                    continue
                k=3-a-b
                for view_index, packet in enumerate(packets):
                    ident=packet['view_id']; before=deepcopy(components[a])
                    step=dict(round=round_index,missing_axis=a,known_face_axis=b,other_axis=k,
                        view_id=ident,renderer=packet['renderer'],timestamp=float(packet['timestamp']),
                        before=before,constraints=[])
                    data=anchors[ident].get(b)
                    if data is None:
                        step.update(skipped='camera_inside_facing_face_slab_no_proof',after=deepcopy(components[a]))
                        trace['rounds'].append(step); continue
                    if not components[a] or not components[k]:
                        step.update(skipped='empty_current_component_no_reset',after=deepcopy(components[a]))
                        trace['rounds'].append(step); continue
                    lo,hi=components[k][0][0],components[k][-1][1]
                    common=[hi-HALF_SIDE,lo+HALF_SIDE]
                    eligible=data['free']&(data['footprint_min'][:,k]>=common[0])&(data['footprint_max'][:,k]<=common[1])
                    low=data['footprint_max'][:,a]-HALF_SIDE; high=data['footprint_min'][:,a]+HALF_SIDE
                    valid=np.isfinite(low)&np.isfinite(high)&(low < high)
                    intersects=np.zeros(len(low),bool)
                    for l,h in components[a]:
                        intersects|=(h > low)&(l < high)
                    used=np.flatnonzero(eligible&valid&intersects); forbidden=[]
                    for i in used:
                        interval=[float(low[i]),float(high[i])]; forbidden.append(interval)
                        step['constraints'].append(dict(pixel_uv=data['pixel_uv'][i].tolist(),forbidden_open=interval,
                            raw_observed_depth=float(data['observed_depth'][i]),
                            observed_depth_lower_bound=float(data['observed_lower_depth'][i]),
                            maximum_expected_camera_depth=float(data['max_expected_depth'][i]),
                            footprint_min=data['footprint_min'][i].tolist(),footprint_max=data['footprint_max'][i].tolist()))
                    components[a]=subtract_open_forbidden(components[a],forbidden)
                    step.update(mandatory_other_axis_span=common,free_pixel_count=int(data['free'].sum()),
                        whole_pixel_other_axis_pass_count=int(eligible.sum()),merged_forbidden_open=_open_union(forbidden),
                        after=deepcopy(components[a]))
                    contribution=trace['view_contributions'][view_index]
                    contribution['constraints']+=len(used)
                    contribution['removed_interval_length_m'][a]+=_length(before)-_length(components[a])
                    if before and not components[a]:
                        trace['contradictions'].append(dict(round=round_index,axis=a,view_id=ident,
                            renderer=packet['renderer'],reason='this_actual_view_excluded_remaining_interval'))
                    trace['rounds'].append(step)
    result['feasible_components']=deepcopy(components); trace['final_components']=deepcopy(components)
    result['intervals']=[[c[0][0],c[-1][1]]if c else[None,None]for c in components]
    if any(not c for c in components):
        result['reasons'].append('all_current_components_excluded')
    if any(len(c)>1 for c in components):
        result['reasons'].append('multiple_current_components_no_selection')
    if any(len(c)!=1 for c in components):
        result['radius_each']=[None]*3; return result
    centre=[lo+.5*(hi-lo)for ((lo,hi),)in components]
    radii=[MARGIN if seen[a] else .5*(c[0][1]-c[0][0])for a,c in enumerate(components)]
    joint=float(np.linalg.norm(radii))
    result.update(diagnostic_centre_only=centre,radius_each=radii,joint_radius=joint)
    if not np.isfinite(centre+radii+[joint]).all():
        result['reasons'].append('nonfinite_derived_geometry'); return result
    if joint > MAXIMUM_RADIUS:
        result['reasons'].append('joint_radius_exceeds_unchanged5mm'); return result
    result.update(centre=centre,support_available=True)
    return result


def refine_multiview_oriented_support(view_packets, frozen_oriented_estimate):
    _validate_views(view_packets)
    if not isinstance(frozen_oriented_estimate,dict) or not isinstance(frozen_oriented_estimate.get('modes'),list):
        raise ValueError('Explicit frozen nuisance-frame estimate required')
    result=deepcopy(frozen_oriented_estimate)
    result.update(version=VERSION,status='UNKNOWN',centre=None,joint_radius=None,
        conditional_geometry_only=True,qualified=False,position_measurement_qualified=False,
        model_qualified=False,raw_mask_model_qualified=False,force_or_contact_qualified=False,
        full_orientation_feasible_set_proven=False,
        current_planar_point_count_unchanged=True,all_actual_view_ids=list(VIEW_IDS))
    observed={packet['view_id']:observed_depth_lower_bound(packet['depth'],packet['renderer'])for packet in view_packets}
    for mode in result['modes']:
        refined=_refine_mode(view_packets,observed,mode)
        mode['ray_support']=refined
        centre=refined['centre']
        mode['conditional_centre_world']=None if centre is None else (
            np.asarray(mode['reference_point'])+np.asarray(mode['nuisance_rotation'])@centre).tolist()
    if len(result['modes']) != 1:
        result['reasons']=['no_unique_current_nuisance_frame']; return result
    mode=result['modes'][0]; support=mode['ray_support']
    if support['support_available'] is not True:
        result['reasons']=list(support['reasons']); return result
    result.update(status='AVAILABLE',centre=mode['conditional_centre_world'],
        joint_radius=support['joint_radius'],reasons=[])
    return result
