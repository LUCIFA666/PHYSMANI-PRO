"""F current-depth negative-evidence research; not a physical certificate.

The caller must first retain the original point/view/confidence/crop gates
and finally apply scope. Only current original GL3 front depth and camera K/T
enter the added constraints. No GT, robot, history or previous centre input.
"""
from copy import deepcopy
import hashlib
import numpy as np

PROTOCOL_SHA256='bfddd160742313810e974968bccadeab90e05397dfeb7f3141115a10f091293f'
HALF_SIDE_M=.025
PADDING_M=.00025
FREE_DEPTH_MARGIN_M=.00025
MAXIMUM_JOINT_RADIUS_M=.005
ROUNDS=2


def _closed_union(intervals):
    result=[]
    for lo,hi in sorted((float(p[0]),float(p[1])) for p in intervals):
        if not np.isfinite([lo,hi]).all() or lo>hi:raise ValueError('Finite nonempty closed intervals required')
        if result and lo<=result[-1][1]:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return result


def _open_union(intervals):
    result=[]
    for lo,hi in sorted((float(p[0]),float(p[1])) for p in intervals):
        if not np.isfinite([lo,hi]).all():raise ValueError('Finite open forbidden interval required')
        if lo>=hi:continue
        # Touching OPEN intervals cannot merge: their common endpoint survives.
        if result and lo<result[-1][1]:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return result


def subtract_open_forbidden(components,forbidden):
    """Subtract only strict interiors, preserving isolated boundary points."""
    current=_closed_union(components)
    for a,b in _open_union(forbidden):
        after=[]
        for lo,hi in current:
            if hi<=a or lo>=b:after.append([lo,hi]);continue
            if lo<=a<=hi:after.append([lo,a])
            if lo<=b<=hi:after.append([b,hi])
        current=_closed_union(after)
    return current


def whole_pixel_top_footprints(depth,K,T,centre_z):
    """Four actual pixel corners times both extreme current top-plane heights.

    The world ray is intentionally NOT unit-normalized. Its parameter is the
    camera-Z depth because the signed pinhole ray has camera z=1.
    """
    depth,K,T=map(np.asarray,(depth,K,T))
    if depth.ndim!=2 or depth.dtype.kind!='f':raise ValueError('Original floating metric frontdepth required')
    if (K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all()
        or abs(np.linalg.det(K))<1e-12 or not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6)
        or not np.isclose(np.linalg.det(T[:3,:3]),1.,rtol=0,atol=1e-6) or not np.isfinite(centre_z)):
        raise ValueError('Finite public signedK/camera-to-world and current faceZ required')
    h,w=depth.shape;vv,uu=np.indices((h,w));u=uu.ravel();v=vv.ravel();n=len(u)
    top=np.asarray([centre_z+HALF_SIDE_M-PADDING_M,centre_z+HALF_SIDE_M+PADDING_M])
    out=dict(pixel_uv=np.c_[u,v],observed_depth=depth.ravel().astype(float),
        valid_observed_depth=np.isfinite(depth.ravel())&(depth.ravel()>0),
        valid_geometry=np.zeros(n,bool),free=np.zeros(n,bool),max_expected_depth=np.full(n,np.nan),
        footprint_min=np.full((n,3),np.nan),footprint_max=np.full((n,3),np.nan),
        top_planes=top,camera_above_planes=bool(T[2,3]>top.max()))
    if not out['camera_above_planes']:return out
    x=u[:,None]+np.array([0.,1.,0.,1.]);y=v[:,None]+np.array([0.,0.,1.,1.])
    pixels=np.stack((x,y,np.ones_like(x)),axis=-1)
    rays=(pixels@np.linalg.inv(K).T)@T[:3,:3].T
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        expected=(top[None,None,:]-T[2,3])/rays[:,:,2,None]
        corners=T[:3,3]+rays[:,:,None,:]*expected[:,:,:,None]
    valid=(np.isfinite(expected).all((1,2))&(expected>0).all((1,2))
           &np.isfinite(corners).all((1,2,3)))
    out['valid_geometry']=valid
    out['max_expected_depth'][valid]=expected[valid].max((1,2))
    out['footprint_min'][valid]=corners[valid].min((1,2))
    out['footprint_max'][valid]=corners[valid].max((1,2))
    out['free']=valid&out['valid_observed_depth']&(out['observed_depth']>out['max_expected_depth']+FREE_DEPTH_MARGIN_M)
    return out


def refine_with_front_free_space(depth_front,Kfront,Tfront,E_support):
    """Preserve E success; otherwise narrow missing horizontal coordinates.

    Original point/view/confidence/crop eligibility is a required caller
    precondition. The output never authorizes overriding those gates or scope.
    """
    if not isinstance(E_support,dict):raise ValueError('Frozen current E geometry dictionary required')
    out=deepcopy(E_support)
    trace=dict(E_passthrough=False,free_space_applied=False,rounds=[],
        original_measurement_guards_required=True,final_scope_guard_required=True,
        front_profile='original_GL3_rawdepth_no_polygon_offset_undo',
        coordinate_padding_m=PADDING_M,free_depth_margin_m=FREE_DEPTH_MARGIN_M,
        physical_or_statistical_certificate=False,GT_robot_history_or_past_centre_used=False)
    out['free_space_trace']=trace;out['feasible_components']=[[],[],[]]
    if E_support.get('support_available') is True:
        out['feasible_components']=[[[float(p[0]),float(p[1])]] for p in E_support['intervals']]
        trace['E_passthrough']=True
        return out
    out.update(support_available=False,centre=None,diagnostic_centre_only=None,reasons=[],joint_radius=None)
    counts=E_support.get('axis_counts',[])
    if len(counts)!=3 or any(type(n) is not int or n<0 for n in counts):raise ValueError('Three actual D face counts required')
    observed=np.asarray(counts)>=3
    if not observed[2]:out['reasons'].append('F_requires_current_visible_Z_face_at_least_three');return out
    try:
        intervals=np.asarray(E_support['intervals'],float)
        if intervals.shape!=(3,2) or not np.isfinite(intervals).all() or np.any(intervals[:,0]>intervals[:,1]):
            raise ValueError('invalid interval')
        components=[[[float(p[0]),float(p[1])]] for p in intervals]
        cz=float(E_support['diagnostic_centre_only'][2])
        if not np.isfinite(cz):raise ValueError('missing faceZ')
    except (ValueError,TypeError,KeyError,IndexError):
        out['reasons'].append('F_requires_finite_nonempty_initial_E_intervals_and_faceZ');return out
    data=whole_pixel_top_footprints(depth_front,Kfront,Tfront,cz)
    trace.update(free_space_applied=True,top_planes=data['top_planes'].tolist(),
        camera_above_top_planes=data['camera_above_planes'],
        pixel_counts=dict(total=len(data['free']),finite_positive=int(data['valid_observed_depth'].sum()),
            valid_eight_intersections=int(data['valid_geometry'].sum()),free=int(data['free'].sum()),
            valid_but_not_free=int((data['valid_geometry']&data['valid_observed_depth']&~data['free']).sum())),
        free_mask_sha256=hashlib.sha256(data['free'].tobytes()).hexdigest(),
        initial_components=deepcopy(components))
    for round_index in range(ROUNDS):
        for j in (0,1):
            step=dict(round=round_index,axis=j,other_axis=1-j,before=deepcopy(components[j]),constraints=[])
            if observed[j]:step['skipped']='visible_face_axis_never_shrunk';step['after']=deepcopy(components[j]);trace['rounds'].append(step);continue
            k=1-j
            if not components[j] or not components[k]:
                step['skipped']='empty_current_candidate_axis';step['after']=deepcopy(components[j]);trace['rounds'].append(step);continue
            lo_k=components[k][0][0];hi_k=components[k][-1][1]
            common=[hi_k-HALF_SIDE_M,lo_k+HALF_SIDE_M]
            step.update(other_axis_hull=[lo_k,hi_k],mandatory_other_axis_span=common)
            eligible=(data['free']&(data['footprint_min'][:,k]>=common[0])&(data['footprint_max'][:,k]<=common[1]))
            a=data['footprint_max'][:,j]-HALF_SIDE_M;b=data['footprint_min'][:,j]+HALF_SIDE_M
            nonempty=np.isfinite(a)&np.isfinite(b)&(a<b)
            touches=np.zeros(len(a),bool)
            for lo,hi in components[j]:touches|=(hi>a)&(lo<b)
            used=np.flatnonzero(eligible&nonempty&touches)
            step.update(whole_pixel_other_axis_pass_count=int(eligible.sum()),
                nonempty_forbidden_count=int((eligible&nonempty).sum()),
                intersecting_constraint_count=len(used))
            forbidden=[]
            for index in used:
                interval=[float(a[index]),float(b[index])];forbidden.append(interval)
                step['constraints'].append(dict(pixel_uv=data['pixel_uv'][index].tolist(),
                    observed_depth=float(data['observed_depth'][index]),max_expected_depth=float(data['max_expected_depth'][index]),
                    footprint_min=data['footprint_min'][index].tolist(),footprint_max=data['footprint_max'][index].tolist(),
                    forbidden_open=interval))
            step['merged_forbidden_open']=_open_union(forbidden)
            components[j]=subtract_open_forbidden(components[j],forbidden)
            step['after']=deepcopy(components[j]);trace['rounds'].append(step)
    out['feasible_components']=deepcopy(components)
    trace['final_components']=deepcopy(components)
    out['intervals']=[[p[0][0],p[-1][1]] if p else [None,None] for p in components]
    if any(len(p)==0 for p in components):out['reasons'].append('F_free_space_removed_all_current_components')
    if any(len(p)>1 for p in components):out['reasons'].append('F_multiple_feasible_components_no_component_choice')
    if any(len(p)!=1 for p in components):out['radius_each']=[None]*3;return out
    radii=[];centre=[]
    for j,p in enumerate(components):
        lo,hi=p[0]
        centre.append(float(E_support['diagnostic_centre_only'][j]) if observed[j] else lo+.5*(hi-lo))
        radii.append(PADDING_M if observed[j] else .5*(hi-lo))
    joint=float(np.linalg.norm(radii))
    out.update(radius_each=radii,joint_radius=joint,diagnostic_centre_only=centre)
    if not np.isfinite(centre+radii+[joint]).all():
        out['reasons'].append('F_nonfinite_final_geometry');return out
    if joint>MAXIMUM_JOINT_RADIUS_M:
        out['reasons'].append('F_joint_radius_exceeds_fixed_5mm');return out
    out.update(support_available=True,centre=centre)
    out['coordinate_sources']=[E_support['coordinate_sources'][j] if observed[j]
        else 'bounded_current_support_plus_current_front_free_space' for j in range(3)]
    return out
