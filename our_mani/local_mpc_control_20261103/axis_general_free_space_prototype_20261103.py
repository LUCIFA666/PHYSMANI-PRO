"""Unqualified axis-general current-depth exclusion, development prototype.

Conditional on a strictly world-axis-aligned opaque50mm Cube, valid current
centre intervals and the unchanged front-GL3 depth profile. The five-degree D
normal acceptance is NOT a proof of those assumptions. No rotation, GT,
robot, history or predicted/controlled position is accepted as input.
"""
from copy import deepcopy
import numpy as np
from ..near_goal_cube_geometry_20261103_v1.free_space import subtract_open_forbidden, _open_union

HALF_SIDE=.025
MARGIN=.00025
MAX_RADIUS=.005
ROUNDS=2
VERSION='axis_general_whole_pixel_free_space_development_20261103_v1'


def whole_pixel_face_footprints(depth,K,T,axis,planes,side):
    depth,K,T=np.asarray(depth),np.asarray(K),np.asarray(T)
    planes=np.asarray(planes,float)
    if (type(axis)is not int or axis not in (0,1,2) or type(side)is not int or side not in (-1,1)
        or depth.ndim!=2 or depth.dtype.kind!='f' or planes.shape!=(2,) or not np.isfinite(planes).all() or planes[0]>planes[1]
        or K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all()
        or abs(np.linalg.det(K))<1e-12 or not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-10)
        or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6)
        or not np.isclose(np.linalg.det(T[:3,:3]),1.,rtol=0,atol=1e-6)):
        raise ValueError('Explicit face axis/side, current depth, finite planes and proper signedK/T required')
    h,w=depth.shape;v,u=np.indices((h,w));u=u.ravel();v=v.ravel();n=len(u)
    out=dict(pixel_uv=np.c_[u,v],free=np.zeros(n,bool),valid_geometry=np.zeros(n,bool),
        valid_depth=np.isfinite(depth.ravel())&(depth.ravel()>0),observed_depth=depth.ravel().astype(float),
        footprint_min=np.full((n,3),np.nan),footprint_max=np.full((n,3),np.nan),max_expected_depth=np.full(n,np.nan),
        axis=axis,planes=planes.tolist(),side=side)
    facing=(T[axis,3]>planes[1] if side==1 else T[axis,3]<planes[0])
    if not facing:return out
    x=u[:,None]+[0.,1.,0.,1.];y=v[:,None]+[0.,0.,1.,1.]
    rays=(np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(K).T)@T[:3,:3].T
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        times=(planes[None,None,:]-T[axis,3])/rays[:,:,axis,None]
        points=T[:3,3]+rays[:,:,None,:]*times[:,:,:,None]
    good=np.isfinite(times).all((1,2))&(times>0).all((1,2))&np.isfinite(points).all((1,2,3))
    out['valid_geometry']=good
    out['max_expected_depth'][good]=times[good].max((1,2))
    out['footprint_min'][good]=points[good].min((1,2))
    out['footprint_max'][good]=points[good].max((1,2))
    out['free']=good&out['valid_depth']&(out['observed_depth']>out['max_expected_depth']+MARGIN)
    return out


def refine_axis_general(depth,K,T,support):
    if not isinstance(support,dict):raise ValueError('Current E support dictionary required')
    out=deepcopy(support)
    trace=dict(version=VERSION,strict_axis_aligned_condition=True,near_axis_normal_gate_does_not_certify_this=True,
        physical_or_statistical_certificate=False,source='current_front_GL3_rawdepth_only',
        additional_GT_robot_history_prediction_or_control_input=False,E_passthrough=False,rounds=[])
    out['axis_general_trace']=trace
    if support.get('support_available')is True:
        trace['E_passthrough']=True
        return out
    out.update(support_available=False,centre=None,diagnostic_centre_only=None,joint_radius=None,reasons=[])
    counts=support.get('axis_counts',[])
    if len(counts)!=3 or any(type(x)is not int or x<0 for x in counts):raise ValueError('Three real current D face counts')
    observed=np.asarray(counts)>=3
    intervals=np.asarray(support.get('intervals'),float)
    if intervals.shape!=(3,2) or not np.isfinite(intervals).all() or np.any(intervals[:,0]>intervals[:,1]):
        out['reasons']=['nonfinite_or_empty_original_support'];return out
    components=[[[float(lo),float(hi)]]for lo,hi in intervals]
    trace['initial_components']=deepcopy(components)
    anchors={}
    trace['anchor_admission']=[]
    for b in range(3):
        if not observed[b]:continue
        origin=float(np.asarray(T)[b,3])
        side=1 if origin>intervals[b,1]+HALF_SIDE else -1 if origin<intervals[b,0]-HALF_SIDE else 0
        trace['anchor_admission'].append(dict(axis=b,actual_D_face_samples=counts[b],side=side,
            usable=bool(side),reason='camera_outside_entire_current_cube_axis_slab'if side else'camera_inside_axis_slab_no_facing_face_proof'))
        if side:
            anchors[b]=whole_pixel_face_footprints(depth,K,T,b,intervals[b]+side*HALF_SIDE,side)
    trace['usable_anchor_axes']=list(anchors)
    for round_i in range(ROUNDS):
        for a in range(3):
            if observed[a]:continue
            for b,data in anchors.items():
                if a==b:continue
                k=3-a-b
                step=dict(round=round_i,missing_axis=a,known_face_axis=b,other_axis=k,before=deepcopy(components[a]),constraints=[])
                if not components[a] or not components[k]:
                    step['skipped']='empty_current_component';trace['rounds'].append(step);continue
                lo,hi=components[k][0][0],components[k][-1][1]
                common=[hi-HALF_SIDE,lo+HALF_SIDE]
                eligible=data['free']&(data['footprint_min'][:,k]>=common[0])&(data['footprint_max'][:,k]<=common[1])
                low=data['footprint_max'][:,a]-HALF_SIDE;high=data['footprint_min'][:,a]+HALF_SIDE
                valid=np.isfinite(low)&np.isfinite(high)&(low<high)
                touches=np.zeros(len(low),bool)
                for l,h in components[a]:touches|=(h>low)&(l<high)
                used=np.flatnonzero(eligible&valid&touches);forbidden=[]
                for i in used:
                    interval=[float(low[i]),float(high[i])];forbidden.append(interval)
                    step['constraints'].append(dict(pixel_uv=data['pixel_uv'][i].tolist(),forbidden_open=interval,
                        max_expected_depth=float(data['max_expected_depth'][i]),observed_depth=float(data['observed_depth'][i]),
                        footprint_min=data['footprint_min'][i].tolist(),footprint_max=data['footprint_max'][i].tolist()))
                components[a]=subtract_open_forbidden(components[a],forbidden)
                step.update(mandatory_other_axis_span=common,free_pixel_count=int(data['free'].sum()),
                    whole_pixel_other_axis_pass_count=int(eligible.sum()),merged_forbidden_open=_open_union(forbidden),after=deepcopy(components[a]))
                trace['rounds'].append(step)
    out['feasible_components']=deepcopy(components);trace['final_components']=deepcopy(components)
    out['intervals']=[[c[0][0],c[-1][1]]if c else[None,None]for c in components]
    if any(not c for c in components):out['reasons'].append('all_current_components_excluded')
    if any(len(c)>1 for c in components):out['reasons'].append('multiple_current_components_no_selection')
    if any(len(c)!=1 for c in components):out['radius_each']=[None]*3;return out
    centre=[lo+.5*(hi-lo)for ((lo,hi),)in components]
    radii=[MARGIN if observed[a]else .5*(components[a][0][1]-components[a][0][0])for a in range(3)]
    joint=float(np.linalg.norm(radii))
    out.update(diagnostic_centre_only=centre,radius_each=radii,joint_radius=joint)
    if not np.isfinite(centre+radii+[joint]).all():out['reasons'].append('nonfinite_derived_geometry');return out
    if joint>MAX_RADIUS:out['reasons'].append('joint_radius_exceeds_unchanged5mm');return out
    out.update(support_available=True,centre=centre)
    out['coordinate_sources']=[support['coordinate_sources'][a]if observed[a]else 'bounded_current_support_plus_axis_general_front_free_space'for a in range(3)]
    return out
