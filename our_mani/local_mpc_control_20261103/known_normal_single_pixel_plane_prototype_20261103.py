"""Conditional single-pixel Cube-plane research, never an admitted sensor.

An already frozen current-plane nuisance frame supplies candidate face
normals. All six face assignments remain explicit. A high depth-influence
probability does not establish a pure Cube surface sample, face identity,
subpixel ownership or a calibrated bound. No NN, pose/robot/GT, previous
world-axis bounds or live qualification is consumed or produced here.
"""
from copy import deepcopy
import numpy as np

VERSION='conditional_known_normal_single_pixel_cube_planes_20261103_v1'
HALF_SIDE=.025
POINT_ALLOWANCE=.00025
MAXIMUM_RADIUS=.005
MAXIMUM_COMPONENT_BOXES=4096
VIEW_IDS=('front','8','16','36')
RENDERERS=('GL3','CLASSIC','CLASSIC','CLASSIC')


def _camera(K,T):
    K,T=np.asarray(K,float),np.asarray(T,float)
    if (K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all()
        or abs(np.linalg.det(K))<1e-12 or not np.allclose(K[2],[0,0,1],atol=1e-10,rtol=0)
        or not np.allclose(T[3],[0,0,0,1],atol=1e-10,rtol=0)
        or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6,rtol=0)
        or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-6,rtol=0)):
        raise ValueError('Actual proper signedK/camera-to-world required')
    return K,T


def plane_from_known_normal(depth_m,pixel_uv,K,T,normal_world,renderer):
    K,T=_camera(K,T);uv=np.asarray(pixel_uv);n=np.asarray(normal_world,float)
    if (renderer not in ('GL3','CLASSIC') or uv.shape!=(2,) or uv.dtype.kind not in 'iu'
        or np.any(uv<0) or n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<=0
        or isinstance(depth_m,(bool,np.bool_)) or not np.isscalar(depth_m)):
        raise ValueError('Explicit renderer, integer pixel, finite candidate normal and scalar depth required')
    result=dict(valid=False,renderer=renderer,pixel_uv=uv.tolist(),raw_depth_m=float(depth_m),
        corrected_depth_m=None,point_world=None,camera_plane_offset=None,world_plane_offset=None,
        normal_world=None,formula_terms=None,reasons=[],conditional_pure_face_sample_assumed=True,
        pure_surface_or_pixel_ownership_verified=False,qualified=False)
    if not np.isfinite(depth_m) or depth_m<=0:
        result['raw_depth_m']=None;result['reasons']=['invalid_current_depth'];return result
    n=n/np.linalg.norm(n);ncam=T[:3,:3].T@n
    B=ncam@np.linalg.inv(K);pixel=np.r_[uv.astype(float)+.5,1.]
    b=float(B@pixel);s=float(np.max(np.abs(B[:2])))
    result['normal_world']=n.tolist()
    result['formula_terms']=dict(B=B.tolist(),b=b,slope_absolute_max=s,
        normal_offset_sign=None if b==0 else float(np.sign(b)),
        polygon_offset_factor=0. if renderer=='GL3'else .5)
    if not np.isfinite([b,s]).all() or b==0:
        result['reasons']=['grazing_or_nonfinite_candidate_plane'];return result
    factor=1. if renderer=='GL3'else 1.-.5*s/abs(b)
    corrected=float(depth_m)*factor
    result['formula_terms']['depth_correction_factor']=factor
    if not np.isfinite(corrected) or corrected<=0:
        result['reasons']=['nonpositive_or_nonfinite_corrected_depth'];return result
    ray=np.linalg.inv(K)@pixel;point_camera=ray*corrected
    point=T[:3,:3]@point_camera+T[:3,3]
    D=float(ncam@point_camera);world_offset=float(n@point)
    if not np.isfinite(point).all() or not np.isfinite([D,world_offset]).all() or D==0:
        result['reasons']=['nonfinite_candidate_point_or_plane_offset'];return result
    result.update(valid=True,corrected_depth_m=corrected,point_world=point.tolist(),
        camera_plane_offset=D,world_plane_offset=world_offset)
    return result


def pixel_cube_face_assignments(depth_m,pixel_uv,K,T,nuisance_rotation,reference_point,renderer):
    R=np.asarray(nuisance_rotation,float);reference=np.asarray(reference_point,float)
    K,T=_camera(K,T)
    if (R.shape!=(3,3) or reference.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(reference).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(R),1.,atol=1e-8,rtol=0)):
        raise ValueError('Unchanged proper frozen nuisanceR/reference required')
    ledger=[]
    for axis in range(3):
        for sign in (-1,1):
            normal=R[:,axis]*sign
            plane=plane_from_known_normal(depth_m,pixel_uv,K,T,normal,renderer)
            row=dict(axis=axis,face_sign=sign,plane=plane,camera_facing=False,centre_box_local=None,
                reason='invalid_candidate_plane')
            if plane['valid']:
                point=np.asarray(plane['point_world']);local=R.T@(point-reference)
                facing=float(normal@(T[:3,3]-point))
                box=np.c_[local-HALF_SIDE-POINT_ALLOWANCE,local+HALF_SIDE+POINT_ALLOWANCE]
                face=local[axis]-sign*HALF_SIDE
                box[axis]=[face-POINT_ALLOWANCE,face+POINT_ALLOWANCE]
                row.update(camera_facing=bool(facing>0),facing_dot=facing,
                    centre_box_local=box.tolist(),point_local=local.tolist(),
                    reason='conditional_visible_face_assignment'if facing>0 else'candidate_face_not_visible_from_this_camera')
            ledger.append(row)
    return ledger


def _intersection(a,b):
    low=np.maximum(a[:,0],b[:,0]);high=np.minimum(a[:,1],b[:,1])
    return None if np.any(low>high)else np.c_[low,high]


def _union_boxes(boxes):
    """Remove exact duplicates/contained boxes, never trim an adverse point."""
    result=[]
    for b in boxes:
        if any(np.all(old[:,0]<=b[:,0])and np.all(old[:,1]>=b[:,1])for old in result):continue
        result=[old for old in result if not(np.all(b[:,0]<=old[:,0])and np.all(b[:,1]>=old[:,1]))]
        result.append(b)
    return sorted(result,key=lambda b:tuple(b.ravel()))


def _connected_sets(boxes):
    seen=set();groups=[]
    for i in range(len(boxes)):
        if i in seen:continue
        pending=[i];seen.add(i);group=[]
        while pending:
            j=pending.pop();group.append(j)
            for k in range(len(boxes)):
                if k not in seen and _intersection(boxes[j],boxes[k])is not None:
                    seen.add(k);pending.append(k)
        groups.append(sorted(group))
    return groups


def refine_current_single_pixels(packets,mode):
    """Intersect all face disjunctions with the fixed current-plane support.

    Every original candidate receives all six entries before intersection.
    Components are closed boxes; there is no favourable-view/face selection.
    The finite-computation guard can only make the result UNKNOWN.
    """
    if not isinstance(packets,(list,tuple))or len(packets)!=4:
        raise ValueError('All four actual ordered views required')
    R=np.asarray(mode['nuisance_rotation'],float);reference=np.asarray(mode['reference_point'],float)
    if (R.shape!=(3,3)or reference.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(reference).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0)or not np.isclose(np.linalg.det(R),1.,atol=1e-8,rtol=0)):
        raise ValueError('Unchanged proper frozen nuisanceR/reference required')
    support=deepcopy(mode['support']);bounds=np.asarray(support['intervals'],float)
    result=dict(version=VERSION,status='UNKNOWN',centre=None,diagnostic_centre_world=None,
        nuisance_rotation=R.tolist(),reference_point=reference.tolist(),initial_support=support,
        feasible_boxes_local=[],connected_box_groups=[],hull_intervals=None,radius_each=None,joint_radius=None,
        pixels=[],intersection_trace=[],reasons=[],all_six_face_assignments_retained=True,
        all_current_pixels_considered=True,orientation_reestimated=False,conditional_geometry_only=True,
        pure_pixel_ownership_verified=False,position_measurement_qualified=False,qualified=False,
        model_qualified=False,raw_mask_model_qualified=False,force_or_contact_qualified=False,
        maximum_radius_m=MAXIMUM_RADIUS,point_allowance_m=POINT_ALLOWANCE,
        maximum_component_boxes=MAXIMUM_COMPONENT_BOXES)
    times=[]
    for packet,ident,renderer in zip(packets,VIEW_IDS,RENDERERS):
        if (set(packet)!={'view_id','renderer','timestamp','depth','pixels_uv','K','T'}or
            packet['view_id']!=ident or packet['renderer']!=renderer):raise ValueError('Real viewID/renderer required')
        t=packet['timestamp'];depth=np.asarray(packet['depth']);uv=np.asarray(packet['pixels_uv'])
        if (isinstance(t,(bool,np.bool_))or not np.isfinite(t)or t<0 or depth.ndim!=2 or depth.dtype.kind!='f'
            or uv.ndim!=2 or uv.shape[1:]!=(2,)or uv.dtype.kind not in 'iu'
            or np.any(uv<0)or np.any(uv[:,0]>=depth.shape[1])or np.any(uv[:,1]>=depth.shape[0])
            or len(set(map(tuple,uv.tolist())))!=len(uv)):
            raise ValueError('Current unique realUV/depth/time required')
        times.append(float(t));_camera(packet['K'],packet['T'])
        for pixel in uv:
            ledger=pixel_cube_face_assignments(float(depth[pixel[1],pixel[0]]),pixel,
                packet['K'],packet['T'],R,reference,renderer)
            result['pixels'].append(dict(view_id=ident,renderer=renderer,timestamp=float(t),
                pixel_uv=pixel.tolist(),raw_depth_m=float(depth[pixel[1],pixel[0]])if np.isfinite(depth[pixel[1],pixel[0]])else None,assignments=ledger))
    if any(t!=times[0]for t in times):raise ValueError('Actual four-view frame timestamp must match')
    if bounds.shape!=(3,2)or not np.isfinite(bounds).all()or np.any(bounds[:,0]>bounds[:,1]):
        result['reasons']=['empty_or_nonfinite_original_nuisance_plane_support'];return result
    if not result['pixels']:
        result['reasons']=['no_original_current_candidate_pixels'];return result
    boxes=[bounds.copy()];overflow=False
    for index,pixel in enumerate(result['pixels']):
        assignments=[np.asarray(a['centre_box_local'],float)for a in pixel['assignments']if a['camera_facing']]
        if not assignments:result['reasons'].append('pixel_without_any_valid_visible_face_assignment_'+str(index))
        new=[]
        for old in boxes:
            for a in assignments:
                b=_intersection(old,a)
                if b is not None:new.append(b)
        boxes=_union_boxes(new)
        result['intersection_trace'].append(dict(pixel_index=index,view_id=pixel['view_id'],pixel_uv=pixel['pixel_uv'],
            visible_face_hypotheses=len(assignments),remaining_boxes=len(boxes)))
        if len(boxes)>MAXIMUM_COMPONENT_BOXES:
            overflow=True;result['reasons'].append('component_budget_exceeded_no_favourable_pruning')
            result['all_current_pixels_considered']=False;break
    result['feasible_boxes_local']=[b.tolist()for b in boxes]
    if overflow:return result
    if not boxes:
        result['reasons'].append('all_current_pixel_face_disjunctions_inconsistent');return result
    groups=_connected_sets(boxes);result['connected_box_groups']=groups
    low=np.stack([b[:,0]for b in boxes]).min(0);high=np.stack([b[:,1]for b in boxes]).max(0)
    radii=(high-low)/2;mid=low+radii;radius=float(np.linalg.norm(radii))
    result.update(hull_intervals=np.c_[low,high].tolist(),radius_each=radii.tolist(),joint_radius=radius,
        diagnostic_centre_world=(reference+R@mid).tolist())
    if len(groups)!=1:result['reasons'].append('multiple_disconnected_current_face_assignment_solutions')
    if not any(np.all(mid>=b[:,0])and np.all(mid<=b[:,1])for b in boxes):
        result['reasons'].append('hull_midpoint_is_not_a_feasible_assignment_solution')
    if not np.isfinite(radius)or radius>MAXIMUM_RADIUS:result['reasons'].append('joint_radius_exceeds_unchanged5mm')
    if not result['reasons']:result.update(status='CONDITIONAL_CANDIDATE',centre=result['diagnostic_centre_world'])
    return result
