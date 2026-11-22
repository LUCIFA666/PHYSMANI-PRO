"""TRAIN-only all-rotation enclosure; current RGBD uses existing sphere fitting.

A circumscribed convex polytope contains the complete radius-r ball, hence
contains every rotation of every source triangle. It does not assume observed
orientation. No world object, collision pair, threshold or physics is removed.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import MethodType
import json
import numpy as np
from scipy.spatial import ConvexHull
from .collision_geometry import mesh_sha256, collision_geometry_for_orientation
from .evaluation import file_sha
from .protocol import MultitaskProtocolError
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='r10_TRAIN_spherical_all_rotation_enclosure_v7_2'
KIND='rotation_uncertainty_circumscribed_polytope'

def _planes(vertices,triangles):
    a,b,c=np.asarray(vertices,dtype=float)[np.asarray(triangles)].transpose(1,0,2)
    n=np.cross(b-a,c-a);n/=np.linalg.norm(n,axis=1)[:,None]
    d=np.einsum('ij,ij->i',n,a)
    n[d<0]*=-1;d=np.abs(d)
    return n,d

def _canonical_faces(points, faces):
    faces=np.asarray(faces,dtype=np.int32).copy()
    raw=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    reverse=np.einsum('ij,ij->i',raw,points[faces[:,0]])<0
    faces[reverse]=faces[reverse][:,[0,2,1]]
    canonical=[]
    for row in faces:canonical.append(tuple(np.roll(row,-int(np.argmin(row)))))
    return np.asarray(sorted(canonical),dtype=np.int32)

@lru_cache(maxsize=1)
def _unit_hull():
    g=(1+np.sqrt(5.))/2
    p=np.array([(0,a,b*g) for a in (-1,1) for b in (-1,1)]+
               [(a,b*g,0) for a in (-1,1) for b in (-1,1)]+
               [(a*g,0,b) for a in (-1,1) for b in (-1,1)],float)
    p/=np.linalg.norm(p,axis=1)[:,None]
    f=_canonical_faces(p,ConvexHull(p).simplices)
    for _ in range(3):
        vertices=p.tolist();edges={};faces=[]
        def midpoint(a,b):
            key=tuple(sorted((int(a),int(b))))
            if key not in edges:
                v=p[a]+p[b];v/=np.linalg.norm(v);edges[key]=len(vertices);vertices.append(v.tolist())
            return edges[key]
        for a,b,c in f:
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            faces.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
        p=np.asarray(vertices);f=np.asarray(faces,dtype=np.int32)
    hull=ConvexHull(p);f=_canonical_faces(p,hull.simplices)
    n,d=_planes(p,f)
    # Orient every stored face outwards as well as checking its halfspace.
    raw=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
    reverse=np.einsum('ij,ij->i',raw,p[f[:,0]])<0
    f[reverse]=f[reverse][:,[0,2,1]]
    return p,f,float(d.min())

def enclosure(vertices,triangles,asset_sha):
    original=collision_geometry_for_orientation(vertices,triangles,orientation_observability='none')
    radius=original.receipt['source_vertices_max_radius_m']
    unit,faces,dmin=_unit_hull()
    scale=np.nextafter(np.float32(radius/dmin),np.float32(np.inf))
    for _ in range(8):
        stored=(unit*float(scale)).astype(np.float32)
        normals,distances=_planes(stored,faces)
        # A closed convex mesh and all inward halfspaces enclosing B(r).
        if distances.min()>radius and np.max(stored.astype(float)@normals.T-distances)<1e-12:break
        scale=np.nextafter(scale,np.float32(np.inf))
    else:raise MultitaskProtocolError('float32 polytope failed all-rotation inclusion proof')
    receipt=dict(schema=SCHEMA,collision_geometry_kind=KIND,orientation_observability='none',
        TRAIN_asset_sha256=asset_sha,original_mesh_sha256=mesh_sha256(vertices,triangles),
        registered_mesh_sha256=mesh_sha256(stored,faces),source_vertices_max_radius_m=radius,
        minimum_face_plane_distance_m=float(distances.min()),maximum_vertex_radius_m=float(np.linalg.norm(stored,axis=1).max()),
        all_rotation_inclusion_proved=True,construction_rule='closed circumscribed subdivided icosahedron; every face plane distance exceeds maximum TRAIN-local vertex radius after float32 storage',
        vertex_count=len(stored),triangle_count=len(faces),cavities_preserved=False,
        full_orientation_promoted=False,native_pose_or_future_state_used=False,TRAIN_asset_unchanged=True)
    return stored,faces,receipt

def geometry_binding(root,protocol,task='moving_basketball_in_hoop'):
    if task!='moving_basketball_in_hoop':raise MultitaskProtocolError('v7 is normal basketball only')
    pointer=protocol['learned_artifacts'][task]['geometry'];p=Path(root)/pointer['path']
    if file_sha(p)!=pointer['sha256']:raise MultitaskProtocolError('TRAIN CAD changed')
    asset=json.loads(p.read_text());body=asset['bodies']['ball']
    vertices,faces,receipt=enclosure(body['collision_vertices_body'],body['collision_triangles'],asset['asset_sha256'])
    # This explicit sphere model is derived from a spherical TRAIN collision mesh.
    norms=np.linalg.norm(body['collision_vertices_body'],axis=1)
    if np.ptp(norms)>32*np.finfo(np.float32).eps*receipt['source_vertices_max_radius_m']:
        raise MultitaskProtocolError('TRAIN CAD does not certify a spherical body-origin surface')
    radius=receipt['source_vertices_max_radius_m']
    return dict(schema=SCHEMA,task=task,role='ball',source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__)),
        TRAIN_geometry=pointer,registered_geometry=receipt,current_RGBD_estimator='existing MultiBodySE3Tracker._symmetry_pose fixed-radius sphere least-squares',
        current_RGBD_sphere_radius_m=radius,rotation_observability='none',collision_permissions_unchanged=True)

@contextmanager
def install_registration(service,protocol,root):
    expected=geometry_binding(root,protocol)
    if protocol.get('spherical_geometry_v7')!=expected:raise MultitaskProtocolError('explicit v7 geometry binding differs')
    if protocol.get('body_sphere_radii',{}).get(expected['task'],{}).get('ball')!=expected['current_RGBD_sphere_radius_m']:
        raise MultitaskProtocolError('existing current RGBD sphere estimator must use the exact bound TRAIN radius')
    asset=json.loads((Path(root)/expected['TRAIN_geometry']['path']).read_text());body=asset['bodies']['ball']
    v,f,r=enclosure(body['collision_vertices_body'],body['collision_triangles'],asset['asset_sha256'])
    cube=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='none')
    original=service.register_estimated_geometry;seen=[]
    def register(instance,object_id,vertices,triangles,**kwargs):
        if object_id=='ball':
            if (object_id in instance.estimated_bodies or kwargs.get('source_sha256')!=asset['asset_sha256'] or
                    mesh_sha256(vertices,triangles)!=cube.receipt['registered_mesh_sha256'] or kwargs.get('fallback_mesh') is not None):
                raise MultitaskProtocolError('only fresh frozen TRAIN ball rotation enclosure may be refined')
            kwargs=dict(kwargs,collision_geometry_receipt=deepcopy(r));vertices,triangles=v,f;seen.append(object_id)
        return original(object_id,vertices,triangles,**kwargs)
    service.register_estimated_geometry=MethodType(register,service)
    try:
        yield expected
        if seen!=['ball']:raise MultitaskProtocolError('v7 ball geometry was not registered exactly once')
    finally:service.register_estimated_geometry=original

def bind_geometry_port(actor):
    original=actor.collision_port._geometry_modes
    def modes(instance):
        selected,reasons=original()
        entry=instance.map.estimated_bodies['ball'];r=entry.collision_geometry_receipt
        if r.get('collision_geometry_kind')!=KIND or r.get('all_rotation_inclusion_proved') is not True or entry.fallback_ghost is not None or selected['ball']!='primary':
            raise MultitaskProtocolError('current query lost the verified all-rotation enclosure')
        reasons['ball']['reason']='verified primary circumscribed polytope contains every TRAIN body rotation'
        return selected,reasons
    actor.collision_port._geometry_modes=MethodType(modes,actor.collision_port)
