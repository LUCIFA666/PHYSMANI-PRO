import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import unittest
from scipy.spatial.transform import Rotation
from our_mani.multitask.spherical_enclosure_v7 import enclosure,_planes,geometry_binding,install_registration
from our_mani.multitask.collision_geometry import collision_geometry_for_orientation
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.contracts import Pose
ROOT=Path(__file__).resolve().parents[1]

class TestSphericalEnclosure(unittest.TestCase):
    def test_float32_closed_convex_all_rotation_bound(self):
        v=np.array([[.03,0,0],[-.03,0,0],[0,.01,0],[0,0,-.02]])
        p,f,r=enclosure(v,np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]]),'a'*64)
        n,d=_planes(p,f)
        assert p.dtype==np.float32 and np.all(d>=np.linalg.norm(v,axis=1).max())
        assert np.max(p.astype(float)@n.T-d)<1e-12
        assert np.all(np.linalg.norm(np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]]),axis=1)>0)
        edges={}
        for tri in f:
            for a,b in zip(tri,np.roll(tri,-1)):
                key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
        assert set(edges.values())=={2}
        for rot in Rotation.random(20,random_state=0):assert np.max(rot.apply(v)@n.T-d)<1e-12
        assert r['maximum_vertex_radius_m']<.0302
    
    def test_cv_recorded_cube_corner_removed_without_shrinking_sphere(self):
        center=np.array([.119950108,.0870006606,.784819901])
        low=np.array([.067998676,.00199910,.75179903]);high=np.array([.133000972,.067001599,.761806087])
        distance=np.linalg.norm(center-np.clip(center,low,high))
        p,_,r=enclosure(np.array([[.03,0,0],[0,.03,0],[0,0,.03]]),np.array([[0,1,2]]),'a'*64)
        assert distance>r['maximum_vertex_radius_m']>.03
        assert np.all(center+.03>low) and np.all(center-.03<high)
    
    def test_existing_current_rgbd_sphere_fitter_recovers_center(self):
        center=np.array([.2,.1,.782]);radius=.03
        angles=np.linspace(-.7,.7,12);points=np.array([center+radius*np.array([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)]) for a in angles for b in angles])
        tracker=object.__new__(MultiBodySE3Tracker)
        tracker.templates={'ball':SimpleNamespace(sphere_radius=radius,orientation_observability='none',finite_rotation_symmetry=None)}
        result=tracker._symmetry_pose('ball',Pose(center+[.002,-.001,-.001]),points,None)
        assert np.linalg.norm(result.position-center)<1e-6
        assert np.array_equal(result.rotation,np.eye(3))
    
    def test_frozen_train_sphere_binding_reproducible(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6_setup/configuration.json').read_text())
        b=geometry_binding(ROOT,c)
        assert b==geometry_binding(ROOT,c) and b['rotation_observability']=='none'
        assert b['registered_geometry']['all_rotation_inclusion_proved']
    
    def test_registration_preserves_other_mesh_and_uses_bound_source(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6_setup/configuration.json').read_text());b=geometry_binding(ROOT,c)
        c['spherical_geometry_v7']=b;c['body_sphere_radii']={b['task']:{'ball':b['current_RGBD_sphere_radius_m']}}
        a=json.loads((ROOT/b['TRAIN_geometry']['path']).read_text());body=a['bodies']['ball'];cube=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='none')
        seen=[]
        class Map:
            estimated_bodies={}
            def register_estimated_geometry(self,*args,**kwargs):seen.append((args,kwargs))
        m=Map();original=m.register_estimated_geometry
        with install_registration(m,c,ROOT):
            m.register_estimated_geometry('ball',cube.vertices,cube.triangles,source_sha256=a['asset_sha256'],collision_geometry_receipt=cube.receipt)
            m.register_estimated_geometry('hoop',cube.vertices,cube.triangles,source_sha256=a['asset_sha256'])
        assert seen[0][1]['collision_geometry_receipt']==b['registered_geometry']
        assert seen[1][0][1] is cube.vertices and m.register_estimated_geometry==original

if __name__=='__main__':unittest.main()
