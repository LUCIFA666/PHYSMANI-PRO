"""Fixed uncertainty envelope and explicit permission failures, CPU only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.multitask.bin_swept_geometry import _prism,points_inside_prototype,verify_stored_sweep_exact
from our_mani.multitask.bin_swept_assets import (ROOT,inflated_axis_yaw_sweep,
    load_axis_swept_collision_geometry,SOURCE_MIGRATION_BOUND_M)
from our_mani.multitask.training_geometry import register_collision_asset
from our_mani.multitask.protocol import MultitaskProtocolError


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class BinSweptInflationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        angles=np.arange(12)*2*np.pi/12;directions=np.c_[np.cos(angles),np.sin(angles)]
        walls,wt=_prism(0.,.2,directions*.15,directions*.1)
        floor,ft=_prism(0.,.015,directions*.15,None)
        cls.vertices=np.r_[walls,floor];cls.triangles=np.r_[wt,ft+len(walls)]
        cls.result=inflated_axis_yaw_sweep(cls.vertices,cls.triangles)

    def test_fixed_error_bounds_keep_cavity_and_exact_enclosure(self):
        result=self.result;receipt=result.receipt
        self.assertEqual(receipt['engineering_tilt_bound_degrees'],10.)
        self.assertEqual(receipt['engineering_translation_bound_m'],.005)
        self.assertEqual(receipt['source_vertex_migration_allowance_m'],1e-6)
        self.assertFalse(receipt['calibrated_safety_guarantee'])
        self.assertGreater(receipt['retained_cavity_slabs'],0)
        self.assertFalse(points_inside_prototype([[.19,0,0]],result)[0])
        proof=verify_stored_sweep_exact(self.vertices,self.triangles,result,
            inflation_m=receipt['total_axial_radial_inflation_m'])
        self.assertTrue(proof['exact_axis_escape_for_every_retained_cavity'])

    def test_surface_yaw_tilt_translation_samples_are_contained(self):
        rng=np.random.default_rng(9274);n=4000
        faces=self.vertices[self.triangles[rng.integers(0,len(self.triangles),n)]]
        points=(faces*rng.dirichlet([1,1,1],n)[:,:,None]).sum(1)
        yaw=rng.uniform(-np.pi,np.pi,n);y,z=points[:,1].copy(),points[:,2].copy()
        points[:,1]=y*np.cos(yaw)-z*np.sin(yaw);points[:,2]=y*np.sin(yaw)+z*np.cos(yaw)
        from scipy.spatial.transform import Rotation
        directions=rng.normal(size=(n,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        points=Rotation.from_rotvec(directions*rng.uniform(0,np.deg2rad(10),n)[:,None]).apply(points)
        translations=rng.normal(size=(n,3));translations/=np.linalg.norm(translations,axis=1)[:,None]
        points+=translations*rng.uniform(0,.005,n)[:,None]
        self.assertTrue(points_inside_prototype(points,self.result).all())

    def test_axis_permission_is_never_implicit_or_cross_role(self):
        for permission in ({},dict(enabled=False),dict(enabled=True,axis_local=[1.,0.,0.],axis_polarity='undirected')):
            with self.assertRaises(ValueError):load_axis_swept_collision_geometry(permission,{})
        with self.assertRaises(ValueError):
            load_axis_swept_collision_geometry(dict(enabled=True,axis_local=[1.,0.,0.],axis_polarity='directed'),{},role_id='cup0')


class FrozenActualTRAINAssetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_path=ROOT/'artifacts/framework/multitask_r10/bin_axis_swept_assets_v1/report.json'
        if not report_path.exists():raise unittest.SkipTest('run the fixed offline builder for actual TRAIN asset checks')
        cls.report=json.loads(report_path.read_text());cls.task='put_rubbish_in_moving_bin'
        cls.permission=cls.report['axis_swept_assets'][cls.task]['bin']
        cls.source=json.loads((ROOT/cls.report['tasks'][cls.task]['source_asset']['path']).read_text())

    def test_actual_raw_train_asset_roundtrip_uses_explicit_inflated_sweep(self):
        geometry=load_axis_swept_collision_geometry(self.permission,self.source)
        self.assertEqual(geometry.receipt['collision_geometry_kind'],'directed_axis_yaw_uncertainty_sweep')
        self.assertEqual(geometry.receipt['registered_source_vertex_difference_max_m'],0.)
        self.assertTrue(geometry.receipt['cavities_preserved'])
        self.assertFalse(geometry.receipt['calibrated_safety_guarantee'])

    def test_changed_permission_hash_topology_flags_and_large_vertex_migration_are_rejected(self):
        bad={**self.permission,'sha256':'0'*64}
        with self.assertRaises(ValueError):load_axis_swept_collision_geometry(bad,self.source)
        for mutate in ('triangles','component_names','large_shift'):
            source=deepcopy(self.source);body=source['bodies']['bin']
            if mutate=='triangles':body['collision_triangles'][0][0]=body['collision_triangles'][0][1]
            elif mutate=='component_names':body['collision_component_names'].append('new_native_shape')
            else:body['collision_vertices_body'][0][0]+=SOURCE_MIGRATION_BOUND_M*1.01
            source['asset_sha256']=digest({k:v for k,v in source.items() if k!='asset_sha256'})
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                load_axis_swept_collision_geometry(self.permission,source)

    def test_sub_micrometre_vertex_migration_remains_inside_explicit_inflation(self):
        source=deepcopy(self.source)
        source['bodies']['bin']['collision_vertices_body'][0][0]+=.5*SOURCE_MIGRATION_BOUND_M
        source['asset_sha256']=digest({k:v for k,v in source.items() if k!='asset_sha256'})
        geometry=load_axis_swept_collision_geometry(self.permission,source)
        self.assertGreater(geometry.receipt['registered_source_vertex_difference_max_m'],0.)
        self.assertLess(geometry.receipt['registered_source_vertex_difference_max_m'],SOURCE_MIGRATION_BOUND_M)
        self.assertTrue(geometry.receipt['source_vertex_migration_bound_already_in_inflation'])

    def test_explicit_axis_registration_keeps_rotation_unknown_cube_alongside_sweep(self):
        calls=[];map_service=SimpleNamespace(register_estimated_geometry=lambda *a,**k:calls.append((a,k)),
            declare_noncolliding_geometry=lambda *a,**k:None)
        observations={name:'none' for name in self.source['bodies']};observations['bin']='axis'
        result=register_collision_asset(map_service,self.source,orientation_observability=observations,
            axis_swept_assets={'bin':self.permission})
        self.assertEqual(result['bin']['collision_geometry_kind'],'directed_axis_yaw_uncertainty_sweep')
        registration=next(k for a,k in calls if a[0]=='bin')
        self.assertEqual(registration['fallback_mesh']['collision_geometry_receipt']['collision_geometry_kind'],
            'rotation_uncertainty_cube')
        self.assertTrue(registration['fallback_mesh']['collision_geometry_receipt']['unknown_rotation_identity_is_not_exact_CAD'])

    def test_missing_explicit_permission_keeps_axis_rotation_cube(self):
        calls=[];map_service=SimpleNamespace(register_estimated_geometry=lambda *a,**k:calls.append((a,k)),
            declare_noncolliding_geometry=lambda *a,**k:None)
        observations={name:'none' for name in self.source['bodies']};observations['bin']='axis'
        result=register_collision_asset(map_service,self.source,orientation_observability=observations)
        self.assertEqual(result['bin']['collision_geometry_kind'],'rotation_uncertainty_cube')
        with self.assertRaises(MultitaskProtocolError):
            register_collision_asset(map_service,self.source,orientation_observability={name:'none' for name in observations},
                axis_swept_assets={'bin':self.permission})


if __name__=='__main__':unittest.main()
