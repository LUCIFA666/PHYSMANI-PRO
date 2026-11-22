"""Two fixed real geometry implementations; synthetic public data, no NN/Native."""
import ast
from copy import deepcopy
import io
import inspect
import json
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.local_mpc_control_20261103 import scope12_support_quality_20261103_v10 as new
from our_mani.cube_position_measurement_20261103_v7 import support_quality as old
from our_mani.near_goal_cube_geometry_20261103_v1 import measurement as geometry
from our_mani.near_goal_cube_geometry_scope12_20261103_v1 import measurement as scope


def public():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.full((4,256,256),1.025,np.float32),
        K=np.repeat(np.array([[-200.,0,128],[0,-200.,128],[0,0,1.]])[None],4,0),
        T=np.repeat(np.eye(4)[None],4,0),public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(['front','8','16','36']),timestamp=np.array(0.))


def fixture(seed=7):
    rng=np.random.default_rng(seed)
    crops=dict(valid_crop=np.ones((4,64,64),bool),valid_depth=np.ones((4,64,64),bool),origins_xy=rng.integers(0,160,(4,2),dtype=np.int64))
    ledger=[]
    for c in range(4):
        n=0 if c==3 else 12
        local=np.stack(np.unravel_index(rng.choice(60*60,n,replace=False),(60,60)),axis=1)+2
        keep=rng.random(n)>.35
        ledger.append(dict(camera=c,baseline_candidate_points=n,accepted_points=int(keep.sum()),pixels_uv=(local+crops['origins_xy'][c]).tolist(),accepted=keep.tolist()))
    n=sum(r['accepted_points']for r in ledger);views=sum(r['accepted_points']>0 for r in ledger)
    m=dict(geometry_profile=geometry.VERSION,camera_point_ledger=ledger,measured_points=n,visible_views=views,
        refiner_available=False,centre=None,diagnostic_centre_only=[0.,0.,0.],
        bounded_support={'intervals':[[-.01,.01]]*3},free_space_receipt=dict(free_space_applied=False,feasible_components=[[[-.01,.01]]for _ in range(3)],free_space_trace=[]))
    semantic=rng.choice(np.array([0,1,255],np.uint8),(4,64,64))
    return dict(crops=crops,semantic=semantic,cube_center_world=np.zeros(3)),m


class SupportCompatibilityV10(unittest.TestCase):
    def test_actual_both_geometry_profiles_to_exact_support_and_serialization(self):
        inputs=public();crops=geometry.make_near_goal_crops(inputs)
        raw=np.zeros((4,64,64),bool);raw[:,27:37,27:37]=True
        a=geometry.measure_current_cube(inputs,crops,raw)
        b=scope.measure_current_cube_scope12(inputs,crops,raw)
        self.assertEqual((a['geometry_profile'],b['geometry_profile']),new.SUPPORTED_GEOMETRY_VERSIONS)
        aa=new.extract_current_support_masks(crops,a);bb=new.extract_current_support_masks(crops,b)
        self.assertGreater(aa['accepted'],0)
        for key in ('points','stencils','domain'):
            np.testing.assert_array_equal(aa[key],bb[key])
            np.testing.assert_array_equal(aa[key],old.extract_current_support_masks(crops,a)[key])
        with self.assertRaises(ValueError):old.extract_current_support_masks(crops,b)
        stream=io.BytesIO();np.savez_compressed(stream,points=bb['points'],stencils=bb['stencils'],domain=bb['domain'])
        stream.seek(0)
        with np.load(stream,allow_pickle=False)as arrays:
            for key in arrays.files:
                self.assertEqual(arrays[key].dtype,np.bool_);np.testing.assert_array_equal(arrays[key],bb[key])
        self.assertEqual(b['geometry_profile'],scope.VERSION) # Never spoof legacy profile.
        json.dumps({k:v for k,v in bb.items()if k not in ('points','stencils','domain')},allow_nan=False)

    def test_seeded_masks_and_scores_are_legacy_exact(self):
        for seed in range(12):
            sample,m=fixture(seed);before=old.extract_current_support_masks(sample['crops'],m);after=new.extract_current_support_masks(sample['crops'],m)
            for key in before:
                if isinstance(before[key],np.ndarray):np.testing.assert_array_equal(before[key],after[key])
                else:self.assertEqual(before[key],after[key])
            self.assertEqual(old.score_current_support(sample,m),new.score_current_support(sample,m))
            # A distinct synthetic packet declaring the other fixed profile changes no support semantics.
            scoped=dict(m,geometry_profile=scope.VERSION)
            self.assertEqual(new.score_current_support(sample,scoped),new.score_current_support(sample,m))

    def test_unknown_profile_and_mutated_geometry_rejected(self):
        sample,m=fixture()
        for profile in ('axis_general','scope18',None,geometry.VERSION+' '):
            with self.assertRaises(ValueError):new.extract_current_support_masks(sample['crops'],dict(m,geometry_profile=profile))
        with patch.object(scope,'SCOPE_M',.13):
            with self.assertRaises(ValueError):new.extract_current_support_masks(sample['crops'],m)
        with patch.object(scope,'_current_face_points',lambda *x:None):
            with self.assertRaises(ValueError):new.extract_current_support_masks(sample['crops'],m)
        with patch.object(scope,'geometry_source_map',return_value={}):
            with self.assertRaises(ValueError):new.checked_geometry_sources()
        self.assertEqual(len(new.checked_geometry_sources()),12);self.assertEqual(len(new.support_source_map()),13)

    def test_malformed_coordinates_and_support_counts_still_fail(self):
        for mode in ('edge0','edge63','duplicate','floatuv','shortuv','intkeep','count','domain','depth'):
            sample,m=fixture();item=m['camera_point_ledger'][0]
            indices=[i for i,v in enumerate(item['accepted'])if v];i,j=indices[:2]
            if mode=='edge0':item['pixels_uv'][i]=(sample['crops']['origins_xy'][0]+[0,10]).tolist()
            if mode=='edge63':item['pixels_uv'][i]=(sample['crops']['origins_xy'][0]+[63,10]).tolist()
            if mode=='duplicate':item['pixels_uv'][j]=item['pixels_uv'][i]
            if mode=='floatuv':item['pixels_uv']=np.asarray(item['pixels_uv'],float)
            if mode=='shortuv':item['pixels_uv']=item['pixels_uv'][:-1]
            if mode=='intkeep':item['accepted']=[int(v)for v in item['accepted']]
            if mode=='count':item['accepted_points']+=1
            if mode in ('domain','depth'):
                x,y=np.asarray(item['pixels_uv'][i])-sample['crops']['origins_xy'][0]
                sample['crops']['valid_crop'if mode=='domain'else'valid_depth'][0,y-1,x-1]=False
            with self.subTest(mode=mode):
                for module in (old,new):
                    with self.assertRaises(ValueError):module.extract_current_support_masks(sample['crops'],m)

    def test_original_positive_pixel_and_stencil_AST_unchanged(self):
        def parse(function,compat=False):
            tree=ast.parse(inspect.getsource(function))
            if compat:
                node=tree.body[0]
                node.body=[n for n in node.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='checked_geometry_sources')]
                class Normalize(ast.NodeTransformer):
                    def visit_Compare(self,node):
                        self.generic_visit(node)
                        if len(node.comparators)==1 and isinstance(node.comparators[0],ast.Name)and node.comparators[0].id=='SUPPORTED_GEOMETRY_VERSIONS':
                            node.ops=[ast.NotEq()];node.comparators=[ast.Name(id='GEOMETRY_VERSION',ctx=ast.Load())]
                        return node
                tree=Normalize().visit(tree)
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(parse(old.extract_current_support_masks),parse(new.extract_current_support_masks,True))
        for name in ('_components_from_intervals','_inside','_free_space_posthoc','score_current_support'):
            self.assertEqual(parse(getattr(old,name)),parse(getattr(new,name)))

if __name__=='__main__':unittest.main()
