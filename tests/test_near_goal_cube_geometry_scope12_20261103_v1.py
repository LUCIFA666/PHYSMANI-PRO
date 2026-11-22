"""Synthetic scope-only delta tests; no model/real captures/calibration/Native."""
from copy import deepcopy
from pathlib import Path
import ast,inspect,unittest
import numpy as np
from our_mani.near_goal_cube_geometry_20261103_v1 import measurement as old
from our_mani.near_goal_cube_geometry_scope12_20261103_v1 import measurement as new


def support(distance):
    p=[distance,0.,0.]
    return dict(centre=p,diagnostic_centre_only=p,support_available=True,reasons=[],
        axis_counts=[20,20,20],observed_axis=[True]*3,coordinate_sources=['visible_face_median']*3,
        intervals=[[x-.00025,x+.00025]for x in p],radius_each=[.00025]*3,joint_radius=np.sqrt(3)*.00025)

def measure(distance,*,n=60,v=3,reasons=None):
    inputs={'public_sphere_center':np.zeros(3)}
    baseline={'reasons':['outside_predeclared_near_scope_0p08m']+(reasons or []),'near_scope_m':.08}
    axes=np.arange(n)%3;g=support(distance);g['axis_counts']=[int((axes==j).sum())for j in range(3)]
    return new._measurement_from_support(inputs,baseline,np.zeros((n,3)),np.arange(n)%v,axes,g)

def public():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.full((4,256,256),1.025,np.float32),
        K=np.repeat(np.array([[-200.,0,128],[0,-200.,128],[0,0,1.]])[None],4,0),
        T=np.repeat(np.eye(4)[None],4,0),public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(['front','8','16','36']),timestamp=np.array(0.))


class ScopeOnlyAST(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value==.08:return ast.Constant('SCOPE_LIMIT')
        if isinstance(node.value,str):return ast.Constant(node.value.replace('scope_0p08m','scope_LIMIT').replace('scope_0p12m','scope_LIMIT'))
        return node
    def visit_Name(self,node):
        if node.id=='SCOPE_M':return ast.Constant('SCOPE_LIMIT')
        return node
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.name=='measure_current_cube_scope12':node.name='measure_current_cube'
        return node
    def visit_Set(self,node):
        self.generic_visit(node);seen=set();out=[]
        for x in node.elts:
            key=ast.dump(x)
            if key not in seen:out.append(x);seen.add(key)
        node.elts=out;return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute)and node.func.attr=='update':node.keywords=[x for x in node.keywords if x.arg!='near_scope_m']
        return node


class Scope12Tests(unittest.TestCase):
    def test_scope_boundary_is_explicit_twelve_not_availability_flip(self):
        for distance in(.079,.08,.1,.12):
            r=measure(distance);self.assertTrue(r['refiner_available']);self.assertEqual(r['centre'],[distance,0.,0.]);self.assertEqual(r['near_scope_m'],.12)
            self.assertNotIn('outside_predeclared_near_scope_0p08m',r['reasons'])
        r=measure(np.nextafter(.12,np.inf));self.assertFalse(r['refiner_available']);self.assertIsNone(r['centre']);self.assertIn('outside_predeclared_near_scope_0p12m',r['reasons'])
    def test_all_non_scope_gates_remain(self):
        for kwargs in ({'n':9},{'v':1},{'n':20},{'reasons':['raw_Cube_mask_touches_crop_boundary_before_erosion']},{'reasons':['cube_outside_observed_crop']}):
            with self.subTest(kwargs=kwargs):self.assertFalse(measure(.1,**kwargs)['refiner_available'])
        g=support(.1);g.update(support_available=False,centre=None,reasons=['missing_current_geometry'])
        out=new._measurement_from_support({'public_sphere_center':np.zeros(3)},{'reasons':[]},np.zeros((60,3)),np.arange(60)%3,np.arange(60)%3,g)
        self.assertFalse(out['refiner_available']);self.assertIsNone(out['centre'])
    def test_full_public_pipeline_inside_oldscope_is_numerically_unchanged(self):
        inputs=public();crops=old.make_near_goal_crops(inputs)
        for raw in(np.zeros((4,64,64),bool),np.pad(np.ones((4,9,9),bool),((0,0),(27,28),(27,28)))):
            before=old.measure_current_cube(inputs,crops,raw);after=new.measure_current_cube_scope12(inputs,crops,raw)
            before.pop('geometry_profile');after.pop('geometry_profile');before.pop('near_scope_m');after.pop('near_scope_m')
            self.assertEqual(before,after)
        raw=np.zeros((4,64,64),bool);raw[:2,30:33,30:33]=True;r=new.measure_current_cube_scope12(inputs,crops,raw)
        self.assertEqual(r['measured_points'],2);self.assertEqual(r['geometry_erosions'],1);self.assertFalse(r['refiner_available'])
    def test_current_input_crop_and_private_channel_guards_unchanged(self):
        inputs=public();crops=old.make_near_goal_crops(inputs);raw=np.zeros((4,64,64),bool)
        with self.assertRaises(ValueError):new.measure_current_cube_scope12(dict(inputs,cube_pose_world=np.zeros(3)),crops,raw)
        changed=deepcopy(crops);changed['depth_m'][0,30,30]+=.01
        with self.assertRaises(ValueError):new.measure_current_cube_scope12(inputs,changed,raw)
        raw[:,0:16,20:40]=True;r=new.measure_current_cube_scope12(inputs,crops,raw)
        self.assertIn('raw_Cube_mask_touches_crop_boundary_before_erosion',r['reasons']);self.assertFalse(r['refiner_available'])
    def test_only_documented_AST_delta_and_exact_original_lower_functions(self):
        for a,b in((old._measurement_from_support,new._measurement_from_support),(old._original_gate_reasons,new._original_gate_reasons),(old.measure_current_cube,new.measure_current_cube_scope12)):
            self.assertEqual(ast.dump(ScopeOnlyAST().visit(ast.parse(inspect.getsource(a)))),ast.dump(ScopeOnlyAST().visit(ast.parse(inspect.getsource(b)))))
        for name in('_current_face_points','_full_current_mask','_initial_components','aggregate_known_cube_support','refine_with_front_free_space'):
            self.assertIs(getattr(new,name),getattr(old,name))
        before=old.geometry_source_map();after=new.geometry_source_map();self.assertEqual(len(before),10);self.assertEqual(len(after),12)
        self.assertTrue(all(after[k]==v for k,v in before.items()))

if __name__=='__main__':unittest.main()
