"""V67 pure capability/routing negatives; no certificate, NN or capture access."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.local_mpc_control_20261103 import depth_scope12_goal_perception_v67 as p
from our_mani.local_mpc_control_20261103 import goal_aware_scope12_binding_v67 as b
from our_mani.local_mpc_control_20261103 import depth_position_goal_perception_v64 as old_tracker
from our_mani.cube_position_measurement_20261103_v10 import prediction as predict
from our_mani.cube_position_measurement_20261103_v8 import prediction as legacy
from our_mani.local_mpc_p4_20261103.runtime import _walk_public
ROOT=Path(__file__).resolve().parents[1]


def measurement(ok,scope=False):
    return dict(refiner_available=ok,centre=[.01,.02,1.]if ok else None,
        geometry_profile=p.SCOPE_GEOMETRY_VERSION if scope else p.LEGACY_GEOMETRY_VERSION,
        reasons=[]if ok else['current_support_unknown'],measured_points=24 if ok else 0,
        visible_views=3 if ok else 0,geometry_confidence=1. if ok else 0.)


def inputs():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.ones((4,256,256),np.float32),
        K=np.repeat(np.array([[-100.,0,128],[0,-100.,128],[0,0,1.]])[None],4,0),
        T=np.repeat(np.eye(4)[None],4,0),public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(['front','8','16','36']),timestamp=np.array(0.))


def prediction(route):
    crops={'valid_crop':np.ones((4,64,64),bool)}
    prob=np.zeros((4,2,64,64),np.float32);prob[:,1]=.999;prob[:,0]=1-prob[:,1]
    if route.startswith('legacy'):
        values=[measurement(True)]if route=='legacy_first'else[measurement(False),measurement(True)]
        with patch.object(legacy,'measure_current_cube',side_effect=values):
            result=legacy.predict_current_position(inputs(),crops,prob)
        with patch.object(predict,'legacy_predict',return_value=result),patch.object(predict,'measure_current_cube_scope12',side_effect=AssertionError('old pass exact')):
            assert predict.predict_current_position(inputs(),crops,prob)is result
        return result
    with patch.object(legacy,'measure_current_cube',side_effect=[measurement(False),measurement(False)]):
        previous=legacy.predict_current_position(inputs(),crops,prob)
    values=[measurement(True,True)]if route=='scope_first'else[measurement(False,True),measurement(route=='scope_second',True)]
    with patch.object(predict,'legacy_predict',return_value=previous),patch.object(predict,'measure_current_cube_scope12',side_effect=values):
        return predict.predict_current_position(inputs(),crops,prob)


class V67Contracts(unittest.TestCase):
    def test_flags_require_real_scope12_and_leave_composite_honest(self):
        flags=dict(position_measurement_qualified=True,scope12_EF_qualified=True,axis_general_qualified=False,
            active_certified_routes=('legacy_v8','scope12_EF'),scope_radius_m=.12,
            qualified=False,model_qualified=False,raw_mask_model_qualified=False,
            composite_global_coverage_qualified=False)
        self.assertTrue(p._position_flags(SimpleNamespace(**flags)))
        for key,value in (('scope12_EF_qualified',False),('axis_general_qualified',True),
            ('active_certified_routes',('legacy_v8',)),('scope_radius_m',.08),('qualified',True),
            ('model_qualified',True),('raw_mask_model_qualified',True),('composite_global_coverage_qualified',None)):
            self.assertFalse(p._position_flags(SimpleNamespace(**dict(flags,**{key:value}))))

    def test_all_five_pure_routes_and_actual_geometry_trace(self):
        for route in ('legacy_first','legacy_second','scope_first','scope_second','unknown'):
            result=prediction(route);evidence=p.validate_position_proposal_trace(result)
            self.assertEqual(evidence['measurement_route'],'legacy_v8'if route.startswith('legacy')else'scope12_EF')
            self.assertTrue(evidence['threshold_change_is_same_position_source'])
            self.assertEqual(evidence['scope_radius_m'],.08 if route.startswith('legacy')else .12)
            self.assertEqual(evidence['proposal_trace'][-1]['reasons'],result['measurement']['reasons'])
            bad=deepcopy(result);bad['measurement']['measured_points']+=1
            with self.assertRaises(ValueError):p.validate_position_proposal_trace(bad)
            if route=='unknown':self.assertIsNone(result['centre'])
        bad=prediction('scope_first');bad['axis_general_runtime_enabled']=True
        with self.assertRaises(ValueError):p.validate_position_proposal_trace(bad)
        bad=prediction('scope_first');bad['legacy_proposal_trace'][0]['geometry_available']=True
        with self.assertRaises(ValueError):p.validate_position_proposal_trace(bad)

    def test_public_projection_private_F_never_reaches_policy(self):
        for route in ('legacy_first','scope_second','unknown'):
            private=dict(timestamp=0.,measurement_available=route!='unknown',
                position_proposals=p.validate_position_proposal_trace(prediction(route)),
                scope12_EF_qualified=True,axis_general_qualified=False,scope_radius_m=.12,
                composite_global_coverage_qualified=False,active_certified_routes=['legacy_v8','scope12_EF'],
                NN_current_measurement={'source':{'GT_or_label_or_robot_or_history_input':False}})
            with self.assertRaises(ValueError):_walk_public(private)
            public=p.project_public_position_evidence(private);_walk_public(public)
            self.assertNotIn('NN_current_measurement',public)
            bad=deepcopy(private);bad['unregistered']=False
            with self.assertRaises(ValueError):p.project_public_position_evidence(bad)
            bad=deepcopy(private);bad['position_proposals']['proposal_trace'][0]['extra']=False
            with self.assertRaises(ValueError):p.project_public_position_evidence(bad)

    def test_no_old_certificate_or_invalid_sha_constructor(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'old.pt';path.write_bytes(b'no model or credential')
            with patch.object(p,'CurrentCubePositionMeasurementV10')as factory:
                with self.assertRaises(ValueError):p.CurrentDepthScope12TrackerV67(path,'')
                factory.assert_not_called()
            # Runtime rejects old version before any model construction. No passing fixture exists.
            with patch.object(p.torch,'load',return_value={'version':'old_V8','metadata':{}}):
                with self.assertRaises(ValueError):p.CurrentDepthScope12TrackerV67(path,hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(TypeError):b.DepthScope12PublicGraspStateAssemblerV67(object())

    def test_actual_cloud_identity_only_certified_profiles(self):
        rows=[dict(camera=i,baseline_candidate_points=1,accepted=[True],point_world=[[i*.01,0.,1.]],accepted_points=1)for i in range(4)]
        points=[r['point_world'][0]for r in rows]
        for profile in (p.LEGACY_GEOMETRY_VERSION,p.SCOPE_GEOMETRY_VERSION):
            value=dict(geometry_profile=profile,camera_point_ledger=rows,current_points_world=deepcopy(points),measured_points=4,visible_views=4)
            cloud,views=p.current_position_depth_cloud(value);np.testing.assert_array_equal(cloud,points);self.assertEqual(views,4)
            bad=deepcopy(value);bad['current_points_world'][0][0]+=.001
            with self.assertRaises(ValueError):p.current_position_depth_cloud(bad)
        with self.assertRaises(ValueError):p.current_position_depth_cloud(dict(geometry_profile='axis_general'))

    def test_history_template_association_AST_unchanged_and_sources_exact(self):
        def method(module,name):
            tree=ast.parse(Path(module.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name.startswith('CurrentDepth'))
            node=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
            # Backend log provenance is the sole intentional change within _accept.
            for n in ast.walk(node):
                if isinstance(n,ast.Constant)and isinstance(n.value,str):n.value=n.value.replace('v67_current','v64_current')
            return ast.dump(node,include_attributes=False)
        for name in ('_association','_accept','_unknown','reset','reset_episode'):
            self.assertEqual(method(p,name),method(old_tracker,name))
        for name,digest in b.APPROVED_COMPONENT_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        self.assertIn('our_mani/local_mpc_control_20261103/depth_scope12_goal_perception_v67.py',b.APPROVED_COMPONENT_SHA256)

if __name__=='__main__':unittest.main()
