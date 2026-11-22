"""CPU-only V9 source/routing/contracts. No real calibration or NN forward."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.cube_position_measurement_20261103_v9 import prediction as p,evaluation as e,qualification as q,runtime as r,data as d
from our_mani.cube_position_measurement_20261103_v8 import prediction as old


def inputs():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.ones((4,256,256),np.float32),
        K=np.repeat(np.array([[-100.,0,128],[0,-100.,128],[0,0,1.]])[None],4,0),
        T=np.repeat(np.eye(4)[None],4,0),public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(['front','8','16','36']),timestamp=np.array(0.))


def measurement(available,scope=False):
    return dict(refiner_available=available,centre=[.01,.02,1.]if available else None,
        geometry_profile=p.SCOPE_GEOMETRY_VERSION if scope else old.GEOMETRY_VERSION,
        reasons=[]if available else['current_support_unknown'],measured_points=24 if available else 0,
        visible_views=3 if available else 0,geometry_confidence=1. if available else 0.)


def stats(n):return dict(count=n,p95_m=.001,max_m=.002)
def group(n,a):return dict(samples=n,available=a,coverage=a/n,centre_error=stats(a))
def positions(control=7,annulus=15,near=22):
    controls=group(8,control)
    controls.update(baseline_available=control,legacy_passes_preserved=control,
        legacy_pass_nonregression=True,independent_coverage_gate=False)
    return dict(samples=80,scope_validation=dict(required24=group(24,control+annulus),
        legacy_control=controls,scope_annulus=group(16,annulus),all56_near=group(56,near)),
        all_available_centre_error=stats(near),negatives=dict(sphere_only=dict(samples=16,falseavailable=0),
        cube_far=dict(samples=8,falseavailable=0)),owner_unobserved_falseavailable=0)


class Scope12V9Tests(unittest.TestCase):
    def setUp(self):
        self.crops={'valid_crop':np.ones((4,64,64),bool)}
        self.prob=np.empty((4,2,64,64),np.float32);self.prob[:,1]=.999;self.prob[:,0]=1-self.prob[:,1]
        self.support=dict(TP=1000,FP=0,U=0)

    def test_both_legacy_pass_routes_preserve_entire_object_and_never_scope(self):
        for fallback in (False,True):
            values=[measurement(False),measurement(True)]if fallback else[measurement(True)]
            with patch.object(old,'measure_current_cube',side_effect=values):
                legacy=old.predict_current_position(inputs(),self.crops,self.prob)
            with patch.object(p,'legacy_predict',return_value=legacy),patch.object(p,'measure_current_cube_scope12',side_effect=AssertionError('no scope')):
                selected,audit=p.predict_with_legacy_audit(inputs(),self.crops,self.prob)
            self.assertIs(selected,legacy);self.assertTrue(audit['legacy_pass_returned_exact_object'])
            p.validate_prediction_route(selected);p.validate_legacy_audit(selected,audit)
            changed=deepcopy(selected);changed['centre'][0]+=.00001
            with self.assertRaises(ValueError):p.validate_legacy_audit(changed,audit)

    def test_scope_only_after_complete_legacy_unknown_fixed_order_and_no_axis(self):
        for outcome in ('first','second','unknown'):
            with patch.object(old,'measure_current_cube',side_effect=[measurement(False),measurement(False)]):
                legacy=old.predict_current_position(inputs(),self.crops,self.prob)
            values=[measurement(True,True)]if outcome=='first'else[measurement(False,True),measurement(outcome=='second',True)]
            with patch.object(p,'legacy_predict',return_value=legacy),patch.object(p,'measure_current_cube_scope12',side_effect=values)as geometry:
                result,audit=p.predict_with_legacy_audit(inputs(),self.crops,self.prob)
            self.assertEqual(geometry.call_count,1 if outcome=='first'else 2)
            p.validate_prediction_route(result);p.validate_legacy_audit(result,audit)
            self.assertFalse(result['axis_general_runtime_enabled']);self.assertFalse(result['raw_mask_qualified'])
            if outcome=='unknown':self.assertIsNone(result['centre']);self.assertIsNone(result['selected_threshold'])
            bad=deepcopy(result);bad['legacy_thresholds_tried']=[.995]
            with self.assertRaises(ValueError):p.validate_prediction_route(bad)

    def test_privileged_input_and_invalid_probability_rejected_before_geometry(self):
        for key in ('q','cube_center_world','semantic_role_ids','future_observation'):
            value=inputs();value[key]=np.zeros(3)
            with patch.object(old,'measure_current_cube',side_effect=AssertionError('no geometry')):
                with self.assertRaises(ValueError):p.predict_current_position(value,self.crops,self.prob)
        with self.assertRaises(ValueError):p.predict_current_position(inputs(),self.crops,self.prob.astype(np.float64))

    def test_required24_annulus16_not_control8_coverage_and_composite_separate(self):
        value=positions()
        decision=e.position_gate(value,self.support,self.support)
        self.assertTrue(decision['scope12_EF_qualified']);self.assertFalse(decision['composite_global_coverage_qualified'])
        self.assertFalse(decision['axis_general_qualified']);self.assertFalse(decision['model_qualified'])
        self.assertTrue(e.position_gate(positions(8,16,56),self.support,self.support)['composite_global_coverage_qualified'])
        for value in (positions(6,15),positions(8,14)):
            self.assertFalse(e.position_gate(value,self.support,self.support)['scope12_EF_qualified'])
        value=positions();value['scope_validation']['legacy_control']['legacy_pass_nonregression']=False
        self.assertFalse(e.position_gate(value,self.support,self.support)['scope12_EF_qualified'])
        for field in ('p95_m','max_m'):
            value=positions();value['all_available_centre_error'][field]=.011
            self.assertFalse(e.position_gate(value,self.support,self.support)['scope12_EF_qualified'])
        for bad in (dict(TP=1000,FP=0,U=6),dict(TP=0,FP=0,U=0)):
            self.assertFalse(e.position_gate(positions(),bad,self.support)['scope12_EF_qualified'])

    def test_all80_ledger_fixed24_and_no_group_relabel(self):
        samples=[dict(parent_id=pid,scene_id=pid+'/'+shot,role='calib',position_band=d.POSITION_BANDS[shot],
            condition=d.CONDITIONS[shot],inputs={},crops={},semantic=np.ones((1,),np.uint8))for pid in d.PARTITION_IDS for shot in d.SHOT_IDS]
        with patch.object(e,'legacy_measurement',return_value={'refiner_available':False}):ledger=e.freeze_legacy_eligibility(samples)
        e._eligibility(ledger)
        self.assertEqual(sum(r['required_position']for r in ledger['records']),24)
        bad=deepcopy(ledger);bad['records'][2]['required_position']=False;bad.pop('content_sha256');bad['content_sha256']=e.canonical_sha(bad)
        with self.assertRaises(ValueError):e._eligibility(bad)

    def test_old_schema_old_runtime_and_axis_certificate_fail_before_model(self):
        with tempfile.TemporaryDirectory(prefix='v9_contract_')as tmp:
            path=Path(tmp)/'manifest.json';path.write_text(json.dumps({'schema':'oldV8'}))
            with self.assertRaises(ValueError):d.PositionCalibration8V9Dataset(path)
            payload=Path(tmp)/'old.pt';payload.write_bytes(b'old version')
            with patch.object(r.torch,'load',return_value={'version':'V8','metadata':{}}),patch.object(r,'TinyDepthInfluenceUNet',side_effect=AssertionError('no model')):
                with self.assertRaises(ValueError):r.CurrentCubePositionMeasurementV9(payload,d.sha256(payload))
        decision=e.position_gate(positions(),self.support,self.support)
        good={k:deepcopy(decision[k])for k in ('position_measurement_qualified','scope12_EF_qualified','axis_general_qualified','composite_global_coverage_qualified','active_certified_routes')}
        r.validate_scope_flags(good,decision)
        bad=dict(good,axis_general_qualified=True)
        with self.assertRaises(ValueError):r.validate_scope_flags(bad,decision)

    def test_source_profile_and_default_cli_only_preflight(self):
        profile=q.check_profile_bindings();self.assertEqual(profile['qualification']['required_position_fixed_denominator'],24)
        self.assertNotIn('legacy_control_minimum_coverage',profile['qualification'])
        sources=q.qualification_source_map()
        self.assertEqual(sources[str(q.PROTOCOL)],q.PROTOCOL_SHA256)
        self.assertEqual(sources[str(d.ROOT/'our_mani/near_goal_cube_geometry_scope12_20261103_v1/measurement.py')],profile['source_geometry']['measurement_sha256'])
        path=d.ROOT/'scripts/qualify_cube_position_measurement_20261103_v9.py'
        spec=importlib.util.spec_from_file_location('v9_cli_fixture',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch.object(cli,'check_qualification_inputs',return_value={'forward_calls':0})as check,patch.object(cli,'qualify_selected_checkpoint',side_effect=AssertionError('no qualification')):
            self.assertEqual(cli.main(['--selected','selected.pt','--expected-sha256','0'*64,'--position-manifest','manifest.json']),0)
        check.assert_called_once()

    def test_new_plan_registry_binding_reads_metadata_only_and_excludes_old48(self):
        registry=d.ROOT/'artifacts/framework/local_mpc_20261102/P3_position_v9_calibration8_TRAIN_q_registry_v1/q_sources.json'
        rows=json.loads(registry.read_text())['rows']
        obj=object.__new__(d.PositionCalibration8V9Dataset);obj.bound_pointers={}
        obj.capture_plan=dict(schema=d.PLAN_SCHEMA,scope=d.SCOPE,split='train',perception_only=True,
            TRAIN_q_registry=dict(path=str(registry),sha256=d.FRESH_REGISTRY_SHA),
            parents=[dict(parent_id=pid,seed=261103648+i,role='calibration',fold='calibration',
                near_direction_ordinal=i,TRAIN_q_source=rows[i],shots=[{'shot_id':s}for s in d.SHOT_IDS])for i,pid in enumerate(d.PARTITION_IDS)])
        with patch.object(d,'load_npz',side_effect=AssertionError('no images')):
            obj._check_position_plan({})
        self.assertEqual(len(obj.bound_pointers),5)
        bad=deepcopy(obj.capture_plan);bad['parents'][1]['TRAIN_q_source']=rows[0];obj.capture_plan=bad
        with self.assertRaises(ValueError):obj._check_position_plan({})


if __name__=='__main__':unittest.main()
