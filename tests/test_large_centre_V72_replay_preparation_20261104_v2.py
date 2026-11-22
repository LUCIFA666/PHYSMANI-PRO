"""Public reconstruction and report-only replay contracts; no actual model/Native/GT."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('large_replay',ROOT/'scripts/replay_current_cube_dual_crop_large_closed_v72_20261104_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class LargeReplayContracts(unittest.TestCase):
 def test_default_no_selected_model_rawframe_or_GT(self):
  with patch.object(m,'load_public_helper',side_effect=AssertionError('no pixels')),patch.object(m,'main_gate',side_effect=AssertionError('no model evidence')),contextlib.redirect_stdout(io.StringIO())as output:self.assertEqual(m.main([]),0)
  value=json.loads(output.getvalue());self.assertEqual(value['expected_frames'],1046);self.assertEqual(value['model_forward_calls'],0);self.assertFalse(value['GT_read']);self.assertFalse(value['old_GPU_Native_resource_locks_acquired'])

 def test_decode_fixed_tie_and_failed_outputs_kept_in_place(self):
  values=dict(centre_world=np.zeros((3,3)),measurement_availability_logit=np.array([0.,8.,5.]),uncertainty_score_m=np.ones((3,3)),uncertainty_score_valid=np.array([True,False,True]))
  values['centre_world'][2,0]=np.nan
  rows=m.decode_outputs(values,3);self.assertEqual(len(rows),3);self.assertTrue(rows[0]['candidate_available']);self.assertEqual(rows[0]['availability_probability'],.5);self.assertFalse(rows[1]['candidate_available']);self.assertIsNone(rows[2]['centre_world']);self.assertFalse(rows[2]['candidate_available'])
  values['uncertainty_score_valid']=np.array([1.,0.,1.])
  with self.assertRaises(ValueError):m.decode_outputs(values,3)

 def test_public_reconstruction_preserves_RGBD_K_and_does_not_claim_bitexact_R_anchor(self):
  helper=m.load_public_helper()
  with tempfile.TemporaryDirectory()as temp:
   raw=Path(temp)/'public.npz';images={};depths={};calibration={}
   K=np.array([[-200.,0.,128.],[0.,-205.,127.],[0.,0.,1.]])
   q=np.array([.1,-.2,.3,.9]);q=q/np.linalg.norm(q)
   for i,c in enumerate(helper.CAMERAS):
    images['images_'+c]=np.full((256,256,3),i,np.uint8);depths['depths_'+c]=np.full((256,256),1.+.1*i,np.float32)
    calibration[c]=dict(intrinsics=K.tolist(),camera_to_world=[i*.1,0.,1.]+q.tolist())
   np.savez_compressed(raw,timestamp=np.array(.1),**images,**depths)
   result=dict(goal_pose={'position':[.2,.3,.4]},goal_metadata=dict(simulator_pose_used=False,backend='rgbd_red_goal_sphere_train_calibrated_frame',marker_to_goal_m=[.01,.02,.03]))
   inputs,proof=m.reconstructed_public_input(raw,calibration,result,helper)
   for i,c in enumerate(helper.CAMERAS):self.assertTrue(np.array_equal(inputs['rgb'][i],images['images_'+c]));self.assertTrue(np.array_equal(inputs['depth_m'][i],depths['depths_'+c]));self.assertTrue(np.array_equal(inputs['K'][i],K))
   self.assertTrue(np.allclose(inputs['public_sphere_center'],[.19,.28,.37],rtol=0,atol=1e-15));self.assertFalse(proof['original_raw7_bitexact_claim']);self.assertFalse(proof['rawRGBD_intrinsics_modified'])
   self.assertEqual(m.public_anchor(result),inputs['public_sphere_center'].tolist())
   result['goal_metadata']['simulator_pose_used']=True
   with self.assertRaises(ValueError):m.public_anchor(result)

 def test_same_actual_time_only_no_interpolation(self):
  self.assertEqual(m.tick_key(3.2000000000000006),64)
  with self.assertRaises(ValueError):m.tick_key(3.21)
  with self.assertRaises(ValueError):m.tick_key(float('nan'))

 def test_score_refuses_incomplete_freeze_before_truth_or_consumption(self):
  with tempfile.TemporaryDirectory()as temp:
   out=Path(temp);(out/'prediction_freeze.json').write_text(json.dumps(dict(complete=False,frames=1045)))
   with patch.object(m,'closed_gate',side_effect=AssertionError('No GT gate reached')),patch.object(m,'main_gate',side_effect=AssertionError('No actual model')):
    with self.assertRaises(ValueError):m.score(out)
   self.assertFalse((out/'score_consumption.json').exists())

 def test_summary_keeps_failures_missing_truth_and_negative_denominators(self):
  base=dict(centre_error_m=.01,original_centre_error_m=.02,candidate_available=True,prediction_status='ok',truth_status='matched',expected_negative=False,actual_in_scope_0p20=True)
  bad=dict(base,centre_error_m=None,original_centre_error_m=None,candidate_available=False,prediction_status='failed',truth_status='missing',actual_in_scope_0p20=None)
  negative=dict(base,expected_negative=True)
  result=m.summarize([base,bad,negative]);self.assertEqual(result['frames'],3);self.assertEqual(result['prediction_failed'],1);self.assertEqual(result['truth_missing'],1);self.assertEqual(result['error_all_returned']['count'],2);self.assertEqual(result['NoProxy_new_available'],1)

 def test_loader_new_family_no_old_model_or_shared_resource_lock(self):
  text=(ROOT/'scripts/replay_current_cube_dual_crop_large_closed_v72_20261104_v2.py').read_text();tree=ast.parse(text)
  self.assertIn('current_cube_rgbd_centre_large_20261104_v2.evaluation import _validate_selected',text);self.assertIn('LargeDualCropCubeCentreV2',text);self.assertNotIn('DualCropCubeCentreHeadV1(',text)
  lock=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='replay_lock')
  self.assertIn('LOCK.open',ast.unparse(lock));self.assertNotIn('fixed_TRAIN104_geometry_native.lock',text);self.assertNotIn('helper.predict(',text);self.assertNotIn('proof.execute(',text)
  self.assertEqual(sum(m.COUNTS.values()),1046);self.assertEqual(sum(v for k,v in m.COUNTS.items()if k!='static_no_proxy'),845)


if __name__=='__main__':unittest.main()
