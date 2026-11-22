"""Position flag, one-use and prediction-freeze contracts; synthetic CPU only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,os,unittest
import numpy as np
import torch
from our_mani.cube_position_measurement_20261103_v8 import qualification as q,runtime as r
from our_mani.near_goal_depth_refiner_20261103_v5 import qualification as oldq
from our_mani.near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from our_mani.cube_position_measurement_20261103_v8.prediction import route_current_public_proposals, GEOMETRY_VERSION
from scripts import qualify_cube_position_measurement_20261103_v8 as cli

class AdaptivePositionQualificationTests(unittest.TestCase):
 def test_native_policy4_environment_and_torch2_are_required(self):
  threads=torch.get_num_threads()
  try:
   torch.set_num_threads(2)
   with patch.dict(os.environ,{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}):
    environment=q.check_execution_environment(TinyDepthInfluenceUNet().float().cpu().eval())
    self.assertEqual(environment['torch_num_threads'],2);self.assertEqual(environment['OMP_NUM_THREADS'],'4')
    with patch.dict(os.environ,{'OMP_NUM_THREADS':'2'}):
     with self.assertRaises(ValueError):q.check_execution_environment()
  finally:torch.set_num_threads(threads)

 def test_same_fit_validator_and_only_new_position_manifest(self):
  self.assertIs(q.validate_selected_checkpoint,oldq.validate_selected_checkpoint)
  profile=q.check_profile_bindings()
  self.assertFalse(profile['raw_mask_model_qualified']);self.assertEqual(profile['output_capability'],'position_measurement_qualified only; generic qualified/model/raw flags stayFalse')
  with TemporaryDirectory()as t:
   wrong=Path(t)/'manifest.json';wrong.write_text('{}')
   with self.assertRaises(ValueError):q.check_profile_bindings(position_manifest=wrong)

 def test_single_consumption_survives_manifest_alias(self):
  with TemporaryDirectory()as t:
   base=Path(t);capture=base/'capture';capture.mkdir();manifest=base/'dataset'/'manifest.json';manifest.parent.mkdir();manifest.write_text('{}')
   first,second=q.claim_calibration_once(manifest,'a'*64,{},base/'out',capture_root=capture)
   self.assertEqual(first['sha256'],second['sha256'])
   alias=base/'alias'/'manifest.json';alias.parent.mkdir();alias.write_text('{}')
   with self.assertRaises(FileExistsError):q.claim_calibration_once(alias,'b'*64,{},base/'newout',capture_root=capture)

 def test_dataset_must_bind_frozen_capture_plan_registry_and_64_count(self):
  profile=q.check_profile_bindings();path=q.ROOT/profile['validation']['capture_plan']
  class MetadataDataset:
   capture_plan_path=path
   capture_plan=json.loads(path.read_text())
   count=64
   def __len__(self):return self.count
  dataset=MetadataDataset();q.validate_position_dataset(dataset)
  dataset.count=63
  with self.assertRaises(ValueError):q.validate_position_dataset(dataset)
  dataset.count=64;dataset.capture_plan['TRAIN_q_registry']['sha256']='0'*64
  with self.assertRaises(ValueError):q.validate_position_dataset(dataset)

 def test_default_cli_does_not_consume(self):
  with patch.object(cli,'check_qualification_inputs',return_value={'complete':True}),patch.object(cli,'qualify_selected_checkpoint',side_effect=AssertionError('no consume'))as call:
   self.assertEqual(cli.main(['--selected','x','--expected-sha256','a'*64,'--position-manifest','y']),0);call.assert_not_called()

 def test_diagnostic_never_calls_position_decoder_or_returns_centre(self):
  obj=object.__new__(r.CurrentCubePositionMeasurementV8);obj.position_measurement_qualified=False
  obj.qualified=obj.model_qualified=False;obj.model=object();obj._sources={};obj.position_calibration_environment={}
  with patch.object(r,'check_execution_environment'),patch.object(r,'qualification_source_map',return_value={}),\
   patch.object(r,'make_near_goal_crops',return_value={'features':None}),patch.object(r,'predict_probabilities',return_value='probabilities'),\
   patch.object(r,'predict_current_position',side_effect=AssertionError('no fallback'))as decode:
   result=obj.predict({});self.assertIsNone(result['centre']);self.assertFalse(result['position_measurement_qualified'])
   self.assertFalse(result['model_qualified']);decode.assert_not_called()

 def test_public_positions_and_masks_frozen_before_single_GT_score(self):
  with TemporaryDirectory()as t:
   base=Path(t);capture=base/'capture';capture.mkdir();manifest=base/'data'/'manifest.json';manifest.parent.mkdir();manifest.write_text('{}')
   selected=base/'selected.pt';selected.write_text('immutable');status=base/'fit_status.json';status.write_text('{}');out=base/'output'
   order=[];features=np.zeros((4,9,64,64),np.float32);mask=np.zeros((4,64,64),bool)
   samples=[dict(parent_id=p,scene_id=p+'/'+s,role='calib',features=features,inputs={},crops={})for p in q.PARTITION_IDS for s in ('shot00','shot01','shot02','shot03','shot04','shot05','shot06','shot07')]
   class Dataset:
    path=manifest;bound_pointers={str(manifest):q.sha256(manifest)}
    records=[{'inputs_path':capture/x['parent_id']/'inputs'/(x['scene_id'].split('/')[-1]+'.npz')}for x in samples]
    def __len__(self):return 64
    def __getitem__(self,i):return samples[i]
   def eligibility(actual):
    self.assertNotIn('forward',order);order.append('eligibility');return {'synthetic':True}
   def predict(model,x):
    self.assertTrue((out/'eligibility_before_predictions.json').exists());order.append('forward')
    p=np.zeros((4,2,64,64),np.float32);p[:,0]=1;return p
   def position(inputs,crops,probabilities):
    index=order.count('public_position');order.append('public_position')
    def m(available):return dict(refiner_available=available,centre=[0.,0.,0.]if available else None,geometry_profile=GEOMETRY_VERSION,reasons=[]if available else ['synthetic_unknown'],measured_points=0,visible_views=0,geometry_confidence=0.)
    proposals=[(.995,m(index%3==0))]
    if index%3:proposals.append((.9,m(index%3==1)))
    return route_current_public_proposals({'valid_crop':np.ones((4,64,64),bool)},probabilities,proposals)
   decision={'position_measurement_qualified':True,'raw_mask_model_qualified':False}
   def score(actual,predictions,oldeligible):
    self.assertEqual(order.count('forward'),64);self.assertEqual(order.count('public_position'),64)
    freeze=json.loads((out/'prediction_freeze.json').read_text());self.assertTrue(freeze['frozen_before_GT_scores'])
    self.assertTrue((out/'public_positions.json').exists());self.assertTrue((out/'public_support_masks.npz').exists())
    self.assertTrue(all(x['raw_Cube_mask'].dtype==np.bool_ for x in predictions))
    self.assertEqual([x['proposal_route'] for x in predictions[:3]],['first_0p995','fallback_0p9','UNKNOWN'])
    self.assertEqual([x['selected_threshold'] for x in predictions[:3]],[.995,.9,None]);self.assertIsNone(predictions[2]['centre'])
    order.append('GT_score')
    return {'positions':{},'used_available_support':{'points':{},'stencils':{}},'decision':decision}
   model=TinyDepthInfluenceUNet();payload={'model_state_dict':model.state_dict(),'metadata':{'identity':{},'model_qualified':False}}
   evidence={'selected':q.pointer(selected),'fit_status':q.pointer(status),'evidence':{}}
   threads=torch.get_num_threads()
   try:
    with patch.object(q,'check_profile_bindings'),patch.object(q,'validate_selected_checkpoint',return_value=(payload,evidence)),\
     patch.object(q,'qualification_source_map',return_value={}),patch.object(q,'check_execution_environment',return_value={}),\
     patch.object(q,'PositionCalibration8Dataset',return_value=Dataset()),patch.object(q,'validate_position_dataset'),patch.object(q,'freeze_legacy_eligibility',side_effect=eligibility),\
     patch.object(q,'predict_probabilities',side_effect=predict),patch.object(q,'predict_current_position',side_effect=position),\
     patch.object(q,'extract_current_support_masks',return_value={'points':mask.copy(),'stencils':mask.copy()}),\
     patch.object(q,'_verify_unchanged'),patch.object(q,'score_frozen_positions',side_effect=score),patch.object(q,'position_gate',return_value=decision):
     result=q.qualify_selected_checkpoint(selected,q.sha256(selected),manifest,out)
   finally:torch.set_num_threads(threads)
   self.assertTrue(result['position_measurement_qualified']);self.assertFalse(result['model_qualified'])
   self.assertEqual(order[-1],'GT_score');self.assertEqual(order.count('GT_score'),1)
   saved=torch.load(out/'calibrated.pt',map_location='cpu',weights_only=True)['metadata']
   self.assertTrue(saved['position_measurement_qualified']);self.assertFalse(saved['qualified']);self.assertFalse(saved['model_qualified']);self.assertFalse(saved['raw_mask_model_qualified'])
   self.assertEqual(selected.read_text(),'immutable')

if __name__=='__main__':unittest.main()
