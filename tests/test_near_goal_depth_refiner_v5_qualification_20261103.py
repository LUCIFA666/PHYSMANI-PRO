"""CPU contracts only; never forward a real calibration sample."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
import torch

from our_mani.near_goal_depth_refiner_20261103_v5 import qualification as q
from our_mani.near_goal_depth_refiner_20261103_v5 import runtime as r
from our_mani.near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from scripts import qualify_near_goal_depth_refiner_20261103_v5 as cli


class QualificationTests(unittest.TestCase):
    def test_one_use_claim_survives_method_output_and_manifest_alias_changes(self):
        with TemporaryDirectory() as t:
            base=Path(t);capture=base/'capture';capture.mkdir()
            manifest=base/'data'/'manifest.json';manifest.parent.mkdir();manifest.write_text('{}')
            request,sibling=q.claim_calibration_once(manifest,'a'*64,{'method':'b'*64},base/'out',capture_root=capture)
            self.assertEqual(request['sha256'],sibling['sha256'])
            self.assertTrue(Path(request['path']).is_file())
            alias=base/'alias'/'manifest.json';alias.parent.mkdir();alias.write_text('{}')
            with self.assertRaises(FileExistsError):
                q.claim_calibration_once(alias,'c'*64,{'method':'d'*64},base/'changed',capture_root=capture)
            self.assertFalse((base/'changed').exists())

    def test_default_cli_only_checks_and_does_not_qualify(self):
        with patch.object(cli,'check_qualification_inputs',return_value={'complete':True}) as check,\
             patch.object(cli,'qualify_selected_checkpoint',side_effect=AssertionError('consume forbidden')) as execute:
            self.assertEqual(cli.main(['--selected','x','--expected-sha256','a'*64,'--fresh-manifest','y']),0)
            check.assert_called_once();execute.assert_not_called()

    def test_state_digest_is_order_invariant_and_preserves_dtype_and_values(self):
        a={'z':torch.tensor([1.]),'a':torch.tensor([2.])}
        self.assertEqual(q.state_digest(a),q.state_digest(dict(reversed(list(a.items())))))
        self.assertNotEqual(q.state_digest(a),q.state_digest({k:v.double() for k,v in a.items()}))
        with self.assertRaises(ValueError):q.state_digest({'a':torch.tensor([float('nan')])})

    def test_selected_only_payload_is_never_runtime_qualified(self):
        with TemporaryDirectory() as t:
            path=Path(t)/'selected.pt';torch.save({'version':q.FIT_VERSION,'metadata':{'model_qualified':False}},path)
            with self.assertRaisesRegex(ValueError,'fresh4-qualified'):
                r.CurrentCubeDepthRefinerV5(path,q.sha256(path),allow_unqualified_diagnostic=True)

    def test_failed_runtime_diagnostic_never_calls_measurement_or_threshold_fallback(self):
        runtime=object.__new__(r.CurrentCubeDepthRefinerV5)
        runtime.qualified=False;runtime.threshold=None;runtime.model=object();runtime._sources={}
        runtime.calibration_environment={};runtime.checkpoint_sha256='a'*64
        probabilities=np.zeros((4,2,64,64),np.float32);probabilities[:,0]=1
        with patch.object(r,'assert_cpu_prediction_contract'),patch.object(r,'qualification_source_map',return_value={}),\
             patch.object(r,'make_near_goal_crops',return_value={'features':np.zeros((4,9,64,64),np.float32)}),\
             patch.object(r,'predict_probabilities',return_value=probabilities),\
             patch.object(r,'probability_to_observation',side_effect=AssertionError('no threshold fallback')) as decode:
            result=runtime.predict({})
            self.assertIsNone(result['centre']);self.assertFalse(result['refiner_available'])
            self.assertIs(result['probabilities'],probabilities);decode.assert_not_called()

    def test_qualification_freezes_eligibility_then_exact32_predictions_before_scoring(self):
        with TemporaryDirectory() as t:
            base=Path(t);capture=base/'capture';capture.mkdir();dataset_dir=base/'dataset';dataset_dir.mkdir()
            manifest=dataset_dir/'manifest.json';manifest.write_text('{}')
            selected=base/'selected.pt';selected.write_text('immutable selected')
            fit=base/'fit_status.json';fit.write_text('{}')
            output=base/'qualification';order=[];features=np.zeros((4,9,64,64),np.float32)
            samples=[dict(parent_id=p,scene_id=p+'/'+s,role='calib',features=features)
                     for p in q.FRESH_PARENT_IDS for s in q.SHOT_IDS]
            class Dataset:
                path=manifest;bound_pointers={str(manifest):q.sha256(manifest)}
                records=[{'inputs_path':capture/s['parent_id']/'inputs'/(s['scene_id'].split('/')[-1]+'.npz')}for s in samples]
                def __len__(self):return 32
                def __getitem__(self,i):return samples[i]
            def eligibility(actual):
                self.assertEqual(actual,samples);self.assertNotIn('forward',order);order.append('eligibility')
                return {'protocol':'synthetic-only','records':[]}
            def predict(model,x):
                self.assertTrue((output/'eligibility_before_predictions.json').exists())
                self.assertIs(x,features);order.append('forward')
                p=np.zeros((4,2,64,64),np.float32);p[:,0]=1;return p
            def score(actual,probs,threshold,eligible):
                self.assertEqual(order.count('forward'),32)
                self.assertTrue((output/'prediction_freeze.json').exists())
                self.assertEqual(probs.shape,(32,4,2,64,64));order.append('score')
                return {'threshold':threshold}
            model=TinyDepthInfluenceUNet()
            payload={'model_state_dict':model.state_dict(),'metadata':{'identity':{},'selected_epoch':1}}
            evidence={'selected':q.pointer(selected),'fit_status':q.pointer(fit),'evidence':{}}
            threads=torch.get_num_threads()
            try:
                with patch.object(q,'validate_selected_checkpoint',return_value=(payload,evidence)),\
                     patch.object(q,'qualification_source_map',return_value={'method':'b'*64}),\
                     patch.object(q,'check_library_environment'),patch.object(q,'Fresh4CalibrationDataset',return_value=Dataset()),\
                     patch.object(q,'freeze_legacy_eligibility',side_effect=eligibility),\
                     patch.object(q,'assert_cpu_prediction_contract',return_value={'torch_version':'synthetic','numpy_version':'synthetic'}),\
                     patch.object(q,'predict_probabilities',side_effect=predict),\
                     patch.object(q,'_verify_unchanged'),patch.object(q,'evaluate_fresh_probabilities',side_effect=score),\
                     patch.object(q,'choose_fresh_calibrated_threshold',return_value={'calibration_passed':False,'selected_threshold':None}):
                    report=q.qualify_selected_checkpoint(selected,q.sha256(selected),manifest,output)
            finally:torch.set_num_threads(threads)
            self.assertEqual(order,['eligibility']+['forward']*32+['score']*5)
            self.assertFalse(report['selection']['calibration_passed'])
            self.assertEqual(selected.read_text(),'immutable selected')
            self.assertTrue((capture/q.CONSUMPTION_NAME).exists())
            status=json.loads((output/'status.json').read_text());self.assertFalse(status['model_qualified'])
            saved=torch.load(output/'calibrated.pt',map_location='cpu',weights_only=True)
            self.assertIsNone(saved['metadata']['selected_threshold'])
            self.assertEqual(q.state_digest(saved['model_state_dict']),q.state_digest(payload['model_state_dict']))


if __name__=='__main__':unittest.main()
