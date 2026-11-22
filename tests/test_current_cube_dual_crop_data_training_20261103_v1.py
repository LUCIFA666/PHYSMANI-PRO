"""Synthetic CF-to-loss contracts; no actual RGBD/weights/model forward."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.test_dual_current_rgbd_centre_features_20261103 import fixture
from our_mani.local_mpc_20261103.depth_owner_capture_sensors import _labels
from our_mani.current_cube_rgbd_centre_20261103_v1 import data,training,evaluation
from our_mani.current_cube_rgbd_centre_20261103_v1.model import centre_presence_loss


def sample(point=None,rgb_only=False,conflict=False):
    inputs=fixture();cube={'rgb':inputs['rgb'].copy(),'depth_m':inputs['depth_m'].copy()}
    goal=deepcopy(cube)
    if point:
        x,y=point
        if rgb_only:cube['rgb'][0,y,x,0]^=np.uint8(1)
        else:cube['depth_m'][0,y,x]=2.
        if conflict:goal['depth_m'][0,y,x]=2.
    labels=_labels(inputs,{'cube':cube,'goal':goal})
    labels['cube_pose_world']=np.array([0.,0.,1.,0.,0.,0.,1.])
    meta=dict(cube_scene_present=True,absence_from_all_camera_frusta_by_construction=False,construction='in_scope')
    return inputs,labels,meta


class DualCropSupervisionContracts(unittest.TestCase):
    def test_near_and_true_stride2_wide_owner_use_actual_pixels(self):
        x,y,m=sample((128,128));t=data.derive_targets(x,y,m,'shot03')
        self.assertEqual(t['presence_target'],1);self.assertTrue(t['centre_valid'])
        self.assertEqual((t['near_owner_pixels'],t['wide_owner_pixels']),(1,1))
        x,y,m=sample((180,128));t=data.derive_targets(x,y,m,'shot11')
        self.assertEqual(t['presence_target'],1);self.assertEqual((t['near_owner_pixels'],t['wide_owner_pixels']),(0,1))
        x,y,m=sample((181,128));t=data.derive_targets(x,y,m,'shot11')
        self.assertEqual(t['presence_target'],0);self.assertIn('image_unobservable',t['reason'])

    def test_rgb_response_or_conflict_without_qualified_owner_remain_unknown(self):
        for options in (dict(rgb_only=True),dict(conflict=True)):
            x,y,m=sample((128,128),**options);t=data.derive_targets(x,y,m,'shot04')
            self.assertEqual(t['presence_target'],-1);self.assertFalse(t['centre_valid'])

    def test_exact_crop_CF_noresponse_is_not_world_absence(self):
        x,y,m=sample((1,1));t=data.derive_targets(x,y,m,'shot04')
        self.assertEqual(t['presence_target'],0);self.assertFalse(t['physical_existence_label'])
        self.assertEqual(t['crop_RGBD_response_counts'],[0,0]);self.assertTrue(m['cube_scene_present'])
        y['cube_pose_world'][:3]=[0.,0.,1.25];m['construction']='outside_scope'
        t=data.derive_targets(x,y,m,'shot01');self.assertEqual(t['reason'],'constructed_outside_measurement_scope')
        x,y,m=sample((128,128));m['absence_from_all_camera_frusta_by_construction']=True;m['_verified_absence_for_supervision']=True
        with self.assertRaises(ValueError):data.derive_targets(x,y,m,'shot00')
        x,y,m=sample();t=data.derive_targets(x,y,m,'shot00')
        self.assertEqual(t['presence_target'],-1);self.assertFalse(t['centre_valid'])
        self.assertIn('unproven_unknown',t['reason'])

    def test_unknown_targets_zero_both_losses_and_all_prediction_gradients(self):
        x,y,m=sample((128,128),rgb_only=True);t=data.derive_targets(x,y,m,'shot04')
        centre=torch.ones((1,3),requires_grad=True);logit=torch.tensor([1000.],requires_grad=True);score=torch.ones((1,3),requires_grad=True)
        value=centre_presence_loss(dict(centre_world=centre,measurement_availability_logit=logit,uncertainty_score_m=score),
            torch.tensor(t['centre_target_world'][None]),torch.tensor([t['centre_valid']]),torch.tensor([t['presence_target']]))
        self.assertEqual(float(value['total'].detach()),0.);self.assertEqual(float(value['centre'].detach()),0.);self.assertEqual(float(value['presence'].detach()),0.)
        value['total'].backward()
        for a in (centre,logit,score):torch.testing.assert_close(a.grad,torch.zeros_like(a),rtol=0,atol=0)

    def test_reserved_loader_rejects_before_any_array_decode(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'manifest.json';path.write_text('{}')
            with patch.object(data,'load_npz',side_effect=AssertionError('No reserved array decode')):
                with self.assertRaisesRegex(ValueError,'Reserved'):data.DualCropCubeDataset(path,'reserved_fresh_centre_VAL')

    def test_real_capture_null_proof_channels_ignore_only_inapplicable_proofs(self):
        # P0 annotation_metadata always serializes both fields, often null.
        metadata=dict(absence_proof=None,scope_negative_proof=None)
        self.assertEqual(data.construction_proofs(metadata,'shot03'),{})
        for shot in ('shot00','shot02'):self.assertEqual(data.construction_proofs(metadata,shot),{})
        with self.assertRaises(ValueError):data.construction_proofs(metadata,'shot01')
        absence=dict(path='labels/shot00.proof.json',sha256='a'*64)
        self.assertEqual(data.construction_proofs(dict(metadata,absence_proof=absence),'shot00'),{'absence_proof':absence})
        scope=dict(path='labels/shot01.scope_negative_proof.json',sha256='b'*64)
        self.assertEqual(data.construction_proofs(dict(metadata,scope_negative_proof=scope),'shot01'),{'scope_negative_proof':scope})
        with tempfile.TemporaryDirectory()as tmp:
            case=Path(tmp);(case/'labels').mkdir();path=case/'labels/proof.json'
            path.write_text('{"verified":false,"all_four_camera_frusta_outside":false}')
            ref=dict(path='labels/proof.json',sha256=data.sha256(path))
            normalized,bound=data.validated_target_metadata(case,dict(metadata,absence_proof=ref),'shot00')
            self.assertFalse(normalized['_verified_absence_for_supervision']);self.assertEqual(bound[str(path)],ref['sha256'])
            x,y,m=sample();m.update(normalized)
            self.assertEqual(data.derive_targets(x,y,m,'shot00')['presence_target'],-1)

    def test_single_fixed_budget_stages_and_global_selection_loss_aggregation(self):
        c=training.fixed_config()
        self.assertEqual((c['epochs'],c['batch_size'],c['head_only_epochs'],c['encoder_and_head_epochs'],c['planned_optimizer_steps']),(40,16,5,35,520))
        self.assertEqual((c['encoder_learning_rate'],c['head_learning_rate']),(3e-5,3e-4))
        calls=iter([dict(centre=torch.tensor(2.),presence=torch.tensor(3.),centre_samples=1,known_presence_samples=2),dict(centre=torch.tensor(6.),presence=torch.tensor(7.),centre_samples=3,known_presence_samples=1)])
        class EvalOnly:
            def eval(self):return self
        with patch.object(training,'tensor_batch',return_value={}),patch.object(training,'loss_batch',side_effect=lambda *args:next(calls)):
            r=training.selection_loss(EvalOnly(),{'presence_target':np.zeros(3)},2,'cpu')
        self.assertEqual(r['centre'],5.);self.assertAlmostEqual(r['presence'],13/3);self.assertAlmostEqual(r['total'],5+13/3)
        self.assertEqual(r['forward_batch_calls'],2)
        r=evaluation.summarize_rows([]);self.assertEqual(r['total'],0);self.assertIsNone(r['known_positive_coverage']);self.assertFalse(r['qualified'])

    def test_invalid_freeze_rejected_before_reserved_loader_or_labels(self):
        with tempfile.TemporaryDirectory()as tmp:
            (Path(tmp)/'prediction_freeze.json').write_text('{"version":"invalid"}')
            with patch.object(evaluation,'DualCropCubeDataset',side_effect=AssertionError('No labels before freeze')):
                with self.assertRaises(ValueError):evaluation.score(tmp)
            self.assertFalse((Path(tmp)/'reserved_score_consumption.request.json').exists())

if __name__=='__main__':unittest.main()
