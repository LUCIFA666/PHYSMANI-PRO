import inspect
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from our_mani.world_tcp_command_warmstart_20261104_v1 import data,features
from our_mani.world_tcp_command_warmstart_20261104_v1.model import (
    CandidateConfig,WorldTCPCommandTransformerV1,command_imitation_loss,parameter_count,
)
from scripts import prepare_world_tcp_command_warmstart_20261104_v1 as cli


def inputs():
    robot=torch.zeros(1,14);robot[:,7:10]=torch.tensor([.2,.3,.4]);robot[:,13]=1
    calibration=torch.cat((torch.eye(3).reshape(1,1,9).expand(1,4,9),
        torch.eye(4)[:3].reshape(1,1,12).expand(1,4,12),torch.tensor([.01,10.]).reshape(1,1,2).expand(1,4,2)),-1)
    return dict(visual_memory=torch.randn(1,86,1024),visual_memory_mask=features.causal_memory_mask(torch.tensor([1])),
        current_robot=robot,camera_calibration=calibration,task_e=torch.full((1,16),1/8))


class CommandWarmstartContracts(unittest.TestCase):
    def test_exact_parameters_and_no_fictitious_motion_time_inputs(self):
        self.assertEqual(parameter_count(),85971467)
        names=set(inspect.signature(WorldTCPCommandTransformerV1.forward).parameters)
        self.assertEqual(names,{'self','visual_memory','visual_memory_mask','current_robot','camera_calibration','task_e'})

    def test_small_real_BF16_model_is_command_not_observation_or_handoff(self):
        model=WorldTCPCommandTransformerV1(CandidateConfig(width=32,layers=1,heads=4,dropout=0.))
        with torch.autocast('cpu',dtype=torch.bfloat16):prediction=model(**inputs())
        self.assertEqual(prediction['world_tcp_positions'].shape,(1,8,3))
        self.assertIsNone(prediction['handoff_tau_seconds'])
        self.assertFalse(prediction['physical_handoff_qualified'])
        loss=command_imitation_loss(prediction,torch.tensor([[.21,.3,.4]]),torch.eye(3)[None],torch.ones(1))
        loss['loss'].backward()
        self.assertTrue(torch.isfinite(loss['loss']))
        self.assertEqual(float(loss['time_loss']),0.)
        self.assertFalse(loss['training_scales_are_confidence_intervals'])

    def test_future_padding_is_ignored_and_nonprefix_rejected(self):
        model=WorldTCPCommandTransformerV1(CandidateConfig(width=32,layers=1,heads=4,dropout=0.)).eval()
        batch=inputs()
        with torch.no_grad():
            first=model(**batch)
            batch['visual_memory'][~batch['visual_memory_mask']]=float('nan')
            second=model(**batch)
        self.assertTrue(torch.equal(first['proposal_logits'],second['proposal_logits']))
        self.assertTrue(torch.equal(first['world_tcp_positions'],second['world_tcp_positions']))
        batch['visual_memory_mask'][0,35]=True
        with self.assertRaises(ValueError):model(**batch)
        self.assertEqual(features.causal_memory_mask(torch.tensor([1,2,5])).sum(-1).tolist(),[18,35,86])

    def test_every_exact_command_run_kept_without_result_or_stage_filter(self):
        values=np.zeros((5,1,8),np.float32);values[:,:,6]=1
        values[2:,0,0]=.1;values[4:,0,7]=1
        self.assertEqual(data.command_run_starts(values).tolist(),[0,2,4])
        with self.assertRaises(ValueError):data.command_run_starts(values[:,0])

    def test_labels_are_desired_intent_not_current_object_pose(self):
        target=data.command_target(np.array([.1,.2,.3,0,0,0,1,1.]))
        self.assertFalse(target['observed_object_pose']);self.assertIsNone(target['handoff_tau_seconds'])
        self.assertEqual(target['time_loss_weight'],0.)
        self.assertIn('not_observed_TCP',target['semantics'])
        with self.assertRaises(ValueError):data.command_target(np.zeros(8))

    def test_mixture_NLL_matches_fixed_training_scales(self):
        p=torch.zeros(1,8,3,requires_grad=True)
        logits=torch.arange(8,dtype=torch.float32)[None].requires_grad_()
        opening=torch.zeros(1,8,requires_grad=True)
        predicted=dict(world_tcp_positions=p,world_tcp_rotations=torch.eye(3).expand(1,8,3,3),
            opening_logits=opening,opening_intent=opening.sigmoid(),proposal_logits=logits)
        result=command_imitation_loss(predicted,torch.tensor([[.03,0,0]]),torch.eye(3)[None],torch.ones(1))
        self.assertAlmostEqual(float(result['loss'].detach()),.5+np.log(2),places=6)
        result['loss'].backward();self.assertIsNotNone(logits.grad)
        self.assertFalse(result['physical_handoff_supervised'])

    def test_default_does_not_read_payload_or_load_a_selected_encoder(self):
        with mock.patch.object(cli,'metadata_check',return_value=dict(NN_forwards=0,RGBD_payloads_read=0))as check, \
             mock.patch.object(cli,'parameter_count',return_value=85971467), \
             mock.patch.object(cli,'prepare_index',side_effect=AssertionError('payload')):
            self.assertEqual(cli.main(['--manifest','synthetic']),0);check.assert_called_once()
        with self.assertRaises(ValueError):features.frozen_encoder_binding('automatic_best_or_compatible')


if __name__=='__main__':unittest.main()
