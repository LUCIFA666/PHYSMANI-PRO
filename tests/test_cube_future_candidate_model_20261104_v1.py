import json
import unittest
import torch
from our_mani.candidate_transformer_cube_20261104_v1.model import (
    CandidateConfig,CubeFutureCandidateTransformer,model_from_spec,parameter_count)
from our_mani.candidate_transformer_cube_20261104_v1.losses import matching_loss,fp32_matching_loss
from our_mani.local_mpc_p4_20261103 import model as original


def features(batch=2):
    future=torch.zeros(batch,17,16);future[:,:,0]=torch.arange(17)/16
    future[:,:,1]=torch.arange(17)*.002
    future[:,:,4:10]=torch.tensor([1.,0.,0.,0.,1.,0.]);future[:,:,10]=.04
    robot=torch.zeros(batch,25);robot[:,17:23]=torch.tensor([1.,0.,0.,0.,-1.,0.]);robot[:,-2:]=1
    return dict(future_tokens=future,robot=robot,geometry=torch.tensor([[.05,.05,.05,.9]]).expand(batch,-1).clone(),
        context_c=torch.full((batch,16),1/8),task_e=torch.full((batch,16),1/8))


def targets(batch=2):
    return (torch.tensor([[.01,.002,-.02]]).expand(batch,-1).clone(),
            torch.diag(torch.tensor([1.,-1.,-1.])).expand(batch,3,3).clone(),torch.full((batch,),.3))


class CubeFutureModel(unittest.TestCase):
    def test_exact_counts_and_json_spec_roundtrip(self):
        self.assertEqual(parameter_count(),3190538)
        self.assertEqual(parameter_count(CandidateConfig(d_model=512,num_layers=6)),18977290)
        cfg=CandidateConfig();self.assertEqual(CandidateConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))),cfg)
        with torch.device('meta'):
            model=CubeFutureCandidateTransformer(cfg);spec=json.loads(json.dumps(model.spec()))
            restored=model_from_spec(spec)
        self.assertEqual(restored.spec(),spec)
        with self.assertRaises(ValueError):CandidateConfig.from_dict(dict(cfg.to_dict(),task_id=1))
        with self.assertRaises(ValueError):CandidateConfig(d_model=True)

    def test_original_FP32_math_and_loss_are_preserved(self):
        torch.manual_seed(20261104)
        cfg=CandidateConfig(d_model=32,num_layers=1,num_heads=4,dropout=.1)
        model=CubeFutureCandidateTransformer(cfg).eval()
        base=original.CandidateTransformerPilot(**cfg.to_dict()).eval();base.load_state_dict(model.state_dict(),strict=True)
        x=features()
        a=model(**x);b=base(**x)
        self.assertEqual(set(a),set(b))
        for key in a:
            if isinstance(a[key],torch.Tensor):self.assertTrue(torch.equal(a[key],b[key]),key)
            else:self.assertEqual(a[key],b[key])
        self.assertIs(matching_loss,original.matching_loss)
        ta=matching_loss(a,*targets());tb=fp32_matching_loss(b,*targets())
        self.assertTrue(torch.equal(ta['per_example'],tb['per_example']))
        self.assertFalse(ta['unmatched_candidates_are_negative_labels']);self.assertFalse(ta['proposal_score_loss_used'])

    def test_default_3M_real_CPU_forward_backward_is_finite(self):
        torch.manual_seed(20261104);model=CubeFutureCandidateTransformer().train()
        output=model(**features());loss=fp32_matching_loss(output,*targets())['per_example'].mean()
        loss.backward()
        for head in (model.position_head,model.time_head):
            self.assertIsNotNone(head.weight.grad);self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()),0)
        self.assertEqual(output['positions'].shape,(2,8,3));self.assertEqual(output['rotation_matrices'].shape,(2,8,3,3))
        self.assertTrue(((output['times']>=.05)&(output['times']<=.8)).all())
        self.assertTrue(torch.equal(output['proposal_logits'],torch.zeros(2,8)))

    def test_Cube_condition_is_fixed_but_motion_context_is_independent(self):
        model=CubeFutureCandidateTransformer(CandidateConfig(d_model=32,num_layers=1,num_heads=4)).eval()
        x=features();x['context_c']=torch.arange(16.).reshape(1,2,8).softmax(-1).reshape(1,16).expand(2,-1)
        self.assertEqual(model(**x)['times'].shape,(2,8))
        x['task_e']=x['context_c']
        with self.assertRaises(ValueError):model(**x)
        x=features();x['future_tokens'][:,0,1]=1
        with self.assertRaises(ValueError):model(**x)
        bad=CubeFutureCandidateTransformer(CandidateConfig(d_model=32,num_layers=1,num_heads=4)).spec()
        bad['output_relative_time_bounds_seconds']=[0,5]
        with self.assertRaises(ValueError):model_from_spec(bad)


if __name__=='__main__':unittest.main()
