"""CPU-only signal/gradient/contract checks of the independent normalized encoder."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from our_mani.semantic_task_e_large_20261104_v2.model import EncoderConfig,LargeSemanticTaskEncoderV2,parameter_count


class NormalizedLargeSemanticTests(unittest.TestCase):
 def model(self):
  torch.manual_seed(20261104)
  return LargeSemanticTaskEncoderV2(EncoderConfig(width=32,layers=1,heads=4,cnn_base=4,dropout=0.))

 def test_large_meta_count_and_all_conv_norms_token_norms(self):
  self.assertEqual(parameter_count(EncoderConfig()),207855640)
  model=self.model();conv=[i for i,m in enumerate(model.camera_encoder)if isinstance(m,nn.Conv2d)]
  self.assertEqual(len(conv),8)
  self.assertTrue(all(isinstance(model.camera_encoder[i+1],nn.GroupNorm)for i in conv))
  self.assertIsInstance(model.visual_token_norm,nn.LayerNorm);self.assertIsInstance(model.robot_token_norm,nn.LayerNorm)
  self.assertFalse(model.spec()['v1_checkpoint_compatible']);self.assertEqual(model.spec()['classifier_input'],'preSimNorm_raw16')

 def test_decoder_identity_and_high_confidence_raw_signal_not_simplex_limited(self):
  model=self.model();expected=torch.cat((torch.eye(8),torch.eye(8)),1)*.5
  self.assertTrue(torch.equal(model.operation_classifier.weight,expected));self.assertTrue(torch.equal(model.operation_classifier.bias,torch.zeros(8)))
  raw=torch.full((1,2,8),-20.);raw[:,:,3]=20.
  p=model.operation_classifier(raw.reshape(1,16)).softmax(-1)
  self.assertGreater(float(p[0,3]),.99999)

 def test_wrong_saturated_raw_signal_keeps_classification_and_aux_gradient(self):
  model=self.model();raw=torch.full((1,2,8),-1000.);raw[:,:,0]=1000.;raw.requires_grad_();label=torch.tensor([1])
  main=F.cross_entropy(model.operation_classifier(raw.reshape(1,16)),label)
  aux=(F.cross_entropy(raw[:,0],label)+F.cross_entropy(raw[:,1],label))/2
  grad_main=torch.autograd.grad(main,raw,retain_graph=True)[0]
  grad_all=torch.autograd.grad(main+.5*aux,raw)[0]
  self.assertTrue(torch.isfinite(main+.5*aux));self.assertGreater(float(grad_main.abs().max()),.49);self.assertGreater(float(grad_all.abs().max()),.74)

 def test_CPU_BF16_full_small_model_finite_e16_raw_and_backward(self):
  torch.set_num_threads(2);model=self.model();rgbd=torch.randn(2,5,4,4,64,64);robot=torch.randn(2,5,15);robot[:,:,-1]=0.;lengths=torch.tensor([5,3]);labels=torch.tensor([1,5])
  with torch.autocast('cpu',dtype=torch.bfloat16):output=model(rgbd=rgbd,robot_history=robot,lengths=lengths)
  self.assertEqual(output['task_e'].shape,(2,16));self.assertEqual(output['raw_group_logits'].dtype,torch.float32)
  self.assertTrue(torch.allclose(output['task_e'].reshape(2,2,8).sum(-1),torch.ones(2,2),atol=1e-6))
  self.assertTrue(all(torch.isfinite(x).all()for x in output.values()))
  loss=F.cross_entropy(output['operation_logits'].float(),labels)+.5*(F.cross_entropy(output['raw_group_logits'][:,0],labels)+F.cross_entropy(output['raw_group_logits'][:,1],labels))/2
  loss.backward();grads=[p.grad for p in model.parameters()if p.grad is not None]
  self.assertTrue(all(torch.isfinite(g).all()for g in grads));self.assertGreater(float(model.camera_encoder[0].weight.grad.norm()),0.)

 def test_padding_does_not_change_current_public_result_or_admit_bad_open(self):
  model=self.model().eval();rgbd=torch.randn(1,5,4,4,64,64);robot=torch.randn(1,5,15);robot[:,:,-1]=0.;lengths=torch.tensor([3])
  with torch.no_grad():a=model(rgbd=rgbd,robot_history=robot,lengths=lengths)
  rgbd[:,3:]=float('nan');robot[:,3:]=float('nan')
  with torch.no_grad():b=model(rgbd=rgbd,robot_history=robot,lengths=lengths)
  self.assertTrue(torch.equal(a['operation_logits'],b['operation_logits']))
  robot[0,0,-1]=1.
  with self.assertRaisesRegex(ValueError,'gripper'):model(rgbd=rgbd,robot_history=robot,lengths=lengths)


if __name__=='__main__':unittest.main()
