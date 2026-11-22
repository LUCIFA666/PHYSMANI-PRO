"""Pure fusion/loss and early guards; no checkpoint/sample/CNN forward."""
import inspect
from unittest.mock import patch
import unittest
import torch
from our_mani.current_cube_rgbd_centre_20261103_v1 import model as m


class CurrentCubeCentreHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=m.DualCropCubeCentreHeadV1()

    def test_explicit_encoder_initialization_and_public_only_dual_input_signature(self):
        f=self.fixture
        self.assertIsNone(f.spec()['initialization'])
        self.assertFalse(f.spec()['model_qualified']);self.assertFalse(f.spec()['runtime_or_control_admitted'])
        self.assertFalse(f.spec()['extent_or_coverage_qualified'])
        self.assertEqual(set(inspect.signature(f.forward).parameters),
                         {'features','wide_features','K','camera_rotations','initial_sphere'})
        with patch.object(f.e0,'forward',side_effect=AssertionError('No CNN')):
            with self.assertRaises(ValueError):f(features=None,wide_features=None,K=None,camera_rotations=None,initial_sphere=None)
            with self.assertRaises(TypeError):f(features=None,wide_features=None,K=None,camera_rotations=None,initial_sphere=None,labels={})
        self.assertTrue(all(not p.requires_grad for n,p in f.named_parameters()if n.startswith(m.ENCODER_PREFIXES)))
        self.assertTrue(all(p.requires_grad for n,p in f.named_parameters()if not n.startswith(m.ENCODER_PREFIXES)))

    def test_selected_encoder_prefix_extraction_never_loads_old_decoder_or_new_heads(self):
        shape={p+'weight':(2,3)for p in m.ENCODER_PREFIXES}
        state={k:torch.zeros(v)for k,v in shape.items()};state['d0.weight']=torch.ones(1);state['head.weight']=torch.ones(1)
        selected=m._selected_encoder_tensors(state,shape)
        self.assertEqual(set(selected),set(shape));self.assertTrue(all(selected[k]is not state[k]for k in shape))
        bad=dict(state);bad.pop(next(iter(shape)))
        with self.assertRaises(ValueError):m._selected_encoder_tensors(bad,shape)
        bad=dict(state);bad[next(iter(shape))]=torch.full((2,3),float('nan'))
        with self.assertRaises(ValueError):m._selected_encoder_tensors(bad,shape)

    def test_loader_wrong_sha_or_missing_source_rejects_before_torch_load(self):
        with patch.object(torch,'load',side_effect=AssertionError('No actual checkpoint load')):
            with self.assertRaises(ValueError):self.fixture.load_selected_v5_encoder('/missing/selected.pt','0'*64)
            with self.assertRaises(FileNotFoundError):self.fixture.load_selected_v5_encoder('/missing/selected.pt',m.SELECTED_V5_SHA256)
        self.assertIsNone(self.fixture.spec()['initialization'])

    def test_dual_view_rotation_fusion_and_no_depth_do_not_fabricate_evidence(self):
        theta=torch.tensor(.4);c,s=theta.cos(),theta.sin()
        R=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        rotations=torch.eye(3).repeat(1,8,1,1);rotations[:,1::2]=R
        world=torch.tensor([.04,-.03,.01]).repeat(1,8,1)
        camera=torch.einsum('bvji,bvj->bvi',rotations,world)
        scale=torch.full_like(camera,.01);logit=torch.zeros(1,8);valid=torch.ones(1,8,dtype=torch.bool)
        centre,score,weights=m.fuse_public_views(camera,scale,logit,rotations,valid)
        torch.testing.assert_close(centre,world[:,0],atol=1e-7,rtol=0)
        self.assertTrue(torch.isfinite(score).all());self.assertTrue((score>0).all())
        torch.testing.assert_close(weights.sum(1),torch.ones(1))
        Q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        moved,_,_=m.fuse_public_views(camera,scale,logit,Q@rotations,valid)
        torch.testing.assert_close(moved,centre@Q.T,atol=1e-7,rtol=0)
        empty,empty_score,empty_weights=m.fuse_public_views(camera,scale,logit,rotations,torch.zeros_like(valid))
        torch.testing.assert_close(empty,torch.zeros_like(empty));torch.testing.assert_close(empty_weights,torch.zeros_like(empty_weights))
        self.assertTrue(torch.isfinite(empty_score).all())
        torch.testing.assert_close(empty_score,torch.full_like(empty_score,m.UNCERTAINTY_SCALE_M))

    def predictions(self,n):
        return dict(centre_world=torch.zeros(n,3,requires_grad=True),
                    measurement_availability_logit=torch.zeros(n,requires_grad=True),
                    uncertainty_score_m=torch.full((n,3),.01,requires_grad=True))

    def test_presence_and_centre_masks_are_separate_and_unknown_never_becomes_absent(self):
        output=self.predictions(3);target=torch.tensor([[.02,0.,0.],[float('nan')]*3,[float('nan')]*3])
        loss=m.centre_presence_loss(output,target,torch.ones(3,dtype=torch.bool),torch.tensor([1,0,-1]))
        self.assertEqual(loss['centre_samples'],1);self.assertEqual(loss['known_presence_samples'],2);self.assertEqual(loss['unknown_samples'],1)
        loss['total'].backward()
        for value in output.values():self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(output['centre_world'].grad[0].abs().sum()>0)
        torch.testing.assert_close(output['centre_world'].grad[1:],torch.zeros(2,3))
        self.assertTrue(output['measurement_availability_logit'].grad[:2].abs().sum()>0)
        self.assertEqual(float(output['measurement_availability_logit'].grad[2]),0.)
        torch.testing.assert_close(output['uncertainty_score_m'].grad[1:],torch.zeros(2,3))

    def test_all_unknown_has_finite_zero_losses_and_zero_gradients(self):
        output=self.predictions(2)
        loss=m.centre_presence_loss(output,torch.full((2,3),float('nan')),torch.ones(2,dtype=torch.bool),torch.full((2,),-1))
        self.assertEqual(float(loss['total'].detach()),0.);self.assertEqual(float(loss['centre'].detach()),0.);self.assertEqual(float(loss['presence'].detach()),0.)
        loss['total'].backward()
        for value in output.values():
            self.assertTrue(torch.isfinite(value.grad).all());torch.testing.assert_close(value.grad,torch.zeros_like(value))
        self.assertFalse(loss['uncertainty_qualified'])


if __name__=='__main__':unittest.main()
