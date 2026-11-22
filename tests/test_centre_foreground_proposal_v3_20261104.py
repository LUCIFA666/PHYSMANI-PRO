"""Synthetic CPU prototype contracts; no real weights/data, fit or qualification."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_dual_current_rgbd_centre_features_20261103 import fixture
from tests.test_current_cube_dual_crop_data_training_20261103_v1 import sample
from our_mani.current_cube_rgbd_centre_20261103_v1.features import public_model_arrays as original_arrays
from our_mani.current_cube_rgbd_centre_large_20261104_v3 import data, model, losses, source_map

torch.set_num_threads(2)


def tensors():
    return {k:torch.from_numpy(v[None].copy()) for k,v in data.public_model_arrays(fixture()).items()}


def geometry_args(values, probability):
    return dict(features=values['features'],wide_features=values['wide_features'],
        camera_rotations=values['camera_rotations'],initial_sphere=values['initial_sphere'],
        source_pixels_uv=values['source_pixels_uv'],foreground_probability=probability)


class ForegroundSurfacePrototype(unittest.TestCase):
    def test_public_arrays_exact_five_originals_and_no_projection_padding(self):
        inputs=fixture();before=deepcopy(inputs)
        arrays=data.public_model_arrays(inputs);old=original_arrays(inputs)
        self.assertEqual(set(arrays),data.PUBLIC_KEYS)
        for k in old:np.testing.assert_array_equal(arrays[k],old[k])
        self.assertEqual(arrays['source_pixels_uv'].shape,(2,4,64,64,2))
        np.testing.assert_array_equal(arrays['source_pixels_uv'][0,0,32,32],[128,128])
        np.testing.assert_array_equal(arrays['source_pixels_uv'][1,0,32,32],[128,128])
        for k in inputs:np.testing.assert_array_equal(inputs[k],before[k])
        inputs['public_sphere_center'][2]=-1
        behind=data.public_model_arrays(inputs)
        self.assertTrue((behind['source_pixels_uv']<0).all())
        self.assertFalse(behind['features'][:,8].any())
        v={k:torch.from_numpy(a[None].copy())for k,a in behind.items()}
        out=model.surface_proposal(**geometry_args(v,torch.ones(1,2,4,64,64)))
        self.assertFalse(out['proposal_supported'].item());self.assertEqual(out['unique_supported_pixels'].item(),0)

    def test_pixel_CF_mapping_unknown_and_scope_are_independent(self):
        inputs,labels,meta=sample((128,128))
        arrays=data.public_model_arrays(inputs)
        targets=data.pixel_targets(inputs,labels,arrays['source_pixels_uv'])
        self.assertEqual(targets[0,0,32,32],1);self.assertEqual(targets[1,0,32,32],1)
        self.assertTrue(np.isin(targets,[0,1,255]).all())
        labels['cube_pose_world'][:3]=[0,0,1.25];meta['construction']='outside_scope'
        scope=data.derive_targets(inputs,labels,meta,'shot01')
        self.assertEqual(scope['presence_target'],0);self.assertFalse(scope['centre_valid'])
        self.assertEqual(data.pixel_targets(inputs,labels,arrays['source_pixels_uv'])[0,0,32,32],1)
        x,y,_=sample((128,128),conflict=True)
        self.assertEqual(data.pixel_targets(x,y,arrays['source_pixels_uv'])[0,0,32,32],255)
        inputs['K'][:,0,2]=5.;inputs['K'][:,1,2]=7.
        edge=data.public_model_arrays(inputs)
        target=data.pixel_targets(inputs,labels,edge['source_pixels_uv'])
        valid=np.stack((edge['features'][:,8],edge['wide_features'][:,8])).astype(bool)
        self.assertTrue(np.all(target[~valid]==255))

    def test_unique_UV_camera_balance_and_detached_probabilities(self):
        v=tensors();v['features'][:,:,4:7]=0;v['wide_features'][:,:,4:7]=0
        v['features'][:,0,4]=1.;v['wide_features'][:,0,4]=1.
        v['features'][:,1,5]=2.;v['wide_features'][:,1,5]=2.
        p=torch.zeros(1,2,4,64,64)
        p[0,0,0,10,10]=.8;p[0,1,0,21,21]=.99  # Same original UV; fixed near wins.
        p[0,0,1,10,11]=.7;p[0,0,1,10,12]=.7
        p.requires_grad_()
        g=model.surface_proposal(**geometry_args(v,p))
        self.assertTrue(g['proposal_supported'].item());self.assertEqual(g['current_views'].item(),2)
        self.assertEqual(g['unique_supported_pixels'].item(),3)
        torch.testing.assert_close(g['proposal_world'],v['initial_sphere']+torch.tensor([[.05,.1,0]]),atol=1e-7,rtol=0)
        self.assertFalse(g['detached_pixel_weights'].requires_grad)
        self.assertFalse(g['proposal_world'].requires_grad)
        self.assertEqual(g['detached_pixel_weights'][0,1,0,21,21].item(),0.)

    def test_current_depth_two_views_and_strict_tie_unknown(self):
        v=tensors();p=torch.full((1,2,4,64,64),.5)
        for q in (p,torch.zeros_like(p)):
            g=model.surface_proposal(**geometry_args(v,q));self.assertFalse(g['proposal_supported'].item())
            torch.testing.assert_close(g['proposal_world'],v['initial_sphere'])
        p[0,0,0,2,2]=.9
        self.assertFalse(model.surface_proposal(**geometry_args(v,p))['proposal_supported'].item())
        p[0,0,1,2,2]=.9
        self.assertTrue(model.surface_proposal(**geometry_args(v,p))['proposal_supported'].item())
        v['features'][:,1,7]=0;v['wide_features'][:,1,7]=0
        self.assertFalse(model.surface_proposal(**geometry_args(v,p))['proposal_supported'].item())
        wrong=deepcopy(v);wrong['source_pixels_uv'][0,1,0,1,1,0]+=1
        with self.assertRaisesRegex(ValueError,'lattice'):model.surface_proposal(**geometry_args(wrong,p))
        with self.assertRaises(ValueError):model.surface_proposal(**geometry_args(v,torch.full_like(p,float('nan'))))

    def test_proposal_common_SE3_and_improper_rotation_rejection(self):
        v=tensors();p=torch.ones(1,2,4,64,64)
        v['features'][:,:,4:7]=torch.tensor([1.,2.,3.])[None,None,:,None,None]
        v['wide_features'][:,:,4:7]=v['features'][:,:,4:7]
        a=model.surface_proposal(**geometry_args(v,p))['proposal_world']
        Q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        move=torch.tensor([[.4,-.2,.1]])
        v['camera_rotations']=Q@v['camera_rotations'];v['initial_sphere']=v['initial_sphere']@Q.T+move
        b=model.surface_proposal(**geometry_args(v,p))['proposal_world']
        torch.testing.assert_close(b,a@Q.T+move,atol=2e-7,rtol=0)
        v['camera_rotations'][0,0,0,0]=3
        with self.assertRaisesRegex(ValueError,'Proper'):model.surface_proposal(**geometry_args(v,p))

    def test_decoder_exact_extraction_and_early_initialization_guard(self):
        shapes={prefix+'weight':(2,3)for prefix in model.DECODER_PREFIXES}
        state={k:torch.zeros(s)for k,s in shapes.items()};state['centre_head.weight']=torch.ones(5)
        result=model._decoder_tensors(state,shapes)
        self.assertEqual(set(result),set(shapes));self.assertTrue(all(result[k]is not state[k]for k in shapes))
        for mutated in ({k:v for k,v in state.items()if k!=next(iter(shapes))},
                        dict(state,**{next(iter(shapes)):torch.full((2,3),float('nan'))})):
            with self.assertRaises(ValueError):model._decoder_tensors(mutated,shapes)
        f=model.LargeDualCropCubeCentreV3(model.CentreConfig(width=32,layers=1,heads=4))
        with patch.object(f.e0,'forward',side_effect=AssertionError('No CNN before initialization')):
            with self.assertRaises(ValueError):f(**tensors())
        with patch.object(torch,'load',side_effect=AssertionError('No checkpoint load')):
            with self.assertRaises(ValueError):f.load_selected_v5_components('/missing','0'*64)

    def test_loss_unknown_padding_and_scope_negative_foreground_gradient(self):
        n=2
        out=dict(centre_world=torch.zeros(n,3,requires_grad=True),
            measurement_availability_logit=torch.zeros(n,requires_grad=True),
            uncertainty_score_m=torch.full((n,3),.05,requires_grad=True),
            pixel_logits=torch.zeros(n,2,4,2,64,64,requires_grad=True))
        labels=torch.full((n,2,4,64,64),255,dtype=torch.uint8)
        labels[1,0,0,0,0]=1
        crop=torch.zeros_like(labels,dtype=torch.bool);crop[1,0,0,0,0]=True
        loss=losses.prototype_loss(out,centre_target_world=torch.full((n,3),float('nan')),
            centre_valid=torch.zeros(n,dtype=torch.bool),presence_target=torch.tensor([-1,0]),
            pixel_labels=labels,inside_original_image=crop)
        self.assertEqual(loss['centre_samples'],0);self.assertEqual(loss['unknown_presence_samples'],1)
        self.assertEqual(loss['pixel_unknown_abstention'].item(),0.)
        loss['total'].backward()
        self.assertEqual(out['measurement_availability_logit'].grad[0].item(),0.)
        self.assertGreater(out['measurement_availability_logit'].grad[1].item(),0.)
        self.assertLess(out['pixel_logits'].grad[1,0,0,1,0,0].item(),0.)
        self.assertEqual(out['pixel_logits'].grad[0].abs().sum().item(),0.)
        self.assertEqual(out['centre_world'].grad.abs().sum().item(),0.)

    def test_synthetic_full_forward_no_decoder_gradient_from_centre_then_pixel_backward(self):
        torch.manual_seed(11)
        f=model.LargeDualCropCubeCentreV3(model.CentreConfig(width=32,layers=1,heads=4,dropout=0.))
        # Unqualified random unit fixture only; no real checkpoint/certificate is created.
        f._encoder_initialization={'synthetic_random_unit_fixture':True}
        f._decoder_initialization={'synthetic_random_unit_fixture':True}
        with torch.no_grad():f.head.weight.zero_();f.head.bias.copy_(torch.tensor([0.,2.]))
        out=f(**tensors())
        self.assertEqual(out['pixel_logits'].shape,(1,2,4,2,64,64))
        self.assertFalse(out['model_qualified']);self.assertFalse(out['position_measurement_qualified'])
        self.assertTrue(torch.isfinite(out['centre_world']).all());self.assertTrue(out['proposal_supported'].item())
        out['centre_world'].sum().backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for n,p in f.named_parameters()if n.startswith(model.DECODER_PREFIXES)))
        self.assertGreater(f.centre_head.weight.grad.abs().sum().item(),0.)
        out['pixel_logits'].float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for n,p in f.named_parameters()if n.startswith(model.DECODER_PREFIXES)))

    def test_default_capacity_explicit_sources_and_no_reserved_loader(self):
        self.assertEqual(model.parameter_count(),25409695+48738)
        self.assertEqual(set(inspect.signature(model.LargeDualCropCubeCentreV3.forward).parameters)-{'self'},data.PUBLIC_KEYS)
        sources=source_map()
        self.assertIn(str(__import__('pathlib').Path(model.__file__).resolve()),sources)
        self.assertTrue(any('v5/losses.py'in name for name in sources))
        with patch.object(data,'DualCropCubeDataset',side_effect=AssertionError('No reserved constructor')):
            with self.assertRaises(ValueError):data.ForegroundProposalDataset('/missing','reserved_fresh_centre_VAL')


if __name__=='__main__':unittest.main()
