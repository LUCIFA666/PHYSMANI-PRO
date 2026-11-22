import unittest
import numpy as np
import torch

from tests.test_world_tcp_command_warmstart_20261104_v1 import inputs
from our_mani.world_tcp_command_warmstart_20261104_v3.model import (
    CandidateConfig,WorldTCPCommandTransformerV3,command_imitation_loss,parameter_count,
)
from our_mani.world_tcp_command_warmstart_20261104_v3.features import pool_recent_maps
from our_mani.world_tcp_command_warmstart_20261104_v3.geometry import nominal_cell_geometry,previous_current_indices
from our_mani.world_tcp_command_warmstart_20261104_v3.resources import ROOT,GLOBAL_LOCKS


def geometry_inputs():
    depths=torch.zeros(1,2,4,256,256)
    K=torch.tensor([[200.,0,128],[0,200.,128],[0,0,1]])[None,None].expand(1,4,3,3).clone()
    T=torch.eye(4)[None,None].expand(1,4,4,4).clone()
    clips=torch.tensor([.01,10.])[None,None].expand(1,4,2).clone()
    present=torch.tensor([[False,True]])
    return depths,K,T,clips,present


def model_inputs():
    value=inputs()
    value.update(fine_visual=torch.randn(1,512,512),fine_geometry=torch.zeros(1,512,6),
                 fine_token_mask=torch.cat((torch.zeros(1,256,dtype=torch.bool),torch.ones(1,256,dtype=torch.bool)),1))
    return value


class FineV3Contracts(unittest.TestCase):
    def test_parameter_count_and_correct_global_lock_inodes(self):
        self.assertEqual(parameter_count(),86772491)
        self.assertEqual([p.parent for p in GLOBAL_LOCKS],[ROOT/'runtime/framework']*2)

    def test_real_nearest_maps_and_missing_previous_never_duplicated(self):
        lengths=torch.tensor([1,2,5]);maps=torch.arange(8*4,dtype=torch.float32)[:,None,None,None].expand(-1,512,16,16).clone()
        pooled=pool_recent_maps(maps,lengths).reshape(3,2,4,64,512)
        self.assertTrue((pooled[0,0]==0).all())
        self.assertEqual(pooled[0,1,:,0,0].tolist(),[0.,1.,2.,3.])
        self.assertEqual(pooled[1,0,:,0,0].tolist(),[4.,5.,6.,7.])
        self.assertEqual(pooled[1,1,:,0,0].tolist(),[8.,9.,10.,11.])
        self.assertEqual(pooled[2,0,:,0,0].tolist(),[24.,25.,26.,27.])
        self.assertEqual(pooled[2,1,:,0,0].tolist(),[28.,29.,30.,31.])
        self.assertEqual(previous_current_indices(lengths).tolist(),[[-1,0],[0,1],[3,4]])

    def test_pixel_centres_subcell_3cm_and_public_calibration_unmodified(self):
        args=geometry_inputs();args[0][0,1,:,80,80]=1.;original=args[1].clone()
        g1,mask=nominal_cell_geometry(*args)
        args[0].zero_();args[0][0,1,:,80,86]=1.;g2,_=nominal_cell_geometry(*args)
        index=256+2*8+2
        self.assertAlmostEqual(float(g2[0,index,0]-g1[0,index,0]),.03,places=6)
        self.assertAlmostEqual(float(g1[0,index,4]),80.5/256,places=6)
        self.assertTrue(torch.equal(args[1],original));self.assertEqual(int(mask.sum()),256)

    def test_invalid_depth_no_fabricated_surface_but_frame_visual_retained(self):
        args=geometry_inputs();args[0].fill_(float('nan'))
        geometry,mask=nominal_cell_geometry(*args)
        self.assertTrue(torch.equal(geometry,torch.zeros_like(geometry)))
        self.assertEqual(int(mask.sum()),256)
        with self.assertRaises(ValueError):previous_current_indices(torch.tensor([0]))

    def test_signedK_and_common_world_SE3_coordinate_equivariance(self):
        args=geometry_inputs();args[1][:,:,0,0]=-200.;args[0][0,1,:,80,80]=1.
        g1,_=nominal_cell_geometry(*args)
        R=torch.tensor([[0.,-1,0],[1.,0,0],[0,0,1.]])
        args[2][:,:,:3,:3]=R;args[2][:,:,:3,3]=torch.tensor([.2,.3,.4])
        g2,_=nominal_cell_geometry(*args);index=256+18
        self.assertTrue(torch.allclose(g2[0,index,:3],R@g1[0,index,:3]+torch.tensor([.2,.3,.4]),atol=1e-6))

    def test_small_BF16_candidate_backward_and_same_unqualified_command_schema(self):
        torch.manual_seed(3)
        model=WorldTCPCommandTransformerV3(CandidateConfig(width=32,layers=1,heads=4,dropout=0.))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=model(**model_inputs())
            loss=command_imitation_loss(result,torch.tensor([[.21,.3,.4]]),torch.eye(3)[None],torch.ones(1))['loss']
        loss.backward();self.assertTrue(torch.isfinite(model.fine_visual_projection.weight.grad).all())
        self.assertIsNone(result['handoff_tau_seconds']);self.assertFalse(result['physical_handoff_qualified'])

    def test_no_fake_previous_or_BF16_geometry_and_padding_ignored(self):
        model=WorldTCPCommandTransformerV3(CandidateConfig(width=32,layers=1,heads=4,dropout=0.)).eval()
        value=model_inputs()
        with torch.no_grad():
            first=model(**value);value['fine_visual'][:,:256]=float('nan')
            value['fine_geometry'][:,:256]=float('nan');second=model(**value)
        self.assertTrue(torch.equal(first['proposal_logits'],second['proposal_logits']))
        value['fine_geometry']=torch.zeros(1,512,6,dtype=torch.bfloat16)
        with self.assertRaises(ValueError):model(**value)
        value['fine_geometry']=torch.zeros(1,512,6);value['fine_token_mask'][0,0]=True
        with self.assertRaises(ValueError):model(**value)


if __name__=='__main__':unittest.main()
