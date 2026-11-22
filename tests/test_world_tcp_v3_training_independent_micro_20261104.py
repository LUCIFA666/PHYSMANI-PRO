"""Pure tensor logical-batch/selection boundaries, no real models/cache/GPU."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from our_mani.world_tcp_command_warmstart_20261104_v3 import training
from our_mani.world_tcp_command_warmstart_20261104_v3.model import command_imitation_loss


class TensorMixture(nn.Module):
    def __init__(self):
        super().__init__();self.delta=nn.Parameter(torch.tensor(.013));self.open=nn.Parameter(torch.tensor(.1))
        self.logits=nn.Parameter(torch.arange(8,dtype=torch.float32)/10)
    def forward(self,x):
        b=len(x);positions=(x[:,None,None]+self.delta).expand(b,8,3)
        opening=self.open.expand(b,8)
        return dict(world_tcp_positions=positions,world_tcp_rotations=torch.eye(3).expand(b,8,3,3),
            opening_logits=opening,opening_intent=opening.sigmoid(),proposal_logits=self.logits.expand(b,8))


def samples(n=16):
    return [dict(inputs={'x':torch.tensor(float(i)/200)},target=dict(position=torch.tensor([.01*i,.002*i,-.003*i]),
        rotation=torch.eye(3),opening=torch.tensor(float(i%2))))for i in range(n)]


def batch(items,device,**kwargs):
    return {name:{key:torch.stack([item[name][key]for item in items])for key in items[0][name]}for name in ('inputs','target')}


class Micro(unittest.TestCase):
    def test_micro4_or2_has_exact_logical16_weight_and_gradient(self):
        torch.set_num_threads(2);data=samples();baseline=TensorMixture()
        all_batch=batch(data,'cpu');loss=command_imitation_loss(baseline(**all_batch['inputs']),**all_batch['target'])['loss']
        loss.backward()
        for micro in (4,2):
            model=TensorMixture();model.load_state_dict(deepcopy(baseline.state_dict()))
            with patch.object(training,'tensor_batch',batch):
                value=training.logical_backward(model,data,list(range(16)),'cpu',micro)
            self.assertAlmostEqual(value['loss'],float(loss.detach()),places=5)
            self.assertEqual(value['model_forward_calls'],16//micro)
            for actual,expected in zip(model.parameters(),baseline.parameters()):
                torch.testing.assert_close(actual.grad,expected.grad,atol=1e-5,rtol=1e-6)

    def test_no_partial_batch_or_unadmitted_micro_silently_trains(self):
        with self.assertRaises(ValueError):training.logical_backward(TensorMixture(),samples(),list(range(15)),'cpu',4)
        with self.assertRaises(ValueError):training.logical_backward(TensorMixture(),samples(),list(range(16)),'cpu',8)

    def test_selection_keeps_all5168_and_does_not_backward(self):
        class Data:
            parent_batches=[list(range(16*i,16*(i+1)))for i in range(323)]
            def __getitem__(self,i):
                if torch.is_inference_mode_enabled():raise ValueError('Cache decoding must remain outside inference mode')
                return samples()[i%16]
        model=TensorMixture()
        with patch.object(training,'tensor_batch',batch):
            value=training.selection_loss(model,Data(),'cpu',4)
        self.assertEqual(value['rows'],5168);self.assertEqual(value['model_forward_calls'],1292)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
