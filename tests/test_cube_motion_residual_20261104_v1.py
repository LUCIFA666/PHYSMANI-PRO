"""Synthetic CPU contracts; no real weights, capture, optimizer or simulator."""
import json
from collections import deque
import unittest

import numpy as np
import torch

from our_mani.cube_motion_residual_20261104_v1.model import (
    CubeMotionResidualGRU, MotionConfig, model_from_spec, public_inputs)
from our_mani.cube_motion_residual_20261104_v1.losses import physical_motion_loss
from our_mani.cube_motion_residual_20261104_v1.data import compile_episode
from our_mani.cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
from our_mani.multitask.context_se3 import SE3ContextNetwork


def sample(lengths=(5,20)):
    count=20
    p=torch.zeros(len(lengths),count,3)
    t=torch.zeros(len(lengths),count,dtype=torch.float64)
    c=torch.zeros(len(lengths),count)
    for i,n in enumerate(lengths):
        q=torch.arange(n)*.05
        p[i,:n]=torch.tensor([.2,-.1,.78])+q[:,None]*torch.tensor([.18,-.11,0.])
        t[i,:n]=2.+torch.arange(n,dtype=torch.float64)*.05
        c[i,:n]=.9
    return dict(positions=p,timestamps=t,confidence=c,lengths=torch.tensor(lengths))


class MotionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(72)
        self.model=CubeMotionResidualGRU(MotionConfig(hidden=8))

    def test_CV_initialization_and_json_spec(self):
        result=self.model.forward_public(**sample())
        self.assertTrue(torch.allclose(result['positions'],result['cv_positions'],atol=2e-7,rtol=0))
        self.assertTrue(torch.allclose(result['velocities'][:,0],torch.tensor([[.18,-.11,0.]]).expand(2,3),atol=1e-6,rtol=0))
        self.assertTrue(torch.equal(result['residual_accelerations'],torch.zeros(2,16,3)))
        self.assertTrue(torch.allclose(result['context'].reshape(2,2,8).sum(-1),torch.ones(2,2)))
        self.assertEqual(result['latents'].shape,(2,17,64))
        self.assertEqual(self.model.model_spec(),model_from_spec(json.loads(json.dumps(self.model.model_spec()))).model_spec())

    def test_legacy_network_math_and_state_shapes_retained(self):
        base=SE3ContextNetwork(**self.model.configuration.to_dict())
        base.load_state_dict(self.model.state_dict(),strict=True)
        history,lengths,state,_=public_inputs(self.model,**sample())
        expected=base(history,lengths,state,steps=16,dt=.05)
        actual=self.model(history,lengths,state,steps=16,dt=.05)
        for key in expected:
            self.assertTrue(torch.equal(expected[key],actual[key]),key)
        self.assertEqual(set(base.state_dict()),set(self.model.state_dict()))

    def test_real_future_loss_has_gradient_and_all_parts_remain_trainable(self):
        # Nonzero synthetic head exposes gradients through the transferred core;
        # no optimizer or fitting step is constructed or executed.
        with torch.no_grad():
            self.model.acceleration[-1].weight.normal_(std=.001)
        output=self.model.forward_public(**sample())
        target=output['cv_positions'].detach().clone()
        tau=torch.arange(17)*.05
        target[:,:,0]+=.5*.8*tau.square()
        valid=torch.ones(2,17,dtype=torch.bool)
        loss=physical_motion_loss(output,target,valid,output['times'])
        loss['per_example'].mean().backward()
        for prefix in ('history_encoder.','context_projection.','state_encoder.','transition.','acceleration.'):
            values=[p.grad.abs().sum().item() for n,p in self.model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertGreater(sum(values),0.,prefix)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertFalse(loss['targets_are_runtime_inputs'])

    def test_missing_and_private_future_do_not_fill_public_history(self):
        inputs=sample()
        bad={k:v.clone() for k,v in inputs.items()}
        bad['timestamps'][0,3]+=.01
        with self.assertRaises(ValueError):self.model.forward_public(**bad)
        bad={k:v.clone() for k,v in inputs.items()};bad['positions'][0,5]=1.
        with self.assertRaises(ValueError):self.model.forward_public(**bad)
        with self.assertRaises(TypeError):self.model.forward_public(**inputs,target_positions=torch.ones(2,17,3))
        before=public_inputs(self.model,**inputs)
        _private_future=torch.full((2,17,3),1000.)
        after=public_inputs(self.model,**inputs)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))

    def test_unknown_label_zero_loss_and_clock_binding(self):
        output=self.model.forward_public(**sample())
        targets=torch.full_like(output['positions'],float('nan'))
        mask=torch.zeros(2,17,dtype=torch.bool)
        losses=physical_motion_loss(output,targets,mask,output['times'])
        self.assertTrue(torch.equal(losses['per_example'],torch.zeros(2)))
        losses['per_example'].sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        with self.assertRaises(ValueError):physical_motion_loss(output,targets,mask,output['times']+.05)

    def test_compiler_retains_missing_resets_and_contact_future_mask(self):
        public=[];labels=[]
        for i in range(30):
            visible=i!=8
            public.append(dict(timestamp=i*.05,available=visible,
                position_world=[i*.01,0.,.8] if visible else None,confidence=.9 if visible else None))
            labels.append(dict(timestamp=i*.05,target_position_world=[i*.01,0.,.8],free_motion=i<20,segment_id='A' if i<20 else 'B'))
        windows,report=compile_episode(public,labels,parent_id='synthetic-fit-parent')
        self.assertEqual(report['rows'],30)
        self.assertEqual(len(report['ledger']),30)
        self.assertFalse(next(x for x in report['ledger'] if abs(x['timestamp']-.4)<1e-8)['eligible'])
        self.assertFalse(next(x for x in report['ledger'] if abs(x['timestamp']-.45)<1e-8)['eligible'])
        late=next(w for w in windows if abs(w['anchor_time']-.9)<1e-8)
        self.assertEqual(late['targets']['target_valid'].sum(),1)
        self.assertTrue(np.isnan(late['targets']['target_positions'][2:]).all())
        poisoned=[dict(x) for x in public];poisoned[0]['Native_position']=[0.,0.,0.]
        with self.assertRaises(ValueError):compile_episode(poisoned,labels,parent_id='synthetic-fit-parent')

    def test_runtime_input_path_matches_training_and_ignores_supplied_twist(self):
        # Deliberately not a passing checkpoint constructor or runtime admission.
        runtime=CubeResidualMotionDynamics.__new__(CubeResidualMotionDynamics)
        import threading
        runtime.model=self.model;runtime.lock=threading.RLock()
        x=sample((20,))
        runtime.history={'target':deque([dict(timestamp=float(x['timestamps'][0,i]),
            position=x['positions'][0,i].numpy(),confidence=.9,velocity=np.ones(3)*999.) for i in range(20)],maxlen=20)}
        expected=public_inputs(self.model,**x)
        actual=runtime.inputs('target')
        for a,b in zip(actual[:3],expected[:3]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(actual[3],float(expected[3][0]))

    def test_actual_label_clock_retained_instead_of_nominal_target_grid(self):
        public=[dict(timestamp=i*.05,available=True,position_world=[i*.01,0.,.8],confidence=.9) for i in range(25)]
        labels=[dict(timestamp=i*.05+2e-7,target_position_world=[i*.01,0.,.8],free_motion=True,segment_id='A') for i in range(25)]
        windows,report=compile_episode(public,labels,parent_id='clock-synthetic')
        window=windows[0]
        self.assertEqual(window['targets']['target_times'][1],labels[5]['timestamp'])
        self.assertNotEqual(window['targets']['target_times'][1],window['anchor_time']+.05)
        self.assertAlmostEqual(report['maximum_actual_label_vs_nominal_query_alignment_seconds'],2e-7,places=12)
        x=window['inputs']; y=window['targets']
        output=self.model.forward_public(**{k:torch.from_numpy(v[None]) if isinstance(v,np.ndarray) else torch.tensor([v]) for k,v in x.items()})
        targets={k:torch.from_numpy(v[None]) for k,v in y.items()}
        self.assertTrue(torch.isfinite(physical_motion_loss(output,**targets)['per_example']).all())


if __name__=='__main__':unittest.main()
