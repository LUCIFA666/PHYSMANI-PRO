"""CPU contracts for the unqualified single-task P4 proposal pilot."""
import inspect
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from our_mani.local_mpc_p4_20261103.model import (
    CandidateTransformerPilot, matching_loss, rotation_6d_to_matrix,
    matrix_to_rotation_6d, parallel_gripper_rotation_distance, SCORE_SEMANTICS)


def features(batch=2):
    torch.manual_seed(2718)
    future=torch.zeros(batch,17,16)
    future[:,:,0]=torch.arange(17)/16.
    future[:,:,1:4]=torch.randn(batch,17,3)*.015
    future[:,0,1:4]=0.
    future[:,:,4:10]=torch.tensor([1.,0.,0.,0.,1.,0.])
    future[:,:,10:13]=torch.randn(batch,17,3)*.08
    robot=torch.randn(batch,25)*.1
    robot[:,17:23]=torch.tensor([1.,0.,0.,0.,-1.,0.])
    robot[:,-2:]=torch.tensor([.7,.3])
    c=torch.softmax(torch.randn(batch,2,8),dim=-1).reshape(batch,16)
    return dict(future_tokens=future,robot=robot,geometry=torch.tensor([.05,.05,.05,.9]).expand(batch,4).clone(),
                context_c=c,task_e=torch.full((batch,16),1./8))


def small_model():
    torch.manual_seed(31415)
    return CandidateTransformerPilot(d_model=32,num_layers=1,num_heads=4).eval()


def informative_heads(model):
    # Contract probe only; no fitting/optimizer. Nonzero readouts ensure the
    # batch-isolation/feature tests cannot pass merely from identical priors.
    with torch.no_grad():
        for head in (model.position_head,model.rotation_head,model.time_head):
            torch.nn.init.normal_(head.weight,std=.03)
    return model


def proposals(batch=1):
    target=torch.tensor(Rotation.from_euler('xyz',[.31,.72,-.46]).as_matrix(),dtype=torch.float32)
    return dict(positions=torch.full((batch,8,3),.10),
                rotation_matrices=target.expand(batch,8,3,3).clone(),
                times=torch.full((batch,8),.4)), target.expand(batch,3,3).clone()


class P4ModelContractTests(unittest.TestCase):
    def test_prior_outputs_are_bounded_proper_and_explicitly_unscored(self):
        model=small_model();data=features();out=model(**data)
        self.assertEqual(out['positions'].shape,(2,8,3))
        torch.testing.assert_close(out['positions'],torch.tensor([0.,0.,-.015]).expand(2,8,3))
        expected=torch.diag(torch.tensor([1.,-1.,-1.])).expand(2,8,3,3)
        torch.testing.assert_close(out['rotation_matrices'],expected)
        self.assertTrue(bool((out['times']>=.05).all() and (out['times']<=.8).all()))
        self.assertFalse(out['proposal_logits'].requires_grad)
        self.assertEqual(float(out['proposal_logits'].abs().max()),0.)
        self.assertEqual(out['proposal_score_semantics'],SCORE_SEMANTICS)
        self.assertFalse(model.spec()['context_and_task']['automatic_task_classification_learned'])
        self.assertEqual(model.spec()['explicit_priors']['pregrasp_offset_m'],.07)
        self.assertEqual(model.spec()['explicit_priors']['required_width_m'],.08)
        self.assertIn('opposite',model.spec()['explicit_prior_semantics']['approach_direction'])
        self.assertIn('normalized',model.spec()['explicit_prior_semantics']['required_opening'])

    def test_rotation_column_packing_roundtrips_general_orientation_and_degenerate_heads(self):
        matrices=torch.tensor(Rotation.from_euler('xyz',[[.7,-.4,1.2],[-.2,.8,-.6]]).as_matrix(),dtype=torch.float64)
        packed=matrix_to_rotation_6d(matrices)
        torch.testing.assert_close(packed[:,:3],matrices[:,:,0])
        torch.testing.assert_close(packed[:,3:],matrices[:,:,1])
        torch.testing.assert_close(rotation_6d_to_matrix(packed),matrices)
        malformed=torch.tensor([[0.,0.,0.,0.,0.,0.],[1.,0.,0.,2.,0.,0.]],dtype=torch.float64)
        rotations=rotation_6d_to_matrix(malformed)
        torch.testing.assert_close(rotations.transpose(-1,-2)@rotations,torch.eye(3,dtype=torch.float64).expand(2,3,3))
        torch.testing.assert_close(torch.linalg.det(rotations),torch.ones(2,dtype=torch.float64))

    def test_extreme_readouts_cannot_leave_position_or_relative_time_domain(self):
        model=small_model();data=features()
        for sign in (-1.,1.):
            with torch.no_grad():
                model.position_head.bias.fill_(sign*100.)
                model.time_head.bias.fill_(sign*100.)
            out=model(**data)
            self.assertTrue(bool((out['positions'].abs()<=.12).all()))
            self.assertTrue(bool((out['times']>=.05).all() and (out['times']<=.8).all()))

    def test_transformer_rows_do_not_exchange_information_across_batch(self):
        model=informative_heads(small_model());data=features(3)
        together=model(**data)
        alone=model(**{k:v[0:1] for k,v in data.items()})
        for key in ('positions','rotation_matrices','times'):
            torch.testing.assert_close(together[key][0:1],alone[key],atol=2e-6,rtol=1e-5)
        changed={k:v.clone() for k,v in data.items()};changed['robot'][1:,:14]+=1.
        changed['future_tokens'][1:,:,1:4]+=.1;changed['future_tokens'][1:,0,1:4]=0.
        changed_out=model(**changed)
        torch.testing.assert_close(together['positions'][0],changed_out['positions'][0],atol=1e-7,rtol=0)
        self.assertGreater(float((together['positions'][1:]-changed_out['positions'][1:]).abs().max().detach()),1e-6)

    def test_SI_inputs_are_immutable_and_motion_c_and_task_e_both_have_gradient_paths(self):
        model=informative_heads(small_model());data=features();before={k:v.clone() for k,v in data.items()}
        data['context_c'].requires_grad_();data['task_e'].requires_grad_()
        out=model(**data)
        loss=out['positions'].square().sum()+out['times'].sum()+out['rotation_matrices'][...,0,1].sum()
        loss.backward()
        for key,value in data.items():torch.testing.assert_close(value,before[key])
        for key in ('context_c','task_e'):
            self.assertTrue(bool(torch.isfinite(data[key].grad).all()))
            self.assertGreater(float(data[key].grad.abs().sum()),0.)
        self.assertIsNot(model.context_projection,model.task_projection)
        self.assertEqual(model.spec()['fixed_input_divisors']['robot_q_rad'],math.pi)

    def test_each_nonfinite_input_and_malformed_clock_is_rejected(self):
        model=small_model()
        for key in features():
            for invalid in (float('nan'),float('inf')):
                data=features();data[key].reshape(-1)[-1]=invalid
                with self.subTest(key=key,invalid=invalid),self.assertRaises((ValueError,TypeError)):model(**data)
        data=features();data['future_tokens'][:,8,0]-=.01
        with self.assertRaisesRegex(ValueError,'clock'):model(**data)
        data=features();data['future_tokens'][:,0,1]=.01
        with self.assertRaisesRegex(ValueError,'anchor'):model(**data)
        data=features();data['task_e'].zero_()
        with self.assertRaisesRegex(ValueError,'SimNorm'):model(**data)

    def test_forward_has_no_label_stage_or_hidden_Native_arguments(self):
        names=set(inspect.signature(CandidateTransformerPilot.forward).parameters)
        self.assertEqual(names,{'self','future_tokens','robot','geometry','context_c','task_e'})
        with self.assertRaises(TypeError):small_model()(**features(),held=True)

    def test_TCP_local_Z_postmultiplication_is_the_parallel_gripper_symmetry(self):
        output,target=proposals();symmetry=torch.diag(torch.tensor([-1.,-1.,1.]))
        right=target@symmetry;wrong=symmetry@target
        torch.testing.assert_close(parallel_gripper_rotation_distance(right,target),torch.zeros(1),atol=1e-6,rtol=0)
        self.assertGreater(float(parallel_gripper_rotation_distance(wrong,target)[0]),.2)
        output['positions'][0,0]=0.;output['rotation_matrices'][0,0]=right[0]
        loss=matching_loss(output,torch.zeros(1,3),target,torch.tensor([.4]))
        self.assertEqual(int(loss['matching_index'][0]),0)
        self.assertLess(float(loss['per_example'][0]),1e-8)

    def test_matching_is_joint_pose_and_time_in_declared_physical_scales(self):
        output,target=proposals()
        output['positions'][0,0]=0.
        rx=torch.tensor(Rotation.from_euler('x',.5).as_matrix(),dtype=torch.float32)
        output['rotation_matrices'][0,0]=target[0]@rx
        output['positions'][0,1]=torch.tensor([.03,0.,0.])
        output['positions'][0,2]=0.;output['times'][0,2]=.8
        loss=matching_loss(output,torch.zeros(1,3),target,torch.tensor([.4]))
        self.assertEqual(int(loss['matching_index'][0]),1)
        torch.testing.assert_close(loss['per_example'],torch.ones(1),atol=1e-5,rtol=0)
        self.assertEqual(loss['pairwise_loss'].shape,(1,8))
        self.assertAlmostEqual(float(loss['matched_position_error_m'][0]),.03,places=6)
        self.assertFalse(loss['unmatched_candidates_are_negative_labels'])
        self.assertFalse(loss['proposal_score_loss_used'])

    def test_unmatched_unknown_modes_have_no_negative_label_gradient(self):
        output,target=proposals()
        output['positions'][0,0]=torch.tensor([.02,0.,0.]);output['times'][0,0]=.3
        output['positions'].requires_grad_();output['times'].requires_grad_()
        loss=matching_loss(output,torch.zeros(1,3),target,torch.tensor([.4]))
        loss['per_example'].sum().backward()
        self.assertEqual(int(loss['matching_index'][0]),0)
        self.assertGreater(float(output['positions'].grad[0,0].abs().sum()),0.)
        self.assertGreater(float(output['times'].grad[0,0].abs()),0.)
        self.assertEqual(float(output['positions'].grad[0,1:].abs().sum()),0.)
        self.assertEqual(float(output['times'].grad[0,1:].abs().sum()),0.)

    def test_loss_rejects_invalid_label_units_rotations_and_nonfinite_predictions(self):
        output,target=proposals()
        with self.assertRaisesRegex(ValueError,'seconds'):
            matching_loss(output,torch.zeros(1,3),target,torch.tensor([.81]))
        bad=target.clone();bad[0,0,0]=2.
        with self.assertRaisesRegex(ValueError,'SO'):
            matching_loss(output,torch.zeros(1,3),bad,torch.tensor([.4]))
        output['positions'][0,7,2]=float('nan')
        with self.assertRaises(ValueError):matching_loss(output,torch.zeros(1,3),target,torch.tensor([.4]))

    def test_initial_identity_symmetry_loss_backpropagates_finite_values(self):
        model=small_model();out=model(**features())
        target=torch.diag(torch.tensor([1.,-1.,-1.])).expand(2,3,3)
        loss=matching_loss(out,torch.tensor([[.02,0.,-.015],[0.,.02,-.015]]),target,torch.tensor([.3,.5]))
        loss['per_example'].mean().backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads);self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertGreater(float(model.position_head.weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
