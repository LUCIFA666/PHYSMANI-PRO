"""Control equivalence, parent budgets and observability boundary checks."""
from collections import Counter
import unittest

import numpy as np
import torch

from our_mani.multitask import fitting
from our_mani.multitask import rotation_training_objective as objective
from our_mani.multitask.context_se3 import SE3ContextNetwork


class RotationTrainingObjectiveTests(unittest.TestCase):
    def rows(self):
        result=[]
        for parent in range(3):
            for i in range(4):
                result.append(dict(source_key=('train','fixture',str(parent)),rotation_mask=np.ones(4,bool),
                    targets={'angular_velocities':np.full((4,3),.2 if i==0 else 0.)}))
        return result

    def test_original_sampler_is_exact_for_unit_multiplier(self):
        rows=self.rows()
        expected=fitting.balanced_parent_indices(rows,8,np.random.default_rng(17))
        actual=objective.balanced_parent_indices(rows,8,np.random.default_rng(17))
        np.testing.assert_array_equal(actual,expected)

    def test_weighting_changes_window_mix_and_retains_exact_parent_draw_budget(self):
        rows=self.rows();indices=objective.balanced_parent_indices(rows,10000,np.random.default_rng(17),
            rotation_sampling_multiplier=3.)
        self.assertEqual(Counter(rows[index]['source_key'][2] for index in indices),{'0':10000,'1':10000,'2':10000})
        # One moving window has weight3; three stationary windows have weight1.
        fraction=np.mean([objective.moving_rotation_window(rows[index]) for index in indices])
        self.assertTrue(.48<fraction<.52)
        self.assertEqual(len(rows),12)

    def test_rotation_sampler_rejects_validation_and_unobservable_targets_do_not_get_weight(self):
        row=self.rows()[0];row['rotation_mask'][:]=False
        self.assertFalse(objective.moving_rotation_window(row))
        row['source_key']=('validation','fixture','0')
        with self.assertRaisesRegex(ValueError,'TRAIN'):objective.moving_rotation_window(row)
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            objective.balanced_parent_indices([row],1,np.random.default_rng(0),rotation_sampling_multiplier=2.)

    def fixture(self):
        torch.manual_seed(4);model=SE3ContextNetwork();n=2;steps=3
        positions=torch.zeros((n,3));rotations=torch.eye(3).repeat(n,1,1)
        state=model.normalized(positions,rotations,torch.zeros((n,3)),torch.zeros((n,3)),torch.zeros(n,dtype=torch.long))
        history=torch.zeros((n,5,21));history[:,:,:19]=state[:,None]
        targets={'positions':torch.zeros((n,4,3)),'rotations':rotations[:,None].repeat(1,4,1,1),
            'linear_velocities':torch.zeros((n,4,3)),'angular_velocities':torch.ones((n,4,3))}
        data=dict(history=history,length=torch.full((n,),5),state=state,mask=torch.ones((n,4),dtype=torch.bool),
            rotation_mask=torch.ones((n,4),dtype=torch.bool),rotation_observable=torch.ones(n),targets=targets)
        prediction=model(history,data['length'],state,steps=steps)
        return model,data,prediction

    def test_unit_objective_matches_original_loss_and_gradients(self):
        model,data,prediction=self.fixture()
        expected=fitting.loss_per_window(model,data,3,prediction)
        actual=objective.loss_per_window(model,data,3,prediction)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        first=torch.autograd.grad(expected.sum(),model.acceleration[-1].bias,retain_graph=True)[0]
        second=torch.autograd.grad(actual.sum(),model.acceleration[-1].bias)[0]
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_rotation_emphasis_keeps_position_gradient_and_respects_original_mask(self):
        model,data,prediction=self.fixture()
        original=fitting.loss_per_window(model,data,3,prediction)
        changed=objective.loss_per_window(model,data,3,prediction,rotation_loss_multiplier=4.,angular_loss_multiplier=4.)
        first=torch.autograd.grad(original.sum(),model.acceleration[-1].bias,retain_graph=True)[0]
        second=torch.autograd.grad(changed.sum(),model.acceleration[-1].bias,retain_graph=True)[0]
        torch.testing.assert_close(first[:3],second[:3],rtol=0,atol=0)
        self.assertGreater(float(torch.linalg.vector_norm(second[3:])),float(torch.linalg.vector_norm(first[3:])))
        data['rotation_mask'][:]=False
        torch.testing.assert_close(objective.loss_per_window(model,data,3,prediction,
            rotation_loss_multiplier=4.,angular_loss_multiplier=4.),fitting.loss_per_window(model,data,3,prediction),rtol=0,atol=0)

    def test_invalid_experiment_weights_are_rejected(self):
        for value in (True,0.,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                objective.balanced_parent_indices(self.rows(),2,np.random.default_rng(0),rotation_sampling_multiplier=value)


if __name__=='__main__':unittest.main()
