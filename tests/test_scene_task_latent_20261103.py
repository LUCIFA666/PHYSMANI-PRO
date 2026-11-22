"""Meaningful CPU scene-only model/split contracts; no optimizer or Native."""
import inspect
import unittest

import numpy as np
import torch

from our_mani.scene_task_latent_20261103.model import SceneTaskEncoder,TASKS
from our_mani.scene_task_latent_20261103.data import assign_splits,normalized_rgbd
from our_mani.scene_task_latent_20261103.training import parent_weights


class SceneTaskLatentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(31)

    def test_scene_forward_has_no_task_robot_stage_or_calibration_input(self):
        self.assertEqual(list(inspect.signature(SceneTaskEncoder.forward).parameters),['self','rgbd'])
        model=SceneTaskEncoder().eval();x=torch.randn(2,4,4,64,64);out=model(x)
        self.assertEqual(out['task_e'].shape,(2,16));self.assertEqual(out['task_logits'].shape,(2,2))
        torch.testing.assert_close(out['task_e'].reshape(2,2,8).sum(-1),torch.ones(2,2))
        self.assertTrue(bool((out['task_e']>0).all()))
        torch.testing.assert_close(out['task_probabilities'].sum(-1),torch.ones(2))
        self.assertFalse(model.spec()['frozen_uniform_e_P4_direct_substitution_allowed'])

    def test_view_permutation_and_batch_isolation_keep_real_scene_sensitivity(self):
        model=SceneTaskEncoder().eval();x=torch.randn(2,4,4,64,64)
        base=model(x)['task_e'];permuted=model(x[:,[2,0,3,1]])['task_e']
        torch.testing.assert_close(base,permuted,atol=1e-6,rtol=1e-5)
        changed=x.clone();changed[1]+=5.
        other=model(changed)['task_e'];torch.testing.assert_close(base[0],other[0],atol=1e-7,rtol=1e-5)
        self.assertGreater(float((base[1]-other[1]).detach().abs().max()),1e-5)
        for views in range(1,5):self.assertEqual(model(x[:,:views])['task_e'].shape,(2,16))

    def test_gradient_flows_through_simnorm_and_shared_visual_path(self):
        model=SceneTaskEncoder();out=model(torch.randn(2,4,4,64,64))
        loss=torch.nn.functional.cross_entropy(out['task_logits'],torch.tensor([0,1]));loss.backward()
        for part in (model.camera_encoder,model.scene_projection,model.classifier):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters()),0.)

    def test_nonfinite_or_extra_features_are_rejected(self):
        model=SceneTaskEncoder();x=torch.zeros(1,4,4,64,64);x[0,0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):model(x)
        with self.assertRaises(TypeError):model(torch.zeros(1,4,4,64,64),task_ID=torch.zeros(1))
        for shape in [(1,5,4,64,64),(1,4,3,64,64),(1,4,4,32,32)]:
            with self.assertRaises(ValueError):model(torch.zeros(shape))

    def test_optional_visual_warmstart_loads_only_matching_camera_weights(self):
        model=SceneTaskEncoder();original={k:v.detach().clone() for k,v in model.scene_projection.state_dict().items()}
        visual={'encoder.camera_encoder.'+k:torch.ones_like(v)*.02 for k,v in model.camera_encoder.state_dict().items()}
        visual['encoder.classifier.weight']=torch.zeros(11,64)
        receipt=model.warmstart_camera_encoder(visual)
        self.assertFalse(receipt['robot_or_classifier_weights_loaded'])
        for key,value in original.items():torch.testing.assert_close(model.scene_projection.state_dict()[key],value)
        with self.assertRaises(ValueError):model.warmstart_camera_encoder({})

    def test_parent_split_is_deterministic_and_speed_is_not_a_label(self):
        rows=[dict(task=task,raw_episode_id=f'episode{n}') for family in TASKS
              for task in (family,family+'_high_speed') for n in range(20)]
        assigned,summary=assign_splits(rows);again,again_summary=assign_splits(rows)
        self.assertEqual(assigned,again);self.assertEqual(summary,again_summary)
        self.assertEqual(summary['parent_count'],40)
        self.assertEqual(summary['counts']['internal_val']['sources'],16)
        for family in TASKS:
            for n in range(20):
                match=[r for r in assigned if r['semantic_task']==family and r['raw_episode_id']==f'episode{n}']
                self.assertEqual(len({r['parent_group'] for r in match}),1)
                self.assertEqual(len({r['split'] for r in match}),1)
                self.assertEqual(len({r['semantic_label'] for r in match}),1)
        idx=np.array([i for i,r in enumerate(assigned) if r['split']=='fit'])
        weights=parent_weights(assigned,idx)
        self.assertAlmostEqual(weights.mean(),1.)

    def test_sensor_normalization_never_consumes_metadata_and_depth_is_metric(self):
        arrays=dict(rgb=np.full((2,4,64,64,3),255,np.uint8),depth=np.full((2,4,64,64),2,np.float32))
        values=normalized_rgbd(arrays,np.array([0,1]))
        self.assertEqual(values.shape,(2,4,4,64,64))
        np.testing.assert_allclose(values[:,:,:3],.5)
        np.testing.assert_allclose(values[:,:,3],np.log(3),rtol=1e-6)


if __name__=='__main__':unittest.main()
