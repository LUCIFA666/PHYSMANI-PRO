import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from our_mani.semantic_task_e_large_20261104_v2 import common, evaluation, training
from our_mani.semantic_task_e_large_20261104_v2 import VERSION
from our_mani.semantic_task_e_large_20261104_v2.model import OPERATIONS
from scripts import train_semantic_task_e_large_20261104_v2 as cli


def sample(index=0,resolution=256):
    x=np.zeros((5,4,4,resolution,resolution),np.float32)
    x[:1,:,3]=1.
    mask=np.zeros((5,15),bool);mask[0,:14]=True
    return dict(rgbd=x,robot_history=np.zeros((5,15),np.float32),lengths=np.int64(1),
        frame_indices=np.array([0,-1,-1,-1,-1],np.int64),robot_observed_mask=mask,
        operation_label=index,task=OPERATIONS[index],source_parent_group='synthetic:'+str(index),
        current_frame_index=0,labels={})


class TrainingContracts(unittest.TestCase):
    def test_default_cli_does_not_construct_optimizer_or_GPU(self):
        report=dict(complete=True,model_forward_calls=0,optimizers_created=0,GPU_started=False)
        with mock.patch.object(cli,'check_preparation',return_value=report) as check, \
             mock.patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer')), \
             mock.patch.object(training,'require_cuda',side_effect=AssertionError('GPU')):
            self.assertEqual(cli.main(['--manifest','synthetic','--data-root','synthetic']),0)
            check.assert_called_once()

    def test_original256_and_unobserved_gripper_are_enforced(self):
        self.assertEqual(tuple(common.collate_public([sample()])['rgbd'].shape),(1,5,4,4,256,256))
        with self.assertRaises(ValueError):common.collate_public([sample(resolution=64)])
        wrong=sample();wrong['robot_history'][0,-1]=1
        with self.assertRaises(ValueError):common.collate_public([wrong])
        wrong=sample();wrong['frame_indices'][1]=1
        with self.assertRaises(ValueError):common.collate_public([wrong])

    def test_full_cohort_and_partial_accumulation_are_not_dropped(self):
        groups=training.training_groups(35)
        self.assertEqual(groups,[(0,16),(16,32),(32,35)])
        self.assertEqual([i for lo,hi in groups for i in range(lo,hi)],list(range(35)))
        x=torch.tensor(2.,requires_grad=True)
        training.weighted_micro_loss(x,1,3).backward()
        self.assertAlmostEqual(float(x.grad),1/3,places=6)
        a=dict(accuracy=.8,cross_entropy=.2)
        self.assertGreater(training.selection_key(a,0),training.selection_key(a,1))
        class WindowFixture:
            class base:
                records=[{'current_frame_index':i}for i in (0,8,0,9,0,10,0,11)]
        self.assertEqual(training.full_history_indices(WindowFixture(),4),([1,3,5,7],[1,3,5,7]))
        self.assertEqual(training.full_history_indices(WindowFixture(),2),([1,3],[1,3,5,7]))
        config=common.fixed_config()
        total=len(training.training_groups(3074))*10
        self.assertEqual(total,1930)
        self.assertAlmostEqual(training.logical_step_learning_rate(1,total,config),6e-7)
        self.assertAlmostEqual(training.logical_step_learning_rate(50,total,config),3e-5)
        self.assertAlmostEqual(training.logical_step_learning_rate(total,total,config),3e-6)
        raw=torch.tensor([[[1.,0.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.,0.]]],requires_grad=True)
        logits=torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.]],requires_grad=True)
        target=torch.tensor([0]);output=dict(raw_group_logits=raw,operation_logits=logits)
        loss=training.combined_operation_loss(output,target,config)
        expected=torch.nn.functional.cross_entropy(logits,target)+.5*torch.nn.functional.cross_entropy(raw[:,0],target)
        self.assertTrue(torch.allclose(loss,expected));loss.backward()
        self.assertGreater(float(raw.grad.abs().sum()),0.)

    def test_reduction_requires_same_source_actual_OOM_and_keeps_effective16(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'manifest.json';manifest.write_text('{}')
            pred=Path(tmp)/'failure.json';sources={'synthetic':'hash'};processed={'synthetic':'completion'}
            value=dict(cuda_oom=True,stage='gpu_preflight',microbatch=4,config=common.pointer(common.CONFIG),
                split_manifest=common.pointer(manifest),source_sha256=sources,preprocess_completion=processed)
            pred.write_text(json.dumps(value))
            self.assertEqual(training.microbatch_contract(2,pred,common.sha(pred),manifest,sources,processed),8)
            with self.assertRaises(ValueError):training.microbatch_contract(1,pred,common.sha(pred),manifest,sources,processed)
            value['cuda_oom']=False;pred.write_text(json.dumps(value))
            with self.assertRaises(ValueError):training.microbatch_contract(2,pred,common.sha(pred),manifest,sources,processed)
            self.assertEqual(training.microbatch_contract(4,None,None,manifest,sources,processed),4)

    def test_Root_native_lifecycle_typed_wait_is_not_task_success_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);manifest=tmp/'manifest.json';manifest.write_text('{}')
            owner=tmp/'actual_exit.json';owner.write_text(json.dumps(dict(actual_process_returncode=1,actual_wait_completed=True)))
            auth=tmp/'Root_authorization.json';sources={'synthetic':'hash'}
            value=dict(schema='semantic_e207M_GPU_authorization_20261104_v2',execution_authorized=True,
                method='current_network',fitted_component=common.COMPONENT,split_manifest=common.pointer(manifest),
                config=common.pointer(common.CONFIG),source_sha256=sources,
                native_lifecycle=dict(all_native_closed=True,all_native_children_reaped=True,owners=[common.pointer(owner)]),
                prior_training_lifecycle=dict(all_training_children_reaped=True,owners=[common.pointer(owner)]))
            auth.write_text(json.dumps(value))
            self.assertEqual(common.root_authorization(auth,common.sha(auth),manifest,sources),common.pointer(auth))
            owner.write_text(json.dumps(dict(actual_process_returncode=True,actual_wait_completed=True)))
            value['native_lifecycle']['owners']=[common.pointer(owner)];auth.write_text(json.dumps(value))
            with self.assertRaises(ValueError):common.root_authorization(auth,common.sha(auth),manifest,sources)

    def test_actual_GPU_preflight_measurements_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);manifest=tmp/'manifest.json';manifest.write_text('{}');path=tmp/'preflight.json'
            sources={'synthetic':'hash'};processed={'synthetic':'completion'}
            value=dict(complete=True,passed=True,source_unchanged=True,stage='gpu_preflight',version=VERSION,
                config=common.pointer(common.CONFIG),split_manifest=common.pointer(manifest),source_sha256=sources,
                preprocess_completion=processed,exact_parameters=207855640,image_size=256,history_frames=5,
                optimizer_steps=1,optimizer_step_is_ephemeral_and_discarded=True,preflight_weights_reused_for_training=False,
                microbatch=4,gradient_accumulation=4,effective_batch_size=16,actual_CUDA_BF16_supported=True,
                device_name='synthetic-not-a-real-certificate',real_RGBD_model_forward_batches=1,
                sample_identities=[dict(length=5)]*4,actual_valid_frames=20,full_history_window_required=True,
                preflight_record_indices=[1,3,5,7],canonical_full_history_record_indices=[1,3,5,7],
                peak_allocated_bytes=16,peak_reserved_bytes=32,
                loss=1.,gradient_norm_before_clip=1.,forward_batch_cuda_ms=1.)
            path.write_text(json.dumps(value))
            self.assertEqual(training.checked_preflight(path,common.sha(path),manifest,sources,processed)['microbatch'],4)
            value.pop('peak_allocated_bytes');path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):training.checked_preflight(path,common.sha(path),manifest,sources,processed)

    def test_public_predictions_freeze_precedes_all8_metric_scoring(self):
        class SyntheticDataset:
            role='selection'
            def __len__(self):return 8
            def __getitem__(self,index):return sample(index)
            def verify(self):return True
        class SyntheticModel:
            def eval(self):return self
            def __call__(self,*,rgbd,robot_history,lengths):
                n=len(rgbd)
                return dict(operation_probabilities=torch.full((n,8),1/8),task_e=torch.full((n,16),1/8))
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'source.py';source.write_text('# synthetic')
            output=tmp/'evaluation';original=evaluation.classification_metrics
            def score(rows):
                self.assertTrue((output/'prediction_freeze.json').is_file())
                return original(rows)
            with mock.patch.object(evaluation,'classification_metrics',side_effect=score):
                metrics=evaluation.evaluate_model(SyntheticModel(),SyntheticDataset(),'cpu',4,output,
                    {str(source):common.sha(source)},dict(synthetic=True),split_manifest=dict(synthetic=True))
            self.assertEqual(metrics['windows'],8)
            self.assertEqual(metrics['accuracy'],1/8)
            self.assertFalse(metrics['current_network_end_to_end_ready'])


if __name__=='__main__':unittest.main()
