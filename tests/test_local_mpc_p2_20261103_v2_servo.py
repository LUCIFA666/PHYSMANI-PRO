from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from our_mani.local_mpc_p2_20261103_v2.dataset import (ServoWindowDataset,build_episode_manifest,
    train_statistics,common_reference_limits,file_sha)
from our_mani.local_mpc_p2_20261103_v2.kinematics import export_native_kinematic_snapshot
from our_mani.local_mpc_p2_20261103_v2.servo import (JointServoResidual,rollout_regenerated_references,
    rollout_recorded_references,ReferenceRolloutInfeasible)
from our_mani.local_mpc_p2_20261103_v2.losses import servo_loss,teacher_forced_prediction,position_precision_gate
from our_mani.local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition
from our_mani.local_mpc_p2_20261103_v2.train_servo import evaluate,SELECTION_RULE,_finalize_training
from tests.test_local_mpc_p2_20261103_kinematics import GetterOnlyArm
from tests.test_local_mpc_p2_20261103_servo import synthetic_episode


def write_json(path,value):path.write_text(json.dumps(value))


def enrich_episode(path,seed,height):
    initial=json.loads((path/'initial_snapshot.json').read_text())
    initial['joint_limits']=dict(lower=[-2.]*7,upper=[2.]*7,velocity=[3.]*7,
                                controller_reference_acceleration=12.,controller_reference_jerk=1000.)
    write_json(path/'initial_snapshot.json',initial)
    run=json.loads((path/'run_inputs.json').read_text())
    run['config'].update(seed=seed,initial_target_position=[.2,-.2,height],task='synthetic',calibration_train_episode='TRAIN_only_fixture',
                         joint_reference_acceleration=12.,joint_reference_jerk=1000.)
    write_json(path/'run_inputs.json',run)


class ReferenceTrackingServoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.snapshot=export_native_kinematic_snapshot(GetterOnlyArm())
        specs=[]
        for name,parent,seed,height,shock in [('a','A',10,.8,0.),('a_height','A',10,.775,.02),('b','B',20,.775,.1)]:
            path=synthetic_episode(self.root/name,self.snapshot,ticks=19,shock=shock)
            enrich_episode(path,seed,height);specs.append(dict(path=str(path),episode_id=name,parent_id=parent))
        self.specs=specs
        self.manifest=build_episode_manifest(specs,dict(A='train',B='validation'))
        self.train=ServoWindowDataset(self.manifest,'train')
        self.stats=train_statistics(self.train)
        self.model=JointServoResidual(self.snapshot,**{k:self.stats[k] for k in
                        ('residual_q_bound','residual_dq_bound','feature_mean','feature_std')})

    def test_zero_joint_head_is_exact_strong_reference_baseline(self):
        state=torch.randn(3,17);state[:,14:16]=.5
        reference=torch.randn(3,22);reference[:,-1]=0.
        prediction=self.model(state,reference)
        torch.testing.assert_close(prediction['q'],reference[:,:7],rtol=0,atol=0)
        torch.testing.assert_close(prediction['dq'],reference[:,7:14],rtol=0,atol=0)
        self.assertTrue(torch.equal(prediction['residual_q'],torch.zeros(3,7)))
        self.assertTrue(self.model.metadata['zero_initial_joint_residual'])

    def test_residual_calibration_is_train_next_minus_reference_not_CV(self):
        residual=np.concatenate([e['states'][1:,:7]-e['references'][:,:7] for e in self.train.episodes])
        expected=np.maximum(np.abs(residual).max(0)*1.25,.001)
        np.testing.assert_array_equal(np.asarray(self.stats['residual_q_bound']),expected)
        self.assertEqual(self.stats['source_split'],'train');self.assertEqual(self.stats['parent_ids'],['A'])
        self.assertEqual(self.stats['residual_base'],'dispatched_reference_q_and_dq')
        self.assertFalse(self.stats['validation_or_holdout_used_for_scaling'])

    def test_same_seed_height_variants_cannot_split_parent(self):
        specs=deepcopy(self.specs);specs[1]['parent_id']='C'
        with self.assertRaisesRegex(ValueError,'height variants'):
            build_episode_manifest(specs,dict(A='train',B='validation',C='validation'))
        self.assertEqual(self.train.summary['parents'],1)
        self.assertEqual(self.train.summary['episodes'],2)

    def test_differentiable_free_reference_rollout_uses_no_future_state_or_memory(self):
        batch=next(iter(DataLoader(self.train,batch_size=3)))
        limits,_=common_reference_limits(self.train)
        anchor_memory={k:v.clone() for k,v in batch['initial_actuator_state'].items()}
        a=rollout_regenerated_references(self.model,batch,limits)
        altered=deepcopy(batch);altered['teacher_states'].fill_(333.);altered['target_states'].fill_(-777.)
        b=rollout_regenerated_references(self.model,altered,limits)
        for k in ('q','dq','opening','tcp_matrix','generated_references'):
            torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        for k in anchor_memory:torch.testing.assert_close(batch['initial_actuator_state'][k],anchor_memory[k],rtol=0,atol=0)
        self.assertTrue(a['generated_references'].requires_grad)
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        loss=servo_loss(a,batch)+.5*servo_loss(rollout_recorded_references(self.model,batch['initial_state'],batch['references']),batch)
        loss.backward()
        self.assertGreater(float(self.model.network[-1].weight.grad.abs().sum()),0.)
        for k,v in self.model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_known_command_reference_failure_is_explicit_never_dropped(self):
        batch=next(iter(DataLoader(self.train,batch_size=2)));batch['command_goal_q'][0,2,0]=5.
        limits,_=common_reference_limits(self.train)
        with self.assertRaisesRegex(ReferenceRolloutInfeasible,'no sample dropped'):
            rollout_regenerated_references(self.model,batch,limits)

    def test_all_sixteen_horizons_enter_precision_gate(self):
        batch=next(iter(DataLoader(self.train,batch_size=2)))
        prediction=teacher_forced_prediction(self.model,batch)
        prediction['tcp_matrix']=batch['target_tcp'].clone();prediction['tcp_matrix'][0,4,0,3]+=.06
        gate=position_precision_gate(prediction,batch)
        self.assertFalse(gate['passed']);self.assertGreater(gate['max_m'],.05)
        self.assertEqual(gate['horizon_seconds'],'every_50ms_through_800ms')
        self.assertEqual(gate['samples'],32)

    def test_validation_selection_key_prioritizes_worst_episode_tail_not_average_loss(self):
        val=ServoWindowDataset(self.manifest,'validation')
        report=evaluate(self.model,val,'cpu',batch_size=7)
        by_episode=report['regenerated']['by_episode_precision']
        expected=[max(r['above_5cm_fraction'] for r in by_episode.values()),
                  max(r['max_m'] for r in by_episode.values()),max(r['mean_m'] for r in by_episode.values()),report['loss']]
        self.assertEqual(report['selection_key'],expected)
        self.assertEqual(report['selection_rule'],SELECTION_RULE)
        self.assertEqual(len(report['regenerated']['horizons']),16)
        self.assertFalse(report['future_actual_state_or_h_used'])
        self.assertEqual(report['windows'],19)
        # A lower average loss never beats fewer >5cm outliers under the
        # predeclared lexicographic criterion.
        self.assertLess((0.,.049,.02,100.),(.01,.04,.001,.01))

    def test_final_checkpoint_keeps_tail_priority_selection_and_epoch(self):
        output=self.root/'completed_unit_fixture';output.mkdir()
        source=self.root/'source.py';source.write_text('# synthetic archival fixture, never executed')
        paths={}
        for name in ('manifest','snapshot','validation'):
            path=self.root/(name+'.json');path.write_text('{}');paths[name]=path
        identity=dict(source_sha256={str(source):file_sha(source)},
            manifest=dict(path=str(paths['manifest']),sha256=file_sha(paths['manifest'])),
            fk=dict(snapshot_path=str(paths['snapshot']),snapshot_sha256=file_sha(paths['snapshot']),
                    validation_path=str(paths['validation']),validation_sha256=file_sha(paths['validation'])))
        write_json(output/'identity.json',identity)
        history=[dict(epoch=1,validation=dict(loss=100.,selection_key=[0.,.04,.02,100.])),
                 dict(epoch=2,validation=dict(loss=.001,selection_key=[.1,.03,.001,.001]))]
        write_json(output/'history.json',history)
        for name,epoch in [('best.pt',1),('last.pt',2)]:
            torch.save(dict(state_dict={'weight':torch.tensor([float(epoch)])},epoch=epoch,optimization_steps=epoch,
                            validation=history[epoch-1]['validation'],model_contract=dict(trained=True)),output/name)
        status=_finalize_training(output,history,identity,epochs=2,optimization_steps=2,
                                  started_at='synthetic_unit_fixture',started_clock=time.perf_counter())
        payload=torch.load(output/'best.pt',weights_only=False)
        self.assertEqual(status['selected_epoch'],1);self.assertEqual(payload['epoch'],1)
        self.assertEqual(payload['optimization_steps'],1)
        self.assertEqual(payload['state_dict']['weight'].item(),1.)
        self.assertEqual(status['selection_value'],[0.,.04,.02,100.])
        self.assertTrue(payload['training_complete'])
        self.assertEqual(status['best_checkpoint_sha256'],file_sha(output/'best.pt'))

    def test_v1_checkpoint_is_rejected_even_when_tensor_shapes_match(self):
        p=self.root/'v1.pt'
        torch.save(dict(model_contract=dict(schema='P2_bounded_joint_servo_residual_v1',training_complete=True,trained=True,formal_TEST_used=False),
                        training_complete=True,training_status=dict(status='complete',training_complete=True,epochs_requested=100,epochs_completed=100),
                        optimization_steps=100,identity=dict(formal_TEST_used=False),formal_TEST_used=False),p)
        with self.assertRaises(ValueError):TorchServoTransition.from_checkpoint(p)


if __name__=='__main__':unittest.main()
