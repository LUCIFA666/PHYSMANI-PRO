from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from our_mani.local_mpc_p2_20261103.dataset import (DOF,DT,ServoWindowDataset,build_episode_manifest,canonical_h,
                      read_episode,train_statistics)
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics,export_native_kinematic_snapshot
from our_mani.local_mpc_p2_20261103.losses import horizon_metrics,servo_loss,teacher_forced_prediction
from our_mani.local_mpc_p2_20261103.servo import JointServoResidual,baseline_rollout,rollout_recorded_references
from tests.test_local_mpc_p2_20261103_kinematics import GetterOnlyArm


def write_json(path,value):path.write_text(json.dumps(value))


def matrix_quaternion(matrix):
    # Synthetic chain quaternions use a stable eigensystem conversion rather
    # than borrowing the tested model's FK/rotation decode implementation.
    m=matrix[:3,:3]
    K=np.array([[m[0,0]-m[1,1]-m[2,2],m[1,0]+m[0,1],m[2,0]+m[0,2],m[2,1]-m[1,2]],
                [m[1,0]+m[0,1],m[1,1]-m[0,0]-m[2,2],m[2,1]+m[1,2],m[0,2]-m[2,0]],
                [m[2,0]+m[0,2],m[2,1]+m[1,2],m[2,2]-m[0,0]-m[1,1],m[1,0]-m[0,1]],
                [m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1],np.trace(m)]])/3
    _,vectors=np.linalg.eigh(K);q=vectors[:,-1]
    return q if q[3]>=0 else -q


def synthetic_episode(path,snapshot,*,ticks=19,failed=False,mode='p1',shock=0.):
    path.mkdir()
    fk=PoEKinematics(snapshot)
    def state(q,dq,opening,tick):
        T=fk(q)
        return dict(time=tick*DT,counted_tick=tick,q=q.tolist(),dq=dq.tolist(),
                    finger_open_amounts=opening.tolist(),opening=float(opening.mean()),
                    tcp_position=T[:3,3].tolist(),tcp_quaternion=matrix_quaternion(T).tolist(),
                    held=bool(tick>=ticks-2),truth_diagnostic_only=True,official_public_input_evaluation=False)
    run=dict(mode=mode,scene='synthetic_CPU_contract',TEST_read=False,
             config=dict(formal_evaluation=False,physics_dt=.05,control_dt=.1,decision_ticks=2),
             source_sha256={'synthetic_executor':'not_a_Native_run'})
    write_json(path/'run_inputs.json',run)
    write_json(path/'worker_exit.json',dict(native_closed=True,exit_code=1 if failed else 0))
    q,dq,opening=np.zeros(7),np.zeros(7),np.ones(2)
    before=state(q,dq,opening,0);write_json(path/'initial_snapshot.json',{'state':before})
    h=dict(reference_valid=False,reference_positions=None,reference_velocities=None,
           reference_accelerations=None,target_joints=None,gripper_target=1.)
    records,native=[],[]
    tick=0
    while tick<ticks:
        start=tick;first=deepcopy(before);h_before=deepcopy(h);parts=[]
        goal=np.full(7,.1+shock+(.01 if (tick//2)%2 else -.01))
        command=1. if tick<ticks//2 else 0.
        for index in range(min(2,ticks-tick)):
            qref=q+.02*(goal-q);vref=(qref-q)/DT;aref=(vref-dq)/DT
            newq=q+.5*(qref-q);newdq=(newq-q)/DT
            newopen=opening+.2*(command-opening)
            tick+=1;after=state(newq,newdq,newopen,tick)
            parts.append(dict(index=index,clock_before=(tick-1)*DT,clock_after=tick*DT,
                              measured_positions_before=q.tolist(),measured_velocities_before=dq.tolist(),
                              measured_positions_after=newq.tolist(),measured_velocities_after=newdq.tolist(),
                              reference_positions=qref.tolist(),reference_velocities=vref.tolist(),
                              reference_accelerations=aref.tolist(),gripper_target=command,tick_completed=True))
            native.append(after);before=after;q,dq,opening=newq,newdq,newopen
            h=dict(reference_valid=True,reference_positions=qref.tolist(),reference_velocities=vref.tolist(),
                   reference_accelerations=aref.tolist(),target_joints=goal.tolist(),gripper_target=command)
        records.append(dict(before=first,after=before,actuator_state_before=h_before,actuator_state_after=h,
                            executor=dict(target_joints=goal.tolist(),executed_ticks=tick-start,ticks=parts)))
    (path/'transitions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (path/'native_ticks.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in native))
    return path


class ServoDatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.snapshot=export_native_kinematic_snapshot(GetterOnlyArm())
        specs=[]
        for name,parent,failed,mode in [('a','A',False,'p1'),('a_restart','A',True,'p1'),
                                        ('b','B',False,'p1'),('c','C',False,'p1'),('p0','P0',False,'p0')]:
            p=synthetic_episode(self.root/name,self.snapshot,failed=failed,mode=mode)
            specs.append(dict(path=str(p),episode_id=name,parent_id=parent))
        self.manifest=build_episode_manifest(specs,dict(A='train',B='validation',C='holdout',P0='train'))
        self.train=ServoWindowDataset(self.manifest,'train')

    def model(self):
        stats=train_statistics(self.train)
        return JointServoResidual(self.snapshot,**{k:stats[k] for k in
                    ('residual_q_bound','residual_dq_bound','feature_mean','feature_std')})

    def test_parent_split_precedes_windows_failed_prefix_and_all_tails_retained(self):
        self.assertEqual(len(self.train),38)
        self.assertEqual(self.train.summary['failed_episodes'],1)
        self.assertEqual(self.train.summary['parents'],1)
        self.assertEqual(ServoWindowDataset(self.manifest,'validation').summary['parents'],1)
        self.assertEqual(self.train[-1]['valid'].tolist(),[True]+[False]*15)
        self.assertEqual(sum(r['excluded_reason']=='P0_is_execution_contract_only' for r in self.manifest['episode_records']),1)
        broken=deepcopy(self.manifest);broken['episode_records'][0]['split']='validation'
        with self.assertRaises(ValueError):ServoWindowDataset(broken,'train')

    def test_native_reference_join_and_midcycle_memory_are_not_future_labels(self):
        first,second=self.train[0],self.train[1]
        self.assertFalse(bool(first['initial_actuator_state']['valid']))
        self.assertTrue(bool(second['initial_actuator_state']['valid']))
        np.testing.assert_array_equal(second['initial_actuator_state']['q_ref'],first['references'][0,:7])
        np.testing.assert_array_equal(first['target_states'][0],first['teacher_states'][1])
        np.testing.assert_array_equal(first['references'][:2,-1],[1.,1.])
        self.assertFalse(np.array_equal(first['references'][0,:7],first['target_states'][0,:7]))

    def test_scaling_is_train_only_and_source_changed_episode_is_rejected(self):
        stats=train_statistics(self.train)
        self.assertEqual(stats['source_split'],'train');self.assertEqual(stats['parent_ids'],['A'])
        with self.assertRaises(ValueError):train_statistics(ServoWindowDataset(self.manifest,'holdout'))
        changed=self.root/'a'/'native_ticks.jsonl';changed.write_text(changed.read_text()+'\n')
        with self.assertRaises(ValueError):ServoWindowDataset(self.manifest,'train')

    def test_formal_TEST_guard_runs_before_initial_truth_is_read(self):
        p=self.root/'b'
        run=json.loads((p/'run_inputs.json').read_text());run['TEST_read']=True
        write_json(p/'run_inputs.json',run);(p/'initial_snapshot.json').write_text('MUST_NOT_PARSE_AS_TRUTH')
        with self.assertRaisesRegex(ValueError,'formal TEST'):read_episode(p)

    def test_alignment_errors_fail_closed_without_fake_displacement(self):
        path=self.root/'b'/'transitions.jsonl'
        lines=path.read_text().splitlines();row=json.loads(lines[0]);row['executor']['ticks'][0]['measured_positions_after'][0]+=1.
        lines[0]=json.dumps(row);path.write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ValueError,'disagrees'):read_episode(self.root/'b')

    def test_one_and_free_multistep_losses_are_finite_differentiable_without_optimizer_step(self):
        model=self.model();batch=next(iter(DataLoader(self.train,batch_size=3)))
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        one=teacher_forced_prediction(model,batch)
        multi=rollout_recorded_references(model,batch['initial_state'],batch['references'])
        loss=servo_loss(one,batch)+servo_loss(multi,batch)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertFalse(model.metadata['trained'])
        torch.testing.assert_close(multi['tcp_matrix'],model.fk(multi['q']))

    def test_bounds_physical_aperture_and_no_future_teacher_state_in_free_rollout(self):
        model=self.model();batch=next(iter(DataLoader(self.train,batch_size=2)))
        prediction=teacher_forced_prediction(model,batch)
        self.assertTrue(bool((prediction['residual_q'].abs()<=model.residual_q_bound+1e-7).all()))
        self.assertTrue(bool((prediction['residual_dq'].abs()<=model.residual_dq_bound+1e-7).all()))
        self.assertTrue(bool(((prediction['opening']>=0)&(prediction['opening']<=1)).all()))
        a=rollout_recorded_references(model,batch['initial_state'],batch['references'])
        batch['teacher_states'].fill_(999.)
        batch['target_states'].fill_(-999.)
        b=rollout_recorded_references(model,batch['initial_state'],batch['references'])
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(a['state'][...,-1],batch['initial_state'][:,-1,None].expand(-1,16))

    def test_invalid_tail_labels_are_masked_and_horizon_cohorts_not_filtered(self):
        model=self.model();batch=next(iter(DataLoader(self.train,batch_size=3)))
        prediction=rollout_recorded_references(model,batch['initial_state'],batch['references'])
        batch['valid'][:,3:]=False
        a=servo_loss(prediction,batch)
        batch['target_states'][:,3:]=float('nan');batch['target_tcp'][:,3:]=float('nan')
        b=servo_loss(prediction,batch);torch.testing.assert_close(a,b)
        report=horizon_metrics(prediction,batch)
        self.assertEqual(report['1']['samples'],3);self.assertEqual(report['16']['samples'],0)
        for name in ('hold','constant_velocity','instant_reference'):
            result=baseline_rollout(model,batch['initial_state'],batch['references'],name)
            self.assertEqual(tuple(result['tcp_matrix'].shape),(3,16,4,4))


if __name__=='__main__':unittest.main()
