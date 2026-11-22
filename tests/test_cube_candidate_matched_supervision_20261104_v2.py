"""Pure normalized stream/tensor contracts, no real model/data/Native."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from our_mani.cube_candidate_matched_20261104_v2 import data,losses,evaluation

SHA='a'*64
PARENT='Cube_native_generated_ordinary_seed2611042000'

def fixture(contact=True):
    episode='synthetic_teacher_pass'
    initial=dict(episode_id=episode,time=0.,physics_time=10.,task_time=5.,held=False,native_success=False,
        requested_opening=1.,object_position=[0.,0.,0.],tcp_position=[0.,0.,.1],tcp_quaternion=[0,1,0,0])
    states=[initial];public=[]
    for index in range(1,45):
        state=deepcopy(initial);now=index*.05;state.update(time=now,physics_time=10.+now,task_time=5.+now)
        if contact and now>=1.45:state['requested_opening']=0.
        if contact and now>=1.5:state.update(held=True,object_position=[0.,0.,.05],tcp_position=[.01,0.,.06],native_success=now>=2.1)
        states.append(state)
    for index,state in enumerate(states):
        now=state['time'];n=min(index+1,20)
        forecast=dict(times=(now+np.arange(17)*.05).tolist(),object_ids=['target'],positions=np.zeros((17,1,3)).tolist(),rotations=np.broadcast_to(np.eye(3),(17,1,3,3)).tolist(),linear_velocities=np.zeros((17,1,3)).tolist(),angular_velocities=np.zeros((17,1,3)).tolist(),metadata=dict(checkpoint_sha256=SHA,observed_until=now,full_orientation_observable={'target':False},history_frames={'target':n},contexts={'target':[1./8]*16}))
        obj=dict(object_id='target',timestamp=now,visible=True,confidence=1.,pose=dict(position=[0,0,0],rotation=np.eye(3).tolist()),points_world=[[-.025,-.025,-.025],[.025,.025,.025],[.025,-.025,.025]])
        result=dict(timestamp=now,mode='FREE',available=True,free_forecast_available=True,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,forecaster_is_injected_test_double=False,forecast=forecast,object_observation=obj)
        robot=dict(timestamp=now,tcp_position=[0,0,.1],tcp_quaternion=[0,1,0,0],q=[0.]*7,dq=[0.]*7,finger_open_amounts=[1.,1.])
        public.append(dict(episode_id=episode,index=index,result=result,public_robot=robot,close_requested=state['requested_opening']<.5,rgbd=f'rgbd/{index:04d}.npz'))
    return public,states[0],states[1:]


class CandidateMatchedContracts(unittest.TestCase):
    def test_partition_exact_disjoint_roles(self):
        p=data.load_partition();self.assertEqual(len(p['rows']),16)
        self.assertEqual([sum(r['role']==role for r in p['rows'])for role in('fit','selection','heldout')],[8,4,4])
        self.assertFalse(set(r['seed']for r in p['rows'])&set(p['old_evaluated_seeds_excluded']))

    def test_motion_without_contact_is_not_candidate_supervision(self):
        rows,initial,ticks=fixture(False)
        arrays,report=data.compile_contact_episode(rows,initial,ticks,parent_id=PARENT,episode_id=initial['episode_id'],expected_forecast_sha256=SHA)
        self.assertEqual(arrays,{});self.assertFalse(report['candidate_training_ready']);self.assertEqual(report['public_frames_total'],45)

    def test_all_preclose_including_far_pose_and_true_unclipped_tau(self):
        rows,initial,ticks=fixture()
        arrays,report=data.compile_contact_episode(rows,initial,ticks,parent_id=PARENT,episode_id=initial['episode_id'],expected_forecast_sha256=SHA)
        self.assertTrue(report['candidate_training_ready']);self.assertGreater(report['pose_only_far_anchors'],0)
        self.assertGreater(float(arrays['target_dt'].max()),.8)
        self.assertFalse(arrays['target_time_valid'][arrays['target_dt']>.8].any())
        self.assertAlmostEqual(report['ledger'][0]['timestamp'],.2)
        self.assertTrue(np.allclose(arrays['target_position'],[.01,0,.01]))
        self.assertFalse(report['ledger'][0]['Native_object_orientation_used_in_target'])
        # Unknown GT/body axes are deliberately not used to rotate the public basis.
        contact=dict(time=1.5,tcp_position=[.01,0,.06],object_position=[0,0,.05],tcp_quaternion=[0,1,0,0],object_quaternion=[0,1,0,0])
        target=data.public_reference_target(rows[4]['result']['forecast'],contact,anchor_time=.2)
        self.assertTrue(np.allclose(target['target_position'],[.01,0,.01]))

    def test_cross_pass_clock_and_heldout_refused(self):
        rows,initial,ticks=fixture()
        with self.assertRaises(ValueError):data.compile_contact_episode(rows,initial,ticks,parent_id='Cube_native_generated_ordinary_seed2611042006',episode_id=initial['episode_id'],expected_forecast_sha256=SHA)
        bad=deepcopy(ticks);bad[2]['episode_id']='other'
        with self.assertRaises(ValueError):data.actual_successful_contacts(initial,bad,episode_id=initial['episode_id'])
        bad=deepcopy(ticks);bad[2]['physics_time']+=.01
        with self.assertRaises(ValueError):data.actual_successful_contacts(initial,bad,episode_id=initial['episode_id'])

    def test_far_pose_requires_constant_unobserved_public_reference(self):
        rows,_,_=fixture();f=rows[4]['result']['forecast'];f=deepcopy(f);f['rotations'][2][0]=np.diag([-1,-1,1]).tolist()
        with self.assertRaises(ValueError):data.public_reference_target(f,dict(time=2.,tcp_position=[0,0,0],object_position=[0,0,0],tcp_quaternion=[0,1,0,0]),anchor_time=.2)

    def test_loss_time_mask_and_compatibility_not_success_score(self):
        p=torch.zeros(2,8,3,requires_grad=True);r=torch.eye(3).repeat(2,8,1,1);tau=torch.full((2,8),.3,requires_grad=True);rank=torch.zeros(2,8,requires_grad=True)
        out=dict(positions=p,rotation_matrices=r,times=tau)
        loss=losses.candidate_loss(out,torch.zeros(2,3),torch.eye(3).repeat(2,1,1),torch.tensor([.6,2.]),torch.tensor([True,False]),compatibility_logits=rank,ranking_weight=1.)
        loss['per_example'].sum().backward();self.assertEqual(float(tau.grad[1].abs().sum()),0.);self.assertGreater(float(tau.grad[0].abs().sum()),0.)
        self.assertFalse(loss['unmatched_candidates_are_negative']);self.assertFalse(loss['calibrated_physical_success_probability'])
        report=evaluation.candidate_metrics(loss,selected_index=np.array([1,-1]));self.assertEqual(report['selected']['missing'],1);self.assertEqual(report['pose_only_far'],1)

    def test_no_fake_arrival_or_dropped_failed_parent(self):
        report=evaluation.live_arrival_metrics([dict(actual_selected_dispatched=False),dict(actual_selected_dispatched=True,exact_physics_time_arrival_available=False)])
        self.assertEqual(report['selected_receipt_denominator'],2);self.assertEqual(report['evaluated'],0)
        with self.assertRaises(ValueError):data.combine_episode_results([({},dict(role='fit',parent_id=PARENT))])


if __name__=='__main__':unittest.main()
