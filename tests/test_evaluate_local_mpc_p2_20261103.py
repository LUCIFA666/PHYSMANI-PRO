from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.evaluate_local_mpc_p2_20261103 import metric_group, metrics, rollout_regenerated_references
from our_mani.local_mpc_p2_20261103.reference import ReferenceLimits, reference_step


class TrackingToy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.inputs=[];self.outputs=[]
    def forward(self,state,reference):
        self.inputs.append((state.clone(),reference.clone()))
        q=state[:,:7]+.5*(reference[:,:7]-state[:,:7]);dq=(q-state[:,:7])/.05
        opening=state[:,14:16]+.1*(reference[:,-1:]-state[:,14:16])
        nxt=torch.cat((q,dq,opening,state[:,-1:]),-1)
        T=torch.eye(4).expand(len(state),4,4).clone();T[:,:3,3]=q[:,:3]
        out=dict(state=nxt,q=q,dq=dq,opening=opening,tcp_matrix=T);self.outputs.append(nxt.clone());return out


def inputs(batch=2,ticks=4):
    state=torch.zeros(batch,17);state[:,14:16]=1.
    h={k:torch.zeros(batch,7) for k in ('goal_q','q_ref','dq_ref','ddq_ref')}
    h.update(valid=torch.zeros(batch,dtype=torch.bool),initialized=torch.zeros(batch,dtype=torch.bool),
             gripper_target=torch.ones(batch))
    goals=torch.full((batch,ticks,7),.1);goals[:,2:]=.2
    return state,h,goals,torch.zeros(batch,ticks),torch.ones(batch,ticks,dtype=torch.bool)


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_reseed_uses_predicted_actual_and_only_anchor_h_without_mutation(self):
        state,h,goals,opening,valid=inputs();h_original=deepcopy(h);model=TrackingToy();actual_inputs=[]
        def spy(q,dq,*args,**kwargs):
            actual_inputs.append((q.copy(),dq.copy()));return reference_step(q,dq,*args,**kwargs)
        limits=ReferenceLimits([-1.]*7,[1.]*7,[2.]*7)
        with patch('scripts.evaluate_local_mpc_p2_20261103.reference_step',side_effect=spy):
            result=rollout_regenerated_references(model,state,h,goals,opening,valid,limits)
        self.assertTrue(result['available'].all())
        for tick in range(1,4):
            np.testing.assert_array_equal(actual_inputs[tick][0],model.outputs[tick-1][:,:7].numpy())
            np.testing.assert_array_equal(actual_inputs[tick][1],model.outputs[tick-1][:,7:14].numpy())
        for k in h:torch.testing.assert_close(h[k],h_original[k],rtol=0,atol=0)
        self.assertFalse(torch.equal(model.outputs[0],state))
        self.assertTrue(torch.equal(result['final_predicted_state'][:,-1],state[:,-1]))

    def test_infeasible_rows_and_all_later_predictions_remain_explicit_not_dropped(self):
        state,h,goals,opening,valid=inputs();goals[0,1:]=2.
        valid[1,3]=False
        result=rollout_regenerated_references(TrackingToy(),state,h,goals,opening,valid,
                                             ReferenceLimits([-1.]*7,[1.]*7,[2.]*7))
        self.assertEqual(result['first_failure_tick'].tolist(),[1,-1])
        self.assertEqual(result['available'].tolist(),[[True,False,False,False],[True,True,True,False]])
        self.assertTrue(torch.isnan(result['prediction']['q'][0,1:]).all())
        self.assertEqual(int(result['failure_codes'][0,1]),2)
        target=np.zeros((2,4,17));target_tcp=np.broadcast_to(np.eye(4),(2,4,4,4))
        report=metrics({k:v.numpy() for k,v in result['prediction'].items()},target,target_tcp,valid.numpy(),result['available'].numpy())
        all_ticks=report['all_available_target_ticks']
        self.assertEqual(all_ticks['expected_target_samples'],7)
        self.assertEqual(all_ticks['available_predictions'],4)
        self.assertEqual(all_ticks['infeasible_or_unavailable_predictions'],3)
        self.assertFalse(all_ticks['precision_gate']['passed'])

    def test_intermediate_horizon_outlier_counts_in_max_and_gate(self):
        b,t=2,16;target=np.zeros((b,t,17));tcp=np.broadcast_to(np.eye(4),(b,t,4,4)).copy()
        prediction={'q':np.zeros((b,t,7)),'dq':np.zeros((b,t,7)),'opening':np.zeros((b,t,2)),
                    'tcp_matrix':tcp.copy()}
        prediction['tcp_matrix'][0,2,0,3]=.06
        mask=np.ones((b,t),dtype=bool)
        report=metrics(prediction,target,tcp,mask,mask)
        self.assertEqual(len(report['horizons']),16)
        self.assertEqual(report['all_available_target_ticks']['expected_target_samples'],32)
        self.assertAlmostEqual(report['all_available_target_ticks']['TCP_error_m']['max'],.06)
        self.assertFalse(report['all_available_target_ticks']['precision_gate']['passed'])
        self.assertFalse(report['horizons']['3']['precision_gate']['passed'])
        self.assertTrue(report['horizons']['2']['precision_gate']['passed'])

    def test_nonfinite_available_samples_reject_and_missing_cohort_never_passes(self):
        prediction={'q':np.zeros((2,7)),'dq':np.zeros((2,7)),'opening':np.zeros((2,2)),
                    'tcp_matrix':np.broadcast_to(np.eye(4),(2,4,4)).copy()}
        target=np.zeros((2,17));tcp=np.broadcast_to(np.eye(4),(2,4,4));expected=np.ones(2,dtype=bool)
        report=metric_group(prediction,target,tcp,expected,np.array([True,False]))
        self.assertFalse(report['precision_gate']['passed']);self.assertFalse(report['complete_cohort'])
        prediction['q'][1,0]=np.nan
        with self.assertRaises(ValueError):metric_group(prediction,target,tcp,expected,expected)


if __name__=='__main__':unittest.main()
