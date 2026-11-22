"""CPU protocol equivalence and autograd checks for P2v2 reference generation."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from our_mani.local_mpc_20261102.executor import JointLimits, ShortHorizonExecutor
from our_mani.local_mpc_p2_20261103_v2.reference import (
    FailureCode, ReferenceLimits, reference_step as numpy_step,
)
from our_mani.local_mpc_p2_20261103_v2.reference_torch import reference_step


FLOAT64_ATOL = 2e-15
# Float32 is the training arithmetic, not an alternative physical protocol.
FLOAT32_ATOL = dict(q_ref=8e-7, dq_ref=3e-6, ddq_ref=8e-5)


def numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key:numpy(item) for key,item in value.items()}
    return deepcopy(value)


def tensor(value, dtype=torch.float64):
    if isinstance(value, dict):
        return {key:tensor(item,dtype) for key,item in value.items()}
    if isinstance(value, (np.ndarray,list,tuple)):
        array=np.asarray(value)
        return torch.as_tensor(array,dtype=torch.bool if array.dtype.kind=='b' else dtype).clone()
    return value


def limits(joints=7, *, acceleration=12., jerk=1000.):
    return ReferenceLimits(np.full(joints,-2.),np.full(joints,2.),
                           np.linspace(1.5,2.7,joints),acceleration,jerk)


def compare(test, actual, expected, atol=FLOAT64_ATOL):
    for key in ('applied','feasible','failure_code','target_changed','reference_reseeded','failed_joints'):
        np.testing.assert_array_equal(numpy(actual[key]),expected[key],err_msg=key)
    for key in ('q_ref','dq_ref','ddq_ref'):
        np.testing.assert_allclose(numpy(actual[key]),expected[key],atol=atol,rtol=0,err_msg=key)
    # Invalid goal bytes carry no semantic goal (the real executor stores
    # None). Torch preserves opaque inactive padding, while NumPy normalizes
    # invalid placeholders to zero. Compare goals where they actually exist.
    valid=expected['next_h']['valid']
    np.testing.assert_allclose(numpy(actual['next_h']['goal_q'])[valid],expected['next_h']['goal_q'][valid],atol=atol,rtol=0)
    for key in ('q_ref','dq_ref','ddq_ref','valid','initialized'):
        np.testing.assert_allclose(numpy(actual['next_h'][key]),expected['next_h'][key],atol=atol,rtol=0,err_msg='h.'+key)


class FakeBackend:
    def __init__(self,joints):
        self.q=np.zeros(joints);self.dq=np.zeros(joints)
        self.target=self.q.copy();self.time=0.;self.remaining=10000
    def get_joint_positions(self):return self.q.copy()
    def get_joint_velocities(self):return self.dq.copy()
    def set_joint_target_positions(self,q):self.target=np.asarray(q).copy()
    def actuate_gripper(self,opening):return True
    def tick(self,phase):
        next_q=self.q+.65*(self.target-self.q)
        self.dq=(next_q-self.q)/.05;self.q=next_q
        self.time+=.05;self.remaining-=1
    def clock(self):return self.time
    def remaining_ticks(self):return self.remaining
    def terminal(self):return False


class TorchReferenceTests(unittest.TestCase):
    def test_float64_multicycle_matches_numpy_with_same_new_goals_and_failure_masks(self):
        rng=np.random.default_rng(20261103)
        batch,joints=12,7
        for constrained in (False,True):
            with self.subTest(low_jerk=constrained):
                lim=limits(joints,acceleration=np.linspace(4.,12.,joints) if constrained else 12.,
                           jerk=np.linspace(20.,1000.,joints) if constrained else 1000.)
                ht=hn=None;goals=np.zeros((batch,joints));q=np.zeros_like(goals);dq=np.zeros_like(goals)
                for cycle in range(50):
                    changes=rng.uniform(size=batch)<.4
                    goals[changes]=rng.uniform(-1.2,1.2,size=(changes.sum(),joints))
                    active=np.ones(batch,dtype=bool)
                    for step in range(2):
                        rt=reference_step(torch.tensor(q),torch.tensor(dq),torch.tensor(goals),ht,lim,active_mask=active)
                        rn=numpy_step(q,dq,goals,hn,lim,active_mask=active)
                        compare(self,rt,rn)
                        ht,hn=rt['next_h'],rn['next_h']
                        active &= rn['feasible']
                        applied=rn['applied']
                        target=rn['q_ref'];next_q=np.where(applied[:,None],q+.65*(target-q),q)
                        dq=np.where(applied[:,None],(next_q-q)/.05,dq);q=next_q

    def test_float64_matches_real_source_executor_dispatch_over_multiple_macros(self):
        rng=np.random.default_rng(7);batch,joints=4,7
        source_limits=JointLimits(np.full(joints,-2.),np.full(joints,2.),np.linspace(1.5,2.7,joints))
        lim=ReferenceLimits.from_executor_limits(source_limits)
        sources=[ShortHorizonExecutor(FakeBackend(joints),source_limits) for _ in range(batch)]
        h=None;goals=np.zeros((batch,joints))
        for cycle in range(25):
            if cycle%3==0:goals=rng.uniform(-.8,.8,size=(batch,joints))
            receipts=[source.execute_joint_target(goal) for source,goal in zip(sources,goals)]
            for step in range(2):
                q=np.array([receipt.ticks[step]['measured_positions_before'] for receipt in receipts])
                dq=np.array([receipt.ticks[step]['measured_velocities_before'] for receipt in receipts])
                result=reference_step(torch.tensor(q),torch.tensor(dq),torch.tensor(goals),h,lim)
                self.assertTrue(bool(result['feasible'].all()))
                for key,field in (('q_ref','reference_positions'),('dq_ref','reference_velocities'),('ddq_ref','reference_accelerations')):
                    expected=np.array([receipt.ticks[step][field] for receipt in receipts])
                    np.testing.assert_allclose(numpy(result[key]),expected,atol=FLOAT64_ATOL,rtol=0)
                h=result['next_h']

    def test_failure_codes_and_recovery_acceleration_are_source_equivalent(self):
        lim=limits(2)
        h=dict(goal_q=np.tile([.3,.4],(7,1)),q_ref=np.tile([.1,.2],(7,1)),
               dq_ref=np.zeros((7,2)),ddq_ref=np.tile([.7,-.4],(7,1)),
               valid=np.array([False]*5+[True,False]),initialized=np.ones(7,dtype=bool))
        q=np.zeros((7,2));dq=q.copy();goal=np.tile([.3,.4],(7,1))
        goal[0,0]=np.nan;goal[1,0]=3.;q[2,0]=np.nan;q[3,0]=2.1;dq[4,0]=20.;q[5,0]=2.1
        result=reference_step(torch.tensor(q),torch.tensor(dq),torch.tensor(goal),tensor(h),lim)
        expected=numpy_step(q,dq,goal,h,lim)
        compare(self,result,expected)
        np.testing.assert_array_equal(numpy(result['failure_code']),[1,2,3,4,5,0,0])
        q[:]=0.;dq[:]=0.;goal[:]=[.5,.4]
        retry=reference_step(torch.tensor(q),torch.tensor(dq),torch.tensor(goal),result['next_h'],lim)
        compare(self,retry,numpy_step(q,dq,goal,expected['next_h'],lim))

    def test_retarget_threshold_uses_exact_detached_source_precision_in_float32(self):
        lim=limits(2)
        h=dict(goal_q=np.zeros((1,2)),q_ref=np.zeros((1,2)),dq_ref=np.zeros((1,2)),
               ddq_ref=np.zeros((1,2)),valid=np.array([True]))
        for dtype in (torch.float64,torch.float32):
            for value in (1e-10,1.000001e-10):
                goal=torch.tensor([[value,0.]],dtype=dtype)
                ht=tensor(h,dtype)
                result=reference_step(torch.zeros_like(goal),torch.zeros_like(goal),goal,ht,lim)
                expected=numpy_step(np.zeros((1,2)),np.zeros((1,2)),numpy(goal),numpy(ht),lim)
                np.testing.assert_array_equal(numpy(result['target_changed']),expected['target_changed'])
        # float32(1e-10) is slightly > the exact protocol literal 1e-10.
        result=reference_step(torch.zeros((1,2)),torch.zeros((1,2)),torch.tensor([[1e-10,0.]]),tensor(h,torch.float32),lim)
        self.assertTrue(bool(result['target_changed'][0]))

    def test_padding_does_not_change_even_invalid_reference_memory(self):
        lim=limits(2)
        h=dict(goal_q=torch.tensor([[.7,.8],[.3,.4]],dtype=torch.float64),
               q_ref=torch.tensor([[.2,.1],[.1,.2]],dtype=torch.float64),
               dq_ref=torch.tensor([[.4,.3],[0.,0.]],dtype=torch.float64),
               ddq_ref=torch.tensor([[.5,.6],[0.,0.]],dtype=torch.float64),
               valid=torch.tensor([False,True]),initialized=torch.tensor([False,True]),
               gripper_target=torch.tensor([0.,1.],dtype=torch.float64))
        before={key:value.clone() for key,value in h.items()}
        result=reference_step(torch.zeros((2,2),dtype=torch.float64),torch.zeros((2,2),dtype=torch.float64),
                              torch.tensor([[.2,.3],[.3,.4]],dtype=torch.float64),h,lim,active_mask=[False,True])
        for key in h:
            self.assertTrue(torch.equal(result['next_h'][key][0],before[key][0]),key)
            self.assertTrue(torch.equal(h[key],before[key]),'input '+key)
        self.assertFalse(bool(result['applied'][0]));self.assertTrue(bool(result['feasible'][0]))

    def test_zero_budget_prepares_goal_without_dispatched_reference(self):
        lim=limits(2);q=torch.tensor([[.4,.2]],dtype=torch.float64);dq=torch.tensor([[.2,-.1]],dtype=torch.float64)
        result=reference_step(q,dq,torch.zeros_like(q),None,lim,apply_mask=False)
        compare(self,result,numpy_step(numpy(q),numpy(dq),np.zeros((1,2)),None,lim,apply_mask=False))
        self.assertFalse(bool(result['applied'][0]));self.assertTrue(bool(result['next_h']['valid'][0]))

    def test_autograd_gradcheck_through_actual_q_dq_and_goal(self):
        lim=limits(2)
        q=torch.tensor([[.03,-.02]],dtype=torch.float64,requires_grad=True)
        dq=torch.tensor([[.02,-.01]],dtype=torch.float64,requires_grad=True)
        goal=torch.tensor([[.1,-.08]],dtype=torch.float64,requires_grad=True)
        def forward(x,v,g):
            result=reference_step(x,v,g,None,lim)
            return result['q_ref'],result['dq_ref'],result['ddq_ref']
        self.assertTrue(torch.autograd.gradcheck(forward,(q,dq,goal),eps=1e-6,atol=2e-8,rtol=2e-6))
        loss=sum(value.square().sum() for value in forward(q,dq,goal))
        loss.backward()
        for value in (q,dq,goal):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().max()),0.)

    def test_same_goal_keeps_reference_graph_and_has_zero_actual_reseed_gradient(self):
        lim=limits(2)
        q=torch.tensor([[.3,.3]],dtype=torch.float64,requires_grad=True)
        dq=torch.tensor([[.1,.1]],dtype=torch.float64,requires_grad=True)
        qref=torch.tensor([[.05,.03]],dtype=torch.float64,requires_grad=True)
        vref=torch.tensor([[.02,.01]],dtype=torch.float64,requires_grad=True)
        h=dict(goal_q=torch.tensor([[.1,.1]],dtype=torch.float64),q_ref=qref,dq_ref=vref,
               ddq_ref=torch.zeros_like(q),valid=torch.tensor([True]))
        result=reference_step(q,dq,h['goal_q'],h,lim)
        result['q_ref'].sum().backward()
        self.assertTrue(torch.equal(q.grad,torch.zeros_like(q)))
        self.assertTrue(torch.equal(dq.grad,torch.zeros_like(dq)))
        self.assertGreater(float(qref.grad.abs().max()),0.)
        self.assertGreater(float(vref.grad.abs().max()),0.)

    def test_invalid_prior_goal_still_carries_acceleration_gradient_on_retarget(self):
        lim=limits(2,jerk=1.)
        acceleration=torch.tensor([[.3,.2]],dtype=torch.float64,requires_grad=True)
        h=dict(goal_q=torch.zeros((1,2),dtype=torch.float64),q_ref=torch.zeros((1,2),dtype=torch.float64),
               dq_ref=torch.zeros((1,2),dtype=torch.float64),ddq_ref=acceleration,
               valid=torch.tensor([False]),initialized=torch.tensor([True]))
        result=reference_step(torch.zeros((1,2),dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64),
                              torch.ones((1,2),dtype=torch.float64),h,lim)
        result['q_ref'].sum().backward()
        np.testing.assert_allclose(numpy(result['ddq_ref']),[[.35,.25]],atol=2e-16,rtol=0)
        np.testing.assert_allclose(numpy(acceleration.grad),np.full((1,2),.5*.05*.05),atol=2e-16,rtol=0)

    def test_two_tick_closed_graph_uses_predicted_actual_on_new_goal_without_teacher(self):
        lim=limits(2)
        gain=torch.tensor(.65,dtype=torch.float64,requires_grad=True)
        q=torch.tensor([[.02,.01]],dtype=torch.float64,requires_grad=True)
        dq=torch.zeros_like(q)
        first=reference_step(q,dq,torch.tensor([[.4,.3]],dtype=torch.float64),None,lim)
        predicted_q=q+gain*(first['q_ref']-q)
        predicted_dq=(predicted_q-q)/.05
        second=reference_step(predicted_q,predicted_dq,torch.tensor([[-.2,-.1]],dtype=torch.float64),first['next_h'],lim)
        self.assertTrue(bool(second['reference_reseeded'][0]))
        second['q_ref'].square().sum().backward()
        self.assertIsNotNone(gain.grad);self.assertTrue(bool(torch.isfinite(gain.grad)))
        self.assertGreater(abs(float(gain.grad)),0.)
        self.assertIsNotNone(q.grad);self.assertGreater(float(q.grad.abs().max()),0.)

    def test_float32_value_rounding_has_explicit_bounds_without_relaxing_protocol(self):
        rng=np.random.default_rng(11);lim=limits(7);maxima={key:0. for key in FLOAT32_ATOL}
        for _ in range(40):
            q=torch.tensor(rng.uniform(-.6,.6,size=(16,7)),dtype=torch.float32)
            dq=torch.tensor(rng.uniform(-.25,.25,size=(16,7)),dtype=torch.float32)
            goal=torch.tensor(rng.uniform(-.9,.9,size=(16,7)),dtype=torch.float32)
            ht=tensor(dict(goal_q=numpy(goal)-.1,q_ref=numpy(q),dq_ref=numpy(dq),
                           ddq_ref=rng.uniform(-3.,3.,size=(16,7)),valid=np.ones(16,dtype=bool)),torch.float32)
            result=reference_step(q,dq,goal,ht,lim)
            expected=numpy_step(numpy(q),numpy(dq),numpy(goal),numpy(ht),lim)
            for key in ('applied','feasible','failure_code','target_changed'):
                np.testing.assert_array_equal(numpy(result[key]),expected[key])
            for key,tolerance in FLOAT32_ATOL.items():
                difference=float(np.max(np.abs(numpy(result[key])-expected[key])))
                maxima[key]=max(maxima[key],difference)
                self.assertLessEqual(difference,tolerance,key)
        print('float32_reference_max_abs_rounding',maxima)

    def test_cpu_only_input_immutability_and_rejected_half_precision(self):
        self.assertFalse(torch.cuda.is_initialized())
        q=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
        h=dict(reference_positions=q+0.,reference_velocities=q+0.,reference_accelerations=q+0.,
               target_joints=torch.ones_like(q),reference_valid=True,
               gripper_target=q.sum(dim=-1))
        before=numpy(h)
        reference_step(q,q,torch.ones_like(q),h,limits(2))
        for key,value in h.items():np.testing.assert_array_equal(numpy(value),before[key])
        with self.assertRaises(TypeError):
            reference_step(q.half(),q.half(),q.half(),None,limits(2))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
