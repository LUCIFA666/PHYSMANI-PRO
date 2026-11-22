"""Completion memory comes from real actuator snapshots, never held/contact."""
from copy import deepcopy
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.completion_response import CompletionAwareFingerResponse


class Base:
    def __init__(self,mutate=False):self.mutate=mutate;self.last=None
    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        n=len(x['q'])
        if self.mutate:x['finger_open_amounts'][:]=.99;references['q_ref'][:]=999.
        self.last=dict(q=x['q']+.1,dq=x['dq']-.2,finger_open_amounts=np.full((n,2),.2),
            feasible=np.arange(n)%2==0,latent={'counter':1+(latent or {}).get('counter',0)},held=np.ones(n,bool))
        return self.last


def actual(time=0.,done=True,target=0.):
    return dict(time=time,finger_open_amounts=np.array([.7,.4]),
        actuator_state=dict(reference_clock=time if done is not None else None,reference_valid=done is not None,
            gripper_done=done,gripper_target=target,reference_is_measured_state=False))


def step_input(n=2,time=0.,target=0.):
    x=dict(q=np.zeros((n,7)),dq=np.zeros((n,7)),time=np.full(n,time),
        finger_open_amounts=np.broadcast_to([.7,.4],(n,2)).copy())
    refs=dict(q_ref=np.zeros((n,7)),dq_ref=np.zeros((n,7)),ddq_ref=np.zeros((n,7)),gripper_target=np.full(n,target))
    return x,refs


class CompletionTests(unittest.TestCase):
    def test_known_done_same_target_preserves_q_dq_feasible_and_only_holds_fingers(self):
        base=Base();response=CompletionAwareFingerResponse(base)
        latent=response.initialize_latent(actual(),2);x,r=step_input()
        result=response.predict_execution(x,r,latent,.05,held_anchor=np.array([False,True]))
        self.assertIs(result['q'],base.last['q']);self.assertIs(result['dq'],base.last['dq'])
        self.assertIs(result['feasible'],base.last['feasible'])
        np.testing.assert_array_equal(result['finger_open_amounts'],x['finger_open_amounts'])
        self.assertTrue(result['completion_hold_mask'].all());self.assertNotIn('held',result)
        self.assertFalse(result['new_protocol_qualified']);self.assertFalse(result['completion_implies_contact'])

    def test_target_change_releases_only_changed_rows_and_never_predicts_relatching(self):
        response=CompletionAwareFingerResponse(Base());latent=response.initialize_latent(actual(),2)
        x,r=step_input();r['gripper_target']=np.array([0.,1.])
        first=response.predict_execution(x,r,latent,.05,held_anchor=np.zeros(2,bool))
        np.testing.assert_array_equal(first['completion_hold_mask'],[True,False])
        np.testing.assert_array_equal(first['finger_open_amounts'][1],[.2,.2])
        x['time'][:]=.05;x['finger_open_amounts']=first['finger_open_amounts'];r['gripper_target'][:]=0.
        second=response.predict_execution(x,r,first['latent'],.05,held_anchor=np.zeros(2,bool))
        np.testing.assert_array_equal(second['completion_hold_mask'],[True,False])
        self.assertFalse(second['latent']['completion_response_state']['done'][1])

    def test_unknown_or_false_done_uses_base_without_synthetic_completion(self):
        for done in (None,False):
            response=CompletionAwareFingerResponse(Base());latent=response.initialize_latent(actual(done=done),2)
            x,r=step_input();result=response.predict_execution(x,r,latent,.05,held_anchor=np.ones(2,bool))
            self.assertFalse(result['completion_hold_mask'].any())
            np.testing.assert_array_equal(result['finger_open_amounts'],np.full((2,2),.2))

    def test_new_real_anchor_reads_fresh_done_and_rejects_old_virtual_latent(self):
        response=CompletionAwareFingerResponse(Base());old=response.initialize_latent(actual(done=False),2)
        x,r=step_input();result=response.predict_execution(x,r,old,.05,held_anchor=np.zeros(2,bool))
        fresh=response.initialize_latent(actual(time=.1,done=True),2)
        x['time'][:]=.1
        with self.assertRaisesRegex(ValueError,'stale/foreign'):
            response.predict_execution(x,r,result['latent'],.05,held_anchor=np.zeros(2,bool))
        self.assertTrue(response.predict_execution(x,r,fresh,.05,held_anchor=np.zeros(2,bool))['completion_hold_mask'].all())

    def test_stale_snapshot_and_explicit_virtual_feedback_are_rejected(self):
        for mutate in ('clock','virtual','latent','integer_done'):
            response=CompletionAwareFingerResponse(Base());state=actual()
            if mutate=='clock':state['actuator_state']['reference_clock']=-.05
            elif mutate=='virtual':state['actuator_state']['predicted']=True
            elif mutate=='latent':state['actuator_state']['latent']={}
            else:state['actuator_state']['gripper_done']=1
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):response.initialize_latent(state,2)

    def test_initializer_never_reads_object_or_held_channels(self):
        class Guard(dict):
            def get(self,key,*default):
                if key in ('held','object_position','native_success','contact'):raise AssertionError(key)
                return super().get(key,*default)
            def __getitem__(self,key):
                if key in ('held','object_position','native_success','contact'):raise AssertionError(key)
                return super().__getitem__(key)
        response=CompletionAwareFingerResponse(Base());latent=response.initialize_latent(Guard(actual()),2)
        x,r=step_input()
        low=response.predict_execution(x,r,latent,.05,held_anchor=np.zeros(2,bool))
        high=response.predict_execution(x,r,latent,.05,held_anchor=np.ones(2,bool))
        np.testing.assert_array_equal(low['completion_hold_mask'],high['completion_hold_mask'])

    def test_base_mutation_cannot_change_current_hold_or_caller_inputs(self):
        response=CompletionAwareFingerResponse(Base(mutate=True));state=actual();latent=response.initialize_latent(state,2)
        x,r=step_input();before_x=deepcopy(x);before_r=deepcopy(r)
        result=response.predict_execution(x,r,latent,.05,held_anchor=np.zeros(2,bool))
        np.testing.assert_array_equal(result['finger_open_amounts'],before_x['finger_open_amounts'])
        for key in x:np.testing.assert_array_equal(x[key],before_x[key])
        for key in r:np.testing.assert_array_equal(r[key],before_r[key])
        self.assertEqual(result['latent']['base_latent']['counter'],1)

    def test_clock_and_binary_command_contract_is_explicit(self):
        response=CompletionAwareFingerResponse(Base());latent=response.initialize_latent(actual(),2)
        x,r=step_input();r['gripper_target'][:]=.5
        with self.assertRaisesRegex(ValueError,'binary'):response.predict_execution(x,r,latent,.05,held_anchor=np.zeros(2,bool))
        r['gripper_target'][:]=0.;x['time'][:]=.05
        with self.assertRaisesRegex(ValueError,'clocks'):response.predict_execution(x,r,latent,.05,held_anchor=np.zeros(2,bool))
        with self.assertRaisesRegex(ValueError,'50ms'):response.predict_execution(x,r,latent,.1,held_anchor=np.zeros(2,bool))

    def test_existing_joint_execution_interface_carries_two_microstep_latent(self):
        from our_mani.local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel
        from our_mani.local_mpc_p2_20261103.reference import ReferenceLimits,reference_step
        response=CompletionAwareFingerResponse(Base())
        def fk(q):return dict(tcp_position=q[:,:3],tcp_quaternion=np.broadcast_to([0.,0.,0.,1.],(len(q),4)).copy())
        model=JointResidualFKExecutionModel(reference_step,ReferenceLimits(np.full(7,-2.),np.full(7,2.),np.full(7,3.)),
            response,fk,latent_initializer=response.initialize_latent)
        state=actual()
        state['actuator_state'].update(reference_positions=np.zeros(7),reference_velocities=np.zeros(7),
            reference_accelerations=np.zeros(7),target_joints=np.zeros(7))
        state.update(q=np.zeros(7),dq=np.zeros(7),tcp_position=np.zeros(3),tcp_quaternion=[0,0,0,1],
            tcp_velocity=np.zeros(3),tcp_angular_velocity=np.zeros(3),object_position=[.2,0,.8],
            object_quaternion=[0,0,0,1],object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),
            opening=.55,requested_opening=0.,held=False)
        x,h=model.initialize_batch(state,1)
        u=dict(joint_goal=np.full((1,7),.05),opening=np.zeros(1))
        x,h,info=model.step_batch(x,u,h,.1)
        np.testing.assert_array_equal(x['finger_open_amounts'],[[.7,.4]])
        self.assertEqual(info['microsteps'],2)
        self.assertEqual(h['latent']['base_latent']['counter'],2)
        np.testing.assert_allclose(h['latent']['completion_response_state']['virtual_time'],[.1],atol=1e-15)
        self.assertFalse(x['held'][0])


if __name__=='__main__':unittest.main()
