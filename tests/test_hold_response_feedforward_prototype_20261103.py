"""Synthetic numerical/protocol checks only; no checkpoint or Native."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.local_mpc_control_20261103.hold_response_feedforward_prototype_20261103 import (
    hold_reference_schedule,inverse_response_nominal,_batch,_row)


def actual():
    return dict(time=7.9,q=np.zeros(7),dq=np.zeros(7),tcp_position=np.zeros(3),object_position=np.zeros(3),
        held=True,grasp_motion_mode='ATTACHED_PUBLIC',Native_object_state_used=False,Native_held_used=False,Native_success_used=False,
        actuator_state=dict(reference_clock=7.9,predicted=False),
        phase_state=dict(phase='hold',hold_tcp_quaternion=np.array([0.,0.,0.,1.]),lift_goal_position=np.array([.042,0.,0.])))


class LinearResponse:
    def __init__(self,feasible=True):
        self.execution=SimpleNamespace(limits=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7)))
        self.helper=SimpleNamespace(adapter=SimpleNamespace(config=SimpleNamespace(iterations=24,damping=.01)))
        self.feasible=feasible;self.calls=0
    def evaluate(self,x,h,goals):
        self.calls+=1;n=len(goals)
        return dict(x=dict(tcp_position=.25*goals[:,:3],tcp_quaternion=np.tile([0.,0.,0.,1.],(n,1))),
            h={'reference':goals.copy()},feasible=np.full(n,self.feasible,bool),micro_angles_rad=np.zeros((n,2)))


class FeedforwardPrototypeTests(unittest.TestCase):
    def test_schedule_first12_then_timeindexed_no_overshoot_static_no_step(self):
        state=actual();before=deepcopy(state)
        a=hold_reference_schedule(state,progressive=False);b=hold_reference_schedule(state,progressive=True)
        np.testing.assert_allclose(a['positions'][:,0],.012,atol=1e-15)
        np.testing.assert_allclose(b['positions'][:,0],[.012,.024,.036,.042,.042,.042,.042,.042],atol=1e-15)
        self.assertLessEqual(float(b['successive_increments_m'].max()),.012+1e-15)
        np.testing.assert_array_equal(state['actuator_state']['reference_clock'],before['actuator_state']['reference_clock'])

    def test_stale_h_free_or_Native_state_rejected(self):
        for key,value in [('time',8.),('held',False),('grasp_motion_mode','FREE_GRU'),('Native_object_state_used',True)]:
            state=actual();state[key]=value
            with self.assertRaises(ValueError):hold_reference_schedule(state,progressive=True)

    def test_fixed_response_inverse_recovers_gain_without_changing_old_IK_config(self):
        model=LinearResponse();goal=np.array([.01,-.015,.02]);nom=np.zeros(7)
        result=inverse_response_nominal(model,{}, {},nom,goal,np.array([0.,0.,0.,1.]))
        self.assertTrue(result['model_feasible_proposal']);self.assertLess(result['selected']['loss'],1e-12)
        np.testing.assert_allclose(result['selected']['joint_goal'][:3],goal/.25,atol=1e-5)
        self.assertEqual(model.helper.adapter.config.iterations,24)
        self.assertEqual(result['numerical_constants']['iterations'],12)
        self.assertFalse(result['Native_collision_qualified']);self.assertLessEqual(model.calls,12)

    def test_all_infeasible_is_diagnostic_only_and_batch_h_matches_state(self):
        model=LinearResponse(False)
        result=inverse_response_nominal(model,{}, {},np.zeros(7),np.ones(3)*.01,np.array([0.,0.,0.,1.]))
        self.assertFalse(result['model_feasible_proposal']);self.assertIsNotNone(result['selected'])
        h=dict(q_ref=np.arange(7)[None],latent={'value':np.array([[3.,4.]])},predicted=True,source='virtual')
        repeated=_batch(h,15);self.assertEqual(repeated['q_ref'].shape,(15,7))
        restored=_row(repeated,7,15)
        np.testing.assert_array_equal(restored['q_ref'],h['q_ref'])
        np.testing.assert_array_equal(restored['latent']['value'],h['latent']['value'])


if __name__=='__main__':unittest.main()
