"""Current predicted trajectory contracts; CPU fixtures, no Native or fit."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Forecast, Pose
from our_mani.local_mpc_20261102 import mpc
from our_mani.local_mpc_20261102.terminal_mpc import terminal_cost
from our_mani.local_mpc_p2_20261103.action_adapter import DLSConfig
from our_mani.local_mpc_control_20261103.forecast_control import (
    BoundForecastExecutionModel, ForecastJointGoalCEMPlanner, GRU_NOMINAL_BASIS)
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture


def forecast_for(state):
    elapsed=np.arange(17)*.05
    p=np.broadcast_to(state['object_position'],(17,1,3)).copy()
    p[:,0,0]+=.02*elapsed+.01*elapsed**2
    p[:,0,1]+=.005*elapsed**2
    v=np.zeros((17,1,3));v[:,0,0]=.02+.02*elapsed;v[:,0,1]=.01*elapsed
    w=np.zeros((17,1,3));w[:,0,2]=.4*elapsed
    initial_rotation=Pose.from_xyzw(np.r_[state['object_position'],state['object_quaternion']]).rotation
    r=(Rotation.from_rotvec(np.c_[np.zeros((17,2)),.2*elapsed**2]).as_matrix()@initial_rotation)[:,None]
    state['object_linear_velocity']=v[0,0].copy();state['object_angular_velocity']=w[0,0].copy()
    return Forecast(('target',),state['time']+elapsed,p,r,v,w,
        metadata={'untrusted_extra_metadata_not_an_input':'fixture'})


def bound_fixture():
    base,state,candidates=fixture()
    # The toy PoE fixture needs more iterations for the nonstationary targets.
    # This changes only test solver effort, never production limits/tolerances.
    base.goal_helper.adapter.config=DLSConfig(iterations=60)
    forecast=forecast_for(state)
    wrapper=BoundForecastExecutionModel(base.execution_model,forecast,anchor_time=state['time'])
    return base,state,candidates,forecast,wrapper


def planner_for(base,wrapper):
    return ForecastJointGoalCEMPlanner(execution_model=wrapper,goal_helper=base.goal_helper,
        terminal_config=base.terminal_config)


class ForecastControlTests(unittest.TestCase):
    def test_all_H8_endpoints_follow_nonlinear_absolute_forecast_not_CV(self):
        base,state,_,forecast,wrapper=bound_fixture()
        x,h=wrapper.initialize_batch(state,3)
        u=dict(joint_goal=np.broadcast_to(state['q'],(3,7)).copy(),opening=np.ones(3))
        for k in range(1,9):
            x,h,info=wrapper.step_batch(x,u,h)
            p,v,w=forecast.at('target',state['time']+.1*k)
            np.testing.assert_allclose(x['object_position'],np.broadcast_to(p.position,(3,3)),atol=1e-14)
            np.testing.assert_allclose(x['object_linear_velocity'],np.broadcast_to(v,(3,3)),atol=1e-14)
            np.testing.assert_allclose(x['object_angular_velocity'],np.broadcast_to(w,(3,3)),atol=1e-14)
            for q in x['object_quaternion']:
                np.testing.assert_allclose(Pose.from_xyzw(np.r_[p.position,q]).rotation,p.rotation,atol=1e-14)
            self.assertEqual(info['microsteps'],2);self.assertEqual(info['Native_steps'],0)
        self.assertGreater(abs(x['object_position'][0,0]-state['object_position'][0]-.8*.02),.006)

    def test_fresh_anchor_and_complete_domain_required_before_execution(self):
        base,state,_,forecast,wrapper=bound_fixture()
        wrong=deepcopy(state);wrong['time']+=.1
        with self.assertRaisesRegex(ValueError,'same anchor'):wrapper.initialize_batch(wrong,2)
        wrong=deepcopy(state);wrong['object_position'][0]+=.001
        with self.assertRaisesRegex(ValueError,'index0'):wrapper.initialize_batch(wrong,2)
        with self.assertRaisesRegex(ValueError,'index0'):
            BoundForecastExecutionModel(base.execution_model,forecast,anchor_time=.1)
        short=Forecast(forecast.object_ids,forecast.times[:-1],forecast.positions[:-1],forecast.rotations[:-1],
            forecast.linear_velocities[:-1],forecast.angular_velocities[:-1])
        with self.assertRaisesRegex(ValueError,'16 predicted'):
            BoundForecastExecutionModel(base.execution_model,short,anchor_time=0.)
        x,h=wrapper.initialize_batch(state,2);x['time'][:]=.8
        with patch.object(base.execution_model,'step_batch',side_effect=AssertionError('must not call model')):
            with self.assertRaisesRegex(ValueError,'outside'):wrapper.step_batch(x,{},h)

    def test_new_binding_rejects_previous_virtual_h_and_does_not_mutate_inputs(self):
        base,state,_,forecast,wrapper=bound_fixture()
        x,h=wrapper.initialize_batch(state,2);before=deepcopy(x);oldh=deepcopy(h)
        u=dict(joint_goal=np.broadcast_to(state['q'],(2,7)).copy(),opening=np.ones(2))
        wrapper.step_batch(x,u,h)
        for key in x:np.testing.assert_array_equal(x[key],before[key])
        self.assertEqual(h['_current_forecast_binding'],oldh['_current_forecast_binding'])
        other=BoundForecastExecutionModel(base.execution_model,forecast,anchor_time=0.)
        with self.assertRaisesRegex(ValueError,'stale virtual h'):other.step_batch(x,u,h)
        self.assertNotIn('untrusted_extra_metadata_not_an_input',wrapper.forecast.metadata)

    def test_anchor_held_and_hypothetical_release_remain_internal_not_free_GRU(self):
        base,state,_,forecast,_=bound_fixture();state['held']=True;state['requested_opening']=0.
        wrapper=BoundForecastExecutionModel(base.execution_model,forecast,anchor_time=0.)
        x,h=wrapper.initialize_batch(state,2)
        u=dict(joint_goal=np.broadcast_to(state['q'],(2,7)).copy(),opening=np.array([0.,1.]))
        expected,_,_=base.execution_model.step_batch(deepcopy(x),deepcopy(u),deepcopy(h))
        actual,_,info=wrapper.step_batch(x,u,h)
        self.assertTrue(actual['held'][0]);self.assertFalse(actual['held'][1])
        for key in ('object_position','object_quaternion','object_linear_velocity','object_angular_velocity'):
            np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(info['forecast_override_rows'],0)

    def test_public_bridge_provenance_and_unobserved_rotation_are_explicit(self):
        base,state,_,forecast,_=bound_fixture()
        result=dict(available=True,free_forecast_available=True,mode='FREE',timestamp=0.,forecast=forecast,
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,rotation_observable=False)
        wrapper=BoundForecastExecutionModel.from_public_bridge(base.execution_model,result)
        self.assertFalse(wrapper.receipt()['orientation_observable'])
        self.assertFalse(wrapper.receipt()['public_closed_loop_qualified'])
        self.assertEqual(wrapper.receipt()['held_source'],'public_estimate')
        for key,value in (('Native_inputs_used',True),('future_observations_used',True),('mode','UNRESOLVED')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    BoundForecastExecutionModel.from_public_bridge(base.execution_model,dict(result,**{key:value}))
        state['held']=True
        with self.assertRaisesRegex(ValueError,'attached'):wrapper.initialize_batch(state,2)

    def test_fixed_deadline_nominals_use_forecast_and_world_rotated_local_offsets(self):
        base,state,candidates,forecast,wrapper=bound_fixture();planner=planner_for(base,wrapper)
        candidates=deepcopy(candidates)
        for candidate in candidates:candidate['object_tcp_position']=np.array([.02,.01,0.])
        calls=[];solve=base.goal_helper.adapter.solve
        with patch.object(base.goal_helper.adapter,'solve',side_effect=lambda q,p,r:(calls.append(q.shape) or solve(q,p,r))):
            planner._prepare_nominals(state,candidates,state['phase_state'])
        self.assertEqual(calls,[(24,7)])
        for i,candidate in enumerate(candidates):
            pose,_,_=forecast.at('target',candidate['deadline'])
            for kind in (0,1):
                local=candidate['object_tcp_position']+(candidate['approach_direction']*candidate['pregrasp_offset'] if kind==0 else 0.)
                np.testing.assert_allclose(planner._nominals['pose'][kind,i,:3],pose.apply(local),atol=1e-14)
                self.assertEqual(planner._nominals['time'][kind,i],candidate['deadline'])
        self.assertEqual(planner.nominal_receipt['basis_version'],GRU_NOMINAL_BASIS)
        expired=deepcopy(candidates);expired[0]['deadline']=-.1
        expired[1]['deadline']=.9
        planner._prepare_nominals(state,expired,state['phase_state'])
        self.assertFalse(planner._nominals['feasible'][:,:2].any())
        np.testing.assert_array_equal(planner._nominals['time'][:2,:2],np.array([[-.1,.9],[-.1,.9]]))
        self.assertFalse(planner._nominals['deadline_valid'][:2].any())
        self.assertTrue(planner._nominals['deadline_valid'][2:].all())

    def test_inherited_phase_stage_terminal_cost_and_budget_remain_exact_on_GRU_states(self):
        base,state,candidates,forecast,wrapper=bound_fixture();planner=planner_for(base,wrapper)
        result=planner.plan(state,candidates)
        self.assertIsNotNone(result['action'])
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80])
        self.assertEqual(result['diagnostics']['virtual_step_IK_calls'],0)
        candidate=next(c for c in candidates if c['id']==result['selected_candidate_id'])
        total=0.;previous=None;phase=deepcopy(state['phase_state'])
        for i,action in enumerate(result['sequence']):
            before,after=result['predicted_states'][i:i+2]
            phase=mpc.update_phase(before,candidate,phase,planner.config)
            self.assertEqual(action['phase_index'],mpc.PHASES.index(phase['phase']))
            cost,_=mpc.stage_cost(before,action,after,candidate,phase,config=planner.config,previous_action=previous)
            total+=cost;previous=action
            np.testing.assert_allclose(after['object_position'],forecast.at('target',after['time'])[0].position,atol=1e-14)
        phi,_=terminal_cost(state,result['predicted_states'][-1],candidate,state['phase_state'],
            config=planner.config,terminal_config=planner.terminal_config)
        self.assertAlmostEqual(total+phi,result['cost'],places=10)
        self.assertEqual(result['warm_start']['nominal_basis_version'],GRU_NOMINAL_BASIS)
        with self.assertRaisesRegex(ValueError,'fresh current Forecast'):planner.plan(state,candidates)

    def test_warm_controls_can_cross_fresh_bindings_but_never_carry_forecast_or_h(self):
        base,state,candidates,_,wrapper=bound_fixture();first=planner_for(base,wrapper).plan(state,candidates)
        warm=first['warm_start'];self.assertIsNotNone(warm)
        next_state=deepcopy(first['predicted_states'][1]);next_state['actuator_state']={}
        next_state['phase_state']=first['phase_state']
        next_forecast=forecast_for(next_state)
        fresh=BoundForecastExecutionModel(base.execution_model,next_forecast,anchor_time=.1)
        second=planner_for(base,fresh).plan(next_state,candidates,warm_start=warm)
        self.assertEqual(second['budget']['sequences'],256)
        self.assertEqual(second['diagnostics']['warm_start_evaluations_counted'],3)
        for bad in (dict(warm,nominal_basis_version='CV'),dict(warm,forecast=next_forecast),dict(warm,h={})):
            new=BoundForecastExecutionModel(base.execution_model,next_forecast,anchor_time=.1)
            with self.assertRaises(ValueError):planner_for(base,new).plan(next_state,candidates,warm_start=bad)

    def test_expired_selected_warm_cannot_bypass_invalid_candidate_mask(self):
        base,state,candidates,_,wrapper=bound_fixture();first=planner_for(base,wrapper).plan(state,candidates)
        warm=first['warm_start'];selected=warm['selected_candidate_id']
        new_time=warm['deadline']+.05
        next_state=deepcopy(state);next_state['time']=new_time
        next_state['phase_state']['updated_at']=new_time-.1
        changed=deepcopy(candidates)
        for row in changed:
            if row['id']!=selected:row['deadline']=new_time+.5
        current=forecast_for(next_state)
        bound=BoundForecastExecutionModel(base.execution_model,current,anchor_time=new_time)
        planner=planner_for(base,bound);result=planner.plan(next_state,changed,warm_start=warm)
        self.assertEqual(result['budget']['sequences'],256)
        self.assertTrue(result['diagnostics']['expired_selected_warm_discarded'])
        self.assertNotEqual(result['selected_candidate_id'],selected)
        index=next(i for i,c in enumerate(changed) if c['id']==selected)
        self.assertFalse(planner._nominals['feasible'][:,index].any())
        self.assertEqual(planner._nominals['time'][0,index],warm['deadline'])
        for iteration in result['diagnostics']['iterations']:
            self.assertTrue(np.isinf(iteration['minimum_cost_by_candidate'][index]))

    def test_float32_quantization_tolerance_is_micrometre_scale_not_task_error(self):
        base,state,_,forecast,wrapper=bound_fixture()
        quantized=Forecast(forecast.object_ids,forecast.times,forecast.positions.astype(np.float32),
            forecast.rotations.astype(np.float32),forecast.linear_velocities.astype(np.float32),
            forecast.angular_velocities.astype(np.float32))
        bound=BoundForecastExecutionModel(base.execution_model,quantized,anchor_time=0.)
        bound.initialize_batch(state,2)
        self.assertLess(bound.last_anchor_errors['position_m'],1e-6)
        for offset in (3e-6,.03):
            wrong=deepcopy(state);wrong['object_position'][0]+=offset
            with self.assertRaisesRegex(ValueError,'quantization'):bound.initialize_batch(wrong,2)


if __name__=='__main__':unittest.main()
