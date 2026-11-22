"""Pure CPU joint-parameterization checks; no optimizer training or Native."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner,JointCEMConfig,PARAM_SPACE
from our_mani.local_mpc_20261102 import mpc
from our_mani.local_mpc_20261102.constraints import DiagnosticConstraints
from our_mani.local_mpc_20261102.terminal_mpc import terminal_cost
from our_mani.local_mpc_control_20261103.joint_goals import PoEJointGoalHelper,JointGoalConfig
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics,export_native_kinematic_snapshot
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from our_mani.local_mpc_p2_20261103.reference import ReferenceLimits,reference_step
from our_mani.local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel
from tests.test_local_mpc_p2_20261103_kinematics import GetterOnlyArm


class ToyLagFingerResponse:
    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        q=x['q']+.7*(references['q_ref']-x['q'])
        fingers=x['finger_open_amounts']+.2*(references['gripper_target'][:,None]-x['finger_open_amounts'])
        return dict(q=q,dq=(q-x['q'])/dt,finger_open_amounts=fingers,feasible=np.ones(len(q),bool),latent=latent)


def fixture():
    arm=GetterOnlyArm();chain=PoEKinematics(export_native_kinematic_snapshot(arm));q=arm.q.copy();pose=chain(q)
    limits=ReferenceLimits(np.full(7,-2.5),np.full(7,2.5),np.full(7,2.))
    helper=PoEJointGoalHelper(chain,limits,JointGoalConfig(lower=(-2.,-2.,-.5),upper=(2.,2.,2.)))
    def fk(q):
        pose=chain(q);return dict(tcp_position=pose[:,:3,3],tcp_quaternion=matrix_quaternion(pose[:,:3,:3]))
    model=JointResidualFKExecutionModel(reference_step,limits,ToyLagFingerResponse(),fk)
    state=dict(tcp_position=pose[:3,3],tcp_quaternion=matrix_quaternion(pose[:3,:3]),tcp_velocity=np.zeros(3),
        object_position=pose[:3,3]-[0.,0.,.02],object_quaternion=np.array([0.,0.,0.,1.]),
        object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),time=0.,opening=1.,finger_open_amounts=np.ones(2),
        q=q,dq=np.zeros(7),held=False,requested_opening=1.,actuator_state={},
        phase_state=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=-.1))
    candidates=[mpc.validate_candidate(dict(id=f'c{i}',object_tcp_position=np.zeros(3),object_tcp_quaternion=state['tcp_quaternion'],
        deadline=.5+.3*(i%2),approach_direction=[0.,0.,1.],pregrasp_offset=.02,opening=0.,required_opening=1.,allowed_relative_speed=.14)) for i in range(8)]
    planner=JointGoalCEMPlanner(execution_model=model,goal_helper=helper)
    return planner,state,candidates


class JointCEMTests(unittest.TestCase):
    def test_one_nominal_batch_fixed_budget_and_action_FK_exact_joint_identity(self):
        planner,state,candidates=fixture();solve=planner.goal_helper.adapter.solve;calls=[]
        def observed(q,p,r):calls.append(q.shape);return solve(q,p,r)
        with patch.object(planner.goal_helper.adapter,'solve',side_effect=observed):result=planner.plan(state,candidates)
        self.assertEqual(calls,[(24,7)])
        self.assertEqual(result['budget']['iteration_sequences'],[88,88,80]);self.assertEqual(result['budget']['sequences'],256)
        self.assertEqual(result['diagnostics']['execution_model_calls'],24)
        self.assertEqual(result['diagnostics']['virtual_step_IK_calls'],0)
        self.assertFalse(result['diagnostics']['trajectory_equivalent_to_pose_CEM'])
        self.assertIsNotNone(result['action'])
        for action in result['sequence']:
            pose=planner.goal_helper.kinematics(action['joint_goal'])
            np.testing.assert_array_equal(action['tcp_position'],pose[:3,3])
            np.testing.assert_allclose(action['tcp_quaternion'],matrix_quaternion(pose[:3,:3]),atol=1e-14)
        goal,reason,receipt=planner.goal_helper.solve_action_target(state,result['action'])
        self.assertIsNone(reason);self.assertEqual(receipt['DLS_solves'],0)
        np.testing.assert_array_equal(goal,result['action']['joint_goal'])
        self.assertTrue(all(not s['held'] for s in result['predicted_states']))
        self.assertEqual(result['warm_start']['knot_means'].shape,(8,2,7))

    def test_same_existing_phase_stage_and_terminal_cost_on_selected_trajectory(self):
        planner,state,candidates=fixture();result=planner.plan(state,candidates)
        candidate=next(c for c in candidates if c['id']==result['selected_candidate_id'])
        phase=deepcopy(state['phase_state']);total=0.;previous=None
        for i,action in enumerate(result['sequence']):
            before,after=result['predicted_states'][i:i+2]
            phase=mpc.update_phase(before,candidate,phase,planner.config)
            self.assertEqual(int(action['phase_index']),mpc.PHASES.index(phase['phase']))
            cost,_=mpc.stage_cost(before,action,after,candidate,phase,config=planner.config,previous_action=previous)
            total+=cost;previous=action
        phi,_=terminal_cost(state,result['predicted_states'][-1],candidate,state['phase_state'],
                            config=planner.config,terminal_config=planner.terminal_config)
        self.assertAlmostEqual(total+phi,result['cost'],places=10)

    def test_invalid_joint_targets_remain_unclipped_infinite_cost_not_fake_holds(self):
        planner,state,candidates=fixture();initial=mpc.validate_state(state)
        initial.update(q=state['q'],dq=state['dq'],finger_open_amounts=state['finger_open_amounts'])
        planner._prepare_nominals(initial,candidates,state['phase_state'])
        scores,_,_,actions=planner._rollout_batch(initial,candidates,np.arange(8),np.full((8,8,7),100.),state['phase_state'],None,None)
        self.assertTrue(np.isinf(scores).all())
        self.assertGreater(float(actions['joint_goal'].max()),90.)
        self.assertFalse(actions['joint_goal_feasible'].any())
        invalid=[dict(c,native_pose_qualified=False) for c in candidates]
        planner.constraint_cost=DiagnosticConstraints(lower=(-2.,-2.,-.5),upper=(2.,2.,2.))
        result=planner.plan(state,invalid)
        self.assertIsNone(result['action']);self.assertEqual(result['budget']['sequences'],256)

    def test_nominal_deadlines_predict_future_world_point_but_endpoint_reference_keeps_real_clock(self):
        planner,state,candidates=fixture();state['object_linear_velocity']=np.array([.02,0.,0.])
        result=planner.plan(state,candidates)
        np.testing.assert_array_equal(planner._nominals['time'][0],[c['deadline'] for c in candidates])
        for i,c in enumerate(candidates):
            target=mpc.contact_target(state,c,c['deadline'],pregrasp=True)
            np.testing.assert_allclose(planner._nominals['pose'][0,i,:3],target['position'],atol=1e-14)
        if result['action'] is not None:
            action=result['action'];before=result['predicted_states'][0]
            candidate=next(c for c in candidates if c['id']==result['selected_candidate_id'])
            ref=mpc.contact_target(before,candidate,.1,pregrasp=int(action['phase_index'])==0)
            np.testing.assert_allclose(action['endpoint_tcp_reference'][:3],ref['position'],atol=1e-14)

    def test_phase_nominal_validity_is_used_only_when_that_phase_is_needed(self):
        planner,state,candidates=fixture()
        initial=deepcopy(state);planner._prepare_nominals(initial,candidates,state['phase_state'])
        planner._nominals['feasible'][:]=True;planner._nominals['feasible'][2]=False
        scores,_,_,actions=planner._rollout_batch(initial,candidates,np.arange(8),np.zeros((8,8,7)),state['phase_state'],None,None)
        self.assertTrue(np.isfinite(scores).any())
        self.assertFalse(np.any(actions['phase_index']==3))
        planner._nominals['feasible'][1]=False
        scores,_,_,_=planner._rollout_batch(initial,candidates,np.arange(8),np.zeros((8,8,7)),state['phase_state'],None,None)
        self.assertTrue(np.isinf(scores).all())

    def test_warm_schema_absolute_shift_and_deadline_causality(self):
        planner,state,candidates=fixture();result=planner.plan(state,candidates);warm=result['warm_start']
        self.assertEqual(warm['param_space'],PARAM_SPACE)
        self.assertNotIn('execution_h',warm)
        bad=deepcopy(warm);del bad['param_space']
        with self.assertRaisesRegex(ValueError,'Cartesian'):planner.plan(state,candidates,warm_start=bad)
        bad=deepcopy(warm);bad['knot_means']=np.zeros((8,2,6))
        with self.assertRaisesRegex(ValueError,'statistics'):planner.plan(state,candidates,warm_start=bad)
        changed=deepcopy(candidates)
        for c in changed:
            if c['id']==warm['selected_candidate_id']:c['deadline']+=.1
        with self.assertRaisesRegex(ValueError,'deadline'):planner.plan(state,changed,warm_start=warm)
        next_state=deepcopy(result['predicted_states'][1]);next_state['actuator_state']={};next_state['phase_state']=result['phase_state']
        replay=planner.plan(next_state,candidates,warm_start=warm)
        self.assertEqual(replay['budget']['sequences'],256)
        if replay['action'] is not None:self.assertEqual(replay['diagnostics']['warm_start_evaluations_counted'],3)

    def test_actual_held_uses_fixed_lift_nominal_and_close_keeps_attachment_at_large_aperture(self):
        planner,state,candidates=fixture()
        state['held']=True;state['requested_opening']=0.;state['opening']=1.
        state['object_position']=state['tcp_position'].copy()
        state['lift_goal_position']=state['object_position']+np.array([0.,0.,.04])
        result=planner.plan(state,candidates)
        self.assertIsNotNone(result['action'])
        self.assertEqual(result['phase_state']['phase'],'hold')
        self.assertTrue(all(s['held'] for s in result['predicted_states']))
        self.assertTrue(all(a['opening']==0. for a in result['sequence']))
        self.assertTrue(all(int(a['nominal_kind'])==2 for a in result['sequence']))
        for pose in planner._nominals['pose'][2]:
            np.testing.assert_array_equal(pose[:3],state['lift_goal_position'])
        self.assertGreater(result['predicted_states'][0]['opening'],.9)
        self.assertLess(result['predicted_states'][1]['opening'],.65)
        self.assertTrue(result['predicted_states'][1]['held'])


if __name__=='__main__':unittest.main()
