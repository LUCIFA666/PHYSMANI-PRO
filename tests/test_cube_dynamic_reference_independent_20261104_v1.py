"""Independent analytic CPU contracts, no learned model/Native/recorded data."""
from copy import deepcopy
import ast,inspect,textwrap
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel,GRU_NOMINAL_BASIS
from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner
from our_mani.cube_dynamic_reference_20261104_v1 import planner as module
from tests.test_local_mpc_forecast_control_20261103 import bound_fixture,forecast_for


def fixture(now=12.3,held=False):
    base,state,candidates,_,_=bound_fixture()
    state=deepcopy(state);candidates=deepcopy(candidates)
    state['time']=now;state['phase_state']['updated_at']=now-.1
    for i,c in enumerate(candidates):
        c['deadline']=now+.5+.3*(i%2)
        c['object_tcp_position']=np.asarray([.001*i,.002,0.])
    state['held']=held
    if held:state['requested_opening']=0.
    forecast=forecast_for(state)
    bound=BoundForecastExecutionModel(base.execution_model,forecast,anchor_time=now)
    planner=module.DynamicReferenceForecastJointGoalCEMPlannerV1(execution_model=bound,goal_helper=base.goal_helper,terminal_config=base.terminal_config)
    return planner,state,candidates,forecast


def node(fn):return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]


class DynamicReferenceIndependent(unittest.TestCase):
    def test_136_nonzero_clock_rotated_offsets_scatter_and_actual_q_seeds(self):
        planner,state,candidates,forecast=fixture();captured={};before=deepcopy(candidates)
        def solve(q,p,r):
            captured.update(q=q.copy(),p=p.copy(),r=r.copy())
            goals=np.repeat((np.arange(136)/10000.)[:,None],7,axis=1)
            mask=np.ones(136,bool);mask[64+2*8+3]=False
            return dict(joint_goal=goals,joint_goal_feasible=mask)
        with patch.object(planner.goal_helper.adapter,'solve',side_effect=solve):planner._prepare_nominals(state,candidates,state['phase_state'])
        self.assertEqual(captured['q'].shape,(136,7))
        np.testing.assert_array_equal(captured['q'],np.broadcast_to(state['q'],(136,7)))
        n=planner._nominals
        self.assertEqual(n['joint_goal'].shape,(3,8,8,7));self.assertEqual(n['feasible'].shape,(3,8,8))
        self.assertEqual(n['pose'].shape,(3,8,8,7));self.assertEqual(n['time'].shape,(3,8,8))
        for kind in (0,1):
            for slot,candidate in enumerate(candidates):
                for k in range(8):
                    at=state['time']+.1*(k+1);pose,_,_=forecast.at('target',at)
                    local=candidate['object_tcp_position']+(candidate['approach_direction']*candidate['pregrasp_offset']if kind==0 else 0.)
                    np.testing.assert_allclose(n['pose'][kind,slot,k,:3],pose.apply(local),atol=1e-14)
                    got=Rotation.from_quat(n['pose'][kind,slot,k,3:]).as_matrix()
                    want=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
                    np.testing.assert_allclose(got,want,atol=1e-14)
                    self.assertEqual(n['time'][kind,slot,k],at)
                    np.testing.assert_allclose(n['joint_goal'][kind,slot,k],(kind*64+k*8+slot)/10000.)
        self.assertFalse(n['feasible'][1,3,2]);self.assertTrue(n['feasible'][1,2,3])
        for k in range(8):
            np.testing.assert_array_equal(n['joint_goal'][2,:,k],n['joint_goal'][2,:,0])
            np.testing.assert_array_equal(n['time'][2,:,k],np.full(8,state['time']))
        for a,b in zip(candidates,before):
            self.assertEqual(a['deadline'],b['deadline']);np.testing.assert_array_equal(a['object_tcp_position'],b['object_tcp_position'])

    def test_deadline_mask_and_malformed_batch_fail_closed(self):
        planner,state,candidates,_=fixture();candidates[0]['deadline']=state['time']-.01;candidates[1]['deadline']=state['time']+.81
        def solve(q,p,r):return dict(joint_goal=np.zeros((136,7)),joint_goal_feasible=np.ones(136,bool))
        with patch.object(planner.goal_helper.adapter,'solve',side_effect=solve):planner._prepare_nominals(state,candidates,state['phase_state'])
        self.assertFalse(planner._nominals['feasible'][:,:2].any());self.assertTrue(planner._nominals['feasible'][:,2:].all())
        self.assertEqual(candidates[0]['deadline'],state['time']-.01)
        for shape,dtype in (((135,),bool),((136,),int)):
            with patch.object(planner.goal_helper.adapter,'solve',return_value=dict(joint_goal=np.zeros((136,7)),joint_goal_feasible=np.ones(shape,dtype=dtype))):
                with self.assertRaises(ValueError):planner._prepare_nominals(state,candidates,state['phase_state'])

    def test_rollout_uses_k_for_q_pose_time_and_feasibility_without_command_clipping(self):
        planner,state,candidates,_=fixture();planner._prepare_nominals(state,candidates,state['phase_state'])
        # The IK test stub supplies finite actual-joint values; this tests table
        # dispatch, not whether a synthetic IK result is physically qualified.
        for kind in range(3):
            for mode in range(8):
                for k in range(8):planner._nominals['joint_goal'][kind,mode,k]=state['q']+1e-4*(kind*64+mode*8+k)
        planner._nominals['feasible'][:]=True
        modes=np.arange(8);controls=np.zeros((8,8,7));old=state['phase_state']
        scores,_,_,actions=planner._rollout_batch(state,candidates,modes,controls,old,None,None)
        for k in range(8):
            kinds=actions['nominal_kind'][:,k]
            np.testing.assert_array_equal(kinds,np.where(actions['phase_index'][:,k]==0,0,np.where(actions['phase_index'][:,k]==3,2,1)))
            for mode in modes:
                kind=kinds[mode]
                for key,field in (('nominal_joint_goal','joint_goal'),('nominal_tcp_reference','pose'),('nominal_reference_time','time')):
                    np.testing.assert_array_equal(actions[key][mode,k],planner._nominals[field][kind,mode,k])
        selected_kind=int(actions['nominal_kind'][0,3]);planner._nominals['feasible'][selected_kind,0,3]=False
        masked,_,_,again=planner._rollout_batch(state,candidates,modes,controls,old,None,None)
        self.assertTrue(np.isinf(masked[0]));self.assertFalse(again['joint_goal_feasible'][0,3])
        invalid,_,_,raw=planner._rollout_batch(state,candidates,modes,np.full((8,8,7),100.),old,None,None)
        self.assertTrue(np.isinf(invalid).all());self.assertFalse(raw['joint_goal_feasible'].any());self.assertGreater(raw['joint_goal'].max(),90.)
        hold,hs,hc,_=fixture(held=True);hold._prepare_nominals(hs,hc,hs['phase_state'])
        hold._nominals['feasible'][:]=True
        _,_,_,ha=hold._rollout_batch(hs,hc,modes,controls,hs['phase_state'],None,None)
        self.assertTrue((ha['nominal_kind']==2).all());self.assertTrue((ha['opening']==0.).all())

    def test_same_frame_and_shift_one_absolute_warm_not_forecast_or_deadline_shift(self):
        planner,state,candidates,_=fixture(now=0.);first=planner.plan(state,candidates);self.assertIsNotNone(first['action'])
        warm=first['warm_start'];self.assertEqual(warm['nominal_basis_version'],module.DYNAMIC_NOMINAL_BASIS_VERSION)
        for advance in (0,1):
            current=deepcopy(state if not advance else first['predicted_states'][1]);current['actuator_state']={};current['phase_state']=deepcopy(first['phase_state'])
            forecast=forecast_for(current)
            bound=BoundForecastExecutionModel(planner.execution_model.execution_model,forecast,anchor_time=current['time'])
            new=module.DynamicReferenceForecastJointGoalCEMPlannerV1(execution_model=bound,goal_helper=planner.goal_helper,terminal_config=planner.terminal_config)
            capture=[];rollout=new._rollout_batch
            def spy(*args):
                capture.append(deepcopy(args[-1]));return rollout(*args)
            before=deepcopy(warm)
            with patch.object(new,'_rollout_batch',side_effect=spy):result=new.plan(current,candidates,warm_start=warm)
            self.assertEqual(len(capture),3);self.assertEqual(result['budget']['iteration_sequences'],[88,88,80])
            self.assertEqual(result['budget']['sequences'],256)
            expected=warm['sequence'][advance:]+[deepcopy(warm['sequence'][-1])for _ in range(advance)]
            for sequences in capture:
                for got,want in zip(sequences,expected):np.testing.assert_array_equal(got['joint_goal'],want['joint_goal'])
            self.assertEqual(warm['deadline'],before['deadline']);self.assertEqual(warm['time'],before['time'])
            self.assertEqual(warm['nominal_basis_version'],module.DYNAMIC_NOMINAL_BASIS_VERSION)
            bad=deepcopy(warm);bad['nominal_basis_version']=GRU_NOMINAL_BASIS
            fresh=module.DynamicReferenceForecastJointGoalCEMPlannerV1(execution_model=bound,goal_helper=planner.goal_helper)
            with self.assertRaises(ValueError):fresh.plan(current,candidates,warm_start=bad)

    def test_expired_warm_dropped_and_cannot_revive_old_deadline(self):
        planner,state,candidates,_=fixture(now=0.);first=planner.plan(state,candidates);warm=first['warm_start'];selected=warm['selected_candidate_id']
        state=deepcopy(state);state['time']=warm['deadline']+.05;state['phase_state']['updated_at']=state['time']-.1
        forecast=forecast_for(state);bound=BoundForecastExecutionModel(planner.execution_model.execution_model,forecast,anchor_time=state['time'])
        for c in candidates:
            if c['id']!=selected:c['deadline']=state['time']+.5
        new=module.DynamicReferenceForecastJointGoalCEMPlannerV1(execution_model=bound,goal_helper=planner.goal_helper)
        capture=[];real=new._rollout_batch
        def spy(*args):capture.append(args[-1]);return real(*args)
        with patch.object(new,'_rollout_batch',side_effect=spy):result=new.plan(state,candidates,warm_start=warm)
        self.assertTrue(all(x is None for x in capture));self.assertTrue(result['diagnostics']['expired_selected_warm_discarded'])
        selected_index=next(i for i,c in enumerate(candidates)if c['id']==selected)
        self.assertFalse(new._nominals['feasible'][:,selected_index].any());self.assertNotEqual(result['selected_candidate_id'],selected)
        self.assertEqual(candidates[selected_index]['deadline'],warm['deadline'])

    def test_original_rollout_and_sampling_math_exact_except_declared_table_indices(self):
        old=node(JointGoalCEMPlanner._rollout_batch);new=node(module._DynamicJointGoalCEMPlannerV1._rollout_batch)
        count=[0]
        class Normalize(ast.NodeTransformer):
            def visit_Subscript(self,n):
                self.generic_visit(n)
                if isinstance(n.value,ast.Subscript)and isinstance(n.value.value,ast.Attribute)and n.value.value.attr=='_nominals':
                    if isinstance(n.slice,ast.Tuple)and len(n.slice.elts)==3 and isinstance(n.slice.elts[-1],ast.Name)and n.slice.elts[-1].id=='k':
                        n.slice.elts=n.slice.elts[:-1];count[0]+=1
                return n
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(Normalize().visit(new),include_attributes=False));self.assertEqual(count[0],4)
        old=node(JointGoalCEMPlanner.plan);new=node(module._DynamicJointGoalCEMPlannerV1.plan)
        # The only new plan statement changes absolute-warm rebasing from one
        # static q to the corresponding kind/slot/k moving q at every endpoint.
        class Dense(ast.NodeTransformer):
            def visit_Subscript(self,n):
                self.generic_visit(n)
                if isinstance(n.value,ast.Subscript)and isinstance(n.value.value,ast.Attribute)and n.value.value.attr=='_nominals':
                    if isinstance(n.slice,ast.Tuple)and len(n.slice.elts)==3:n.slice.elts=n.slice.elts[:2]
                return n
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(Dense().visit(new),include_attributes=False))

if __name__=='__main__':unittest.main()
