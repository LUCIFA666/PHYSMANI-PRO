"""Independent V72 phase precision contracts; synthetic CPU fixtures only."""
from copy import deepcopy
import ast
import inspect
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_20261102 import mpc
from our_mani.local_mpc_control_20261103 import free_phase_dwell_v72 as stable
from our_mani.local_mpc_control_20261103 import lazy_free_joint_goal_cem_v71 as old_lazy
from our_mani.local_mpc_control_20261103 import lazy_free_joint_goal_cem_v72 as new_lazy
from our_mani.local_mpc_control_20261103 import public_candidate_lazy_free_v71 as old_producer
from our_mani.local_mpc_control_20261103 import public_candidate_stable_dwell_v72 as new_producer
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner
from tests.test_local_mpc_forecast_control_20261103 import bound_fixture
from tests.test_lazy_free_planner_v71_independent_20261103 import QueryOnlyBackend, equal


def phase_fixture(last=5.6000000000000005):
    state=dict(time=5.7,held=False,opening=1.,requested_opening=1.,
        object_position=np.array([.3,.1,.5]),object_quaternion=np.array([0.,0.,0.,1.]),
        object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),
        tcp_position=np.array([.32423,.1,.5]),tcp_quaternion=mpc.quat_exp(np.array([0.,0.,.05047])),
        tcp_velocity=np.array([.04382,0.,0.]))
    candidate=dict(object_tcp_position=np.zeros(3),object_tcp_quaternion=np.array([0.,0.,0.,1.]),
        approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.03,
        allowed_relative_speed=.14,required_opening=1.,opening=0.,closing_opening=0.,deadline=5.8)
    phase=dict(phase='align',dwell=0.,lost_hold=0.,updated_at=last)
    return state,candidate,phase


def function(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]


class StableDwellIndependentContracts(unittest.TestCase):
    def test_actual_subtraction_tail_passes_without_mutating_time_or_threshold(self):
        state,candidate,phase=phase_fixture();before=deepcopy((state,candidate,phase))
        self.assertEqual(state['time']-phase['updated_at'],.09999999999999964)
        old=mpc.update_phase(state,candidate,phase)
        self.assertEqual(old['phase'],'align')
        result=stable.update_phase(state,candidate,phase)
        self.assertEqual(result['phase'],'close')
        self.assertEqual(result['close_entered_at'],5.7)
        self.assertEqual(result['updated_at'],5.7)
        self.assertEqual(mpc.MPCConfig().align_dwell,.1)
        equal(self,(state,candidate,phase),before)

    def test_99ms_and_more_than_numeric_tolerance_shortfall_still_rejected(self):
        for elapsed in (.099,.1-2e-9):
            state,candidate,phase=phase_fixture(last=5.7-elapsed)
            result=stable.update_phase(state,candidate,phase)
            self.assertEqual(result['phase'],'align')
            self.assertLess(result['dwell'],.1)
        state,candidate,phase=phase_fixture(last=5.7-.1)
        self.assertEqual(stable.update_phase(state,candidate,phase)['phase'],'close')

    def test_original_geometry_aperture_and_deadline_gates_still_reject(self):
        for name in ('position','rotation','speed','aperture','deadline','reopening'):
            with self.subTest(name=name):
                state,candidate,phase=phase_fixture()
                if name=='position':state['tcp_position'][0]=state['object_position'][0]+.025001
                elif name=='rotation':state['tcp_quaternion']=mpc.quat_exp(np.array([0.,0.,.200001]))
                elif name=='speed':state['tcp_velocity'][0]=.140001
                elif name=='aperture':state['opening']=.949999
                elif name=='deadline':candidate['deadline']=state['time']+.250001
                elif name=='reopening':phase['reopening']=True;state['opening']=.89
                expected=mpc.update_phase(state,candidate,phase)
                result=stable.update_phase(state,candidate,phase)
                equal(self,result,expected)
                self.assertNotEqual(result['phase'],'close')

    def test_other_phases_duplicate_time_and_lost_hold_are_original_exact(self):
        for name in ('approach','close','hold','held','duplicate','reopening'):
            with self.subTest(name=name):
                state,candidate,phase=phase_fixture()
                if name in mpc.PHASES:phase['phase']=name
                elif name=='held':state['held']=True
                elif name=='duplicate':phase['updated_at']=state['time']
                elif name=='reopening':phase['reopening']=True
                before=deepcopy((state,candidate,phase))
                equal(self,stable.update_phase(state,candidate,phase),mpc.update_phase(state,candidate,phase))
                equal(self,(state,candidate,phase),before)
        state,candidate,phase=phase_fixture(last=5.8)
        with self.assertRaises(ValueError):stable.update_phase(state,candidate,phase)

    def test_vector_and_complete_toy_plan_agree_on_first_opening(self):
        for elapsed,expected in ((.09999999999999964,0.),(.099,1.)):
            with self.subTest(elapsed=elapsed):
                base,state,candidates,forecast,_=bound_fixture()
                state.update(grasp_motion_mode='FREE_GRU',Native_object_state_used=False,
                    Native_held_used=False,Native_success_used=False)
                state['phase_state']=dict(phase='align',dwell=0.,lost_hold=0.,updated_at=-elapsed)
                for c in candidates:c['deadline']=.2
                result=dict(available=True,free_forecast_available=True,mode='FREE',timestamp=0.,forecast=forecast,
                    Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,rotation_observable=False)
                model=BoundForecastExecutionModel.from_public_bridge(base.execution_model,result)
                backend=QueryOnlyBackend()
                planner=new_lazy.LazyQualifiedForecastJointGoalCEMPlannerV72(execution_model=model,
                    goal_helper=base.goal_helper,terminal_config=base.terminal_config,backend=backend)
                calls=[];real=planner._rollout_batch
                def call(*args):
                    value=real(*args);calls.append((len(args[2]),value[3]['opening'][:,0].copy()))
                    return value
                before=deepcopy(state)
                with patch.object(planner,'_rollout_batch',side_effect=call):answer=planner.plan(state,candidates)
                self.assertEqual([n for n,_ in calls],[88,88,80])
                for _,opening in calls:np.testing.assert_array_equal(opening,np.full(len(opening),expected))
                self.assertIsNotNone(answer['action'])
                self.assertEqual(answer['action']['opening'],expected)
                self.assertEqual(answer['phase_state']['phase'],'close'if expected==0. else'align')
                self.assertEqual(answer['budget']['sequences'],256)
                self.assertEqual(len(backend.queries),8)
                equal(self,state,before)
                self.assertIsNone(planner._caller_state)

    def test_scalar_and_vector_only_alignment_compare_changes_and_CEM_plan_AST_exact(self):
        class UndoOnlyAlignmentPredicate(ast.NodeTransformer):
            def __init__(self):self.count=0
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name)and node.func.id=='alignment_dwell_reached':
                    self.count+=1
                    self_outer.assertEqual(len(node.args),2);self_outer.assertEqual(node.keywords,[])
                    return ast.Compare(left=node.args[0],ops=[ast.GtE()],comparators=[node.args[1]])
                return node
        self_outer=self
        for a,b in ((mpc.update_phase,stable.update_phase),
                    (JointGoalCEMPlanner._rollout_batch,new_lazy._LazyJointGoalCEMPlannerV72._rollout_batch)):
            transform=UndoOnlyAlignmentPredicate();after=transform.visit(function(b))
            self.assertEqual(transform.count,1)
            self.assertEqual(ast.dump(function(a),include_attributes=False),ast.dump(after,include_attributes=False))
        self.assertEqual(stable.ALIGNMENT_DWELL_NUMERICAL_ATOL_SECONDS,1e-9)
        self.assertEqual(ast.dump(function(old_lazy._LazyJointGoalCEMPlannerV71.plan),include_attributes=False),
            ast.dump(function(new_lazy._LazyJointGoalCEMPlannerV72.plan),include_attributes=False))
        self.assertIs(new_lazy.RankedRolloutPool,old_lazy.RankedRolloutPool)
        self.assertIs(new_lazy.select_first_qualified,old_lazy.select_first_qualified)
        self.assertIs(new_lazy.validate_feedback_state,old_lazy.validate_feedback_state)
        self.assertIs(new_lazy.LazyQualifiedForecastJointGoalCEMPlannerV72._prepare_nominals,
            old_lazy.LazyQualifiedForecastJointGoalCEMPlannerV71._prepare_nominals)

    def test_dispatch_only_FREE_class_changes_hold_probe_and_MRO_preserved(self):
        class Normalize(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='LazyQualifiedForecastJointGoalCEMPlannerV72':node.id='LazyQualifiedForecastJointGoalCEMPlannerV71'
                return node
        old=function(old_producer._V71V4Dispatch.decide)
        new=Normalize().visit(function(new_producer._V72V4Dispatch.decide))
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        self.assertIs(new_producer.ResponseSeedHoldPlannerV69,old_producer.ResponseSeedHoldPlannerV69)
        self.assertIs(new_producer.JointGoalCEMPlanner,old_producer.JointGoalCEMPlanner)
        order=new_producer.CandidateStableDwellV72Coordinator.mro()
        self.assertLess(order.index(new_producer._V72V4Dispatch),order.index(old_producer._V71V4Dispatch))


if __name__=='__main__':unittest.main()
