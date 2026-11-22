"""New V66 isolated CPU contracts; no learned forward/CEM scan or Native."""
import ast
from copy import deepcopy
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_control_20261103 import public_candidate_local_hold_v66 as subject
from our_mani.local_mpc_control_20261103.joint_goal_cem import JointGoalCEMPlanner
from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualGraspCoordinator
from our_mani.local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Coordinator
from our_mani.local_mpc_control_20261103.hold_orientation_guard_prototype_20261103 import HoldMicroOrientationGuard
from our_mani.local_mpc_20261102.records import plain
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture as pure_fixture


class Oracle:
    """Explicit synthetic no-step service; never a real collision certificate."""
    def __init__(self):self.now=0.;self.calls=[];self.result=False;self.advance=False
    def clock(self):return self.now
    def probe_joint_collision(self,q):
        self.calls.append(q.copy())
        if self.advance:self.now+=.05
        return self.result


def fixture():
    base,state,candidates=pure_fixture()
    state.update(held=True,requested_opening=0.,opening=.5,finger_open_amounts=np.full(2,.5),grasp_motion_mode='ATTACHED_PUBLIC')
    state['actuator_state']=dict(reference_clock=state['time'],reference_valid=False,gripper_target=0.)
    state['lift_goal_position']=state['object_position']+np.array([.1,.08,.15])
    state['phase_state']=dict(phase='hold',hold_tcp_quaternion=state['tcp_quaternion'].copy(),
        lift_goal_position=state['lift_goal_position'].copy(),updated_at=state['time'],dwell=0.,lost_hold=0.)
    oracle=Oracle()
    # Real model protocol type, but only synthetic PoE/lag fixture callbacks.
    # No initialize/step/forward is invoked by these isolated plan tests.
    planner=subject.LocalHoldPlannerV66(base.config,execution_model=base.execution_model,
        goal_helper=base.goal_helper,terminal_config=base.terminal_config,backend=oracle)
    return planner,state,candidates,oracle,base


def solved(state,*,bad_algebraic=None,bad_bound=None):
    q=np.broadcast_to(state['q'],(24,7)).copy();q[:,0]+=np.arange(24)*1e-5
    flags=np.ones(24,bool)
    if bad_algebraic is not None:flags[16+bad_algebraic]=False
    if bad_bound is not None:q[16+bad_bound,0]=99.
    return dict(joint_goal=q,joint_goal_feasible=flags)


class V66Independent(unittest.TestCase):
    def test_new_local_target_overwrites_far_false_preserves_only_undefined_mask(self):
        p,state,cs,oracle,_=fixture();before=deepcopy(state)
        for c in cs:c['native_pose_qualified']=False;c['deadline']=-1.
        cs[3]['parallel_gripper_symmetry_choice']=None
        result=solved(state);calls=[]
        def solve(q,pos,quat):calls.append((q.copy(),pos.copy(),quat.copy()));return deepcopy(result)
        with patch.object(p.goal_helper.adapter,'solve',side_effect=solve):p._prepare_nominals(state,cs,state['phase_state'])
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][0].shape,(24,7))
        self.assertEqual(len(oracle.calls),7)
        self.assertEqual([c['native_pose_qualified']for c in cs],[True,True,True,False,True,True,True,True])
        delta=state['lift_goal_position']-state['object_position'];target=state['tcp_position']+.012*delta/np.linalg.norm(delta)
        np.testing.assert_allclose(calls[0][1][16:],np.tile(target,(8,1)),atol=1e-15)
        np.testing.assert_array_equal(calls[0][2][16:],np.tile(state['phase_state']['hold_tcp_quaternion'],(8,1)))
        for i,case in enumerate(p.fresh_nominal_qualification['cases']):
            np.testing.assert_array_equal(case['newly_solved_nominal_joint_target'],result['joint_goal'][16+i])
            self.assertFalse(case['old_far_qualification_used']);self.assertEqual(case['actuator_state'],state['actuator_state'])
        self.assertEqual(plain(state),plain(before))

    def test_old_true_does_not_bypass_new_algebraic_bounds_or_collision(self):
        p,state,cs,oracle,_=fixture()
        for c in cs:c['native_pose_qualified']=True
        result=solved(state,bad_algebraic=1,bad_bound=2);oracle.result=True
        with patch.object(p.goal_helper.adapter,'solve',return_value=result):p._prepare_nominals(state,cs,state['phase_state'])
        self.assertFalse(any(c['native_pose_qualified']for c in cs))
        self.assertEqual(len(oracle.calls),6)
        self.assertIsNone(p.fresh_nominal_qualification['cases'][1]['static_collision'])
        self.assertIsNone(p.fresh_nominal_qualification['cases'][2]['static_collision'])

    def test_unknown_models_clock_and_collision_types_rejected(self):
        p,state,cs,oracle,base=fixture()
        unknown=SimpleNamespace(requires_joint_state=True,requires_joint_goal=True)
        with self.assertRaises(TypeError):subject.LocalHoldPlannerV66(base.config,execution_model=unknown,goal_helper=base.goal_helper,backend=oracle)
        for bad in (float('nan'),True,.1):
            oracle.now=bad
            with self.subTest(clock=bad),self.assertRaises(ValueError):p.plan(state,cs)
        for bad in (None,0,1):
            oracle.now=0.;oracle.result=bad;oracle.calls=[]
            with patch.object(p.goal_helper.adapter,'solve',return_value=solved(state)):
                with self.subTest(collision=bad),self.assertRaises(TypeError):p._prepare_nominals(state,cs,state['phase_state'])
        oracle.now=0.;oracle.result=False;oracle.advance=True
        with patch.object(p.goal_helper.adapter,'solve',return_value=solved(state)):
            with self.assertRaises((ValueError,RuntimeError)):p._prepare_nominals(state,cs,state['phase_state'])

    def test_fresh_hold_binding_cold_warm_and_action_identity_without_rollout(self):
        p,state,cs,oracle,_=fixture();before=deepcopy(state);old_execution=p.execution_model
        selected_q=np.asarray(state['q']).copy();action=dict(joint_goal=selected_q,opening=0.)
        seen=[]
        def original_plan(self,actual,candidates,*,warm_start):
            seen.append(warm_start)
            if type(self.execution_model)is not HoldMicroOrientationGuard:raise AssertionError('fresh guard absent')
            self.execution_model.initialize_batch(actual,1)
            return dict(action=action,diagnostics={},warm_start=dict(sequence=[]))
        # Initialization is synthetic callback code; no model step/forward/CEM.
        for incoming in ({'nominal_basis_version':'old_far_basis'}, {'nominal_basis_version':subject.HOLD_BASIS_VERSION}):
            with patch.object(JointGoalCEMPlanner,'plan',new=original_plan):result=p.plan(state,cs,warm_start=incoming)
            self.assertIs(result['action'],action);self.assertIs(result['action']['joint_goal'],selected_q)
            self.assertTrue(result['warm_start']['hold_cold_initialization_required'])
            self.assertEqual(result['warm_start']['nominal_basis_version'],subject.HOLD_BASIS_VERSION)
            self.assertIs(p.execution_model,old_execution)
        self.assertEqual(seen,[None,None]);self.assertEqual(plain(state),plain(before))
        stale=deepcopy(state);stale['actuator_state']['reference_clock']-=.05
        with patch.object(JointGoalCEMPlanner,'plan',new=original_plan):
            with self.assertRaises(ValueError):p.plan(stale,cs)
        with patch.object(JointGoalCEMPlanner,'plan',side_effect=RuntimeError('isolated failure')):
            with self.assertRaises(RuntimeError):p.plan(state,cs)
        self.assertIs(p.execution_model,old_execution)

    def test_mro_observer_cost_and_nonhold_dispatch_AST(self):
        mro=subject.CandidateLocalHoldV66Coordinator.__mro__
        self.assertLess(mro.index(ActiveVisualGraspV41Coordinator),mro.index(subject._V66V4Dispatch))
        self.assertLess(mro.index(subject._V66V4Dispatch),mro.index(ActiveVisualGraspCoordinator))
        self.assertIs(subject.ObservedPublicCandidateLocalHoldV66Backend.sample,subject.base.ObservedPublicCandidateReverificationV65Backend.sample)
        self.assertIs(subject.LocalHoldPlannerV66._rollout_batch,JointGoalCEMPlanner._rollout_batch)

    def test_control_loop_and_nonhold_decide_ast_exact_after_declared_changes(self):
        import textwrap
        old=ast.parse(textwrap.dedent(inspect.getsource(ActiveVisualGraspCoordinator.decide)))
        new=ast.parse(textwrap.dedent(inspect.getsource(subject._V66V4Dispatch.decide)))
        class RemoveHoldOnly(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.unparse(node.test)=="mode == 'ATTACHED_PUBLIC'":
                    text=ast.unparse(node)
                    if 'LocalHoldPlannerV66('in text:return node.orelse
                    if 'planner.fresh_nominal_qualification'in text:return None
                return self.generic_visit(node)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(RemoveHoldOnly().visit(new),include_attributes=False))
        old=ast.parse(inspect.getsource(subject.base.run_public_candidate_reverification_v65))
        new=ast.parse(inspect.getsource(subject.run_public_candidate_local_hold_v66))
        oldloop=next(x for x in ast.walk(old)if isinstance(x,ast.While));newloop=next(x for x in ast.walk(new)if isinstance(x,ast.While))
        class Names(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,str):node.value=node.value.replace('v66','v65')
                return node
        self.assertEqual(ast.dump(oldloop,include_attributes=False),ast.dump(Names().visit(newloop),include_attributes=False))


if __name__=='__main__':unittest.main()
