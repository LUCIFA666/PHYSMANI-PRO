from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.joint_goals import JointGoalConfig,PoEJointGoalHelper
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from our_mani.local_mpc_20261102 import mpc
from tests.test_local_mpc_p2_20261103_kinematics import planar_snapshot


class NoIKBackend:
    def __init__(self,collide_at=None):self.calls=[];self.collide_at=collide_at
    def solve_ik(self,*args):raise AssertionError('Native IK forbidden')
    def step(self,*args):raise AssertionError('physical stepping forbidden')
    def probe_joint_collision(self,q):
        self.calls.append(np.asarray(q).copy())
        return len(self.calls)==self.collide_at


def fixture():
    snapshot=planar_snapshot();snapshot['tcp_reference_matrix'][2,3]=1.
    chain=PoEKinematics(snapshot);q=chain.q_ref.copy();pose=chain(q)
    limits=SimpleNamespace(lower=np.full(2,-2.5),upper=np.full(2,2.5))
    helper=PoEJointGoalHelper(chain,limits)
    state=dict(q=q,dq=np.zeros(2),tcp_position=pose[:3,3],tcp_quaternion=matrix_quaternion(pose[:3,:3]),tcp_velocity=np.zeros(3),
               object_position=pose[:3,3]-[0.,0.,.06],object_quaternion=np.array([0.,0.,0.,1.]),
               object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),time=0.,opening=1.,held=False,
               phase_state=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=None))
    candidates=[dict(id=f'c{i}',object_tcp_position=np.zeros(3),object_tcp_quaternion=state['tcp_quaternion'],
                     deadline=.5+.3*(i%2),approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.06,
                     opening=0.,required_opening=1.,allowed_relative_speed=.1) for i in range(8)]
    action=dict(tcp_position=state['tcp_position'].copy(),tcp_quaternion=state['tcp_quaternion'].copy(),opening=1.,joint_goal=q.copy())
    # The synthetic fixture is outside the default diagnostic x bound; use
    # wider explicit bounds only for these algebraic unit checks.
    helper=PoEJointGoalHelper(chain,limits,JointGoalConfig(lower=(-2.,-2.,.76),upper=(2.,2.,1.65)))
    return helper,state,candidates,action


class JointGoalTests(unittest.TestCase):
    def test_candidates_exactly_eight_once_batch_no_native_IK_or_step(self):
        helper,state,candidates,_=fixture();backend=NoIKBackend()
        original=deepcopy(state);solve=helper.adapter.solve
        calls=[]
        def observed(q,p,r):calls.append(q.shape);return solve(q,p,r)
        with patch.object(helper.adapter,'solve',side_effect=observed):
            values,receipt=helper.qualify_candidates(backend,state,candidates)
        self.assertEqual(calls,[(8,2)])
        self.assertEqual(len(values),8);self.assertEqual(len(receipt['cases']),8)
        self.assertEqual(len(backend.calls),8)
        self.assertTrue(all(v['native_pose_qualified'] for v in values))
        self.assertEqual(receipt['Native_IK_calls'],0);self.assertEqual(receipt['physical_steps'],0)
        np.testing.assert_array_equal(state['q'],original['q'])
        with self.assertRaises(ValueError):helper.solve_candidate_targets(state,candidates[:7])

    def test_candidate_geometry_time_matches_original_and_failure_ledger_retained(self):
        helper,state,candidates,_=fixture();state['object_linear_velocity']=np.array([.01,0.,0.])
        result=helper.solve_candidate_targets(state,candidates)
        for c,target in zip(candidates,result['targets']):
            old=mpc.contact_target(mpc.validate_state(state),mpc.validate_candidate(c),c['deadline'],pregrasp=True)
            np.testing.assert_array_equal(target['position'],old['position'])
        state['object_position']=np.array([10.,0.,1.])
        backend=NoIKBackend();values,receipt=helper.qualify_candidates(backend,state,candidates)
        self.assertEqual(len(values),8);self.assertEqual(len(receipt['cases']),8)
        self.assertFalse(any(v['native_pose_qualified'] for v in values))
        self.assertEqual(backend.calls,[])

    def test_selected_joint_goal_reused_bitwise_never_resolved_different_branch(self):
        helper,state,_,action=fixture();backend=NoIKBackend()
        with patch.object(helper.adapter,'solve',side_effect=AssertionError('second IK solve forbidden')):
            goal,reason,receipt=helper.qualify_action(backend,state,action)
        self.assertIsNone(reason)
        np.testing.assert_array_equal(goal,action['joint_goal'])
        self.assertTrue(receipt['model_joint_goal_match']);self.assertEqual(receipt['DLS_solves'],0)
        self.assertEqual([r['alpha'] for r in receipt['collision_probes']],[1.,.125,.25,.375,.5,.625,.75,.875])
        self.assertFalse(receipt['whole_path_certified'])

    def test_missing_model_goal_requires_explicit_cartesian_baseline_and_one_solve(self):
        helper,state,_,action=fixture();del action['joint_goal']
        goal,reason,receipt=helper.solve_action_target(state,action)
        self.assertIsNone(goal);self.assertEqual(reason,'missing_selected_model_joint_goal')
        solve=helper.adapter.solve;calls=[]
        def observed(q,p,r):calls.append(q.shape);return solve(q,p,r)
        with patch.object(helper.adapter,'solve',side_effect=observed):
            goal,reason,receipt=helper.qualify_action(NoIKBackend(),state,action,require_predicted_joint_goal=False)
        self.assertIsNone(reason);self.assertEqual(calls,[(1,2)])
        self.assertFalse(receipt['model_joint_goal_match']);self.assertEqual(receipt['DLS_solves'],1)

    def test_fk_tolerance_bounds_workspace_and_static_path_collision_fail_closed(self):
        helper,state,_,action=fixture()
        incorrect=deepcopy(action);incorrect['joint_goal']+=.2
        goal,reason,_=helper.solve_action_target(state,incorrect)
        self.assertIsNone(goal);self.assertEqual(reason,'PoE_action_pose_or_joint_bounds_infeasible')
        bad=deepcopy(action);bad['tcp_position'][0]=3.
        goal,reason,_=helper.qualify_action(NoIKBackend(),state,bad)
        self.assertIsNone(goal);self.assertEqual(reason,'action_outside_diagnostic_workspace')
        goal,reason,receipt=helper.qualify_action(NoIKBackend(collide_at=3),state,action)
        self.assertIsNone(goal);self.assertEqual(reason,'sampled_joint_path_arm_static_collision')
        self.assertEqual(len(receipt['collision_probes']),3)
        self.assertEqual(helper.config.position_tolerance_m,.002);self.assertEqual(helper.config.angle_tolerance_rad,.02)
        with self.assertRaises(ValueError):JointGoalConfig(path_samples=(.5,))


if __name__=='__main__':unittest.main()
