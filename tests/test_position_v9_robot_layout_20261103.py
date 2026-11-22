"""Robot-only mesh and synthetic layout tests; no simulation or task data."""
from copy import deepcopy
from unittest.mock import patch
import json
import unittest
import numpy as np
from our_mani.local_mpc_20261103 import position_v9_robot_layout as subject


def fixture(ordinal=0,feasible=True):
    plan=dict(DLS=subject.fixed.DLS,tool_quaternion_xyzw=[1.,0.,0.,0.],world_outside_position_m=[100.,100.,-100.])
    parent=dict(parent_id='synthetic',seed=1,near_direction_ordinal=ordinal,
        near_direction_unit_world=subject.fixed.DIRECTIONS[ordinal].tolist(),
        TRAIN_q_source=dict(joint_positions=[0.]*7),shots=subject.shot_specs_for_parent(ordinal))
    cameras={name:dict(K=[[100,0,128],[0,100,128],[0,0,1]],T=np.eye(4),near=.1,far=10.,width=256,height=256)for name in subject.fixed.b.CAMERAS}
    targets=[]
    class Kinematics:
        def __call__(self,q):
            value=np.eye(4);value[:3,:3]=np.diag([1.,-1.,-1.]);value[:3,3]=targets[int(q[0])];return value
    def solve(adapter,seed,positions,quaternions):
        selfseed=seed.copy();targets.extend(positions.copy());q=np.zeros((2,7));q[1,0]=1.
        assert len(positions)==2
        np.testing.assert_array_equal(seed,selfseed)
        return dict(joint_goal=q,joint_goal_feasible=np.full(2,feasible,dtype=bool),all_starts=dict(fixed9=True))
    def call(p=parent,pl=plan,native=np.r_[np.zeros(3),[0,0,0,1]]):
        targets.clear()
        with patch.object(subject.fixed,'BatchPoseJointAdapter',return_value=object()),patch.object(subject.fixed,'solve_fixed_starts',side_effect=solve):
            return subject.materialize_v9_shots(pl,p,dict(center=[0.,0.,1.]),native,[-.025,.025]*3,cameras,Kinematics(),dict(lower=[-10.]*7,upper=[10.]*7))
    return plan,parent,call


class V9RobotLayoutTests(unittest.TestCase):
    def test_new_plan_uses_only_new_executable_routes_and_all_fixed_parent_recipes(self):
        p=subject.ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration8_v9_20261103_v1.json'
        plan=json.loads(p.read_text())
        self.assertEqual(plan['collector_script'],'scripts/collect_near_goal_position_v9_calibration8_20261103.py')
        self.assertEqual(plan['owner_launcher'],'scripts/launch_near_goal_position_v9_calibration8_20261103.py')
        self.assertEqual(plan['owner_shell'],'scripts/run_near_goal_position_v9_calibration8_20261103.sh')
        self.assertEqual([x['seed']for x in plan['parents']],list(range(261103648,261103656)))
        self.assertEqual([x['TRAIN_q_source']['raw_episode_id']for x in plan['parents']],
            ['episode'+str(i)for i in (316,320,322,326,328,331,369,383)])
        for i,parent in enumerate(plan['parents']):
            self.assertEqual(parent['shots'],subject.shot_specs_for_parent(i))
            self.assertEqual(parent['role'],'calibration');self.assertEqual(parent['shot_ids'],list(subject.SHOT_IDS))

    def test_real_asset_full_open_exception_and_exact_60_50_no_clipping(self):
        geometry=subject.robot_asset_geometry();maximum=subject.finger_layout(geometry,'full_open_asset_max')
        self.assertAlmostEqual(maximum['requested_gap_m'],.08)
        self.assertAlmostEqual(maximum['actual_force_pad_gap_m'],.07974060707888797,places=14)
        np.testing.assert_array_equal(maximum['finger_positions_m'],np.asarray(geometry['joint_intervals_m'])[:,1])
        self.assertTrue(maximum['requested80_is_explicit_asset_maximum_exception']);self.assertFalse(maximum['joint_targets_clipped'])
        for kind,gap in [('gap060',.06),('gap050',.05)]:
            row=subject.finger_layout(geometry,kind);self.assertAlmostEqual(row['actual_force_pad_gap_m'],gap,places=14)
            self.assertTrue(row['not_contact_or_grasp_proof'])
            self.assertTrue(all(0<x<.04 for x in row['finger_positions_m']))
        self.assertTrue(all('force_contact'in f['shape']for f in geometry['faces']))

    def test_wrong_asset_or_unfixed_gap_rejected(self):
        with patch.object(subject,'ASSET_SHA256','0'*64),self.assertRaises(ValueError):subject.robot_asset_geometry()
        with self.assertRaises(ValueError):subject.finger_layout(subject.robot_asset_geometry(),'gap080_exact')

    def test_all_eight_directions_ten_preserved_recipes_and_bands(self):
        for ordinal in range(8):
            rows=subject.shot_specs_for_parent(ordinal)
            self.assertEqual([x['shot_id']for x in rows],list(subject.SHOT_IDS))
            self.assertEqual(sum(x['condition']=='near_goal'for x in rows),7)
            self.assertEqual([x['position_band']for x in rows[7:9]],['annulus_08_10','annulus_10_12'])
            self.assertTrue(all(x['static_layout_not_attached_or_grasp']for x in rows))
            self.assertIn('full_open_asset_max',rows[3]['condition_detail'])
        with self.assertRaises(ValueError):subject.shot_specs_for_parent(True)

    def test_synthetic_materialization_sphere_distances_and_private_native_pose_unused(self):
        for i in range(8):
            _,_,call=fixture(i);out=call();self.assertTrue(out['complete']);self.assertEqual(len(out['shots']),10)
            for j,expected in [(1,.20),(2,.06),(7,.09),(8,.11)]:
                self.assertAlmostEqual(out['shots'][j]['distance_from_initial_public_sphere_m'],expected,places=12)
            self.assertTrue(out['shots'][7]['annulus_is_not_claimed_static_inhand'])
            self.assertTrue(out['shots'][8]['annulus_is_not_claimed_static_inhand'])
            self.assertEqual(out['shots'][6]['joint_positions'],out['shots'][9]['joint_positions'])
            self.assertEqual(out['shots'][6]['finger_positions_m'],out['shots'][9]['finger_positions_m'])
            self.assertEqual(out,call(native=np.full(7,123.)))

    def test_infeasible_solver_retains_all_shots_and_fails_without_replacement(self):
        _,_,call=fixture(feasible=False);out=call()
        self.assertFalse(out['complete']);self.assertEqual(len(out['shots']),10)
        self.assertIn('fixed9start_asset_hand_DLS_infeasible_no_retry',out['failures'])
        self.assertTrue(out['no_adaptive_retry'])

    def test_mutated_direction_recipe_or_tool_not_silently_accepted(self):
        plan,parent,call=fixture();altered=deepcopy(parent);altered['near_direction_unit_world']=[1.,0.,0.]
        with self.assertRaises(ValueError):call(p=altered)
        altered=deepcopy(parent);altered['shots'].pop()
        with self.assertRaises(ValueError):call(p=altered)
        altered=deepcopy(plan);altered['tool_quaternion_xyzw']=[0.,0.,0.,1.]
        with self.assertRaises(ValueError):call(pl=altered)


if __name__=='__main__':unittest.main()
