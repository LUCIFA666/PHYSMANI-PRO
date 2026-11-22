"""Pure scene-recipe tests; no images, models, simulator, or task GT."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_20261103 import current_cube_dual_crop_layout_cap16_v2 as layout

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    meta = json.loads((ROOT/'artifacts/framework/local_mpc_20261102/current_RGBD_centre_TRAIN_data_audit_v1/dual_crop_capture_preparation_manifest_v2.json').read_text())
    registry = json.loads((ROOT/'artifacts/framework/local_mpc_20261102/P3_dual_crop_CF_pilot40_TRAIN_q_registry_v1/q_sources.json').read_text())
    qrow = registry['rows'][0]
    original = next(p for p in meta['jobs'] if p['parent_id'] == qrow['parent_id'])
    parent = dict(parent_id=original['parent_id'], seed=original['seed'],
        near_direction_ordinal=original['direction_ordinal'],
        near_direction_unit_world=original['direction_unit_world'],
        cube_rotation_degrees=original['cube_rotation_degrees'],
        cube_rotation_axis=original['cube_rotation_axis'], TRAIN_q_source=qrow)
    parent['shots'] = layout.shot_specs_for_parent(parent)
    base = json.loads((ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration8_v10_20261103_v1.json').read_text())
    kin = layout.fixed.PoEKinematics(json.loads((ROOT/base['kinematics']['path']).read_text()))
    limits = json.loads((ROOT/base['joint_limits']['path']).read_text())['joint_limits']
    cameras = {name: dict(K=[[-200., 0., 128.], [0., -200., 128.], [0., 0., 1.]],
        T=np.eye(4).tolist(), near=.01, far=10., width=256, height=256)
        for name in ('front', '8', '16', '36')}
    plan = dict(DLS=layout.fixed.DLS, tool_quaternion_xyzw=[1., 0., 0., 0.],
        world_outside_position_m=[100., 100., -100.], hand_pad_radius_cap_m=.16)
    return original, parent, plan, kin, limits, cameras


class DualCropStaticCap16LayoutTests(unittest.TestCase):
    def run_layout(self, parent, plan, kin, limits, cameras, feasible=True):
        q = np.asarray(parent['TRAIN_q_source']['joint_positions'])
        def solved(adapter, seed, targets, quaternions):
            np.testing.assert_array_equal(seed, q)
            self.assertEqual(targets.shape, (4, 3))
            np.testing.assert_array_equal(quaternions, np.tile([1., 0., 0., 0.], (4, 1)))
            goals=np.tile(q,(4,1));goals[:,6]+=np.arange(4)*.001
            return dict(joint_goal=goals, joint_goal_feasible=np.full(4, feasible),
                ik_position_error=[0.]*4, ik_angle_error=[0.]*4)
        with patch.object(layout.fixed, 'solve_fixed_starts', side_effect=solved) as call:
            # Deliberately opaque native pose must never be read for layout.
            result = layout.materialize_dual_crop_shots(plan, parent,
                {'center': [.2, .1, 1.]}, object(), [-.025, .025]*3,
                cameras, kin, limits)
        self.assertEqual(call.call_count, 1)
        return result

    def test_thirteen_conditions_and_distances_equal_frozen_metadata(self):
        original, parent, *_ = fixture()
        specs = layout.shot_specs_for_parent(parent)
        self.assertEqual([s['shot_id'] for s in specs], list(layout.SHOT_IDS))
        self.assertEqual([s['condition'] for s in specs], [r['condition'] for r in original['recipes']])
        self.assertEqual([s['requested_distance_from_public_sphere_m'] for s in specs[3:]],
            [.04, .04, .08, .08, .12, .12, .16, .16, .20, .20])
        self.assertTrue(all(s['visibility_label_not_inferred_from_layout'] for s in specs))
        parent['cube_rotation_degrees'] = -15.
        self.assertEqual(layout.shot_specs_for_parent(parent)[0]['cube_rotation_degrees'], -15.)

    def test_world_centre_does_not_borrow_FK_error_and_static_pose_not_grasp(self):
        _, parent, plan, kin, limits, cameras = fixture()
        result = self.run_layout(parent, plan, kin, limits, cameras)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['shots']), 13)
        S = np.asarray(result['public_Sphere_center'])
        direction = np.asarray(parent['near_direction_unit_world'])
        for i, shot in enumerate(result['shots']):
            if i in (0, 2):
                self.assertTrue(shot['absence_from_all_camera_frusta_by_construction'])
                np.testing.assert_array_equal(shot['materialized_cube_pose_world'][:3], [100., 100., -100.])
            else:
                np.testing.assert_array_equal(shot['materialized_cube_pose_world'][:3],
                    S + shot['requested_distance_from_public_sphere_m']*direction)
            self.assertFalse(shot['geometry_or_ownership_or_grasp_qualified'])
            self.assertTrue(shot['Native_robot_and_Cube_pose_readback_still_required'])
            self.assertFalse(shot['joint_targets_clipped'])
        self.assertTrue(result['derived_before_any_shot'])

    def test_infeasible_parent_keeps_all_thirteen_and_never_retries(self):
        _, parent, plan, kin, limits, cameras = fixture()
        result = self.run_layout(parent, plan, kin, limits, cameras, feasible=False)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['shots']), 13)
        self.assertEqual([s['shot_id'] for s in result['shots']], list(layout.SHOT_IDS))
        self.assertTrue(result['no_adaptive_retry'])
        bad = deepcopy(parent); bad['TRAIN_q_source']['joint_positions'][0] = 100.
        with patch.object(layout.fixed, 'solve_fixed_starts', side_effect=AssertionError('No clipped solve')):
            with self.assertRaises(ValueError):
                layout.materialize_dual_crop_shots(plan, bad, {'center': [.2, .1, 1.]}, object(),
                    [-.025, .025]*3, cameras, kin, limits)

    def test_far_reuses_full_same_parent_palm_asset_and_missing_clip_is_unknown(self):
        _, parent, plan, kin, limits, _ = fixture()
        result = self.run_layout(parent, plan, kin, limits, None)
        self.assertTrue(result['complete'])
        self.assertFalse(result['outside_frusta_proof']['verified'])
        for i in (0, 2):
            self.assertFalse(result['shots'][i]['absence_from_all_camera_frusta_by_construction'])
            self.assertTrue(result['shots'][i]['unavailable_absence_proof_is_unknown_not_negative'])
        reference = result['shots'][10]
        for i in (1, 12):
            shot = result['shots'][i]
            self.assertEqual(shot['complete_hand_asset_reused_from_shot_id'], 'shot10')
            self.assertEqual(shot['joint_positions'], reference['joint_positions'])
            self.assertEqual(shot['finger_positions_m'], reference['finger_positions_m'])
            self.assertEqual(shot['finger_layout'], reference['finger_layout'])
            np.testing.assert_array_equal(kin(shot['joint_positions']),kin(reference['joint_positions']))
        self.assertEqual(result['DLS']['unique_hand_targets'], 4)
        self.assertEqual(result['DLS']['fixed_solver_rows'], 36)
        self.assertEqual(result['DLS']['planned_slot_target_indices'], [3,0,1,2,3,3])

    def test_rotated_bbox_palm_clearance_and_illegal_metadata(self):
        _, parent, plan, kin, limits, cameras = fixture()
        parent['cube_rotation_degrees'] = 15.
        parent['cube_rotation_axis'] = 'y'
        parent['shots'] = layout.shot_specs_for_parent(parent)
        result = self.run_layout(parent, plan, kin, limits, cameras)
        target = np.asarray(result['DLS']['target_positions'][1])
        centre = np.asarray(result['public_Sphere_center']) + .08*np.asarray(parent['near_direction_unit_world'])
        Rtool = np.diag([1., -1., -1.])
        local_centre = Rtool.T@(centre-target)
        rotation = layout.Rotation.from_rotvec([0., np.deg2rad(15.), 0.]).as_matrix()
        vertices = np.asarray(list(layout.product(*[(-.025, .025)]*3)))
        closest_z = (vertices@rotation.T@Rtool)[:, 2].min()+local_centre[2]
        self.assertAlmostEqual(closest_z-result['robot_asset_geometry']['palm_front_coordinate_TCP_z_m'], .002)
        for key, value in [('near_direction_ordinal', True), ('cube_rotation_degrees', 16.), ('cube_rotation_axis', 'TCP')]:
            bad = deepcopy(parent); bad[key] = value
            with self.assertRaises(ValueError): layout.shot_specs_for_parent(bad)


if __name__ == '__main__':
    unittest.main()
