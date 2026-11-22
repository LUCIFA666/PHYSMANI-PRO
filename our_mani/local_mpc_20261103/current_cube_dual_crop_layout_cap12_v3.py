"""Fixed asset-only static layouts for the original forty-parent CF pilot.

Cube placement is a visual intervention, never a physical grasp or trajectory.
The current public RGBD initial Sphere is the only world anchor. All six hand
targets and all thirteen scenes are materialized before any augmented capture.
"""
from copy import deepcopy
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

from . import depth_owner_full32 as fixed
from . import position_v9_robot_layout as asset

VERSION = 'static_dual_crop_CF_fixed13_far_reuse_d120_pad_asset_20261104_v3'
SHOT_IDS = tuple(f'shot{i:02d}' for i in range(13))
DISTANCES = (.04, .08, .12, .16, .20)
HAND_PAD_RADIUS_CAP_M = .12
HAND_TARGETS = ((.25, 'offset_hand_asset_reuse_d120_pad_full_open'), (.04, 'pad_full_open'),
                (.08, 'palm_gap060'), (.12, 'pad_full_open'),
                (.16, 'offset_hand_asset_reuse_d120_pad_full_open'), (.20, 'offset_hand_asset_reuse_d120_pad_full_open'))
UNIQUE_HAND_TARGETS = ((.04, 'pad_full_open'), (.08, 'palm_gap060'),
                       (.12, 'pad_full_open'))
HAND_SLOT_TARGET_INDICES = (2, 0, 1, 2, 2, 2)
KINEMATIC_SCOPE = 'independent_static_visual_intervention_not_grasp_or_attached_or_motion_replay'


def _design(parent):
    ordinal = parent['near_direction_ordinal']
    if type(ordinal) is not int or not 0 <= ordinal < 8:
        raise ValueError('Original preassigned octant required')
    direction = fixed.b.vector(parent['near_direction_unit_world'], 3, 'fixed direction')
    if not np.array_equal(direction, fixed.DIRECTIONS[ordinal]):
        raise ValueError('Original normalized octant cannot change')
    angle, axis = parent['cube_rotation_degrees'], parent['cube_rotation_axis']
    if isinstance(angle, (bool, np.bool_)) or abs(angle) not in (0., 3., 5., 10., 15.) or axis not in ('x', 'y', 'z'):
        raise ValueError('Metadata-first Cube rotation required')
    vector = np.zeros(3)
    vector['xyz'.index(axis)] = np.deg2rad(float(angle))
    rotation = Rotation.from_rotvec(vector)
    return ordinal, direction, rotation.as_quat(), rotation.as_matrix()


def shot_specs_for_parent(parent):
    ordinal, direction, _, _ = _design(parent)
    specs = [
        ('sphere_only_parked', 'parked_source_q0_full_open', None, 'outside_all_frusta'),
        ('cube_outside_scope_nearhand', 'offset_hand_asset_reuse_d120_pad_full_open', .25, 'outside_scope_0p20'),
        ('sphere_only_nearhand', 'same_d120_pad_full_open', None, 'outside_all_frusta'),
    ]
    for i, distance in enumerate(DISTANCES):
        band = 'near_scope_0p08' if distance <= .08 else 'transition_scope_0p20'
        specs.extend((('current_cube_near_transition', 'parked_source_q0_full_open', distance, band),
                      ('current_cube_near_transition', HAND_TARGETS[i+1][1], distance, band)))
    suffix = f'_dir{ordinal:02d}_{fixed.DIRECTION_NAMES[ordinal]}_rot{parent["cube_rotation_axis"]}{int(parent["cube_rotation_degrees"]):02d}'
    rows = []
    for i, (condition, hand, distance, band) in enumerate(specs):
        detail = condition + '_' + hand + ('' if distance is None else f'_d{int(round(distance*1000)):03d}') + suffix
        rows.append(dict(shot_id=SHOT_IDS[i], condition=condition, condition_detail=detail,
            hand_layout=hand, requested_distance_from_public_sphere_m=distance,
            position_band=band, near_direction_ordinal=ordinal,
            near_direction_unit_world=direction.tolist(),
            cube_rotation_degrees=float(parent['cube_rotation_degrees']),
            cube_rotation_axis=parent['cube_rotation_axis'], independent_scene=True,
            static_layout_not_attached_or_grasp=True,
            visibility_label_not_inferred_from_layout=True))
    return rows


def materialize_dual_crop_shots(plan, parent, sphere, native_cube_pose, bbox, cameras, kinematics, joint_limits):
    """No task pose is consumed; ``native_cube_pose`` is only legacy signature compatibility."""
    ordinal, direction, quaternion, cube_rotation = _design(parent)
    if parent['shots'] != shot_specs_for_parent(parent):
        raise ValueError('All thirteen predeclared recipes/order must match')
    if plan['DLS'] != fixed.DLS or plan['tool_quaternion_xyzw'] != [1., 0., 0., 0.]:
        raise ValueError('Fixed original DLS and tool frame required')
    if plan.get('hand_pad_radius_cap_m') != HAND_PAD_RADIUS_CAP_M:
        raise ValueError('Root fixed uniform120mm hand-anchor cap required')
    S = fixed.b.vector(sphere['center'], 3, 'initial public Sphere')
    seed = fixed.b.vector(parent['TRAIN_q_source']['joint_positions'], 7, 'preassigned TRAIN q0')
    lower = fixed.b.vector(joint_limits['lower'], 7, 'joint lower')
    upper = fixed.b.vector(joint_limits['upper'], 7, 'joint upper')
    if np.any(seed < lower) or np.any(seed > upper):
        raise ValueError('TRAIN q0 cannot be clipped or replaced')
    box = np.asarray(bbox, dtype=float)
    if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[::2] >= box[1::2]):
        raise ValueError('Actual finite asset local bounding box required')
    corners = np.asarray(list(product(*zip(box[::2], box[1::2]))))
    Rtool = np.diag([1., -1., -1.])
    geometry = asset.robot_asset_geometry()
    layouts = {name: asset.finger_layout(geometry, name) for name in ('full_open_asset_max', 'gap060')}
    pad = np.asarray(layouts['full_open_asset_max']['pad_midpoint_TCP_m'])
    palm = pad.copy()
    # Actual rotated asset extent, not a task/VAL pose or a visible-face assumption.
    cube_corners_tool = corners @ cube_rotation.T @ Rtool
    palm[2] = geometry['palm_front_coordinate_TCP_z_m'] + .002 - cube_corners_tool[:, 2].min()
    if .002 != fixed.DLS['position_tolerance']:
        raise ValueError('Frozen asset clearance reference changed')
    targets = np.stack([S + distance*direction - Rtool @ (palm if hand == 'palm_gap060' else pad)
        for distance, hand in UNIQUE_HAND_TARGETS])
    quaternions = np.tile(plan['tool_quaternion_xyzw'], (len(targets), 1))
    adapter = fixed.BatchPoseJointAdapter(kinematics,
        type('Limits', (), dict(lower=lower, upper=upper))(), fixed.DLSConfig(**fixed.DLS))
    solved = fixed.solve_fixed_starts(adapter, seed, targets, quaternions)
    failures = []
    if not np.all(solved['joint_goal_feasible']):
        failures.append('fixed9start_asset_hand_DLS_infeasible_no_retry_or_source_change')
    hand_q = {(distance, hand): solved['joint_goal'][i] for i, (distance, hand) in enumerate(UNIQUE_HAND_TARGETS)}
    outside = fixed.b.vector(plan['world_outside_position_m'], 3, 'fixed outside position')
    if cameras is None:
        proof = dict(verified=False, all_four_camera_frusta_outside=False,
            source='pending_actual_public_sensor_camera_near_far',
            reason='No assumed or synthetic clipping planes provide absence evidence', views={})
    else:
        proof = fixed.outside_frusta_proof(outside, quaternion, box, cameras)
    # An unproven outside construction is UNKNOWN, never a negative or a dropped row.
    shots = []
    for i, spec in enumerate(parent['shots']):
        absence = i in (0, 2)
        distance = spec['requested_distance_from_public_sphere_m']
        position = outside if absence else S + distance*direction
        hand = spec['hand_layout']
        if hand == 'parked_source_q0_full_open':
            q, finger = seed, 'full_open_asset_max'
        elif hand == 'same_d120_pad_full_open':
            q, finger = hand_q[(.12, 'pad_full_open')], 'full_open_asset_max'
        elif hand == 'offset_hand_asset_reuse_d120_pad_full_open':
            q, finger = hand_q[(.12, 'pad_full_open')], 'full_open_asset_max'
        else:
            q = hand_q[(distance, hand)]
            finger = 'gap060' if hand == 'palm_gap060' else 'full_open_asset_max'
        measured_distance = float(np.linalg.norm(position-S))
        if not absence and abs(measured_distance-distance) > 1e-12:
            failures.append(spec['shot_id']+'_fixed_Sphere_distance_mismatch')
        if i == 1 and not measured_distance > .20:
            failures.append('shot01_not_actual_outside_proposed_scope')
        if np.any(q < lower) or np.any(q > upper) or not np.isfinite(q).all():
            failures.append(spec['shot_id']+'_illegal_joint_target_no_clip')
        shot = deepcopy(spec)
        shot.update(cube_mode='public_sphere_offset', offset_world_m=(position-S).tolist(),
            quaternion_xyzw=quaternion.tolist(), materialized_cube_pose_world=np.r_[position, quaternion].tolist(),
            joint_positions=q.tolist(), finger_positions_m=layouts[finger]['finger_positions_m'],
            finger_layout=layouts[finger], cube_scene_present=True,
            absence_from_all_camera_frusta_by_construction=bool(absence and proof['verified']),
            outside_all_frusta_requested=absence,
            unavailable_absence_proof_is_unknown_not_negative=bool(absence and not proof['verified']),
            hand_anchor_radius_cap_m=HAND_PAD_RADIUS_CAP_M,
            hand_anchor_distance_from_public_sphere_m=None if hand == 'parked_source_q0_full_open' else .12 if hand == 'same_d120_pad_full_open' else min(distance, HAND_PAD_RADIUS_CAP_M),
            offset_hand_not_static_inhand_or_guaranteed_occlusion=i in (1,10,12),
            complete_hand_asset_reused_from_shot_id='shot08' if i in (1,10,12) else None,
            far_asset_reuses_same_q_TCP_and_fingers=i in (1,10,12),
            distance_from_initial_public_sphere_m=measured_distance,
            cube_centre_from_original_Sphere_recipe_not_FK_achieved_position=True,
            kinematic_layout_scope=KINEMATIC_SCOPE,
            source_task_is_q_source_only_not_replayed_motion=True,
            initial_public_Sphere_cached_without_refit=True,
            Native_robot_and_Cube_pose_readback_still_required=True,
            joint_targets_clipped=False, geometry_or_ownership_or_grasp_qualified=False)
        shots.append(shot)
    return dict(layout_version=VERSION, hand_pad_radius_cap_m=HAND_PAD_RADIUS_CAP_M, complete=not failures, failures=failures, parent_id=parent['parent_id'],
        seed=parent['seed'], public_Sphere_center=S.tolist(), shots=shots,
        robot_asset_geometry=geometry, finger_layouts=layouts, outside_frusta_proof=proof,
        derived_before_any_shot=True, no_adaptive_retry=True, not_a_continuous_trajectory=True,
        static_layout_not_grasp=True, bootstrap_initial_public_Sphere_reused_without_refit=True,
        DLS=dict(config=fixed.DLS, target_order=[dict(hand_anchor_distance_m=d, hand_layout=h) for d, h in UNIQUE_HAND_TARGETS],
            unique_hand_targets=3, planned_hand_slots=6, fixed_solver_rows=27,
            planned_slot_target_indices=list(HAND_SLOT_TARGET_INDICES),
            planned_slot_feasible=[bool(solved['joint_goal_feasible'][j]) for j in HAND_SLOT_TARGET_INDICES],
            target_positions=targets.tolist(), target_quaternions=quaternions.tolist(),
            results=fixed.b.plain(solved), Native_IK_called=False, physical_steps=0,
            collision_or_reachability_qualified=False))


materialize_parent = materialize_dual_crop_shots
