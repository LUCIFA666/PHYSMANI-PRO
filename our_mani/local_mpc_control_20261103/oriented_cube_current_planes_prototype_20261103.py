"""Unqualified current-plane Cube centre research with nuisance orientation.

Inputs are actual same-frame pre-axis-gate planar points/normals and public
camera origins. This module never accepts world-axis E intervals, a previous
centre, robot/TCP, forecast, object pose or labels. Finite frame hypotheses are
not a confidence set for orientation. Any reported centre remains conditional
geometry only, with no semantic, position, control or contact qualification.
"""
from copy import deepcopy
from itertools import permutations, product
import numpy as np

from .axis_general_free_space_prototype_20261103 import refine_axis_general

VERSION = 'current_oriented_plane_cube_conditional_research_20261103_v1'
HALF_SIDE = .025
POINT_ALLOWANCE = .00025
MAXIMUM_RADIUS = .005
NORMAL_COMPATIBILITY_DEGREES = 5.
MINIMUM_FACE_SAMPLES = 3
FIXED_FRAME_REFINEMENTS = 3
COSINE = np.cos(np.deg2rad(NORMAL_COMPATIBILITY_DEGREES))
SINE = np.sin(np.deg2rad(NORMAL_COMPATIBILITY_DEGREES))


def _symmetries():
    result = []
    for order in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            matrix = np.eye(3)[:, order]*np.asarray(signs)
            if np.linalg.det(matrix) > 0:
                result.append(matrix)
    return result


CUBE_SYMMETRIES = _symmetries()


def _canonical_frame(rotation):
    choices = [rotation@g for g in CUBE_SYMMETRIES]
    return min(choices, key=lambda r: tuple(np.round(r.ravel(), 12))).copy()


def _proper_frame(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u@np.diag([1., 1., np.linalg.det(u@vt)])@vt


def _normal_groups(normals):
    # Connected components are order-independent. Reject a chained component
    # that is not itself entirely within the fixed angular budget.
    adjacency = np.abs(normals@normals.T) >= COSINE
    seen = np.zeros(len(normals), bool); groups = []
    for seed in range(len(normals)):
        if seen[seed]:
            continue
        stack = [seed]; seen[seed] = True; ids = []
        while stack:
            item = stack.pop(); ids.append(item)
            neighbours = np.flatnonzero(adjacency[item] & ~seen)
            seen[neighbours] = True; stack.extend(neighbours.tolist())
        ids = np.asarray(sorted(ids), dtype=int)
        local = np.abs(normals[ids]@normals[ids].T)
        spread = float(np.rad2deg(np.arccos(np.clip(local.min(), 0., 1.))))
        _, vectors = np.linalg.eigh(normals[ids].T@normals[ids])
        axis = vectors[:, -1]
        first = np.flatnonzero(np.abs(axis) > 1e-12)[0]
        if axis[first] < 0:
            axis = -axis
        groups.append(dict(indices=ids.tolist(), count=len(ids),
            unsigned_normal=axis.tolist(), maximum_pair_angle_degrees=spread,
            complete_link_consistent=bool(local.min() >= COSINE)))
    return groups


def _frame_support(points, normals, origins, rotation, reference):
    q = (points-reference)@rotation
    aligned = normals@rotation
    axes = np.abs(aligned).argmax(1)
    cosines = np.abs(aligned[np.arange(len(points)), axes])
    counts = [int((axes == j).sum()) for j in range(3)]
    observed = np.asarray(counts) >= MINIMUM_FACE_SAMPLES
    support = dict(centre=None, diagnostic_centre_only=None, support_available=False,
        intervals=[[None, None] for _ in range(3)], radius_each=[None]*3,
        joint_radius=None, axis_counts=counts, observed_axis=observed.tolist(),
        coordinate_sources=['oriented_current_support_containment']*3,
        reasons=[], point_compatibility_padding_m=POINT_ALLOWANCE,
        maximum_development_radius_m=MAXIMUM_RADIUS,
        known_5cm_axis_aligned_cube_prior='only_in_this_nuisance_frame',
        orientation_observed=False, statistical_confidence_interval=False,
        certified_physical_bound=False, original_frame_guards_still_required=True)
    if np.any(cosines < COSINE):
        support['reasons'].append('current_planar_normal_not_explained_by_this_frame')
        return support, axes, np.rad2deg(np.arccos(np.clip(cosines, 0., 1.)))
    signs = np.sign(((origins-points)@rotation)[np.arange(len(points)), axes])
    normal_facing = np.sign(np.sum(normals*(origins-points), axis=1))
    normal_signs = normal_facing*np.sign(aligned[np.arange(len(points)), axes])
    if np.any(signs == 0) or np.any(signs != normal_signs):
        support['reasons'].append('grazing_or_inconsistent_current_face_sign')
        return support, axes, np.rad2deg(np.arccos(np.clip(cosines, 0., 1.)))
    for j in range(3):
        if observed[j]:
            values = q[axes == j, j]-signs[axes == j]*HALF_SIDE
            # Intersect all actual current face-point bands. No rolling mean,
            # world-axis median or discarded current outlier can fill a gap.
            lo = float(values.max()-POINT_ALLOWANCE)
            hi = float(values.min()+POINT_ALLOWANCE)
            support['coordinate_sources'][j] = 'oriented_current_visible_face_intersection'
        else:
            lo = float(q[:, j].max()-HALF_SIDE-POINT_ALLOWANCE)
            hi = float(q[:, j].min()+HALF_SIDE+POINT_ALLOWANCE)
        support['intervals'][j] = [lo, hi]
        if not np.isfinite([lo, hi]).all() or lo > hi:
            support['reasons'].append('empty_or_nonfinite_current_oriented_interval_'+str(j))
            continue
        support['radius_each'][j] = .5*(hi-lo)
    if any(value is None for value in support['radius_each']):
        return support, axes, np.rad2deg(np.arccos(np.clip(cosines, 0., 1.)))
    centre = [lo+.5*(hi-lo) for lo, hi in support['intervals']]
    radius = float(np.linalg.norm(support['radius_each']))
    support['diagnostic_centre_only'] = centre
    support['joint_radius'] = radius
    if not np.isfinite(centre+[radius]).all():
        support['reasons'].append('nonfinite_current_oriented_geometry')
    elif radius > MAXIMUM_RADIUS:
        support['reasons'].append('joint_radius_exceeds_unchanged5mm')
    if not support['reasons']:
        support.update(centre=centre, support_available=True)
    return support, axes, np.rad2deg(np.arccos(np.clip(cosines, 0., 1.)))


def estimate_oriented_support(points_world, normals_world, camera_origins, view_ids):
    p, n, o, views = map(np.asarray, (points_world, normals_world, camera_origins, view_ids))
    if (p.ndim != 2 or p.shape[1:] != (3,) or p.dtype.kind not in 'fiu'
            or n.shape != p.shape or n.dtype.kind not in 'fiu' or o.shape != p.shape or o.dtype.kind not in 'fiu'
            or views.shape != (len(p),) or views.dtype.kind not in 'iu'
            or not np.isfinite(p).all() or not np.isfinite(n).all() or not np.isfinite(o).all()):
        raise ValueError('Only finite current points/normals/camera origins and integer view ids')
    p, n, o = (x.astype(np.float64, copy=False) for x in (p, n, o))
    result = dict(version=VERSION, status='UNKNOWN', centre=None, modes=[], rejected_modes=[],
        normal_groups=[], current_planar_points=len(p), current_views=len(set(views.tolist())),
        orientation_observed=False, nuisance_frame_is_certified_pose=False,
        conditional_geometry_only=True, position_measurement_qualified=False,
        qualified=False, model_qualified=False, raw_mask_model_qualified=False,
        force_or_contact_qualified=False,
        original_frame_guards_still_required=True, reasons=[])
    if not len(p):
        result['reasons'].append('no_current_planar_points'); return result
    lengths = np.linalg.norm(n, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError('Nondegenerate actual planar normals required')
    n = n/lengths[:, None]
    groups = _normal_groups(n); result['normal_groups'] = groups
    if any(not g['complete_link_consistent'] for g in groups):
        result['reasons'].append('normal_chain_exceeds_fixed_angular_compatibility'); return result
    stable = [g for g in groups if g['count'] >= MINIMUM_FACE_SAMPLES]
    if len(stable) < 2:
        result['reasons'].append('fewer_than_two_independent_current_normal_families'); return result
    reference = p[0]+(p-p[0]).mean(axis=0)
    if not np.isfinite(reference).all():
        result['reasons'].append('nonfinite_current_reference_point'); return result
    seeds = []
    for a in range(len(stable)):
        for b in range(a+1, len(stable)):
            left = np.asarray(stable[a]['unsigned_normal'])
            right = np.asarray(stable[b]['unsigned_normal'])
            if abs(float(left@right)) > SINE:
                continue
            third = np.cross(left, right); third /= np.linalg.norm(third)
            rotation = _proper_frame(np.column_stack((left, right, third)))
            for _ in range(FIXED_FRAME_REFINEMENTS):
                projections = n@rotation
                assigned = np.abs(projections).argmax(axis=1)
                columns = rotation.copy()
                for j in range(3):
                    ids = assigned == j
                    if int(ids.sum()) < MINIMUM_FACE_SAMPLES:
                        continue
                    values = n[ids]*np.sign(projections[ids, j])[:, None]
                    direction = values.mean(axis=0)
                    columns[:, j] = direction/np.linalg.norm(direction)
                rotation = _proper_frame(columns)
            rotation = _canonical_frame(rotation)
            if not any(np.allclose(rotation, old, rtol=0, atol=1e-10) for old in seeds):
                seeds.append(rotation)
    for rotation in seeds:
        support, axes, angles = _frame_support(p, n, o, rotation, reference)
        mode = dict(nuisance_rotation=rotation.tolist(), reference_point=reference.tolist(),
            support=support, all_current_point_axis_assignments=axes.tolist(),
            normal_fit_angle_degrees=dict(max=float(angles.max()), rms=float(np.sqrt(np.mean(angles**2)))),
            centre_world_from_planes=None if support['centre'] is None else (reference+rotation@support['centre']).tolist())
        if (support['reasons'] and support['reasons'] != ['joint_radius_exceeds_unchanged5mm']):
            result['rejected_modes'].append(mode)
        else:
            result['modes'].append(mode)
    if not seeds:
        result['reasons'].append('no_current_near_orthogonal_normal_pair')
    elif not result['modes']:
        result['reasons'].append('no_frame_explains_all_current_planar_points')
    elif len(result['modes']) > 1:
        result['reasons'].append('multiple_non_cube_equivalent_current_frames')
    # Estimate stage intentionally never advertises a complete observation.
    return result


def refine_current_oriented_support(depth_front, K, T, estimate):
    result = deepcopy(estimate)
    result.update(status='UNKNOWN', centre=None, joint_radius=None,
        axis_general_applied_in_nuisance_frame=True, full_orientation_feasible_set_proven=False)
    for mode in result['modes']:
        rotation = np.asarray(mode['nuisance_rotation']); reference = np.asarray(mode['reference_point'])
        transform = np.asarray(T, dtype=float).copy()
        transform[:3, :3] = rotation.T@np.asarray(T)[:3, :3]
        transform[:3, 3] = rotation.T@(np.asarray(T)[:3, 3]-reference)
        refined = refine_axis_general(depth_front, K, transform, deepcopy(mode['support']))
        mode['ray_support'] = refined
        centre = refined.get('centre')
        mode['conditional_centre_world'] = None if centre is None else (reference+rotation@centre).tolist()
    if len(result['modes']) != 1:
        if not result['reasons']:
            result['reasons'].append('no_unique_current_nuisance_frame')
        return result
    mode = result['modes'][0]; support = mode['ray_support']
    if support['support_available'] is not True:
        result['reasons'] = list(support['reasons']); return result
    if not np.isfinite(support['joint_radius']) or support['joint_radius'] > MAXIMUM_RADIUS:
        result['reasons'] = ['joint_radius_exceeds_unchanged5mm']; return result
    result.update(status='AVAILABLE', centre=mode['conditional_centre_world'],
        joint_radius=support['joint_radius'], reasons=[], conditional_geometry_only=True)
    return result
