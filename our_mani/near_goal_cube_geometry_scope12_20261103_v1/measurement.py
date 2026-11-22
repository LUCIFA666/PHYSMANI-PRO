"""Independent scope12 wrapper around unchanged current D/E/F geometry.

Only the declared centre-to-initial-public-Sphere scope changes from.08 to.12m.
The original complete pipeline must get first refusal in a future V9 producer;
this pure wrapper provides no model/control or calibration qualification.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from ..near_goal_cube_geometry_20261103_v1.measurement import (
    aggregate_known_cube_support, refine_with_front_free_space,
    _legacy_mask_guards, validate_public, make_near_goal_crops,
    _full_current_mask, _current_face_points, _initial_components,
    PROTOCOL_SHA256, geometry_source_map as _original_geometry_source_map)

VERSION='near_goal_cube_current_depth_geometry_F_scope12_20261103_v1'
SCOPE_M=.12


def geometry_source_map():
    """Explicit old ten source/protocol identities plus only these two files."""
    result=_original_geometry_source_map()
    package=Path(__file__).resolve().parent
    for name in ('__init__.py','measurement.py'):
        path=package/name
        result[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _measurement_from_support(inputs,baseline,points,views,axes,geometry):
    if geometry['axis_counts']!=[int((axes==j).sum()) for j in range(3)]:raise ValueError('Current D support axis mismatch')
    measurement=deepcopy(baseline)
    dynamic={'insufficient_current_depth_points','insufficient_current_camera_views',
        'original_geometry_confidence_below_half','outside_predeclared_near_scope_0p08m',
        'outside_predeclared_near_scope_0p12m'}
    reasons=[r for r in baseline['reasons'] if r not in dynamic];reasons.extend(geometry['reasons'])
    n,v=len(points),len(set(views.tolist()));confidence=min(1.,v/3)*min(1.,n/60)
    if n<12:reasons.append('insufficient_current_depth_points')
    if v<2:reasons.append('insufficient_current_camera_views')
    if confidence<.5:reasons.append('original_geometry_confidence_below_half')
    centre=geometry['centre']
    if centre is not None:
        if np.asarray(centre).shape!=(3,) or not np.isfinite(centre).all():raise ValueError('No nonfinite support centre')
        if np.linalg.norm(np.asarray(centre)-inputs['public_sphere_center'])>SCOPE_M:
            reasons.append('outside_predeclared_near_scope_0p12m')
    available=geometry['support_available'] is True and centre is not None and not reasons
    measurement.pop('completion_used',None)
    measurement.update(near_scope_m=SCOPE_M,reasons=reasons,refiner_available=available,centre=centre if available else None,
        diagnostic_centre_only=centre,measured_points=n,visible_views=v,geometry_confidence=confidence,
        bounded_support=geometry,axis_coordinate_samples=geometry['axis_counts'],known_5cm_axis_aligned_shape_prior=True,
        orientation_observed=False,coordinate_sources=geometry['coordinate_sources'],
        conditional_missing_axis_estimation_used=not all(geometry['observed_axis']),amodal_point_cloud_generated=False,
        learned_or_temporal_completion_used=False,no_missing_face_or_normal_fabricated=True,
        radius_is_conditional_compatibility_not_CI_or_certified_bound=True)
    return measurement


def _original_gate_reasons(inputs,baseline,points,views,e):
    dynamic={'insufficient_current_depth_points','insufficient_current_camera_views',
        'original_geometry_confidence_below_half','outside_predeclared_near_scope_0p08m',
        'outside_predeclared_near_scope_0p12m'}
    reasons=[r for r in baseline['reasons'] if r not in dynamic]
    n,v=len(points),len(set(views.tolist()))
    if not np.isfinite(points).all():reasons.append('nonfinite_current_D_support')
    if n<12:reasons.append('insufficient_current_depth_points')
    if v<2:reasons.append('insufficient_current_camera_views')
    if min(1.,v/3)*min(1.,n/60)<.5:reasons.append('original_geometry_confidence_below_half')
    centre=e['bounded_support']['centre']
    if centre is not None and np.linalg.norm(np.asarray(centre)-inputs['public_sphere_center'])>SCOPE_M:
        reasons.append('outside_predeclared_near_scope_0p12m')
    return reasons


def measure_current_cube_scope12(inputs,crops,raw_cube_mask):
    """Current F measurement plus complete point/axis/interval/free-evidence trace."""
    validate_public(inputs);raw=np.asarray(raw_cube_mask)
    if raw.shape!=(4,64,64) or raw.dtype!=np.bool_:raise ValueError('Current raw bool4x64x64Cube mask required')
    expected=make_near_goal_crops(inputs)
    if not isinstance(crops,dict) or set(crops)!=set(expected):raise ValueError('Exact current public crop dictionary required')
    for key,value in expected.items():
        if np.asarray(crops[key]).dtype!=value.dtype or not np.array_equal(crops[key],value,equal_nan=True):
            raise ValueError('Stale/mutated crop differs from the current public input: '+key)
    baseline=_legacy_mask_guards(inputs,crops,raw)
    full=_full_current_mask(crops,raw)
    points,views,axes,proposals,ledger=_current_face_points(inputs,crops,raw,full)
    geometry=aggregate_known_cube_support(points,axes,proposals)
    e=_measurement_from_support(inputs,baseline,points,views,axes,geometry)
    if e['refiner_available'] is True:
        result=deepcopy(e)
        receipt=dict(free_space_called=False,free_space_applied=False,E_passed_measurement_preserved_exact=True,
            skip_reason='original_E_measurement_passes',feasible_components=_initial_components(geometry),free_space_trace=[])
    else:
        gates=_original_gate_reasons(inputs,baseline,points,views,e)
        if gates:
            result=deepcopy(e)
            receipt=dict(free_space_called=False,free_space_applied=False,E_passed_measurement_preserved_exact=False,
                skip_reason='original_frame_gate_failed',original_gate_reasons=gates,
                feasible_components=_initial_components(geometry),free_space_trace=[])
        else:
            if geometry['support_available'] is True:raise ValueError('E geometry/frame availability inconsistent')
            refined=refine_with_front_free_space(inputs['depth_m'][0],inputs['K'][0],inputs['T'][0],deepcopy(geometry))
            result=_measurement_from_support(inputs,baseline,points,views,axes,refined)
            result['current_front_free_space_conditional_support']=True
            receipt=dict(free_space_called=True,free_space_applied=refined['free_space_trace']['free_space_applied'] is True,
                E_passed_measurement_preserved_exact=False,original_gate_reasons=[],
                feasible_components=refined['feasible_components'],free_space_trace=refined['free_space_trace'])
    result.update(available=result['refiner_available'],geometry_profile=VERSION,
        geometry_protocol_sha256=dict(PROTOCOL_SHA256),camera_point_ledger=ledger,free_space_receipt=receipt,
        visible_face_support=dict(axis_counts=[int((axes==j).sum()) for j in range(3)],
            original_eroded_candidate_points=sum(row['baseline_candidate_points'] for row in ledger),
            accepted_current_points=len(points),current_point_provenance='current_mask_metricdepth_immutable_D_extractor'),
        current_points_world=points.tolist(),
        source=dict(current_RGBD_only=True,current_raw_mask_only=True,GT_or_label_or_robot_or_history_input=False,
            known_opaque_near_axis_50mm_Cube_prior=True,not_general_6DOF_pose=True,
            conditional_geometry_not_statistical_or_physical_certificate=True,model_or_control_qualified=False))
    return result

