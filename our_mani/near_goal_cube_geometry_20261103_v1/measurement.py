"""Current visible-face/support/front-free-space Cube measurement, fixed F.

Only the seven public input arrays, their current crops and a current raw mask
are accepted. No label, GT pose, robot, history or old centre is an argument.
Position is conditional on a known opaque near-axis 50 mm Cube, not general
6D pose estimation or a statistical/physical certificate.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np

from .face_points import point_face_proposals
from .support import aggregate_known_cube_support
from .free_space import refine_with_front_free_space
from ..near_goal_depth_refiner_20261103.data import erode_one_pixel,make_measurement as _legacy_mask_guards
from ..near_goal_segmentation_20261103.data import validate_public,make_near_goal_crops

VERSION='near_goal_cube_current_depth_geometry_F_20261103_v1'
ROOT=Path(__file__).resolve().parents[2]
LOWER_SHA256={
 'face_points.py':'75108de1859e4efb527beaec644134c02e70735718c91aa325da3366a3c5a6b8',
 'support.py':'5488046b3e045179ba2651631604ed2bc862f4b3b332cb8a952b248a91189eca',
 'free_space.py':'5c949ee82da9a308312c372f556269a1d577e2197eb62f24fa7d2cbfdb19077d'}
PROTOCOL_SHA256={
 'visible_cube_faces_measurement_D_protocol_v1':'efb13a6181f3d5bb8ecb3502232a637361212282af2de71fe88e16e9e0cc09ad',
 'bounded_cube_support_measurement_E_protocol_v1':'841bcc030b0ffa9dc7caa9a64c4e04ff45455e792ee28878f68a2980f3ffda53',
 'current_depth_free_space_measurement_F_protocol_v1':'bfddd160742313810e974968bccadeab90e05397dfeb7f3141115a10f091293f'}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_source_map():
    """Complete geometry implementation/protocol identity for model freezing."""
    package=Path(__file__).resolve().parent
    sources={str(package/name):_sha(package/name) for name in
        ('__init__.py','face_points.py','support.py','free_space.py','measurement.py')}
    for name,expected in LOWER_SHA256.items():
        if sources[str(package/name)]!=expected:raise ValueError('Fixed D/E/F byte identity changed: '+name)
    for name,expected in PROTOCOL_SHA256.items():
        path=ROOT/'artifacts/framework/near_goal_segmentation_20261103'/name/'protocol.json'
        if _sha(path)!=expected:raise ValueError('Fixed D/E/F protocol changed: '+name)
        sources[str(path)]=expected
    for name in ('our_mani/near_goal_depth_refiner_20261103/data.py',
                 'our_mani/near_goal_segmentation_20261103/data.py'):
        sources[str(ROOT/name)]=_sha(ROOT/name)
    return sources


def _json_array(array):
    a=np.asarray(array)
    if a.ndim==1:return [None if not np.isfinite(x) else float(x) for x in a]
    return [_json_array(row) for row in a]


def _full_current_mask(crops,raw):
    result=np.zeros((4,256,256),bool);yy,xx=np.indices((64,64))
    for camera in range(4):
        valid=crops['valid_crop'][camera]
        x=xx+crops['origins_xy'][camera,0];y=yy+crops['origins_xy'][camera,1]
        result[camera,y[valid],x[valid]]=raw[camera,valid]
    return result


def _current_face_points(inputs,crops,raw,full):
    # One erosion from the RAW mask; the independent baseline guard calculation
    # also starts from raw, never from this eroded mask.
    mask=erode_one_pixel(raw)&crops['valid_depth'];yy,xx=np.indices((64,64))
    points,views,axes,proposals,ledger=[],[],[],[],[]
    for camera in range(4):
        selected=mask[camera];u=xx[selected]+crops['origins_xy'][camera,0];v=yy[selected]+crops['origins_xy'][camera,1]
        pixels=np.c_[u,v].astype(np.int64)
        result=point_face_proposals(inputs['depth_m'][camera],full[camera],pixels,
            inputs['K'][camera],inputs['T'][camera],classic=(camera!=0))
        keep=result['accepted'];p=result['point_world'];axis=result['axis'];value=result['coordinate_proposal']
        points.append(p[keep]);views.extend([camera]*int(keep.sum()))
        axes.extend(axis[keep].tolist());proposals.extend(value[keep].tolist());reasons=result['reasons']
        ledger.append(dict(camera=camera,baseline_candidate_points=len(pixels),accepted_points=int(keep.sum()),
            rejected_points=int((~keep).sum()),reason_counts={str(k):int((reasons==k).sum()) for k in np.unique(reasons)},
            axis_counts={str(j):int((axis[keep]==j).sum()) for j in range(3)},pixels_uv=pixels.tolist(),
            accepted=keep.tolist(),reasons=reasons.tolist(),original_depth_m=crops['depth_m'][camera,selected].astype(float).tolist(),
            corrected_depth_m=_json_array(result['depth_corrected']),metric_fit_max_residual_m=_json_array(result['metric_residual_max']),
            inverse_depth_coefficients=_json_array(result['coefficients']),normals_world=_json_array(result['normals_world']),
            axis=axis.tolist(),sign=_json_array(result['sign']),coordinate_proposal=_json_array(value),point_world=_json_array(p)))
    return np.concatenate(points),np.asarray(views),np.asarray(axes),np.asarray(proposals),ledger


def _measurement_from_support(inputs,baseline,points,views,axes,geometry):
    if geometry['axis_counts']!=[int((axes==j).sum()) for j in range(3)]:raise ValueError('Current D support axis mismatch')
    measurement=deepcopy(baseline)
    dynamic={'insufficient_current_depth_points','insufficient_current_camera_views',
        'original_geometry_confidence_below_half','outside_predeclared_near_scope_0p08m'}
    reasons=[r for r in baseline['reasons'] if r not in dynamic];reasons.extend(geometry['reasons'])
    n,v=len(points),len(set(views.tolist()));confidence=min(1.,v/3)*min(1.,n/60)
    if n<12:reasons.append('insufficient_current_depth_points')
    if v<2:reasons.append('insufficient_current_camera_views')
    if confidence<.5:reasons.append('original_geometry_confidence_below_half')
    centre=geometry['centre']
    if centre is not None:
        if np.asarray(centre).shape!=(3,) or not np.isfinite(centre).all():raise ValueError('No nonfinite support centre')
        if np.linalg.norm(np.asarray(centre)-inputs['public_sphere_center'])>.08:
            reasons.append('outside_predeclared_near_scope_0p08m')
    available=geometry['support_available'] is True and centre is not None and not reasons
    measurement.pop('completion_used',None)
    measurement.update(reasons=reasons,refiner_available=available,centre=centre if available else None,
        diagnostic_centre_only=centre,measured_points=n,visible_views=v,geometry_confidence=confidence,
        bounded_support=geometry,axis_coordinate_samples=geometry['axis_counts'],known_5cm_axis_aligned_shape_prior=True,
        orientation_observed=False,coordinate_sources=geometry['coordinate_sources'],
        conditional_missing_axis_estimation_used=not all(geometry['observed_axis']),amodal_point_cloud_generated=False,
        learned_or_temporal_completion_used=False,no_missing_face_or_normal_fabricated=True,
        radius_is_conditional_compatibility_not_CI_or_certified_bound=True)
    return measurement


def _initial_components(geometry):
    result=[]
    for bounds in geometry['intervals']:
        valid=(bounds is not None and len(bounds)==2 and all(x is not None for x in bounds)
            and np.isfinite(bounds).all() and bounds[0]<=bounds[1])
        result.append([list(bounds)] if valid else [])
    return result


def _original_gate_reasons(inputs,baseline,points,views,e):
    dynamic={'insufficient_current_depth_points','insufficient_current_camera_views',
        'original_geometry_confidence_below_half','outside_predeclared_near_scope_0p08m'}
    reasons=[r for r in baseline['reasons'] if r not in dynamic]
    n,v=len(points),len(set(views.tolist()))
    if not np.isfinite(points).all():reasons.append('nonfinite_current_D_support')
    if n<12:reasons.append('insufficient_current_depth_points')
    if v<2:reasons.append('insufficient_current_camera_views')
    if min(1.,v/3)*min(1.,n/60)<.5:reasons.append('original_geometry_confidence_below_half')
    centre=e['bounded_support']['centre']
    if centre is not None and np.linalg.norm(np.asarray(centre)-inputs['public_sphere_center'])>.08:
        reasons.append('outside_predeclared_near_scope_0p08m')
    return reasons


def measure_current_cube(inputs,crops,raw_cube_mask):
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
