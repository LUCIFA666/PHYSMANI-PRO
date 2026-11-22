"""Labels-only quality of the actual positive support of a current measurement.

Never call this module to produce a policy observation. Semantic/centre GT is
used only after public prediction freezing. U is not relabelled background.
Free space is negative evidence and is never counted as Cube-positive pixels.
"""
import hashlib
from pathlib import Path

import numpy as np

from ..near_goal_depth_refiner_20261103.evaluation import conservative_pixel_metrics
from ..near_goal_cube_geometry_20261103_v1 import measurement as _legacy_geometry
from ..near_goal_cube_geometry_scope12_20261103_v1 import measurement as _scope_geometry
from ..near_goal_cube_geometry_20261103_v1 import face_points as _face, support as _support, free_space as _free

VERSION = 'current_depth_actual_support_scope12_compat_20261103_v10'
GEOMETRY_VERSION = 'near_goal_cube_current_depth_geometry_F_20261103_v1'
SCOPE12_GEOMETRY_VERSION = 'near_goal_cube_current_depth_geometry_F_scope12_20261103_v1'
SUPPORTED_GEOMETRY_VERSIONS = (GEOMETRY_VERSION, SCOPE12_GEOMETRY_VERSION)



ROOT = Path(__file__).resolve().parents[2]
# Explicit source identities of the original ten files and only the scope-only
# wrapper's two files. A profile string alone never approves another algorithm.
GEOMETRY_SOURCE_SHA256 = {'our_mani/near_goal_cube_geometry_20261103_v1/__init__.py': '683e4b8c075c28f0bbab97740c52e1a0bef66f212da4434bfa440892bfaf3b7b', 'our_mani/near_goal_cube_geometry_20261103_v1/face_points.py': '75108de1859e4efb527beaec644134c02e70735718c91aa325da3366a3c5a6b8', 'our_mani/near_goal_cube_geometry_20261103_v1/support.py': '5488046b3e045179ba2651631604ed2bc862f4b3b332cb8a952b248a91189eca', 'our_mani/near_goal_cube_geometry_20261103_v1/free_space.py': '5c949ee82da9a308312c372f556269a1d577e2197eb62f24fa7d2cbfdb19077d', 'our_mani/near_goal_cube_geometry_20261103_v1/measurement.py': '417c2aae7adf736588f58d438d95c730c8512a43e38175840a998bbb204450bb', 'artifacts/framework/near_goal_segmentation_20261103/visible_cube_faces_measurement_D_protocol_v1/protocol.json': 'efb13a6181f3d5bb8ecb3502232a637361212282af2de71fe88e16e9e0cc09ad', 'artifacts/framework/near_goal_segmentation_20261103/bounded_cube_support_measurement_E_protocol_v1/protocol.json': '841bcc030b0ffa9dc7caa9a64c4e04ff45455e792ee28878f68a2980f3ffda53', 'artifacts/framework/near_goal_segmentation_20261103/current_depth_free_space_measurement_F_protocol_v1/protocol.json': 'bfddd160742313810e974968bccadeab90e05397dfeb7f3141115a10f091293f', 'our_mani/near_goal_depth_refiner_20261103/data.py': '7e0a540075fb4078a185b6577f9a218b5aeb82fd7947e8ccf22a19575c53b119', 'our_mani/near_goal_segmentation_20261103/data.py': 'e4db1a685b8bc28d924ee7d1e90950a2985d6f5fa26de00da62e813a2a0d78e6', 'our_mani/near_goal_cube_geometry_scope12_20261103_v1/__init__.py': 'a504f0fa9378d54c6cf0cf4a9e3f64ae9c078ac116b8a7b16a8329bbfab4a14b', 'our_mani/near_goal_cube_geometry_scope12_20261103_v1/measurement.py': '6852b7adc3e918f032bc0c768591062b4d09306a59434488dd293e46d376b1e9'}
_FUNCTION_BINDINGS = tuple((module, name, getattr(module,name), getattr(module,name).__code__)
    for module,names in (
        (_legacy_geometry, ('point_face_proposals','aggregate_known_cube_support','refine_with_front_free_space',
            '_full_current_mask','_current_face_points','_initial_components','_legacy_mask_guards',
            'validate_public','make_near_goal_crops','_measurement_from_support','_original_gate_reasons','measure_current_cube')),
        (_scope_geometry, ('aggregate_known_cube_support','refine_with_front_free_space',
            '_full_current_mask','_current_face_points','_initial_components','_legacy_mask_guards',
            'validate_public','make_near_goal_crops','_measurement_from_support','_original_gate_reasons','measure_current_cube_scope12')))
    for name in names)


def checked_geometry_sources():
    """Check the two fixed versions, byte/source map and unchanged lower objects.

    This is algorithm compatibility, not ownership or measurement provenance.
    The caller still authenticates the real runtime/current measurement source.
    """
    expected={str(ROOT/name):digest for name,digest in GEOMETRY_SOURCE_SHA256.items()}
    actual=_scope_geometry.geometry_source_map()
    if (actual!=expected or len(actual)!=12
        or _legacy_geometry.VERSION!=GEOMETRY_VERSION
        or _scope_geometry.VERSION!=SCOPE12_GEOMETRY_VERSION or _scope_geometry.SCOPE_M!=.12):
        raise ValueError('Only exact original F and fixed scope12-only EF source profiles accepted')
    for name,digest in expected.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Fixed geometry source changed: '+name)
    for module,name,function,code in _FUNCTION_BINDINGS:
        if getattr(module,name)is not function or function.__code__ is not code:
            raise ValueError('Fixed geometry callable changed: '+name)
    for name in ('_full_current_mask','_current_face_points','_initial_components','aggregate_known_cube_support',
                 'refine_with_front_free_space','_legacy_mask_guards','validate_public','make_near_goal_crops'):
        if getattr(_scope_geometry,name)is not getattr(_legacy_geometry,name):
            raise ValueError('Scope-only wrapper must share original geometry function: '+name)
    if (_legacy_geometry.point_face_proposals is not _face.point_face_proposals
        or _legacy_geometry.aggregate_known_cube_support is not _support.aggregate_known_cube_support
        or _legacy_geometry.refine_with_front_free_space is not _free.refine_with_front_free_space):
        raise ValueError('Exact original D/E/F lower function identities required')
    return expected


def support_source_map():
    result=checked_geometry_sources()
    result[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def extract_current_support_masks(crops, measurement):
    """One centre per accepted D pixel, and UNION of its nine actual pixels."""
    checked_geometry_sources()
    domain = np.asarray(crops['valid_crop'])
    depth = np.asarray(crops['valid_depth'])
    origins = np.asarray(crops['origins_xy'])
    if (domain.shape != (4, 64, 64) or domain.dtype != np.bool_
            or depth.shape != domain.shape or depth.dtype != np.bool_
            or origins.shape != (4, 2) or origins.dtype.kind not in 'iu'
            or measurement.get('geometry_profile') not in SUPPORTED_GEOMETRY_VERSIONS):
        raise ValueError('Exact F geometry and observed four-view crop domain required')
    ledger = measurement.get('camera_point_ledger')
    if not isinstance(ledger, list) or [x.get('camera') for x in ledger] != [0, 1, 2, 3]:
        raise ValueError('Exact complete four-camera current D ledger required')
    points = np.zeros_like(domain); stencils = np.zeros_like(domain)
    original_candidates = accepted = 0
    for item in ledger:
        cam = item['camera']; n = item['baseline_candidate_points']
        if type(n) is not int or n < 0:
            raise ValueError('Integer original candidate-point count required')
        uv = np.asarray(item['pixels_uv'])
        if n == 0 and uv.size == 0:
            uv = np.empty((0, 2), np.int64)
        if uv.shape != (n, 2) or uv.dtype.kind not in 'iu':
            raise ValueError('Original integer pixel coordinates, no interpolated support')
        stored = item['accepted']
        if isinstance(stored, list) and any(type(v) is not bool for v in stored):
            raise ValueError('Actual boolean D acceptance values required')
        if isinstance(stored, np.ndarray) and stored.dtype != np.bool_:
            raise ValueError('Actual boolean D acceptance dtype required')
        keep = np.asarray(stored, dtype=bool)
        if keep.shape != (n,) or int(keep.sum()) != item['accepted_points']:
            raise ValueError('Current accepted-point ledger/count differs')
        local = uv[keep]-origins[cam]
        if len(local):
            if ((local < 1) | (local > 62)).any() or len(np.unique(local, axis=0)) != len(local):
                raise ValueError('Actual one-erosion D support must be unique and have its full3x3 crop stencil')
            points[cam, local[:, 1], local[:, 0]] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    stencils[cam, local[:, 1]+dy, local[:, 0]+dx] = True
        original_candidates += n; accepted += len(local)
    if (int(points.sum()) != accepted or accepted != measurement.get('measured_points')
            or np.any(points & ~domain) or np.any(stencils & ~domain)
            or np.any(points & ~depth) or np.any(stencils & ~depth)):
        raise ValueError('No accepted centre or used stencil can be hidden outside current valid depth/image domain')
    views = int(points.any(axis=(1, 2)).sum())
    if views != measurement.get('visible_views'):
        raise ValueError('Current support view count mismatch')
    return dict(points=points, stencils=stencils, domain=domain,
                original_candidates=original_candidates, accepted=accepted, views=views)


def _components_from_intervals(intervals):
    result = []
    for bounds in intervals:
        valid = (bounds is not None and len(bounds) == 2 and all(v is not None for v in bounds)
                 and np.isfinite(bounds).all() and bounds[0] <= bounds[1])
        result.append([list(bounds)] if valid else [])
    return result


def _inside(value, components):
    if not components:
        return None
    for lo, hi in components:
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError('Finite nonempty conditional interval components required')
    return any(lo <= value <= hi for lo, hi in components)


def _free_space_posthoc(measurement, gt):
    support = measurement['bounded_support']
    receipt = measurement['free_space_receipt']
    applied = receipt.get('free_space_applied') is True
    trace = receipt['free_space_trace']
    final = receipt['feasible_components']
    initial = (trace['initial_components'] if applied else _components_from_intervals(support['intervals']))
    if len(initial) != 3 or len(final) != 3:
        raise ValueError('Every coordinate must retain its entire candidate set')
    initial_in = [_inside(gt[j], initial[j]) for j in range(3)]
    final_in = [_inside(gt[j], final[j]) for j in range(3)]
    removed = [applied and initial_in[j] is True and final_in[j] is not True for j in range(3)]
    forbidden_hits = proof_failures = constraints = 0
    if applied:
        for step in trace['rounds']:
            j, k = step['axis'], step['other_axis']
            for item in step['constraints']:
                constraints += 1
                lo, hi = item['forbidden_open']
                forbidden_hits += int(lo < gt[j] < hi)
                mandatory = step['mandatory_other_axis_span']
                if not (item['observed_depth'] > item['max_expected_depth']+.00025
                        and item['footprint_min'][k] >= mandatory[0]
                        and item['footprint_max'][k] <= mandatory[1]):
                    proof_failures += 1
    evaluable = all(value is not None for value in final_in)
    return dict(free_space_applied=applied, initial_components=initial, final_components=final,
        GT_in_initial_axis_components=initial_in, GT_in_final_axis_components=final_in,
        GT_final_all_axes_inside=all(final_in) if evaluable else None,
        GT_final_interval_evaluable=evaluable,
        GT_excluded_by_free_space_axis=removed, GT_excluded_by_free_space_any=any(removed),
        constraint_records=constraints, GT_strictly_in_forbidden_records=forbidden_hits,
        current_pixel_proof_invalid_records=proof_failures,
        GT_used_only_posthoc=True, conditional_geometry_not_CI_or_physical_certificate=True)


def score_current_support(sample, measurement):
    """JSONable labels-only point/stencil precision and position diagnostics.

    Caller must retain every source row and aggregate available measurements
    separately from all measurements, including rejected frames. The helper
    never grants raw-model or position-measurement qualification.
    """
    labels = np.asarray(sample['semantic'])
    if (labels.shape != (4, 64, 64) or labels.dtype != np.uint8
            or not set(np.unique(labels)) <= {0, 1, 255}):
        raise ValueError('Separate observed binary0/1/255 pixel supervision required')
    gt = np.asarray(sample['cube_center_world'], float)
    if gt.shape != (3,) or not np.isfinite(gt).all():
        raise ValueError('Finite labels-only current Cube centre required')
    masks = extract_current_support_masks(sample['crops'], measurement)
    p, s, domain = masks['points'], masks['stencils'], masks['domain']
    available = measurement.get('refiner_available')
    if type(available) is not bool:
        raise ValueError('Explicit geometry availability, not inferred from point counts')
    centre = measurement.get('centre')
    if available and (np.asarray(centre).shape != (3,) or not np.isfinite(centre).all()):
        raise ValueError('Available measurement must retain its finite actual centre')
    error = None if not available else float(np.linalg.norm(np.asarray(centre)-gt))
    diagnostic = measurement.get('diagnostic_centre_only')
    if diagnostic is not None and (np.asarray(diagnostic).shape != (3,) or not np.isfinite(diagnostic).all()):
        raise ValueError('Diagnostic centre must be finite or explicitly absent')
    unknown = (labels == 255) & domain
    padded = np.pad(unknown, ((0, 0), (1, 1), (1, 1)))
    context = np.zeros_like(labels, dtype=np.int16)
    for dy in range(3):
        for dx in range(3): context += padded[:, dy:dy+64, dx:dx+64]
    reasons = {}
    for name, mask in sample.get('unknown_reason_masks', {}).items():
        mask = np.asarray(mask)
        if mask.shape != labels.shape or mask.dtype != np.bool_ or np.any(mask & ~unknown):
            raise ValueError('UNKNOWN reason mask must be an observed labels-only subset')
        reasons[name] = dict(point_U=int((p & mask).sum()), stencil_U=int((s & mask).sum()))
    return dict(geometry_available=available,
        point_pixels=conservative_pixel_metrics(labels, p, domain),
        stencil_pixels=conservative_pixel_metrics(labels, s, domain),
        per_camera={str(c): dict(point_pixels=conservative_pixel_metrics(labels[c], p[c], domain[c]),
            stencil_pixels=conservative_pixel_metrics(labels[c], s[c], domain[c])) for c in range(4)},
        counts=dict(original_candidates=masks['original_candidates'], actual_D_accepted=masks['accepted'],
            actual_views=masks['views'], used_stencil_union=int(s.sum()),
            centres_with_UNKNOWN_stencil=int((p & (context > 0)).sum()),
            UNKNOWN_stencil_occurrences=int(context[p].sum()), total_stencil_occurrences=9*int(p.sum())),
        mask_sha256=dict(points=hashlib.sha256(p.tobytes()).hexdigest(),
                         stencils=hashlib.sha256(s.tobytes()).hexdigest()),
        UNKNOWN_reasons=reasons, UNKNOWN_reasons_may_overlap=True,
        owner_unobserved=not bool((labels == 1).any()),
        owner_unobserved_falseavailable=available and not bool((labels == 1).any()),
        position_posthoc=dict(GT_center_world=gt.tolist(), actual_centre=centre,
            error_m=error, diagnostic_centre_only=diagnostic,
            diagnostic_error_m=None if diagnostic is None else float(np.linalg.norm(np.asarray(diagnostic)-gt)),
            GT_not_used_to_produce_measurement=True),
        free_space_posthoc=_free_space_posthoc(measurement, gt),
        score_scope='actual_positive_support_only_not_complete_instance_segmentation',
        stencil_union_deduplicated_per_camera_frame=True,
        free_space_is_negative_evidence_not_in_positive_PPV=True,
        orientation_observed=False, known_axis_aligned_50mm_Cube_prior=True,
        raw_model_quality_not_promoted=True, position_qualification_not_granted_by_helper=True)
