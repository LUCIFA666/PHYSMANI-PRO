"""Explicit frozen TRAIN bin sweep assets, with fixed engineering inflation.

There are no native handles/poses or policy outcomes in this API. Registration
does not authorize a collision query: a current causal directed-axis readiness
gate is additionally required by the caller for every queried forecast.
"""
from collections.abc import Mapping
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .bin_swept_geometry import (SweptPrototype,axis_yaw_swept_prototype,
    verify_stored_sweep_exact,_outward_float32,_radial_polygons,_prism)
from .collision_geometry import CollisionGeometry,mesh_sha256

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='r10_frozen_TRAIN_bin_directed_axis_sweep_v1'
TILT_BOUND_DEGREES=10.
TRANSLATION_BOUND_M=.005
SOURCE_MIGRATION_BOUND_M=1e-6
ALLOWED_TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')


def _require(value,message):
    if not value:raise ValueError(message)
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _pointer(pointer,root):
    _require(isinstance(pointer,Mapping) and isinstance(pointer.get('path'),str)
        and isinstance(pointer.get('sha256'),str) and len(pointer['sha256'])==64,'explicit artifact path and SHA256 required')
    path=(Path(root)/pointer['path']).resolve()
    _require(path.is_relative_to(Path(root).resolve()) and _sha(path)==pointer['sha256'],
        'frozen TRAIN artifact path/hash differs')
    return path


def inflated_axis_yaw_sweep(vertices,triangles):
    """Bound yaw plus <=10deg tilt and <=5mm translation, as an assumption.

    For any point of radius <=r, relative rotation theta displaces it by at
    most 2*r*sin(theta/2). The extra 1um covers a strictly checked corresponding
    source-vertex perturbation; rotation preserves that perturbation norm.
    This assumption is not a calibrated probability or safety guarantee.
    """
    vertices=np.asarray(vertices,float);triangles=np.asarray(triangles)
    base=axis_yaw_swept_prototype(vertices,triangles,slab_count=32,angular_segments=64)
    radius=float(np.linalg.norm(vertices,axis=1).max());radius_upper=_outward_float32(radius,1)
    # Certificate of the radius used by the engineering formula, independent
    # of floating norm under/overestimation.
    rr=Fraction.from_float(radius_upper)**2
    _require(all(sum(Fraction.from_float(float(v))**2 for v in point)<=rr for point in vertices),
        'float32-upward TRAIN radius does not contain all source vertices')
    formula=2*radius_upper*math.sin(math.radians(TILT_BOUND_DEGREES)/2)+TRANSLATION_BOUND_M
    pose_delta=_outward_float32(formula,1)
    delta=_outward_float32(pose_delta+SOURCE_MIGRATION_BOUND_M,1)
    slabs=[];all_v=[];all_t=[];offset=0
    for original in base.slabs:
        row=copy.deepcopy(original);low,high=row['source_axial_interval']
        old_low,old_high=row['radial_lower_bound'],row['radial_upper_bound']
        lower=max(0.,_outward_float32(old_low-delta,-1)) if old_low>delta else 0.
        upper=_outward_float32(old_high+delta,1)
        outer,inner,rout,rin=_radial_polygons(lower,upper,64)
        stored_low,stored_high=_outward_float32(low-delta,-1),_outward_float32(high+delta,1)
        v,t=_prism(stored_low,stored_high,outer,inner);all_v.append(v);all_t.append(t+offset);offset+=len(v)
        row.update(uninflated_radial_bounds=[old_low,old_high],axial_radial_inflation_m=delta,
            radial_lower_bound=lower,radial_upper_bound=upper,stored_axial_interval=[stored_low,stored_high],
            outer_vertex_radius=rout,inner_vertex_radius=rin,retained_cavity=inner is not None,
            outer_polygon_yz=outer.tolist(),inner_polygon_yz=inner.tolist() if inner is not None else None)
        slabs.append(row)
    receipt=dict(base.receipt,schema=SCHEMA,runtime_installed=False,
        source_vertices_max_radius_m=radius,source_max_radius_float32_upper_m=radius_upper,
        engineering_tilt_bound_degrees=TILT_BOUND_DEGREES,engineering_translation_bound_m=TRANSLATION_BOUND_M,
        engineering_formula='delta_pose = 2 * TRAIN_max_radius_upper * sin(10deg/2) + 0.005m',
        engineering_delta_formula_m=formula,engineering_delta_float32_up_m=pose_delta,
        source_vertex_migration_allowance_m=SOURCE_MIGRATION_BOUND_M,total_axial_radial_inflation_m=delta,
        engineering_pose_error_assumption=True,calibrated_safety_guarantee=False,
        retained_cavity_slabs=sum(row['retained_cavity'] for row in slabs),
        construction='source triangle/slab radial bounds; x bounds +/-delta, rmax+delta, max(0,rmin-delta); '
                     'circumscribed outer and inscribed inner polygons; all bounds float32 outward',
        current_directed_axis_readiness_required_for_every_collision_query=True)
    return SweptPrototype(np.concatenate(all_v),np.concatenate(all_t),tuple(slabs),receipt)


def _asset_identity(asset):
    _require(asset.get('version')=='r10_shared_native_TRAIN_mesh_v2' and asset.get('split')=='train'
        and asset.get('task') in ALLOWED_TASKS and asset.get('shared_with_baseline') is True
        and asset.get('world_pose_or_motion_schedule_in_asset') is False
        and asset.get('asset_sha256')==_digest({k:v for k,v in asset.items() if k!='asset_sha256'}),
        'immutable same-task TRAIN-local CAD identity/hash required')


def _certificate(pointer,task,root):
    path=_pointer(pointer,root);report=json.loads(path.read_text())
    _require(report.get('version')=='r10_TRAIN_bin_axis_yaw_swept_actual_CAD_diagnostic_v1'
        and report.get('complete') is True and report.get('TRAIN_only') is True
        and report.get('GT_pose_used') is False and report.get('native_simulator_started') is False
        and report.get('VAL_used') is False and report.get('TEST_used') is False,
        'completed native-pose-free TRAIN topology/surface evidence required')
    row=report['tasks'][task]
    _require(row['sampled_containment_failures']==0 and row['exact_certificate']['exact_axis_escape_for_every_retained_cavity'] is True,
        'source geometry certificate is not accepted')
    for component in row['component_reports']:
        if component['flags']['respondable']:
            shape=component['topology']
            _require(shape['all_edges_incident_to_two_faces'] is True
                and shape['all_shared_edges_oppositely_oriented'] is True and shape['degenerate_index_faces']==0,
                'respondable component lacks a closed oriented source shell')
    return report,row


def make_frozen_axis_swept_asset(source_asset_pointer,certificate_pointer,*,root=ROOT,construction_sources=()):
    source_path=_pointer(source_asset_pointer,root);source=json.loads(source_path.read_text());_asset_identity(source)
    task=source['task'];report,cert=_certificate(certificate_pointer,task,root);body=source['bodies']['bin']
    _require(cert['source_asset_sha256']==source_asset_pointer['sha256']
        and cert['TRAIN_asset_internal_sha256']==source['asset_sha256'],
        'source shell/surface certificate binds a different TRAIN asset')
    vertices=np.asarray(body['collision_vertices_body'],float);triangles=np.asarray(body['collision_triangles'],int)
    prototype=inflated_axis_yaw_sweep(vertices,triangles)
    exact=verify_stored_sweep_exact(vertices,triangles,prototype,inflation_m=prototype.receipt['total_axial_radial_inflation_m'])
    sources={str(Path(path).resolve().relative_to(Path(root).resolve())):_sha(path) for path in
        (Path(__file__),Path(__file__).with_name('bin_swept_geometry.py'),Path(__file__).with_name('collision_geometry.py'),
         *map(Path,construction_sources))}
    result=dict(schema=SCHEMA,TRAIN_only=True,task=task,body_id='bin',source_group=source['source_group'],
        source_asset=dict(source_asset_pointer),source_asset_internal_sha256=source['asset_sha256'],
        source_vertices_body=vertices.tolist(),source_triangles=triangles.tolist(),
        source_mesh_sha256=mesh_sha256(vertices,triangles),
        source_component_annotations=body['owned_component_annotations'],
        source_visual_component_names=body['visual_component_names'],source_collision_component_names=body['collision_component_names'],
        source_certificate=dict(certificate_pointer),source_material_and_open_surface_evidence=cert['component_reports'],
        analytic_proof=dict(path=report['analytic_proof'],sha256=report['inputs_sha256'][report['analytic_proof']]),
        engineering_inflation_proof=dict(path='artifacts/framework/multitask_r10/bin_axis_swept_inflation_proof_v1.txt',
            sha256=_sha(Path(root)/'artifacts/framework/multitask_r10/bin_axis_swept_inflation_proof_v1.txt')),
        vertices=prototype.vertices.tolist(),triangles=prototype.triangles.tolist(),slabs=list(prototype.slabs),
        mesh_sha256=mesh_sha256(prototype.vertices,prototype.triangles),construction_receipt=prototype.receipt,
        exact_inclusion_certificate=exact,construction_sources_sha256=sources,
        axis_local=[1.,0.,0.],axis_polarity='directed',requires_current_causal_axis_readiness=True,
        native_pose_or_future_state_used=False,calibrated_safety_guarantee=False,
        enabled_by_default=False,material_interpretation_conditional_on_non_self_intersecting_bounded_source=True)
    result['asset_sha256']=_digest(result);return result


def _reconstruct_and_verify(asset):
    v=[];t=[];offset=0
    for row in asset['slabs']:
        outer=np.asarray(row['outer_polygon_yz'],np.float32)
        inner=np.asarray(row['inner_polygon_yz'],np.float32) if row['inner_polygon_yz'] is not None else None
        low,high=row['stored_axial_interval'];vertices,triangles=_prism(low,high,outer,inner)
        v.append(vertices);t.append(triangles+offset);offset+=len(vertices)
    vertices=np.concatenate(v);triangles=np.concatenate(t)
    _require(np.array_equal(vertices,np.asarray(asset['vertices'],np.float32))
        and np.array_equal(triangles,np.asarray(asset['triangles'],np.int32))
        and mesh_sha256(vertices,triangles)==asset['mesh_sha256'],
        'registered float32 mesh does not equal its certified closed slab prisms')
    prototype=SweptPrototype(vertices,triangles,tuple(asset['slabs']),asset['construction_receipt'])
    receipt=asset['construction_receipt'];delta=receipt['total_axial_radial_inflation_m']
    _require(receipt['engineering_tilt_bound_degrees']==TILT_BOUND_DEGREES
        and receipt['engineering_translation_bound_m']==TRANSLATION_BOUND_M
        and receipt['source_vertex_migration_allowance_m']==SOURCE_MIGRATION_BOUND_M
        and receipt['engineering_pose_error_assumption'] is True and receipt['calibrated_safety_guarantee'] is False,
        'fixed engineering uncertainty assumptions changed')
    expected=inflated_axis_yaw_sweep(asset['source_vertices_body'],asset['source_triangles'])
    _require(expected.receipt==receipt and np.array_equal(expected.vertices,vertices)
        and np.array_equal(expected.triangles,triangles) and list(expected.slabs)==asset['slabs'],
        'frozen geometry differs from the fixed conservative construction')
    exact=verify_stored_sweep_exact(asset['source_vertices_body'],asset['source_triangles'],prototype,inflation_m=delta)
    _require(exact==asset['exact_inclusion_certificate'],'exact source/inflation inclusion certificate differs')
    return prototype


def load_axis_swept_collision_geometry(permission,registered_source_asset,*,root=ROOT,role_id='bin'):
    """Strict read-only loader; no automatic task/shape/polarity migration."""
    _require(isinstance(permission,Mapping) and permission.get('enabled') is True and role_id=='bin'
        and permission.get('axis_local')==[1.,0.,0.] and permission.get('axis_polarity')=='directed',
        'explicit enabled bin/direct-local-X artifact permission required')
    path=_pointer(permission,root);asset=json.loads(path.read_text());_asset_identity(registered_source_asset)
    _require(asset.get('schema')==SCHEMA and asset.get('asset_sha256')==_digest({k:v for k,v in asset.items() if k!='asset_sha256'})
        and asset.get('TRAIN_only') is True and asset.get('task')==registered_source_asset['task']
        and asset.get('body_id')=='bin' and asset.get('axis_local')==[1.,0.,0.] and asset.get('axis_polarity')=='directed'
        and asset.get('requires_current_causal_axis_readiness') is True
        and asset.get('native_pose_or_future_state_used') is False and asset.get('enabled_by_default') is False,
        'frozen same-task directed-axis sweep artifact identity differs')
    for name,expected in asset['construction_sources_sha256'].items():
        _require(_sha(Path(root)/name)==expected,'sweep construction source changed: '+name)
    for field in ('analytic_proof','engineering_inflation_proof'):_pointer(asset[field],root)
    source_path=_pointer(asset['source_asset'],root);frozen_source=json.loads(source_path.read_text());_asset_identity(frozen_source)
    _require(frozen_source['task']==asset['task'] and frozen_source['asset_sha256']==asset['source_asset_internal_sha256'],
        'frozen original TRAIN source identity changed')
    _,certificate=_certificate(asset['source_certificate'],asset['task'],root)
    _require(certificate['source_asset_sha256']==asset['source_asset']['sha256']
        and certificate['TRAIN_asset_internal_sha256']==asset['source_asset_internal_sha256']
        and certificate['component_reports']==asset['source_material_and_open_surface_evidence'],
        'frozen source component/material evidence differs')
    source_body=frozen_source['bodies']['bin'];current=registered_source_asset['bodies']['bin']
    _require(source_body['collision_vertices_body']==asset['source_vertices_body']
        and source_body['collision_triangles']==asset['source_triangles']
        and mesh_sha256(asset['source_vertices_body'],asset['source_triangles'])==asset['source_mesh_sha256'],
        'raw original source mesh is not bound to the derived asset')
    for field,stored in (('owned_component_annotations','source_component_annotations'),
        ('visual_component_names','source_visual_component_names'),('collision_component_names','source_collision_component_names')):
        _require(current[field]==asset[stored]==source_body[field], 'source component flags/topology permission changed')
    original=np.asarray(asset['source_vertices_body'],float);vertices=np.asarray(current['collision_vertices_body'],float)
    _require(vertices.shape==original.shape and np.isfinite(vertices).all()
        and np.array_equal(np.asarray(current['collision_triangles']),np.asarray(asset['source_triangles'])),
        'source mesh topology or vertex inventory changed; no implicit migration')
    bound2=Fraction.from_float(SOURCE_MIGRATION_BOUND_M)**2
    _require(all(sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2 for a,b in zip(first,second))<=bound2
        for first,second in zip(vertices,original)),'source local vertex change exceeds the explicit 1um enclosure bound')
    maximum=float(np.linalg.norm(vertices-original,axis=1).max())
    prototype=_reconstruct_and_verify(asset)
    receipt=dict(prototype.receipt,runtime_installed=True,collision_geometry_kind='directed_axis_yaw_uncertainty_sweep',
        orientation_observability='axis',source_frozen_sweep_asset=str(path.relative_to(Path(root).resolve())),
        source_frozen_sweep_asset_file_sha256=permission['sha256'],source_frozen_sweep_asset_internal_sha256=asset['asset_sha256'],
        frozen_source_TRAIN_asset_sha256=asset['source_asset_internal_sha256'],
        registered_source_TRAIN_asset_sha256=registered_source_asset['asset_sha256'],
        original_mesh_sha256=mesh_sha256(vertices,current['collision_triangles']),
        frozen_original_mesh_sha256=asset['source_mesh_sha256'],registered_mesh_sha256=asset['mesh_sha256'],
        registered_source_vertex_difference_max_m=maximum,source_vertex_migration_exact_rational_bound_passed=True,
        source_vertex_migration_bound_already_in_inflation=True,exact_inclusion_certificate=asset['exact_inclusion_certificate'],
        vertex_count=len(prototype.vertices),triangle_count=len(prototype.triangles),cavities_preserved=True,
        cavities_preserved_semantics='conservative subset after fixed engineering uncertainty inflation, not exact CAD',
        unknown_rotation_identity_is_not_exact_CAD=True,TRAIN_asset_unchanged=True)
    return CollisionGeometry(prototype.vertices,prototype.triangles,receipt)
