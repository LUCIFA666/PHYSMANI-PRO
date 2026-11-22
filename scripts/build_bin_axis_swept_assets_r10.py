"""Freeze two explicit TRAIN bin yaw-sweep assets; no simulator or policy run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.bin_swept_assets import (
    ALLOWED_TASKS,SCHEMA,make_frozen_axis_swept_asset,load_axis_swept_collision_geometry,
)

BASE=ROOT/'artifacts/framework/multitask_r10'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('all inputs/outputs must remain inside our-mani')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run(output,certificate):
    output=Path(output).resolve();certificate=Path(certificate).resolve()
    if not output.is_relative_to(BASE) or output.exists():raise ValueError('fresh R10 artifact directory required')
    report=json.loads(certificate.read_text())
    if set(report.get('tasks',{}))!=set(ALLOWED_TASKS):raise ValueError('fixed ordinary/high TRAIN certificate required')
    certificate_pointer=pointer(certificate);output.mkdir(parents=True)
    permissions={};tasks={};inputs={certificate_pointer['path']:certificate_pointer['sha256']}
    for task in ALLOWED_TASKS:
        row=report['tasks'][task];source=ROOT/row['source_asset'];source_pointer=pointer(source)
        if source_pointer['sha256']!=row['source_asset_sha256']:raise ValueError('certified raw TRAIN source changed')
        asset=make_frozen_axis_swept_asset(source_pointer,certificate_pointer,
            construction_sources=(Path(__file__),))
        path=output/(task+'.json');write(path,asset)
        permit=dict(enabled=True,**pointer(path),axis_local=[1.,0.,0.],axis_polarity='directed')
        geometry=load_axis_swept_collision_geometry(permit,json.loads(source.read_text()))
        permissions[task]={'bin':permit}
        cavities=[r for r in asset['slabs'] if r['retained_cavity']]
        tasks[task]=dict(source_asset=source_pointer,sweep_asset=pointer(path),
            asset_internal_sha256=asset['asset_sha256'],mesh_sha256=asset['mesh_sha256'],
            vertices=len(geometry.vertices),triangles=len(geometry.triangles),
            source_max_radius_float32_upper_m=asset['construction_receipt']['source_max_radius_float32_upper_m'],
            total_axial_radial_inflation_m=asset['construction_receipt']['total_axial_radial_inflation_m'],
            retained_cavity_slabs=len(cavities),
            minimum_retained_radial_exclusion_bound_m=min(r['radial_lower_bound'] for r in cavities) if cavities else None,
            roundtrip_loader_verified=True,exact_inclusion_certificate=asset['exact_inclusion_certificate'])
        inputs[source_pointer['path']]=source_pointer['sha256']
        print(json.dumps(dict(task=task,roundtrip_loader_verified=True,
            retained_cavity_slabs=len(cavities),inflation_m=tasks[task]['total_axial_radial_inflation_m'])),flush=True)
    for path in (Path(__file__),ROOT/'our_mani/multitask/bin_swept_assets.py',
                 ROOT/'our_mani/multitask/bin_swept_geometry.py',ROOT/'our_mani/multitask/collision_geometry.py'):
        item=pointer(path);inputs[item['path']]=item['sha256']
    write(output/'axis_swept_assets.json',permissions)
    result=dict(version='r10_frozen_TRAIN_bin_directed_axis_sweep_build_v1',schema=SCHEMA,
        complete=True,TRAIN_only=True,native_simulator_started=False,VAL_used=False,TEST_used=False,
        policy_outcomes_used=False,native_pose_or_future_state_used=False,
        enabled_by_default=False,runtime_integration_verified=False,current_causal_directed_axis_gate_required=True,
        engineering_pose_error_assumption=True,calibrated_safety_guarantee=False,
        fixed_tilt_bound_degrees=10.,fixed_translation_bound_m=.005,source_vertex_migration_bound_m=1e-6,
        axis_swept_assets=permissions,permission_file=pointer(output/'axis_swept_assets.json'),
        tasks=tasks,inputs_sha256=inputs,
        limitations=['Fixed engineering pose error bounds are assumptions, not calibrated guarantees.',
            'Source geometry, closed-shell evidence and conservative enclosure are checked offline; native collision/contact is not tested here.',
            'Runtime must choose the conservative full-rotation fallback whenever current directed-axis evidence is unavailable.',
            'The source closed-material interpretation remains conditional on non-self-intersection.'])
    write(output/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(BASE/'bin_axis_swept_assets_v1'))
    parser.add_argument('--certificate',default=str(BASE/'bin_swept_geometry_prototype_v1/report.json'))
    arguments=parser.parse_args();result=run(arguments.out,arguments.certificate)
    print(json.dumps(dict(complete=result['complete'],tasks=len(result['tasks']))))
