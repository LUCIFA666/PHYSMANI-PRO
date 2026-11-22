"""Offline TRAIN CAD-only validation of an uninstalled bin sweep prototype.

Reads the fixed ordinary/high V6 hold assets. No native simulator, GT poses,
WM, teacher, policy outcomes, VAL or TEST inputs are loaded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.bin_swept_geometry import (
    axis_yaw_swept_prototype,verify_stored_sweep_exact,points_inside_prototype,
)
BASE=ROOT/'artifacts/framework/multitask_r10'
VERSION='r10_TRAIN_bin_axis_yaw_swept_actual_CAD_diagnostic_v1'
TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def topology(vertices,triangles):
    """Audit the actual float32 mesh representation; no tolerance welding."""
    _,inverse=np.unique(np.asarray(vertices,np.float32),axis=0,return_inverse=True)
    faces=inverse[triangles];counts=Counter();orient=Counter()
    degenerate=0
    for face in faces:
        if len(set(map(int,face)))!=3:degenerate+=1
        for a,b in zip(face,np.roll(face,-1)):
            a,b=int(a),int(b);key=tuple(sorted((a,b)))
            counts[key]+=1;orient[key]+=1 if a<b else -1
    return dict(float32_identical_vertex_welding_only=True,degenerate_index_faces=degenerate,
        edge_incidence_histogram={str(k):v for k,v in sorted(Counter(counts.values()).items())},
        all_edges_incident_to_two_faces=bool(counts) and all(v==2 for v in counts.values()),
        all_shared_edges_oppositely_oriented=all(v==0 for v in orient.values()),
        self_intersection_checked=False,
        material_claim='bounded closed native material interpretation is conditional on non-self-intersecting source; '
                       'open collidable-only component interpreted as triangle surfaces')


def selected_asset(task,inputs):
    candidates=[]
    for path in (BASE/'TRAIN_smoke_collection_v6/cases/train'/task).glob('*/case.json'):
        case=json.loads(path.read_text())
        if case['entry']['behavior']=='TRAIN_hold_motion_identification':candidates.append((path,case))
    require(len(candidates)==1,'one fixed pre-existing TRAIN hold asset per task')
    path,case=candidates[0]
    require(case['status']=='completed' and case['split']=='train' and case['task']==task
        and case['teacher_actions_used'] is False and case['policy_history_contains_native_GT'] is False,
        'TRAIN hold identity changed')
    child=ROOT/case['parent_process_receipt']['child_case_path']
    require(sha(child)==case['parent_process_receipt']['child_case_sha256'],'original child record changed')
    original=json.loads(child.read_text());require(all(case.get(k)==v for k,v in original.items()),'canonical child fields changed')
    files=[ROOT/k for k in case['files_sha256'] if Path(k).name=='TRAIN_geometry.json']
    require(len(files)==1,'one immutable original TRAIN geometry required');asset_path=files[0]
    require(sha(asset_path)==case['files_sha256'][str(asset_path.relative_to(ROOT))],'TRAIN geometry file changed')
    asset=json.loads(asset_path.read_text());payload={k:v for k,v in asset.items() if k!='asset_sha256'}
    require(asset['version']=='r10_shared_native_TRAIN_mesh_v2' and asset['split']=='train'
        and asset['task']==task and asset['shared_with_baseline'] is True
        and asset['world_pose_or_motion_schedule_in_asset'] is False
        and hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==asset['asset_sha256'],
        'TRAIN CAD annotation/hash boundary changed')
    for pointer in (path,child,asset_path):inputs[str(pointer.relative_to(ROOT))]=sha(pointer)
    return path,case,asset_path,asset


def polygon_inradius(polygon):
    p=np.asarray(polygon,float);q=np.roll(p,-1,axis=0)
    return float(np.min((p[:,0]*q[:,1]-p[:,1]*q[:,0])/np.linalg.norm(q-p,axis=1)))


def run(output):
    output=Path(output).resolve();require(output.is_relative_to(BASE.resolve()) and not output.exists(),'fresh diagnostic directory only')
    inputs={};records={}
    audit_path=BASE/'component_flag_audit_v2'/TASKS[0]/'report.json'
    audit=json.loads(audit_path.read_text());require(audit['complete'] is True and audit['TRAIN_only'] is True
        and audit['counted_ticks']==0 and audit['native_property_setters_used'] is False,'fixed native flag audit identity differs')
    inputs[str(audit_path.relative_to(ROOT))]=sha(audit_path)
    stats={row['name']:row for row in audit['bodies']['bin']['components']}
    output.mkdir(parents=True)
    for task in TASKS:
        case_path,case,asset_path,asset=selected_asset(task,inputs);body=asset['bodies']['bin']
        vertices=np.asarray(body['collision_vertices_body'],float);triangles=np.asarray(body['collision_triangles'],int)
        annotations={row['name']:row for row in body['owned_component_annotations']}
        require(body['collision_component_names']==sorted(stats),'audited component ordering changed')
        offset=0;component_reports=[]
        for name in body['collision_component_names']:
            row=stats[name];annotation=annotations[name];count=row['mesh']['vertex_count']
            require(annotation['flags']==row['flags'] and annotation['native_getter_values']==row['native_getter_values']
                and (annotation['flags']['collidable'] or annotation['flags']['respondable']),
                'per-task native TRAIN component flags differ from audited native model')
            mask=np.all((triangles>=offset)&(triangles<offset+count),axis=1);faces=triangles[mask]-offset
            require(len(faces)==row['mesh']['triangle_count'],'component vertex-block/triangle inventory differs')
            shape=topology(vertices[offset:offset+count],faces)
            if annotation['flags']['respondable']:
                require(shape['all_edges_incident_to_two_faces'] and shape['all_shared_edges_oppositely_oriented']
                    and shape['degenerate_index_faces']==0,'respondable material lacks closed oriented shell evidence')
            component_reports.append(dict(name=name,flags=annotation['flags'],vertex_count=count,
                triangle_count=len(faces),topology=shape));offset+=count
        require(offset==len(vertices) and sum(r['triangle_count'] for r in component_reports)==len(triangles),
            'combined physical mesh has unaccounted component vertices/triangles')
        prototype=axis_yaw_swept_prototype(vertices,triangles,slab_count=32,angular_segments=64)
        exact=verify_stored_sweep_exact(vertices,triangles,prototype)
        rng=np.random.default_rng(7423)
        # Every triangle gets 16 independent points/yaws; all vertices get 16 yaws.
        faces=np.repeat(vertices[triangles],16,axis=0)
        points=(faces*rng.dirichlet([1.,1.,1.],len(faces))[:,:,None]).sum(1)
        points=np.r_[points,np.repeat(vertices,16,axis=0)]
        angles=rng.uniform(-np.pi,np.pi,len(points));y,z=points[:,1].copy(),points[:,2].copy()
        points[:,1]=y*np.cos(angles)-z*np.sin(angles);points[:,2]=y*np.sin(angles)+z*np.cos(angles)
        contained=points_inside_prototype(points,prototype);require(contained.all(),'random yaw triangle/vertex containment counterexample found')
        first,last=prototype.slabs[0],prototype.slabs[-1]
        cavity=[row for row in prototype.slabs if row['retained_cavity']]
        require(cavity and last['retained_cavity'] and last['axis_escape_positive_x']
            and not first['retained_cavity'],'actual bin must preserve +X mouth and closed -X bottom')
        axis_probes=np.asarray([[np.mean(first['source_axial_interval']),0.,0.],
                               [np.mean(last['source_axial_interval']),0.,0.]])
        require(points_inside_prototype(axis_probes,prototype).tolist()==[True,False], 'actual bottom/mouth axis probes differ')
        stem=output/task;stem.mkdir()
        np.savez_compressed(stem/'derived_mesh_and_samples.npz',vertices=prototype.vertices,triangles=prototype.triangles,
            yaw_sample_points=points,yaw_sample_contained=contained,axis_probes=axis_probes)
        write(stem/'slabs.json',list(prototype.slabs))
        records[task]=dict(source_case=str(case_path.relative_to(ROOT)),source_group=case['parent_source_group'],
            source_asset=str(asset_path.relative_to(ROOT)),source_asset_sha256=sha(asset_path),
            TRAIN_asset_internal_sha256=asset['asset_sha256'],component_reports=component_reports,
            source_vertices=len(vertices),source_triangles=len(triangles),
            source_local_bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
            prototype_receipt=prototype.receipt,exact_certificate=exact,
            sampled_points=len(points),sampled_containment_failures=int((~contained).sum()),
            aperture=dict(open_direction_local=[1.,0.,0.],last_slab_clear_disk_radius_m=polygon_inradius(last['inner_polygon_yz']),
                last_slab_clear_disk_diameter_m=2*polygon_inradius(last['inner_polygon_yz']),
                minimum_clear_radius_over_retained_cavity_slabs_m=min(polygon_inradius(s['inner_polygon_yz']) for s in cavity),
                retained_cavity_slabs=len(cavity),total_slabs=len(prototype.slabs),
                cavity_axial_interval=[cavity[0]['source_axial_interval'][0],cavity[-1]['source_axial_interval'][1]],
                negative_end_bottom_axis_occupied=True,positive_end_mouth_axis_free=True,
                body_centre_default_cube_occupied=True,body_centre_sweep_occupied=bool(points_inside_prototype([[0,0,0]],prototype)[0])),
            derived_npz=str((stem/'derived_mesh_and_samples.npz').relative_to(ROOT)),
            derived_npz_sha256=sha(stem/'derived_mesh_and_samples.npz'),
            slabs=str((stem/'slabs.json').relative_to(ROOT)),slabs_sha256=sha(stem/'slabs.json'))
        print(json.dumps(dict(task=task,exact_certificate=True,samples=len(points),cavities=len(cavity)),ensure_ascii=False),flush=True)
    proof=ROOT/'artifacts/framework/multitask_r10/bin_axis_yaw_swept_proof_v1.txt'
    for pointer in (Path(__file__),ROOT/'our_mani/multitask/bin_swept_geometry.py',
                    ROOT/'tests/test_bin_swept_geometry_prototype_r10.py',proof):
        inputs[str(pointer.relative_to(ROOT))]=sha(pointer)
    result=dict(version=VERSION,complete=True,TRAIN_only=True,VAL_used=False,TEST_used=False,
        teacher_used=False,native_simulator_started=False,GT_pose_used=False,policy_outcomes_used=False,
        default_cube_changed=False,runtime_installed=False,orientation_annotation_promoted=False,
        fixed_discretization=dict(axial_slabs=32,angular_segments=64,random_seed=7423),tasks=records,
        inputs_sha256=inputs,analytic_proof=str(proof.relative_to(ROOT)),
        proposed_readiness=dict(axis_local=[1.,0.,0.],axis_observable=True,axis_polarity='directed',
            current_causal_RGBD_evidence_required=True,unknown_axis_action='fail closed; retain default full-rotation cube'),
        limitations=['One fixed TRAIN CAD asset for each speed; no contact/native collision-engine experiment.',
            'This is a conservative yaw-swept approximation, not exact CAD or a proof of pose-estimation accuracy.',
            'Closed oriented edge incidence is checked; source self-intersection and simulator volume conventions are not proved.',
            'Open collidable visual mesh is enclosed as triangle surfaces; respondable closed shell supports conditional material interpretation.',
            'Current directed axis and polarity must pass independent perception evidence/accuracy validation before runtime use.',
            'No native live orientation, future state, goal or held attachment is supplied to the construction.',
            'Contained surfaces alone would be insufficient for material: every retained cavity also has an exact axis escape certificate.'])
    write(output/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    result=run(parser.parse_args().out)
    print(json.dumps(dict(complete=result['complete'],tasks=len(result['tasks']))))
