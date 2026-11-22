"""Fit rigid tool-to-ring transforms from all declared TRAIN acquisition parents."""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
BASE=ROOT/'artifacts/framework/multitask_r10'


def compose(a,b):return Pose(a.position+a.rotation@b.position,a.rotation@b.rotation)
def inverse(a):return Pose(-a.rotation.T@a.position,a.rotation.T)


def main():
    out=BASE/'ring_grasp_prior_v419'
    if out.exists():raise FileExistsError('preserve previously fitted priors')
    manifest_path=BASE/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/TRAIN_fit_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['fit_split']!='train' or manifest['VAL_used'] or manifest['TEST_used']:raise ValueError('TRAIN only')
    files={str(manifest_path):file_sha(manifest_path),str(Path(__file__)):file_sha(__file__)}
    for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
        config_path=BASE/'registered_tasks_v408_setup'/(task+'_configuration.json')
        c=json.loads(config_path.read_text());lock=json.loads((config_path.parent/'frozen_inputs.json').read_text())['files']
        verify_inputs(ROOT,lock)
        pointer=c['ring_affordances'][task]
        if file_sha(ROOT/pointer['path'])!=pointer['sha256']:raise ValueError('ring permission changed')
        permission=json.loads((ROOT/pointer['path']).read_text())
        sp=permission['finite_symmetry'];cert=json.loads((ROOT/sp['path']).read_text())
        if file_sha(ROOT/sp['path'])!=sp['sha256']:raise ValueError('TRAIN finite symmetry changed')
        matrices=np.asarray(cert['group_rotations_body'],float);center=np.asarray(cert['center_body'],float)
        group=[Pose(center-r@center,r) for r in matrices]
        rows=[r for r in manifest['entries'] if r['task']==task and r.get('teacher_used',True)
            and r['source_kind']=='TRAIN_native_generated_augmentation']
        parents=defaultdict(list);counts=defaultdict(int);missing=[];anchor=None
        if len({r['parent_source_group'] for r in rows})!=100:raise ValueError('all100 original parents required')
        for row in rows:
            path=ROOT/row['cache'];sha=file_sha(path)
            if sha!=row['cache_sha256'] or lock.get(str(path))!=sha:raise ValueError('original locked TRAIN label cache changed')
            files[str(path)]=sha
            with np.load(path,allow_pickle=False) as arrays:
                ids=list(map(str,arrays['body_ids']));ring=ids.index('ring')
                mask=(arrays['native_pose_valid'][:,ring]&arrays['mode_valid'][:,ring]&
                    (arrays['modes'][:,ring]==1)&arrays['robot_contact']&(arrays['teacher_applied_opening']<.5))
                for i in np.flatnonzero(mask):
                    tcp=Pose.from_xyzw(arrays['measured_tcp_xyzw'][i])
                    body=Pose(arrays['target_positions'][i,ring],arrays['target_rotations'][i,ring])
                    local=compose(inverse(tcp),body)
                    if anchor is None:anchor=local
                    alternatives=[compose(local,s) for s in group]
                    chosen=min(alternatives,key=lambda p:Rotation.from_matrix(p.rotation@anchor.rotation.T).magnitude())
                    parents[row['parent_source_group']].append(chosen);counts[row['parent_source_group']]+=1
        if len(parents)<8:raise ValueError('insufficient observed TRAIN capture support')
        per_parent=[]
        for parent,poses in sorted(parents.items()):
            mean=Pose(np.mean([p.position for p in poses],axis=0),Rotation.from_matrix(
                np.asarray([p.rotation for p in poses])).mean().as_matrix())
            per_parent.append((parent,mean))
        mean=Pose(np.mean([p.position for _,p in per_parent],axis=0),Rotation.from_matrix(
            np.asarray([p.rotation for _,p in per_parent])).mean().as_matrix())
        errors=[dict(parent=parent,position_error_m=float(np.linalg.norm(p.position-mean.position)),
            orientation_error_rad=float(Rotation.from_matrix(p.rotation@mean.rotation.T).magnitude()),
            qualified_held_frames=counts[parent]) for parent,p in per_parent]
        all_parents=sorted({r['parent_source_group'] for r in rows})
        model=dict(schema='r10_all_parent_TRAIN_rigid_ring_grasp_prior_v1',task=task,
            tool_to_ring_xyzw=mean.xyzw().tolist(),D4_center_body=center.tolist(),D4_group_rotations_body=matrices.tolist(),
            TRAIN_manifest=dict(path=str(manifest_path.relative_to(ROOT)),sha256=file_sha(manifest_path)),
            original_declared_parent_count=len(all_parents),declared_parents=all_parents,
            captured_parent_count=len(parents),missing_capture_parents=sorted(set(all_parents)-set(parents)),
            all_captured_parents_used_equally=True,native_success_used_for_selection=False,
            sample_masks='offline TRAIN native_pose_valid/mode_valid/attached/public_contact/applied_closed; no success or completion mask',
            offline_native_labels_used_for_training_only=True,parent_residuals=errors,
            runtime_requires_current_public_measured_attachment=True,
            original_grasp_physics_collision_and_native_success_unchanged=True,
            finite_symmetry=sp,source_sha256=file_sha(__file__),formal_VAL_TEST=False,VAL_used=False,TEST_used=False)
        atomic_json(out/task/'model.json',model)
        print(json.dumps(dict(task=task,captured_parents=len(parents),total_parents=100,
            local_translation=mean.position.tolist(),maximum_parent_position_residual=max(e['position_error_m'] for e in errors),
            maximum_parent_orientation_residual=max(e['orientation_error_rad'] for e in errors))),flush=True)
    atomic_json(out/'inputs.json',dict(files=files,TRAIN_only=True,VAL_used=False,TEST_used=False))


if __name__=='__main__':main()
