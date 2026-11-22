"""Calibrate the author's missing wp0 intent from every original TRAIN parent.

Use all valid original wp0 command IDs, balanced by parent. Source parents with
no wp0 command remain explicitly listed; no outcome or completion selects data.
"""
from collections import defaultdict
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask.evaluation import atomic_json,file_sha


def main():
    base=ROOT/'artifacts/framework/multitask_r10';out=base/'remove_wp0_command_v636'
    if out.exists():raise FileExistsError('preserve every calibration version')
    source=base/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/TRAIN_fit_manifest.json'
    manifest=json.loads(source.read_text())
    if manifest['fit_split']!='train' or manifest['VAL_used'] or manifest['TEST_used']:raise ValueError('original TRAIN only')
    files={str(source):file_sha(source),str(Path(__file__)):file_sha(__file__)};reports=[]
    for task in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'):
        rows=[r for r in manifest['entries'] if r['task']==task and r.get('teacher_used',True)
            and r['source_kind']=='TRAIN_native_generated_augmentation']
        declared=sorted({r['parent_source_group'] for r in rows})
        if len(declared)!=100:raise ValueError('all100 original TRAIN parents must remain declared')
        parents=defaultdict(list);counts=defaultdict(int)
        for row in rows:
            path=ROOT/row['cache']
            if file_sha(path)!=row['cache_sha256']:raise ValueError('immutable original TRAIN cache changed')
            files[str(path)]=row['cache_sha256']
            with np.load(path,allow_pickle=False) as a:
                ids=list(map(str,a['body_ids']));seen=set()
                indices=np.flatnonzero((a['teacher_stage']=='wp0')&a['teacher_action_valid'])
                for i in indices:
                    command=int(a['teacher_command_id'][i])
                    if command in seen:continue
                    seen.add(command);body=str(a['teacher_reference_body'][i])
                    if (str(a['teacher_reference_role'][i])!='selected_cup' or body not in ids
                        or str(a['teacher_reference_kind'][i])!='observed_se3'):
                        raise ValueError('wp0 must retain original selected-cup reference annotation')
                    j=ids.index(body)
                    if not a['native_pose_valid'][i,j]:raise ValueError('original wp0 native TRAIN reference missing')
                    ref=Pose(a['target_positions'][i,j],a['target_rotations'][i,j]);goal=Pose.from_xyzw(a['teacher_tcp_xyzw'][i])
                    parents[row['parent_source_group']].append(Pose(ref.inverse_apply(goal.position),ref.rotation.T@goal.rotation))
                    counts[row['parent_source_group']]+=1
        if len(parents)<8:raise ValueError('at least eight independent original command parents required')
        means=[(parent,Pose(np.mean([p.position for p in poses],axis=0),Rotation.from_matrix(
            np.asarray([p.rotation for p in poses])).mean().as_matrix())) for parent,poses in sorted(parents.items())]
        pose=Pose(np.mean([p.position for _,p in means],axis=0),Rotation.from_matrix(np.asarray([p.rotation for _,p in means])).mean().as_matrix())
        residuals=[dict(parent=parent,position_error_m=float(np.linalg.norm(p.position-pose.position)),
            angle_error_rad=float(Rotation.from_matrix(p.rotation@pose.rotation.T).magnitude()),commands=counts[parent]) for parent,p in means]
        position=max(r['position_error_m'] for r in residuals);angle=max(r['angle_error_rad'] for r in residuals)
        if position>.03 or angle>.15:raise ValueError('original3cm/.15rad command reconstruction caps failed')
        model=dict(schema='r10_all_original_TRAIN_remove_wp0_command_geometry_v636',task=task,
            source_stage='wp0',reference_role='selected_cup',reference_kind='observed_se3',
            local_pose_xyzw=pose.xyzw().tolist(),desired_opening=1.,lookahead_seconds=0.,
            declared_parents=declared,declared_parent_count=100,independent_command_parent_support=len(parents),
            missing_wp0_command_parents=sorted(set(declared)-set(parents)),all_available_wp0_command_parents_used_equally=True,
            original_failed_censored_parents_retained=True,native_success_or_completion_used_for_selection=False,
            original_position_cap_m=.03,original_orientation_cap_rad=.15,maximum_parent_position_error_m=position,
            maximum_parent_angle_error_rad=angle,parent_residuals=residuals,
            source_manifest=dict(path=str(source.relative_to(ROOT)),sha256=file_sha(source)),
            completed_skill_support_claimed=False,physical_task_success_capability_claimed=False,
            source_sha256=file_sha(__file__),VAL_used=False,TEST_used=False)
        atomic_json(out/task/'model.json',model);reports.append(dict(task=task,declared_parents=100,
            command_parents=len(parents),maximum_position_error_m=position,maximum_angle_error_rad=angle))
    atomic_json(out/'inputs.json',dict(files=files,TRAIN_only=True,VAL_used=False,TEST_used=False))
    atomic_json(out/'report.json',dict(complete=True,tasks=reports,independent_source_command_models=True,
        every_declared_parent_retained=True,no_success_selection=True,VAL_used=False,TEST_used=False))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
