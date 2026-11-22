"""Audit an initial-only RGBD branch prior on all original 200 TRAIN parents.

Native poses are scored here only; they are never an online prior input.
No cutoff is fitted or searched and no unsuccessful parent is removed.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.two_center_pose import load_two_center_pose_prior,estimate_two_center_pose,PublicCenter,checked_json
BASE=ROOT/'artifacts/framework/multitask_r10';OUT=BASE/'buzz_initial_branch_source_audit_v61/report.json'

def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))

def main():
    if OUT.exists():raise RuntimeError('preserve previous audit')
    tasks={};inputs={str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}
    for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed'):
        model=BASE/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/models'/task/'model.json'
        m=json.loads(model.read_text());prior=load_two_center_pose_prior(ROOT,pointer(model),canonical_asset_pointer=m['canonical_basis']['TRAIN_CAD'])
        receipt=checked_json(ROOT,m['TRAIN_evidence']['training_model_receipt']);rows=[]
        for pt in [pointer(model),m['canonical_basis']['TRAIN_CAD'],m['helper_source'],*m['TRAIN_evidence'].values(),receipt['actual_registration_list']]:
            assert file_sha(ROOT/pt['path'])==pt['sha256'];inputs[pt['path']]=pt['sha256']
        for source in receipt['actual_case_provenance']:
            for key in ('canonical_receipt','raw_child_receipt','actual_parent_receipt','cache'):
                p=source[key];assert file_sha(ROOT/p['path'])==p['sha256'];inputs[p['path']]=p['sha256']
            with np.load(ROOT/source['cache']['path'],allow_pickle=False) as z:
                assert float(z['times'][0])==0. and bool(z['robot_gripper_open'][0]) and not bool(z['robot_contact'][0])
                ids=z['body_ids'].tolist();centers={role:PublicCenter(z['positions'][0,ids.index(role)],bool(z['valid'][0,ids.index(role)]),
                    bool(z['mode_valid'][0,ids.index(role)]),'external',0.) for role in ('frame','wand')}
                assert all(int(z['modes'][0,ids.index(role)])==0 for role in centers)
                estimate=estimate_two_center_pose(prior,centers,current_timestamp=0.);assert estimate.valid
                errors={role:float(Rotation.from_matrix(getattr(estimate,role+'_rotation')@z['target_rotations'][0,ids.index(role)].T).magnitude()) for role in centers}
                rows.append(dict(parent_source_group=source['parent_source_group'],cache=source['cache'],initial_timestamp=0.,
                    errors_rad=errors,passes_existing_point15_rad_gate=all(v<=.15 for v in errors.values())))
        assert len(rows)==100
        tasks[task]=dict(original_parents=100,all_parents_retained=True,all_initials_passed=all(r['passes_existing_point15_rad_gate'] for r in rows),
            maximum_initial_error_rad={role:max(r['errors_rad'][role] for r in rows) for role in ('frame','wand')},rows=rows)
    report=dict(schema='r10_buzz_initial_only_public_branch_prior_source_audit_v61',tasks=tasks,
        all_200_original_TRAIN_parents_retained=True,all_initials_passed=all(t['all_initials_passed'] for t in tasks.values()),
        initial_timestamp_only=0.,prior_after_action_authorized=False,online_native_labels_used=False,
        conditional_on_original_initialization=True,original_angular_tolerance_rad=.15,threshold_search=False,
        runtime_orientation_permission_granted=False,inputs_sha256=inputs)
    atomic_json(OUT,report);print(json.dumps(dict(report=str(OUT.relative_to(ROOT)),all_initials_passed=report['all_initials_passed'],
        tasks={task:{k:v for k,v in row.items() if k!='rows'} for task,row in tasks.items()})))

if __name__=='__main__':main()
