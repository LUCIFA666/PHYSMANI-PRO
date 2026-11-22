"""Audit immutable V6 hold/teacher rigid evidence without native execution.

Prefer each fixed TRAIN hold layout, but retain teacher counterevidence. Only
frame-to-peg/spoke may become a permanent relation. Recompute the original
target_valid fingerprint exactly; a false boundary sample never drops a source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.specs import TASK_SPECS
BASE=ROOT/'artifacts/framework/multitask_r10'
HOLD='TRAIN_hold_motion_identification'
TEACHER='TRAIN_teacher_committed_waypoint'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write(path,value):
    path=Path(path).resolve();require(path.is_relative_to(BASE.resolve()) and not path.exists(),'new TRAIN evidence output only')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_layout(layout,arrays):
    require(layout.get('version')=='r10_TRAIN_observed_rigid_layout_v1'
        and layout.get('split')=='train' and layout.get('shared_with_baseline') is True
        and layout.get('evidence_sha256')==digest({k:v for k,v in layout.items() if k!='evidence_sha256'}),
        'rigid artifact identity/boundary changed')
    require(layout['thresholds']==dict(minimum_samples=5,position_tolerance_m=.002,orientation_tolerance_rad=.02),
            'original rigid thresholds changed; no relaxation permitted')
    fingerprint=digest({key:np.asarray(arrays[key]).tolist() for key in
        ('body_ids','times','target_positions','target_rotations','target_valid')})
    require(fingerprint==layout['arrays_sha256'],'raw stored layout fingerprint differs')
    ids=list(map(str,arrays['body_ids']));valid=np.asarray(arrays['target_valid'])
    require(valid.dtype==np.bool_ and valid.shape==(len(arrays['times']),len(ids)), 'stored label validity changed')
    for relation in layout['accepted']:
        p,c=ids.index(relation['parent']),ids.index(relation['child'])
        mask=valid[:,p]&valid[:,c]
        require(relation['samples']==int(mask.sum())
            and relation['samples']>=5 and relation['position_error_max_m']<=.002
            and relation['orientation_error_max_rad']<=.02,'accepted relation violates original stored masks/thresholds')
        parent=np.asarray(arrays['target_rotations'])[mask,p];child=np.asarray(arrays['target_rotations'])[mask,c]
        local_positions=np.einsum('tji,tj->ti',parent,
            np.asarray(arrays['target_positions'])[mask,c]-np.asarray(arrays['target_positions'])[mask,p])
        local_rotations=np.einsum('tji,tjk->tik',parent,child)
        centre=np.median(local_positions,axis=0);rotation=Rotation.from_matrix(local_rotations).mean().as_matrix()
        position_error=float(np.linalg.norm(local_positions-centre,axis=1).max())
        angle_error=float(Rotation.from_matrix(local_rotations@rotation.T).magnitude().max())
        require(np.linalg.norm(centre-np.asarray(relation['child_to_parent_position']))<=1e-10
            and Rotation.from_matrix(rotation@np.asarray(relation['child_to_parent_rotation']).T).magnitude()<=1e-10
            and abs(position_error-relation['position_error_max_m'])<=1e-10
            and abs(angle_error-relation['orientation_error_max_rad'])<=1e-10,
            'stored rigid transform/error does not recompute from original masked TRAIN poses')
    return dict(frames=len(valid),first_frame_target_valid={body:bool(valid[0,j]) for j,body in enumerate(ids)},
        original_mask_fingerprint_recomputed=True,first_false_boundary_does_not_drop_source=True)


def select_frame_relations(hold,teacher):
    """A short hold cannot erase measured controlled non-rigidity."""
    teacher_rejections={(r['parent'],r['child']):r for r in teacher['rejected']}
    accepted=[];rejected=[]
    for row in hold['accepted']:
        pair=row['parent'],row['child']
        if not (pair[0]=='frame' and pair[1].startswith(('peg','spoke'))):
            rejected.append(dict(parent=pair[0],child=pair[1],reason='only independently TRAIN frame-to-peg/spoke permanent relations allowed',
                hold_positive_evidence_retained=True,teacher_counterevidence=teacher_rejections.get(pair)))
        elif pair in teacher_rejections:
            rejected.append(dict(parent=pair[0],child=pair[1],reason='teacher controlled evidence contradicts permanent rigidity',
                hold_positive_evidence_retained=True,teacher_counterevidence=teacher_rejections[pair]))
        else:accepted.append(row)
    return accepted,rejected


def audit(source_directory,output):
    source_directory=Path(source_directory).resolve();inputs={};cases={};layouts={};statuses={};audits={}
    for task in TASK_SPECS:
        selected={}
        for path in sorted((source_directory/'cases/train'/task).glob('*/case.json')):
            case=json.loads(path.read_text());entry=case.get('entry',{})
            require(case.get('status')=='completed' and case['split']=='train' and case['task']==task
                and case.get('policy_history_contains_native_GT') is False
                and entry.get('source_kind') in ('TRAIN_native_generated_augmentation','TRAIN_native_generated_hold'),
                'complete TRAIN evidence only')
            behavior=entry['behavior'];require(behavior in (HOLD,TEACHER) and behavior not in selected,'one fixed hold and teacher per task')
            require(entry['source_kind']==('TRAIN_native_generated_hold' if behavior==HOLD else 'TRAIN_native_generated_augmentation'),
                    'hold/teacher provenance labels differ')
            parent=case['parent_process_receipt'];original=ROOT/parent['child_case_path']
            require(sha(original)==parent['child_case_sha256'],'original child case changed')
            child=json.loads(original.read_text());require(all(case.get(k)==v for k,v in child.items()),'canonical TRAIN fields changed')
            inputs[str(path.resolve())]=sha(path);inputs[str(original.resolve())]=sha(original)
            for name,expected in case['files_sha256'].items():
                pointer=(ROOT/name).resolve();require(sha(pointer)==expected,'TRAIN raw evidence changed: '+name)
                inputs[str(pointer)]=expected
            pointers=[ROOT/name for name in case['files_sha256'] if Path(name).name=='TRAIN_rigid_layout.json']
            require(len(pointers)==1,'one original rigid-layout audit required')
            pointer=pointers[0].resolve();layout=json.loads(pointer.read_text())
            require(layout['task']==task and layout['source_group']==case['parent_source_group'],'layout task/source differs')
            with np.load(ROOT/entry['cache'],allow_pickle=False) as arrays:boundary=validate_layout(layout,arrays)
            selected[behavior]=dict(case=case,path=path,layout=layout,layout_path=pointer,boundary=boundary)
        require(set(selected)=={HOLD,TEACHER},'all 28 fixed TRAIN behavior cases required')
        hold,teacher=selected[HOLD],selected[TEACHER]
        require(hold['case']['parent_source_group']==teacher['case']['parent_source_group'],'hold/teacher parent differs')
        accepted,rejected=select_frame_relations(hold['layout'],teacher['layout'])
        audits[task]=dict(hold_layout_path=str(hold['layout_path'].relative_to(ROOT)),
            teacher_layout_path=str(teacher['layout_path'].relative_to(ROOT)),hold_boundary=hold['boundary'],teacher_boundary=teacher['boundary'],
            eligible_relations=[dict(parent=r['parent'],child=r['child']) for r in accepted],
            rejected_permanent_relations=rejected,hold_original_rejections=hold['layout']['rejected'],
            teacher_original_rejections=teacher['layout']['rejected'],
            source_selected_by_outcome=False)
        if accepted:
            require(len(accepted)==len(hold['layout']['accepted']),'do not silently load unaccepted relations from original layout')
            layouts[task]=dict(path=str(hold['layout_path'].relative_to(ROOT)),sha256=sha(hold['layout_path']),
                validated_on_TRAIN=True,source_group=hold['layout']['source_group'],
                source_case=str(hold['path'].relative_to(ROOT)),
                accepted_relations=[dict(parent=r['parent'],child=r['child']) for r in accepted],
                evidence_scope='one fixed independent task/source hold plus its teacher countercheck; multi-source validation still pending')
            statuses[task]='independent same-task TRAIN hold validated; original teacher counterevidence checked'
        else:statuses[task]='no eligible permanent frame relation; all raw positive/negative evidence retained'
        cases[task]={behavior:str(value['path'].relative_to(ROOT)) for behavior,value in selected.items()}
    inputs[str(Path(__file__).resolve())]=sha(__file__)
    result=dict(version='r10_completed_TRAIN_rigid_prior_manifest_v2_from_v6',TRAIN_only=True,TEST_used=False,
        source_directory=str(source_directory.relative_to(ROOT)),layout_map=layouts,task_status=statuses,
        case_audits=audits,source_cases=cases,inputs_sha256=inputs,
        source_script_sha256=sha(__file__),native_run_performed=False,task_success_used_for_prior_selection=False,
        per_task_transfer_permitted=False,original_layouts_modified=False,
        old_collection_current_source_reverification_performed=False,
        permanent_relation_whitelist='frame -> peg/spoke only; button-wrap hold cannot override teacher deformation')
    write(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-directory',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args();value=audit(args.source_directory,args.out)
    print(json.dumps(dict(TRAIN_only=True,audited_cases=28,layout_tasks=len(value['layout_map']))))


if __name__=='__main__':main()
